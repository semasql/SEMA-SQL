import json
import re
import os
import ast
import sqlite3
import yaml
from typing import Optional, Dict, Tuple
from pathlib import Path

from pandas.core.arrays import boolean

# Import symbolic verification (optional - will handle ImportError gracefully)
try:
    from .symbolic_verification import symbolic_verify
    SYMBOLIC_VERIFICATION_AVAILABLE = True
except ImportError:
    SYMBOLIC_VERIFICATION_AVAILABLE = False
    symbolic_verify = None

class PlanNode:
    """
    Structure of a node in the query plan.
    """
    def __init__(self, node_dict: dict, downstream_node=None):
        self.node_type = node_dict.get("Node Type", "Unknown")
        
        attributes = node_dict.copy()
        attributes.pop("Node Type", None)
        attributes.pop("Input", None)
        attributes.pop("Inputs", None)  
        self.attribute_dict = attributes

        # The node that consumes this node's output (its parent in the JSON tree)
        self.downstream = downstream_node
        
        # A list of nodes that provide input to this node (its children in the JSON tree)
        self.upstream = []
        
        # Cardinality tracking for cost estimation
        self.input_cardinality: Optional[int] = None
        self.output_cardinality: Optional[int] = None

    def __repr__(self):
        """String representation for plan node."""
        return f"<PlanNode: {self.node_type}, Upstream: {len(self.upstream)}>"
    
    def has_udf(self) -> bool:
        """Check if this node has a UDF."""
        return 'UDF' in self.attribute_dict


class CostModel:
    """
    Cost model for query optimization.
    Loads cost coefficients and selectivity estimates from configuration file.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize cost model from configuration file.
        
        Args:
            config_path: Path to cost_config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to cost_config.yaml in the conf directory
            package_dir = Path(__file__).parent.parent
            config_path = package_dir / "conf" / "cost_config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cost_per_row = self.config['cost_per_row']
        self.selectivity = self.config['selectivity']
        self.join_cost = self.config.get('join_cost', {})
    
    def get_cost_coefficient(self, operator_type: str, has_udf: bool = False) -> float:
        """
        Get cost coefficient for an operator type.
        
        Args:
            operator_type: Type of operator (selection, projection, join, etc.)
            has_udf: Whether the operator has a UDF
            
        Returns:
            Cost coefficient per row
        """
        if has_udf:
            key = f"{operator_type}_udf"
        else:
            key = operator_type.lower()
        
        return self.cost_per_row.get(key, 0.001)
    
    def get_selectivity(self, operator_type: str, has_udf: bool = False) -> float:
        """
        Get selectivity estimate for an operator type.
        
        Args:
            operator_type: Type of operator
            has_udf: Whether the operator has a UDF
            
        Returns:
            Selectivity (fraction of rows passing through)
        """
        if has_udf:
            key = f"{operator_type}_udf"
        else:
            key = operator_type.lower()
        
        return self.selectivity.get(key, 1.0)
    
    def estimate_output_cardinality(self, node: PlanNode, input_cardinality: int) -> int:
        """
        Estimate output cardinality after applying operator.
        
        Args:
            node: The operator node
            input_cardinality: Input cardinality
            
        Returns:
            Estimated output cardinality
        """
        if input_cardinality is None or input_cardinality <= 0:
            return 0
        
        selectivity = self.get_selectivity(node.node_type, node.has_udf())
        return max(1, int(input_cardinality * selectivity))
    
    def compute_operator_cost(self, node: PlanNode, input_cardinality: Optional[int] = None) -> float:
        """
        Compute execution cost for an operator.
        
        Args:
            node: The operator node
            input_cardinality: Input cardinality (if None, uses node.input_cardinality)
            
        Returns:
            Estimated cost
        """
        if input_cardinality is None:
            input_cardinality = node.input_cardinality
        
        if input_cardinality is None or input_cardinality <= 0:
            return 0.0
        
        operator_type = node.node_type.lower()
        has_udf = node.has_udf()
        cost_coeff = self.get_cost_coefficient(operator_type, has_udf)
        
        # Special handling for join operators
        if operator_type == 'join':
            if has_udf:
                # UDF join: consider smart batching efficiency
                # For now, assume nested loop with smart batching
                left_card = node.upstream[0].output_cardinality if node.upstream and node.upstream[0].output_cardinality else input_cardinality
                right_card = node.upstream[1].output_cardinality if len(node.upstream) > 1 and node.upstream[1].output_cardinality else input_cardinality
                batch_efficiency = self.join_cost.get('smart_batch_efficiency', 0.1)
                cost = cost_coeff * (left_card * right_card) * batch_efficiency
            else:
                # Hash join: |L| + |R|
                left_card = node.upstream[0].output_cardinality if node.upstream and node.upstream[0].output_cardinality else input_cardinality
                right_card = node.upstream[1].output_cardinality if len(node.upstream) > 1 and node.upstream[1].output_cardinality else input_cardinality
                hash_factor = self.join_cost.get('hash_join_factor', 1.0)
                cost = cost_coeff * (left_card + right_card) * hash_factor
        else:
            # Unary operators: cost = coefficient * input_cardinality
            cost = cost_coeff * input_cardinality
        
        return cost
    
    def compute_subtree_cost(self, node: PlanNode) -> float:
        """
        Compute total cost of executing a subtree rooted at node.
        Recursively computes cost of all nodes in the subtree.
        
        Args:
            node: Root of the subtree
            
        Returns:
            Total cost of the subtree
        """
        if not node:
            return 0.0
        
        # Compute cost of current node
        node_cost = self.compute_operator_cost(node)
        
        # Recursively compute cost of upstream nodes
        upstream_cost = sum(self.compute_subtree_cost(child) for child in node.upstream)
        
        return node_cost + upstream_cost


def udf_deferral(plan: str, db_name_param: str, path_param: str, cost_model: Optional[CostModel] = None) -> str:
    """
    Optimize query plan by deferring UDF execution.
    
    Args:
        plan: JSON string of the query plan
        db_name_param: Database name
        path_param: Path to database
        cost_model: Optional cost model for cost-based optimization
        
    Returns:
        Optimized query plan as JSON string
    """
    global db_name, path
    db_name = db_name_param
    path = path_param
    root_plan_node = parse_plan_to_tree(plan)
    reordered_root_plan_node = reorder_udf_nodes_in_tree(root_plan_node, cost_model)
    return reconstruct_plan_from_tree(reordered_root_plan_node)

def reorder_udf_nodes_in_tree(root: PlanNode, cost_model: Optional[CostModel] = None) -> PlanNode:
    """
    Reorders the PlanNode tree by moving UDF nodes upstream (towards the root)
    as long as swap conditions are met. Uses cost-based optimization if cost_model is provided.

    Args:
        root: The root node of the PlanNode tree.
        cost_model: Optional cost model for cost-based optimization.

    Returns:
        The new root of the modified tree.
    """
    if not root:
        return None

    # Initialize cost model if not provided
    if cost_model is None:
        try:
            cost_model = CostModel()
        except Exception:
            # If cost model initialization fails, continue without it
            cost_model = None

    # Step 1: Collect all nodes in a stable order (post-order traversal)
    # This prevents issues from modifying the tree while traversing it.
    nodes_to_check = []
    
    def post_order_collect(node):
        if not node: return
        for child in node.upstream:
            post_order_collect(child)
        nodes_to_check.append(node)
        
    post_order_collect(root)
    nodes_to_check.reverse()

    # Step 2: Iterate through the collected nodes and apply the reordering logic
    for node in nodes_to_check:
        if 'UDF' in node.attribute_dict:
            # Keep swapping the UDF node with its downstream as long as rules allow
            current_udf_node = node
            while current_udf_node.downstream and should_swap(current_udf_node, current_udf_node.downstream, cost_model) and 'UDF' not in current_udf_node.downstream.attribute_dict:
                parent = current_udf_node.downstream
                _swap_with_downstream(current_udf_node)
                # If the parent was the root, the UDF node is the new root
                if parent == root:
                    root = current_udf_node

    return root


def should_swap(udf_node: PlanNode, downstream_node: PlanNode, cost_model: Optional[CostModel] = None, verify: bool = True) -> bool:
    """
    Determines if a UDF node should be swapped with its downstream consumer.
    Uses semantic rules, symbolic verification, and cost-based optimization.
    
    Args:
        udf_node: The UDF node to potentially swap
        downstream_node: The downstream node to swap with
        cost_model: Optional cost model for cost-based decisions. If None, uses rule-based only.
        verify: If True, perform symbolic verification after rule-based checks pass.
    
    Returns:
        True if swap should be performed, False otherwise
    """
    # Step 1: Fast rule-based rejection (must pass)
    if not _check_semantic_swap_rules(udf_node, downstream_node):
        return False
    
    # Step 2: Symbolic verification (safety net for cases that pass rules)
    if verify:
        if not SYMBOLIC_VERIFICATION_AVAILABLE:
            # Z3 not available - skip verification
            print("Symbolic verification: SKIPPED (z3-solver not available)")
        else:
            try:
                is_equivalent = symbolic_verify(udf_node, downstream_node)
                if not is_equivalent:
                    # Rules said OK, but symbolic check failed - reject swap
                    return False
            except Exception as e:
                # On any error, conservatively reject the swap
                print(f"Symbolic verification: EXCEPTION - {e}")
                return False
    
    # Step 3: Cost-based decision (if cost model provided)
    if cost_model is not None:
        return _should_swap_cost_based(udf_node, downstream_node, cost_model)
    
    # Fallback to rule-based decision (original behavior)
    return True


def _check_semantic_swap_rules(udf_node: PlanNode, downstream_node: PlanNode) -> bool:
    """
    Check if swap preserves semantic equivalence based on rules.
    This is the original rule-based logic.
    """
    if udf_node.node_type == 'Selection':
        if downstream_node.node_type in ['Aggregation', 'TopK']:
            return False
        
        if downstream_node.node_type in ['Join', 'Selection']:
            return True
        

        if downstream_node.node_type == 'Projection':
            derived_projection_cols, source_projection_cols = parse_projection_columns(downstream_node.attribute_dict.get("Columns", ""))
            downstream_output_cols = set(derived_projection_cols) | set(source_projection_cols)
            
            udf_cols = set(udf_node.attribute_dict.get("UDF", {}).get("Input Columns", []))
            if not udf_cols:
                return False
            return udf_cols.issubset(downstream_output_cols)
        
    elif udf_node.node_type == 'Projection':
        udf_projection_source_cols = udf_node.attribute_dict.get("UDF", {}).get("Input Columns", [])
        udf_projection_derived_col = udf_node.attribute_dict.get("UDF", {}).get("Output Column", "")
        udf_projection_source_cols_set = set(udf_projection_source_cols)
        
        if downstream_node.node_type == 'Aggregation':
            return False
        
        if downstream_node.node_type == 'TopK':
            orderby_cols = extract_column_names_from_Topk_input(downstream_node.attribute_dict.get("Ranking criteria", ""))
            orderby_set = set(orderby_cols)
            return udf_projection_derived_col not in orderby_set
        
        if downstream_node.node_type == 'Join': 
            join_cols = extract_join_columns(downstream_node.attribute_dict.get("Condition", ""))
            join_set = set(join_cols)
            return udf_projection_derived_col not in join_set
        
        if downstream_node.node_type == 'Selection': 
            selection_cols = extract_columns_from_selection(downstream_node.attribute_dict.get("Condition", ""))
            selection_set = set(selection_cols)
            return udf_projection_derived_col not in selection_set
           
        if downstream_node.node_type == 'Projection':
            derived_projection_cols, source_projection_cols = parse_projection_columns(downstream_node.attribute_dict.get("Columns", ""))
            downstream_output_cols = set(derived_projection_cols) | set(source_projection_cols)
            
            if udf_projection_derived_col in downstream_output_cols:
                return False
            elif not udf_projection_source_cols_set.issubset(downstream_output_cols):
                return False
            else:
                return True
                      
    elif udf_node.node_type == 'Join':
        if downstream_node.node_type in ['Aggregation','TopK', 'Selection']:
            return False
        
        if downstream_node.node_type in ['Join']:
            return True

        if downstream_node.node_type == 'Projection':
            derived_projection_cols, source_projection_cols = parse_projection_columns(downstream_node.attribute_dict.get("Columns", ""))
            downstream_output_cols = set(derived_projection_cols) | set(source_projection_cols)
            udf_join_cols_set = set(extract_udf_join_keys(udf_node.attribute_dict.get("UDF", {}).get("Input Columns", []))) 

            if _determine_which_branch(udf_node.upstream[0], source_projection_cols) and _determine_which_branch(udf_node.upstream[1], source_projection_cols):
                return False

            print(downstream_output_cols)
            print(udf_join_cols_set)     
            return udf_join_cols_set.issubset(downstream_output_cols)
                     
    elif udf_node.node_type == 'TopK':
        if downstream_node.node_type in ['Aggregation', 'Selection', 'TopK', 'Join']:
            return False
        
        if downstream_node.node_type == 'Projection':
            rank_cols_set = set(udf_node.attribute_dict.get("UDF", {}).get("Input Columns", []))
            derived_projection_cols, source_projection_cols = parse_projection_columns(downstream_node.attribute_dict.get("Columns", ""))
            downstream_output_cols = set(derived_projection_cols) | set(source_projection_cols)
            
            if not rank_cols_set.issubset(downstream_output_cols):
                return False
            else:
                return True
   
    elif udf_node.node_type == 'Aggregation':
        agg_input_cols = udf_node.attribute_dict.get("UDF", {}).get("Input Columns", [])
        if downstream_node.node_type == 'Aggregation':
            return False
        
        if downstream_node.node_type in['TopK', 'Join', 'Selection', 'Projection']:
            return False   
        
    return True


def _should_swap_cost_based(udf_node: PlanNode, downstream_node: PlanNode, cost_model: CostModel) -> bool:
    """
    Determine if swap should be performed based on cost comparison.
    
    Args:
        udf_node: The UDF node to potentially swap
        downstream_node: The downstream node to swap with
        cost_model: Cost model for computing costs
        
    Returns:
        True if swap reduces cost, False otherwise
    """
    # Estimate cardinalities if not already set
    _estimate_cardinalities(udf_node, cost_model)
    _estimate_cardinalities(downstream_node, cost_model)
    
    # Compute cost of current plan (UDF before downstream)
    current_cost = _compute_swap_cost(udf_node, downstream_node, cost_model, swapped=False)
    
    # Compute cost of swapped plan (downstream before UDF)
    swapped_cost = _compute_swap_cost(udf_node, downstream_node, cost_model, swapped=True)
    
    # Swap if it reduces cost (with small threshold to avoid unnecessary swaps)
    cost_reduction_threshold = 0.01  # 1% cost reduction required
    if swapped_cost < current_cost * (1 - cost_reduction_threshold):
        return True
    
    return False


def _compute_swap_cost(udf_node: PlanNode, downstream_node: PlanNode, 
                       cost_model: CostModel, swapped: bool) -> float:
    """
    Compute total cost of executing the two nodes in given order.
    
    Args:
        udf_node: The UDF node
        downstream_node: The downstream node
        cost_model: Cost model
        swapped: If True, compute cost with downstream before UDF; if False, UDF before downstream
        
    Returns:
        Total execution cost
    """
    if swapped:
        # Downstream executes first, then UDF
        # Get input cardinality for downstream
        downstream_input = _get_input_cardinality(downstream_node)
        downstream_cost = cost_model.compute_operator_cost(downstream_node, downstream_input)
        downstream_output = cost_model.estimate_output_cardinality(downstream_node, downstream_input)
        
        # UDF executes on downstream's output
        udf_cost = cost_model.compute_operator_cost(udf_node, downstream_output)
        
        return downstream_cost + udf_cost
    else:
        # UDF executes first, then downstream
        # Get input cardinality for UDF
        udf_input = _get_input_cardinality(udf_node)
        udf_cost = cost_model.compute_operator_cost(udf_node, udf_input)
        udf_output = cost_model.estimate_output_cardinality(udf_node, udf_input)
        
        # Downstream executes on UDF's output
        downstream_cost = cost_model.compute_operator_cost(downstream_node, udf_output)
        
        return udf_cost + downstream_cost


def _get_input_cardinality(node: PlanNode) -> int:
    """
    Get input cardinality for a node.
    If node has upstream nodes, use their output cardinality.
    Otherwise, use a default estimate.
    """
    if node.upstream:
        # For unary operators, use first upstream's output
        if len(node.upstream) == 1:
            if node.upstream[0].output_cardinality is not None:
                return node.upstream[0].output_cardinality
        # For binary operators (join), combine both inputs
        elif len(node.upstream) >= 2:
            left_card = node.upstream[0].output_cardinality or 1000
            right_card = node.upstream[1].output_cardinality or 1000
            return left_card + right_card  # Approximate for join input
    
    # Default estimate if no upstream or cardinality not set
    return node.input_cardinality or 1000


def _estimate_cardinalities(node: PlanNode, cost_model: CostModel):
    """
    Estimate and set cardinalities for a node if not already set.
    Recursively estimates for upstream nodes first.
    """
    # Recursively estimate upstream nodes first
    for upstream_node in node.upstream:
        if upstream_node.output_cardinality is None:
            _estimate_cardinalities(upstream_node, cost_model)
    
    # Estimate input cardinality from upstream
    if node.input_cardinality is None:
        node.input_cardinality = _get_input_cardinality(node)
    
    # Estimate output cardinality
    if node.output_cardinality is None and node.input_cardinality is not None:
        node.output_cardinality = cost_model.estimate_output_cardinality(node, node.input_cardinality)


def extract_udf_join_keys(condition_string: str) -> list[str]:
    """
    Extracts only the unique column names (without table names) from a
    string representing nested lists of join keys.

    Args:
        condition_string: A string in the format "[[key1, ...], [key2, ...]]",
                          where each key is 'TableName.ColumnName'.
    """
    try:
        # Step 1: Safely parse the string into a Python list of lists
        condition_string = str(condition_string)
        nested_list = ast.literal_eval(condition_string)
        if not isinstance(nested_list, list):
             raise TypeError("Input string does not represent a list.")
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing input string: {e}")
        return []

    unique_columns = set()

    # Step 2: Iterate through the nested structure
    for key_list in nested_list:
        if isinstance(key_list, list):
            for full_column_name in key_list:
                if isinstance(full_column_name, str) and '.' in full_column_name:
                    # Step 3: Split and get the column name part
                    column_name = full_column_name.split('.')[-1]
                    unique_columns.add(column_name)

    # Step 4: Return a sorted list of unique columns
    return sorted(list(unique_columns))

def extract_columns_from_selection(condition_string: str) -> list[str]:
    """
    Extracts all unique column names from a SQL-like selection condition string.

    It correctly identifies columns on the left side of comparison operators
    and handles both 'ColumnName' and 'TableName.ColumnName' formats.

    Args:
        condition_string: A string like "col1 = val1 and T2.col2 > 100".
    """
    regex = r'(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)\s*(?=[=><!])'

    # re.findall will return a list of all captured groups.
    all_matches = re.findall(regex, condition_string)

    # Use a set to get unique names, then convert to a sorted list.
    unique_columns = sorted(list(set(all_matches)))

    return unique_columns

def extract_column_names_from_Topk_input(order_by_string: str) -> list:
    """
    Robustly extracts column names from an 'ORDER BY' style string by splitting it.
    This is the recommended approach.
    """
    if not order_by_string:
        return []
        
    parts = order_by_string.split(',')
    column_names = []
    
    for part in parts:
        # Get the first "word" from the trimmed part, which is the full column name
        # e.g., "  col1 asc  " -> "col1"
        # e.g., "  drivers.name desc  " -> "drivers.name"
        full_col_name = part.strip().split()[0]
        
        # extract just the column name from the full name
        if '.' in full_col_name:
            # It's in TableName.ColumnName format
            column_name = full_col_name.split('.')[-1]
        else:
            column_name = full_col_name
            
        column_names.append(column_name)
        
    return column_names

def parse_projection_columns(columns_list: list) -> tuple[list[str], list[str]]:
    """
    Parses a list of projection column strings and separates them into
    derived columns and the source columns they depend on.

    Args:
        columns_list: A list of strings, e.g., 
                      ["a", "b", "c = a * b", "users.id"].

    Returns:
        A tuple containing two lists:
        1. A list of derived column names.
        2. A list of unique source column names used.
    """
    derived_columns_set = set()
    source_columns_set = set()

    # Regex to find all column-like identifiers (handles Table.Column and Column).
    # The capturing group ensures we only get the final column name.
    # This regex assumes column/table names start with a letter or underscore.
    column_finder_regex = r'(?:[a-zA-Z_][a-zA-Z0-9_]*\.)?([a-zA-Z_][a-zA-Z0-9_]*)'

    for item in columns_list:
        # Check if it's a derived column (contains '=')
        if '=' in item:
            try:
                # Split the string into the derived name and the expression
                derived_name_part, expression_part = item.split('=', 1)
                derived_name = derived_name_part.strip()
                
                # Add the new column name to the derived set
                derived_columns_set.add(derived_name)
                
                # Find all source columns used in the expression part
                used_in_expression = re.findall(column_finder_regex, expression_part)
                source_columns_set.update(used_in_expression)

            except ValueError:
                # Handle cases where the split might fail, though unlikely
                print(f"Warning: Could not parse derived column string: {item}")

        # Otherwise, it's a source column
        else:
            full_col_name = item.strip()
            # Handle both 'TableName.ColumnName' and 'ColumnName' formats
            if '.' in full_col_name:
                source_name = full_col_name.split('.')[-1]
            else:
                source_name = full_col_name
            
            # Add to the source set
            source_columns_set.add(source_name)
            
    # Convert sets to sorted lists for consistent output
    return sorted(list(derived_columns_set)), sorted(list(source_columns_set))


def extract_join_columns(condition_string: str) -> list[str]:
    """
    Extracts only the unique column names (without table names) from a
    strict 'JOIN' condition string.
    """
    if not condition_string:
        return []
        
    all_pure_columns = set()

    # Split the entire string by 'and'
    conditions = condition_string.split(' and ')

    for condition in conditions:
        # Split each condition by '='
        parts = [part.strip() for part in condition.split('=')]
        
        for full_column_name in parts:
            # Extract only the part after the dot '.'
            try:
                # split('.', 1) is slightly more efficient if names could contain '.'
                table_name, column_name = full_column_name.split('.', 1)
                all_pure_columns.add(column_name)
            except ValueError:
                # Handle cases where a part might not have a '.'
                print(f"Warning: Skipping part that is not in 'Table.Column' format: {full_column_name}")

    return sorted(list(all_pure_columns))


def _swap_with_downstream(node_to_move_up: PlanNode):
    """
    Performs the pointer surgery to swap a node with its downstream parent.
    Crucially, it passes the alias of the moving node down to its child,
    ensuring that join conditions remain valid after the swap.
    Assumes the swap condition has already been met.
    """
    parent = node_to_move_up.downstream
    if not parent:
        return  # Cannot swap if there's no parent

    grandparent = parent.downstream

    # Handle Alias Inheritance by the Child Node
    # The alias is a name for the *result* at a specific point in the plan.
    # When the UDF node moves up, its child takes its place as a direct
    # input to the original parent (e.g., the Join). To keep the plan valid,
    # the child must inherit the UDF node's alias.

    if node_to_move_up.node_type == 'Join' and parent.node_type == 'Projection':
        _swap_join_with_projection(node_to_move_up, parent, grandparent)
        return

    if node_to_move_up.node_type == 'Join' and parent.node_type == 'Projection':
        _swap_join_with_join(node_to_move_up, parent, grandparent)
        return

    if 'Alias' in node_to_move_up.attribute_dict:
        if len(node_to_move_up.upstream) == 1:
            alias_to_inherit = node_to_move_up.attribute_dict.pop('Alias')
            child_node = node_to_move_up.upstream[0]
            child_node.attribute_dict['Alias'] = alias_to_inherit

    # Step 1: Grandparent now points to the node_to_move_up
    if grandparent:
        try:
            index = grandparent.upstream.index(parent)
            grandparent.upstream[index] = node_to_move_up
        except ValueError:
            pass  # Should not happen in a consistent tree
    node_to_move_up.downstream = grandparent

    # Step 2: The node_to_move_up's original children now belong to the parent
    if node_to_move_up in parent.upstream:
        parent.upstream.remove(node_to_move_up)
    parent.upstream.extend(node_to_move_up.upstream)
    for child in node_to_move_up.upstream:
        child.downstream = parent

    # Step 3: The parent becomes the child of node_to_move_up
    node_to_move_up.upstream = [parent]
    parent.downstream = node_to_move_up

def _swap_join_with_join(udf_join_node: PlanNode, join_node: PlanNode, grandparent: PlanNode):
    #todo
    return

def _swap_join_with_projection(join_node: PlanNode, projection_node: PlanNode, grandparent: PlanNode):
    """
    Special handling for swapping Join with Projection.
    Pushes the projection down to the appropriate branch of the Join.
    """
    # Step 1: Analyze column sources of Projection
    derived_projection_cols, source_projection_cols = parse_projection_columns(projection_node.attribute_dict.get("Columns", ""))
    target_branch_index = None
    if _determine_which_branch(join_node.upstream[0], source_projection_cols):
        target_branch_index = 0
    else:
        target_branch_index = 1    
    
    if target_branch_index is None:
        print("Warning: Cannot determine which branch the projection columns come from")
        return
    
    # Step 2: Reconnect grandparent
    if grandparent:
        try:
            index = grandparent.upstream.index(projection_node)
            grandparent.upstream[index] = join_node
        except ValueError:
            pass
    join_node.downstream = grandparent
    
    # Step 3: Insert Projection into target branch
    target_branch = join_node.upstream[target_branch_index]
    
    # Insert Projection into target branch
    projection_node.upstream = [target_branch]
    projection_node.downstream = join_node
    target_branch.downstream = projection_node
    
    # Update Join's upstream list
    join_node.upstream[target_branch_index] = projection_node

########################
def _determine_which_branch(rootNode: PlanNode, columns_list: list) -> bool:
    """
    Determine if all columns in the given list are available in the specified branch.
    
    Args:
        rootNode: Root node of the branch to check
        columns_list: List of column names to check
        
    Returns:
        bool: True if all columns are available in this branch, False otherwise
    """
    if not columns_list:
        return True  # Empty list defaults to True
    
    # Get all available columns in this branch
    available_columns = _get_available_columns_in_branch(rootNode)
    
    # Check if all required columns are in available columns
    required_columns = set(columns_list)
    return required_columns.issubset(available_columns)


def _get_available_columns_in_branch(node: PlanNode) -> set:
    """
    Recursively get all available column names in the node and its entire upstream branch.
    
    Args:
        node: Node to traverse
        
    Returns:
        set: Set of all available column names in this branch
    """
    available_columns = set()
    
    # Use stack for depth-first traversal to avoid recursion stack overflow
    nodes_to_visit = [node]
    visited_nodes = set()
    
    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        
        # Avoid revisiting the same node
        node_id = id(current_node)
        if node_id in visited_nodes:
            continue
        visited_nodes.add(node_id)
        
        # Get column information from current node
        node_columns = _extract_columns_from_node(current_node)
        available_columns.update(node_columns)
        
        # Add upstream nodes to visit list
        nodes_to_visit.extend(current_node.upstream)
    
    return available_columns


def _extract_columns_from_node(node: PlanNode) -> set:
    """
    Extract all available column names from a single node.
    
    Args:
        node: Node to extract columns from
        
    Returns:
        set: Set of column names provided by this node
    """
    columns = set()
    
    # 1. Handle Table node - get all table columns
    if node.node_type == "Table":
        table_name = node.attribute_dict.get("Table Name", "")
        if table_name:
            table_columns = _get_table_columns(table_name)
            columns.update(table_columns)
    
    # 2. Handle Projection node - get projected columns
    elif node.node_type == "Projection":
        projection_columns = node.attribute_dict.get("Columns", [])
        if projection_columns:
            # Parse projection columns, including derived and source columns
            derived_cols, source_cols = parse_projection_columns(projection_columns)
            columns.update(derived_cols)
            columns.update(source_cols)
    
    # 3. Handle Selection node - usually doesn't change columns, but may have UDF output
    elif node.node_type == "Selection":
        # If UDF exists, may generate new columns
        if "UDF" in node.attribute_dict:
            udf_info = node.attribute_dict["UDF"]
            # UDF input columns
            input_columns = udf_info.get("Input Columns", [])
            columns.update(input_columns)
            # UDF output column (if exists)
            output_column = udf_info.get("Output Column", "")
            if output_column:
                columns.add(output_column)
    
    # 4. Handle Join node - may generate new columns through UDF
    elif node.node_type == "Join":
        if "UDF" in node.attribute_dict:
            udf_info = node.attribute_dict["UDF"]
            # Extract column names from UDF input columns
            input_columns_nested = udf_info.get("Input Columns", [])
            for column_group in input_columns_nested:
                if isinstance(column_group, list):
                    for col in column_group:
                        # Handle "TableName.ColumnName" format
                        if isinstance(col, str):
                            if '.' in col:
                                pure_column = col.split('.')[-1]
                                columns.add(pure_column)
                            else:
                                columns.add(col)
            
            # UDF output column (if exists)
            output_column = udf_info.get("Output Column", "")
            if output_column:
                columns.add(output_column)
    
    # 5. Handle TopK node - usually doesn't change column structure
    elif node.node_type == "TopK":
        # TopK may have ranking column information
        ranking_criteria = node.attribute_dict.get("Ranking criteria", "")
        if ranking_criteria:
            rank_columns = extract_column_names_from_Topk_input(ranking_criteria)
            columns.update(rank_columns)
    
    # 6. Handle Aggregation node - may have group by and aggregate columns
    elif node.node_type == "Aggregation":
        # Group by columns
        group_by = node.attribute_dict.get("Group By", [])
        if group_by:
            columns.update(group_by)
        
        # Aggregate columns
        aggregates = node.attribute_dict.get("Aggregates", [])
        if aggregates:
            # Need to parse aggregate expressions, similar to projection
            # Simplified: assume aggregate columns are directly available
            columns.update(aggregates)
    
    # 7. Handle generic Columns attribute
    if "Columns" in node.attribute_dict:
        node_columns = node.attribute_dict["Columns"]
        if isinstance(node_columns, list):
            for col in node_columns:
                if isinstance(col, str):
                    # Handle "TableName.ColumnName" format
                    if '.' in col:
                        pure_column = col.split('.')[-1]
                        columns.add(pure_column)
                    else:
                        columns.add(col)
    
    # 8. Clean column names - remove table name prefix, keep only pure column names
    cleaned_columns = set()
    for col in columns:
        if isinstance(col, str):
            if '.' in col:
                cleaned_columns.add(col.split('.')[-1])
            else:
                cleaned_columns.add(col)
    
    return cleaned_columns


def _get_table_columns(table_name: str, db_path: str = None) -> list:
    """
    Get column information for a table.
    
    Args:
        table_name: Table name
        db_path: Full path to database file (optional)
        
    Returns:
        list: List of column names, returns predefined columns if error occurs
    """
    # If path not provided, try to construct it
    if db_path is None:
        if not path or not db_name:
            print(f"Error: No database path provided and globals not set")
            return _get_fallback_columns(table_name)
        db_path = os.path.join(path, db_name, f"{db_name}.sqlite")
    
    # Verify file exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found: {db_path}")
        return _get_fallback_columns(table_name)
    
    # Use context manager to automatically handle connection
    try:
        with sqlite3.connect(db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            table_info = cursor.fetchall()
            
            if not table_info:
                print(f"Warning: Table '{table_name}' not found")
                return _get_fallback_columns(table_name)
            
            column_names = [column[1] for column in table_info]
            print(f"Found columns in '{table_name}': {column_names}")
            return column_names
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return _get_fallback_columns(table_name)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return _get_fallback_columns(table_name)

def _get_fallback_columns(table_name: str) -> list:
    """Return predefined table structure as fallback."""
    fallback_schemas = {
        "state_expenditures": ["Agency Name", "Payments Total", "Year"],
        "rm-mr-2009-eng": ["DEPT_EN_DESC", "MINE", "Amount"],
        # Add more predefined structures as needed...
    }
    return fallback_schemas.get(table_name, ["id", "name", "value"])


#############

def reconstruct_plan_from_tree(root_node: PlanNode, indent: int = 2) -> str:
    """
    Reconstructs a nested JSON query plan string from the root of a PlanNode tree.

    This function is the inverse of `parse_plan_to_tree`.

    Args:
        root_node: The root PlanNode of the execution plan tree.
        indent: The indentation level for the output JSON string.

    Returns:
        A string representing the nested query plan in JSON format.
    """
    if not root_node:
        return "{}"

    # Call the recursive helper function to build the dictionary
    plan_dict = _build_dict_from_node(root_node)
    
    # Serialize the final dictionary into a nicely formatted JSON string
    return json.dumps(plan_dict, indent=indent)


def _build_dict_from_node(node: PlanNode) -> dict:
    """
    A recursive helper function that converts a PlanNode and its descendants
    into a nested Python dictionary.
    """
    # 1. Create the base dictionary for the current node.
    #    The ** operator elegantly merges the attribute dictionary.
    
    attrs = dict(node.attribute_dict) if getattr(node, "attribute_dict", None) else {}

    # extract alias.
    alias_key = None
    alias_val = None
    for k in ("alias", "Alias"):
        if k in attrs:
            alias_key = k
            alias_val = attrs.pop(k)
            break
        
    current_dict = {
        "Node Type": node.node_type,
        **attrs
    }

    # 2. If there are upstream nodes (inputs), recursively process them.
    if node.upstream:
        # Recursively build dictionaries for all children (upstream nodes)
        upstream_dicts = [_build_dict_from_node(child) for child in node.upstream]
        
        # 3. Attach the reconstructed children to the "Input" key.
        #    Handle the case of single vs. multiple inputs (e.g., Join).
        if len(upstream_dicts) > 1:
            current_dict["Inputs"] = upstream_dicts 
        else:
            current_dict["Input"] = upstream_dicts[0]
    
    if alias_key is not None:
        current_dict[alias_key] = alias_val   
          
    # 4. Return the fully constructed dictionary for this sub-tree.
    return current_dict

def parse_plan_to_tree(plan_json_string: str) -> PlanNode:
    """
    Parses a JSON string representing a SQL execution plan into a tree of PlanNode objects,
    where each node is linked to its upstream and downstream dependencies.

    Args:
        plan_json_string: A string containing the execution plan in JSON format.

    Returns:
        The root PlanNode of the execution plan tree. Returns None if parsing fails.
    """
    try:
        plan_dict = json.loads(plan_json_string)
    except json.JSONDecodeError:
        print("Error: Input string is not a valid JSON format.")
        return None

    # Start the recursive tree-building process from the root of the plan.
    # The root node has no downstream consumer, so we pass None.
    root_node = _build_tree(plan_dict, downstream_node=None)
    # root_node = _merge_topk_selection_nodes(root_node)
    return root_node

def _build_tree(node_dict: dict, downstream_node: PlanNode) -> PlanNode:
    """
    A recursive helper function to build the plan tree. (Pre-order traversal)
    """
    current_node = PlanNode(node_dict, downstream_node)

    input_key = None
    if "Input" in node_dict:
        input_key = "Input"
    elif "Inputs" in node_dict:
        input_key = "Inputs"

    if input_key:
        inputs = node_dict[input_key]
        
        if isinstance(inputs, list):
            for child_dict in inputs:
                upstream_node = _build_tree(child_dict, downstream_node=current_node)
                current_node.upstream.append(upstream_node)
        else:
            upstream_node = _build_tree(inputs, downstream_node=current_node)
            current_node.upstream.append(upstream_node)
            
    return current_node

def _merge_topk_selection_nodes(root_node: PlanNode) -> PlanNode:
    """
    Post-process the tree to merge TopK nodes with their upstream UDF Selection nodes.
    """
    def _collect_all_nodes(node: PlanNode, visited: set = None) -> list:
        """Collect all nodes in the tree using DFS."""
        if visited is None:
            visited = set()
        
        if id(node) in visited:
            return []
        
        visited.add(id(node))
        nodes = [node]
        
        for upstream_node in node.upstream:
            nodes.extend(_collect_all_nodes(upstream_node, visited))
        
        return nodes
    
    # Collect all nodes in the tree
    all_nodes = _collect_all_nodes(root_node)
    
    # Find TopK nodes that have UDF Selection nodes as upstream
    nodes_to_merge = []
    for node in all_nodes:
        if node.node_type == "TopK":
            # Check if any upstream node is UDF Selection
            for upstream_node in node.upstream:
                if upstream_node.node_type == "Selection" and 'UDF' in upstream_node.attribute_dict:
                    nodes_to_merge.append((node, upstream_node))  # (topk_node, selection_node)
    
    # Perform the merging
    new_root = root_node
    for topk_node, selection_node in nodes_to_merge:
        merged_node = _create_merged_node(topk_node, selection_node)
        new_root = _replace_node_in_tree(new_root, topk_node, merged_node)
    
    return new_root

def _create_merged_node(topk_node: PlanNode, selection_node: PlanNode) -> PlanNode:
    """
    Create a merged TopK-Selection node from TopK and UDF Selection nodes.
    """
    # Create merged attributes dictionary
    merged_attributes = {}
    merged_attributes.update(selection_node.attribute_dict)
    merged_attributes.update(topk_node.attribute_dict)
    
    # Create the merged node dictionary
    merged_dict = {"Node Type": "TopK-Selection"}
    merged_dict.update(merged_attributes)
    
    # Create new merged node
    merged_node = PlanNode(merged_dict, downstream_node=topk_node.downstream)
    
    # Set upstream connections (from UDF Selection node's upstream)
    merged_node.upstream = selection_node.upstream[:]
    
    # Update upstream nodes' downstream references
    for upstream_node in merged_node.upstream:
        # Remove selection_node from upstream's downstream references
        if hasattr(upstream_node, 'downstream_list'):
            if selection_node in upstream_node.downstream_list:
                upstream_node.downstream_list.remove(selection_node)
                upstream_node.downstream_list.append(merged_node)
    
    return merged_node

def _replace_node_in_tree(root_node: PlanNode, old_node: PlanNode, new_node: PlanNode) -> PlanNode:
    """
    Replace old_node with new_node in the tree structure.
    """
    # If the old_node is the root, return the new_node as the new root
    if root_node == old_node:
        return new_node
    
    # Update downstream node's upstream references
    if old_node.downstream:
        downstream_node = old_node.downstream
        # Replace old_node with new_node in downstream's upstream list
        for i, upstream_node in enumerate(downstream_node.upstream):
            if upstream_node == old_node:
                downstream_node.upstream[i] = new_node
                break
    
    # Update new_node's downstream reference
    if new_node.downstream and new_node.downstream != old_node.downstream:
        new_node.downstream = old_node.downstream
    
    # Update upstream nodes' downstream references (if they exist)
    for upstream_node in new_node.upstream:
        # This assumes each node might have a reference to its downstream nodes
        # Update any downstream references from upstream nodes
        pass  # This depends on your specific tree structure requirements
    
    return root_node

def print_plan_tree(node: PlanNode, indent=""):
    """print the reconstructed tree structure."""
    if not node:
        return
    
    # Print current node
    print(f"{indent}- {node.node_type} (Attributes: {json.dumps(node.attribute_dict)})")
    
    # Recursively print upstream nodes
    if node.upstream:
        print(f"{indent}  [Upstream]:")
        for upstream_node in node.upstream:
            print_plan_tree(upstream_node, indent + "    ")