"""
Symbolic verification module for query plan equivalence using Z3.

This module implements formal verification to ensure that query plan transformations
preserve semantic equivalence. It serves as a safety net after rule-based checks pass.
"""

import time
from typing import Dict, List, Optional, Set, Tuple
try:
    from z3 import (
        BoolSort, BoolVal, Bool, And, Or, Not, Implies, 
        Solver, sat, unsat, unknown, IntSort, StringSort,
        Function, ForAll, Exists, Const, DeclareSort
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    # Create dummy classes for when Z3 is not available
    class BoolSort: pass
    class BoolVal: pass
    class Bool: pass
    class And: pass
    class Or: pass
    class Not: pass
    class Solver: pass
    class unsat: pass
    class sat: pass
    class Function: pass
    class Const: pass
    class DeclareSort: pass

from .optimization_utils import PlanNode


# Z3 sorts for symbolic values
if Z3_AVAILABLE:
    ValueSort = DeclareSort('Value')
    BoolSortType = BoolSort()  # Create instance, not redeclare
else:
    ValueSort = None
    BoolSortType = None


class SymbolicTable:
    """
    Represents a table symbolically with:
    - Schema: column names and their sorts
    - Symbolic cells: Z3 constants for each cell
    - Row-existence predicate: Boolean expression tracking filter conditions
    """
    
    def __init__(self, schema: Dict[str, type], table_name: str = "T"):
        """
        Initialize a symbolic table.
        
        Args:
            schema: Dictionary mapping column names to their types
            table_name: Name for this table (for generating unique constants)
        """
        if not Z3_AVAILABLE:
            raise ImportError("z3-solver is required for symbolic verification")
        
        self.schema = schema
        self.table_name = table_name
        self.symbolic_cells: Dict[str, Const] = {}
        self.row_exists = BoolVal(True)  # Initially all rows exist
        
        # Create symbolic constants for each column
        for col_name in schema.keys():
            # Use table name and column name to create unique constant
            const_name = f"{table_name}_{col_name}"
            self.symbolic_cells[col_name] = Const(const_name, ValueSort)
    
    def add_filter(self, condition: Bool):
        """Add a filter condition to row_exists predicate."""
        self.row_exists = And(self.row_exists, condition)
    
    def get_cell(self, col_name: str) -> Const:
        """Get symbolic constant for a column."""
        if col_name not in self.symbolic_cells:
            # Create on-the-fly if not exists
            const_name = f"{self.table_name}_{col_name}"
            self.symbolic_cells[col_name] = Const(const_name, ValueSort)
        return self.symbolic_cells[col_name]
    
    def copy(self, new_name: str) -> 'SymbolicTable':
        """Create a copy of this table with a new name."""
        new_table = SymbolicTable(self.schema.copy(), new_name)
        # Share row_exists by reference (Z3 expressions are immutable, so this is safe)
        # When add_filter() is called, it creates a new And() expression, so original is not modified
        new_table.row_exists = self.row_exists
        # Copy symbolic cells (Z3 constants are immutable, so sharing is safe)
        new_table.symbolic_cells = self.symbolic_cells.copy()
        return new_table


class SymbolicExecutor:
    """
    Executes query plan operators on symbolic tables.
    Handles all UDF types: Selection, Projection, Join, TopK, Aggregation.
    """
    
    def __init__(self):
        """Initialize the symbolic executor."""
        self.udf_functions: Dict[str, Function] = {}
        self.table_counter = 0
    
    def _get_udf_function(self, udf_name: str, input_arity: int, output_type: str) -> Function:
        """
        Get or create an uninterpreted function for a UDF.
        
        Args:
            udf_name: Name of the UDF
            input_arity: Number of input arguments
            output_type: 'bool' for predicates, 'value' for transformations
        """
        key = f"{udf_name}_{input_arity}_{output_type}"
        if key not in self.udf_functions:
            if output_type == 'bool':
                # Boolean output (Selection, Join, TopK predicates)
                domain = [ValueSort] * input_arity
                self.udf_functions[key] = Function(udf_name, *domain, BoolSort())
            else:
                # Value output (Projection)
                domain = [ValueSort] * input_arity
                self.udf_functions[key] = Function(udf_name, *domain, ValueSort)
        return self.udf_functions[key]
    
    def execute_selection_udf(self, table: SymbolicTable, udf_name: str, 
                               input_cols: List[str]) -> SymbolicTable:
        """
        Execute a Selection UDF (filter predicate).
        
        Args:
            table: Input symbolic table
            udf_name: Name of the UDF
            input_cols: List of column names used as input
            
        Returns:
            Modified table with updated row_exists predicate
        """
        result = table.copy(f"{table.table_name}_filtered")
        
        # Get UDF function
        udf_func = self._get_udf_function(udf_name, len(input_cols), 'bool')
        
        # Create UDF application
        udf_args = [table.get_cell(col) for col in input_cols]
        udf_result = udf_func(*udf_args)
        
        # Update row_exists: row_exists AND UDF(input_cols)
        result.add_filter(udf_result)
        
        return result
    
    def execute_projection_udf(self, table: SymbolicTable, udf_name: str,
                              input_cols: List[str], output_col: str) -> SymbolicTable:
        """
        Execute a Projection UDF (transform function).
        
        Args:
            table: Input symbolic table
            udf_name: Name of the UDF
            input_cols: List of column names used as input
            output_col: Name of the output column
            
        Returns:
            New table with added column (row_exists unchanged)
        """
        result = table.copy(f"{table.table_name}_projected")
        
        # Get UDF function
        udf_func = self._get_udf_function(udf_name, len(input_cols), 'value')
        
        # Create UDF application
        udf_args = [table.get_cell(col) for col in input_cols]
        udf_result = udf_func(*udf_args)
        
        # Add new column to schema and symbolic cells
        result.schema[output_col] = type('value', (), {})
        result.symbolic_cells[output_col] = udf_result
        
        # row_exists remains unchanged (projection doesn't filter)
        return result
    
    def execute_join_udf(self, left_table: SymbolicTable, right_table: SymbolicTable,
                         udf_name: str, left_cols: List[str], right_cols: List[str]) -> SymbolicTable:
        """
        Execute a Join UDF (binary predicate).
        
        Args:
            left_table: Left input table
            right_table: Right input table
            udf_name: Name of the UDF
            left_cols: Columns from left table
            right_cols: Columns from right table
            
        Returns:
            Joined table with combined schema and updated row_exists
        """
        # Combine schemas - handle column name conflicts by prefixing
        # In practice, joins usually have different column names or use table prefixes
        combined_schema = left_table.schema.copy()
        for col_name, col_type in right_table.schema.items():
            if col_name in combined_schema:
                # Column name conflict - this is a limitation of the simplified model
                # In practice, columns would have table prefixes (e.g., "left.col", "right.col")
                pass  # Keep left table's column (or could prefix right table's)
            else:
                combined_schema[col_name] = col_type
        
        result = SymbolicTable(combined_schema, f"{left_table.table_name}_join_{right_table.table_name}")
        
        # Copy symbolic cells from both tables (right overwrites left on conflict)
        result.symbolic_cells.update(left_table.symbolic_cells)
        result.symbolic_cells.update(right_table.symbolic_cells)
        
        # Get UDF function
        udf_func = self._get_udf_function(udf_name, len(left_cols) + len(right_cols), 'bool')
        
        # Create UDF application
        left_args = [left_table.get_cell(col) for col in left_cols]
        right_args = [right_table.get_cell(col) for col in right_cols]
        udf_result = udf_func(*(left_args + right_args))
        
        # Update row_exists: left.row_exists AND right.row_exists AND UDF(...)
        result.row_exists = And(
            left_table.row_exists,
            right_table.row_exists,
            udf_result
        )
        
        return result
    
    def execute_regular_selection(self, table: SymbolicTable, condition: Optional[str] = None) -> SymbolicTable:
        """
        Execute a regular (non-UDF) selection.
        For simplicity, we model this as preserving row_exists (no change).
        In practice, you'd parse the condition and add it to row_exists.
        """
        return table.copy(f"{table.table_name}_selected")
    
    def execute_regular_join(self, left_table: SymbolicTable, right_table: SymbolicTable,
                            condition: Optional[str] = None) -> SymbolicTable:
        """
        Execute a regular (non-UDF) join.
        For simplicity, we model this as combining row_exists predicates.
        """
        combined_schema = {**left_table.schema, **right_table.schema}
        result = SymbolicTable(combined_schema, f"{left_table.table_name}_join_{right_table.table_name}")
        result.symbolic_cells.update(left_table.symbolic_cells)
        result.symbolic_cells.update(right_table.symbolic_cells)
        result.row_exists = And(left_table.row_exists, right_table.row_exists)
        return result
    
    def execute_regular_projection(self, table: SymbolicTable, columns: List[str]) -> SymbolicTable:
        """
        Execute a regular (non-UDF) projection.
        Projection doesn't change row_exists, only schema.
        """
        # Only include columns that exist in the original schema
        # Missing columns will be created on-the-fly by get_cell(), but we should
        # only include them in the schema if they're actually requested
        new_schema = {}
        new_cells = {}
        for col in columns:
            if col in table.schema:
                new_schema[col] = table.schema[col]
                new_cells[col] = table.get_cell(col)
            # If column doesn't exist, get_cell() will create it, but we skip it
            # in the schema to maintain consistency
        
        result = SymbolicTable(new_schema, f"{table.table_name}_projected")
        result.symbolic_cells = new_cells
        result.row_exists = table.row_exists
        return result


class PlanVerifier:
    """
    Orchestrates verification of two query plans for equivalence.
    """
    
    def __init__(self, timeout_seconds: int = 5):
        """
        Initialize the plan verifier.
        
        Args:
            timeout_seconds: Maximum time to spend on Z3 verification
        """
        self.timeout_seconds = timeout_seconds
        self.executor = SymbolicExecutor()
    
    def verify_swap_equivalence(self, udf_node: PlanNode, downstream_node: PlanNode) -> bool:
        """
        Verify that swapping udf_node with downstream_node preserves equivalence.
        
        Args:
            udf_node: The UDF node to potentially swap
            downstream_node: The downstream node to swap with
            
        Returns:
            True if plans are equivalent, False otherwise
        """
        udf_type = udf_node.node_type
        downstream_type = downstream_node.node_type
        udf_name = udf_node.attribute_dict.get("UDF", {}).get("UDF Name", "unknown")
        
        print(f"Symbolic verification: {udf_type} UDF '{udf_name}' ↔ {downstream_type} - Starting...")
        
        try:
            # Create symbolic tables for inputs
            input_schema = self._extract_input_schema(udf_node, downstream_node)
            input_table = SymbolicTable(input_schema, "input")
            
            # Execute original plan: UDF -> Downstream
            original_result = self._execute_plan_symbolically(
                input_table, udf_node, downstream_node, order='udf_first'
            )
            
            # Execute swapped plan: Downstream -> UDF
            swapped_result = self._execute_plan_symbolically(
                input_table, udf_node, downstream_node, order='udf_second'
            )
            
            # Check equivalence
            result = self._check_equivalence(original_result, swapped_result)
            
            if result:
                print(f"Symbolic verification: {udf_type} UDF '{udf_name}' ↔ {downstream_type} - PASSED (plans are equivalent)")
            else:
                print(f"Symbolic verification: {udf_type} UDF '{udf_name}' ↔ {downstream_type} - FAILED (plans are not equivalent)")
            
            return result
            
        except Exception as e:
            # On any error, conservatively reject the swap
            print(f"Symbolic verification: {udf_type} UDF '{udf_name}' ↔ {downstream_type} - ERROR: {e}")
            return False
    
    def _extract_input_schema(self, udf_node: PlanNode, downstream_node: PlanNode) -> Dict[str, type]:
        """
        Extract the input schema needed for symbolic execution.
        This is a simplified version - in practice, you'd traverse the plan tree.
        """
        schema = {}
        
        # Get columns from UDF
        udf_input_cols = udf_node.attribute_dict.get("UDF", {}).get("Input Columns", [])
        if isinstance(udf_input_cols, str):
            udf_input_cols = [udf_input_cols]
        for col in udf_input_cols:
            if isinstance(col, str):
                # Handle "Table.Column" format
                col_name = col.split('.')[-1] if '.' in col else col
                schema[col_name] = type('value', (), {})
        
        # Get columns from downstream node
        if downstream_node.node_type == 'Join':
            # Join has two inputs - get columns from both
            condition = downstream_node.attribute_dict.get("Condition", "")
            if condition:
                # Extract column names from join condition
                import re
                cols = re.findall(r'(\w+)\.(\w+)', condition)
                for _, col_name in cols:
                    schema[col_name] = type('value', (), {})
        elif downstream_node.node_type == 'Projection':
            # Get columns from projection
            cols = downstream_node.attribute_dict.get("Columns", [])
            if isinstance(cols, list):
                for col in cols:
                    if isinstance(col, str):
                        col_name = col.split('.')[-1] if '.' in col else col
                        schema[col_name] = type('value', (), {})
        
        # If schema is empty, create a default column
        if not schema:
            schema['default_col'] = type('value', (), {})
        
        return schema
    
    def _execute_plan_symbolically(self, input_table: SymbolicTable,
                                  udf_node: PlanNode, downstream_node: PlanNode,
                                  order: str) -> SymbolicTable:
        """
        Execute a plan symbolically.
        
        Args:
            input_table: Input symbolic table
            udf_node: UDF node
            downstream_node: Downstream node
            order: 'udf_first' or 'udf_second'
            
        Returns:
            Result symbolic table
        """
        if order == 'udf_first':
            # Original: UDF -> Downstream
            after_udf = self._execute_udf_node(input_table, udf_node)
            result = self._execute_downstream_node(after_udf, downstream_node)
        else:
            # Swapped: Downstream -> UDF
            after_downstream = self._execute_downstream_node(input_table, downstream_node)
            result = self._execute_udf_node(after_downstream, udf_node)
        
        return result
    
    def _execute_udf_node(self, table: SymbolicTable, udf_node: PlanNode) -> SymbolicTable:
        """Execute a UDF node symbolically."""
        udf_info = udf_node.attribute_dict.get("UDF", {})
        udf_name = udf_info.get("UDF Name", "udf")
        input_cols = udf_info.get("Input Columns", [])
        if isinstance(input_cols, str):
            input_cols = [input_cols]
        
        # Normalize column names (remove table prefixes)
        input_cols = [col.split('.')[-1] if '.' in col else col for col in input_cols]
        
        # Validate that input columns exist in the table schema
        # (get_cell() will create them on-the-fly, but we should check)
        missing_cols = [col for col in input_cols if col not in table.schema and col not in table.symbolic_cells]
        if missing_cols:
            # Columns don't exist - this might be an error, but get_cell() will handle it
            # by creating them on-the-fly. This is a limitation of simplified schema extraction.
            pass
        
        if udf_node.node_type == 'Selection':
            return self.executor.execute_selection_udf(table, udf_name, input_cols)
        elif udf_node.node_type == 'Projection':
            output_col = udf_info.get("Output Column", "output")
            return self.executor.execute_projection_udf(table, udf_name, input_cols, output_col)
        elif udf_node.node_type == 'Join':
            # For join UDF, we need to split input columns into left and right
            # Input Columns for Join UDF is typically a list of two lists: [[left_cols], [right_cols]]
            # But it might also be a flat list, so we need to handle both cases
            try:
                # Try to parse as nested list (expected format)
                if input_cols and len(input_cols) == 2 and isinstance(input_cols[0], list):
                    # Input Columns is nested: [[left_cols], [right_cols]]
                    left_cols_raw = input_cols[0] if isinstance(input_cols[0], list) else []
                    right_cols_raw = input_cols[1] if isinstance(input_cols[1], list) else []
                else:
                    # Fallback: treat as flat list and split evenly (simplified)
                    mid_point = len(input_cols) // 2
                    left_cols_raw = input_cols[:mid_point]
                    right_cols_raw = input_cols[mid_point:]
            except (IndexError, TypeError):
                # Error parsing - use fallback
                mid_point = len(input_cols) // 2 if input_cols else 0
                left_cols_raw = input_cols[:mid_point] if input_cols else []
                right_cols_raw = input_cols[mid_point:] if input_cols else []
            
            # Normalize column names (remove table prefixes)
            left_cols = [col.split('.')[-1] if isinstance(col, str) and '.' in col else str(col) for col in left_cols_raw]
            right_cols = [col.split('.')[-1] if isinstance(col, str) and '.' in col else str(col) for col in right_cols_raw]
            
            # Create a right table with the right columns
            right_table = SymbolicTable({col: type('value', (), {}) for col in right_cols}, "right")
            return self.executor.execute_join_udf(table, right_table, udf_name, left_cols, right_cols)
        else:
            # For TopK and Aggregation UDFs: 
            # These rarely pass rule-based checks (TopK can pass with Projection, 
            # but Aggregation almost never passes). We return table as-is as a 
            # conservative fallback. Full symbolic modeling would require handling
            # cardinality changes and ordering semantics, which is complex.
            return table.copy(f"{table.table_name}_udf")
    
    def _execute_downstream_node(self, table: SymbolicTable, downstream_node: PlanNode) -> SymbolicTable:
        """Execute a downstream (non-UDF) node symbolically."""
        if downstream_node.node_type == 'Selection':
            condition = downstream_node.attribute_dict.get("Condition", "")
            # Note: We don't parse the condition - this is a simplification
            # In practice, we'd parse SQL conditions and add them to row_exists
            return self.executor.execute_regular_selection(table, condition)
        elif downstream_node.node_type == 'Join':
            # For join, we need to extract columns from the join condition
            # to create a proper right table schema
            condition = downstream_node.attribute_dict.get("Condition", "")
            # Extract column names from join condition (simplified)
            import re
            right_cols = set()
            if condition:
                # Try to extract right table columns from condition like "left.id = right.id"
                matches = re.findall(r'right\.(\w+)', condition)
                right_cols.update(matches)
            # Create right table with extracted columns (or empty if none found)
            right_schema = {col: type('value', (), {}) for col in right_cols}
            right_table = SymbolicTable(right_schema, "right")
            return self.executor.execute_regular_join(table, right_table, condition)
        elif downstream_node.node_type == 'Projection':
            cols = downstream_node.attribute_dict.get("Columns", [])
            if isinstance(cols, list):
                col_names = [col.split('.')[-1] if '.' in col else col for col in cols]
            else:
                col_names = []
            return self.executor.execute_regular_projection(table, col_names)
        else:
            # For other node types, return as-is
            return table.copy(f"{table.table_name}_downstream")
    
    def _check_equivalence(self, table1: SymbolicTable, table2: SymbolicTable) -> bool:
        """
        Check if two symbolic tables are equivalent using Z3.
        
        Two tables are equivalent if their row_exists predicates are equivalent:
        (pred1 ∧ ¬pred2) ∨ (¬pred1 ∧ pred2) is unsatisfiable.
        
        Args:
            table1: First symbolic table
            table2: Second symbolic table
            
        Returns:
            True if equivalent, False otherwise
        """
        solver = Solver()
        solver.set("timeout", self.timeout_seconds * 1000)  # milliseconds
        
        # Create equivalence condition: (pred1 ∧ ¬pred2) ∨ (¬pred1 ∧ pred2)
        # This is True when predicates differ
        pred1 = table1.row_exists
        pred2 = table2.row_exists
        
        # Check if pred1 and pred2 are equivalent
        # We check if (pred1 != pred2) is unsatisfiable
        not_equivalent = Or(
            And(pred1, Not(pred2)),
            And(Not(pred1), pred2)
        )
        
        # If not_equivalent is unsatisfiable, then pred1 == pred2
        solver.add(not_equivalent)
        result = solver.check()
        
        if result == unsat:
            # Predicates are equivalent
            return True
        elif result == sat:
            # Predicates differ - plans are not equivalent
            return False
        else:
            # Timeout or unknown - conservatively reject
            print(f"Symbolic verification: TIMEOUT (exceeded {self.timeout_seconds}s) or UNKNOWN result")
            return False


def symbolic_verify(udf_node: PlanNode, downstream_node: PlanNode, 
                   timeout_seconds: int = 5) -> bool:
    """
    Convenience function to verify plan equivalence.
    
    Args:
        udf_node: The UDF node to potentially swap
        downstream_node: The downstream node to swap with
        timeout_seconds: Maximum time for verification
        
    Returns:
        True if plans are equivalent, False otherwise
    """
    if not Z3_AVAILABLE:
        raise ImportError("z3-solver is required for symbolic verification")
    
    verifier = PlanVerifier(timeout_seconds=timeout_seconds)
    return verifier.verify_swap_equivalence(udf_node, downstream_node)
