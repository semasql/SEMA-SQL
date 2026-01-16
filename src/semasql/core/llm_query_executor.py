# Standard library
import itertools
import json
import re
import time

# Third-party
import pandas as pd

# Local imports
from src.semasql.core import udf_deferral
from src.semasql.core.optimization_utils import CostModel
from src.semasql.db_utils import (
    BaseDBExecutor,
    obtain_column_description,
    obtain_db_schema,
    load_schema_dict,
)
from src.semasql.llm.LLMCaller import LLMCaller
from src.semasql.parser import (
    extract_column_mappings_from_sql,
    parse_column_list,
    parse_dataframe_to_str,
    parse_json_response,
    parse_json_response_to_df,
    sql_create_parse,
    sql_parse,
    update_column_lst,
    update_query_plan,
)
from src.semasql.prompts import *


class LLMEnhancedDBExecutor(BaseDBExecutor):
    def __init__(self, path, db_name, question, model, enable_column_selector, enable_question_decomposition, optimization_lazy_llm, optimization_udf_rewrite):
        super().__init__(db_name, path)
        self.question = question
        self.enable_column_selector = enable_column_selector
        self.enable_question_decomposition = enable_question_decomposition
        self.optimization_lazy_llm = optimization_lazy_llm
        self.optimization_udf_rewrite = optimization_udf_rewrite
        self.llm_model = LLMCaller(model=model)
        self.schema = obtain_db_schema(self.path, self.db_name)
        self.schema_simple = ''
        self.schema_dict = load_schema_dict(self.schema)
        self.extended_schema = []
        self.eq_primitive = []
        self.connection = None
        self.json_tree = None 
        self.Hint = None
        self.primitive = None
        self.imputation_result = None


    def obtain_concise_schema(self):
        '''
        Obtian a concise schema as [table.column]
        '''
        print("The concise schema is")
        schema_dict = load_schema_dict(self.schema)
        tables = schema_dict['tables']
        schema = [f"{table['name']}.{col['name']}" 
                for table in tables 
                for col in table['columns']]
        self.schema_simple = schema
        print(self.schema_simple)
        return

    def filter_schema(self):
        """
        Filters database schema to retain only information relevant to the question.
        """
        db_schema = self.schema
        if not self.enable_column_selector:
            return

        try:
            Filter_Schema = [{"role": "system", "content": "You are an expert data analyst."}, {'role': 'user', 'content': FilterSchema_reasoning_prompt.format(DATABASE_SCHEMA = db_schema, QUESTION = self.question)}]
            filtered_schema = self.llm_model.call(Filter_Schema)
            match = re.search(r'<filtered_schema>(.*?)</filtered_schema>', 
                            filtered_schema, re.DOTALL | re.IGNORECASE)
            if match:
                self.schema = match.group(1).strip()
            self.obtain_concise_schema()
        except Exception as e:
            print(e)
            pass
          

    def query_optimization(self):
        if not self.optimization_lazy_llm:
            return self.json_tree
        query_plan_str = json.dumps(self.json_tree, indent=2, ensure_ascii=False)
        
        # Initialize cost model for cost-based optimization
        try:
            cost_model = CostModel()
            print("=====Using cost-based optimization=====")
        except Exception as e:
            print(f"Warning: Could not initialize cost model ({e}). Using rule-based optimization.")
            cost_model = None
        
        udf_deferred_plan_str = udf_deferral(query_plan_str, self.db_name, self.path, cost_model)
        self.json_tree = parse_json_response(udf_deferred_plan_str)
        
        print("=====The plan after udf optimization is=====")
        if udf_deferred_plan_str == query_plan_str:
            print("The udf optimized plan is the SAME as the original plan")
        else:
            print("The udf optimized plan is DIFFERENT from the original plan")

        return self.json_tree

    def question_decomposition(self):
        '''
        Analyze the question to determine necessary operators
        '''
        if not self.enable_question_decomposition:
            return
        try:
            QuestionAnalysis = [{"role": "system", "content": "You are an expert data analyst."}, {'role': 'user', 'content': Question_analysis_prompt.format(db_schema = self.schema, question = self.question)}]
            self.Hint = self.llm_model.call(QuestionAnalysis)
        except Exception as e:
            print(e)
            pass

    def NL2JSON(self, few_shot_examples=None):
        """
        Convert natural language question to JSON query execution plan.
        """
        if few_shot_examples is None:
            few_shot_examples = NL2SQL_few_shot_examples
        
        NL2JSON_prompt = NL2SQL_instruct.format(
            db_schema=self.schema, 
            few_shot=few_shot_examples, 
            question=self.question, 
            Hint=self.Hint
        )
        query = [
            {"role": "system", "content": "You are a SQL expert to construct a query execution plan in JSON format."}, 
            {'role': 'user', 'content': NL2JSON_prompt}
        ]
        json_result = self.llm_model.call(query, temperature=0)
        self.json_tree = parse_json_response(json_result)
        return self.json_tree
    
    async def execute_json(self):
        self.drop_all_temp_table()
        new_json = await self.execute_node(self.json_tree)
        print("The final plan is: ", new_json)

        PLAN2SQL_prompt = PLAN2SQL_instruct.format(db_schema = self.schema_simple, few_shot = PLAN2SQL_examples, extended_table = '\n'.join(self.extended_schema), primitive = '\n'.join(self.eq_primitive), json_tree = json.dumps(new_json))
        query = [{"role": "system", "content": "You are a helpful assistant to transfer a execution plan in JSON format to executable SQL. "}, {'role': 'user', 'content': PLAN2SQL_prompt}]
        sql_query = self.llm_model.call(query)
        print("=====The SQL query is=====")
        print(sql_query)
        sql_query = sql_parse(sql_query)
        print("=====The SQL query after parsing is=====")
        print(sql_query)
        final_result = self.execute_query(sql_query)

        # Drop all temp table
        self.drop_all_temp_table()

        # Close the connection after execution
        self.close()
        return final_result
    
    def create_table_from_json(self, table_name, json_data):
        """Create a table in the SQLite database from JSON data."""
        create_table_instruct = PLAN2SQL_gen_table_instruct.format(db_schema = self.schema_simple, extended_table = '\n'.join(self.extended_schema), primitive = '\n'.join(self.eq_primitive), new_table_name = table_name, table_json = json_data)
        query = [{"role": "system", "content": "You are a helpful assistant to transfer the execution plan in JSON format to SQL."}, {'role': 'user', 'content': create_table_instruct}]
        sql_result = self.llm_model.call(query)
        sql_query = sql_create_parse(sql_result)
        print(sql_query)
        column_mappings = extract_column_mappings_from_sql(sql_query)
        try:
            self.execute_query(sql_query)
        except Exception as e:
            print(e)
        self.extended_schema.append(f"\nTable: {table_name}\ncreated by sql:\n{sql_query}")
        return column_mappings


    def UDF_rewrite(self, NodeType, UDF_name, col_lst, col_description_lst, value_list):
        column_values = value_list.replace('{', '').replace('}', '')
        if NodeType == 'Selection' or NodeType == 'Projection': # Only selection and projection udf can be inlined.
            if NodeType == 'Selection':
                udf_prompt = 'Selection predicate (use an LLM to filter rows based on a natural language condition)'
            if NodeType == 'Projection':
                udf_prompt = 'Data transformation (use an LLM to generate new columns based on a transformation function)'
            udf_info = f"{UDF_name}({','.join(col_lst)})"
            col_info = ", ".join(f"{col}:{desc}" for col, desc in zip(col_lst, col_description_lst))
            query = [
            {"role": "system", "content": "You are a helpful assistant that converts LLM UDFs into equivalent SQL operations."},
            {'role': 'user', 'content': UDF_inline.format(question = self.question, udf_prompt = udf_prompt, udf = udf_info, input_cols = col_info, sampled_values = column_values)}]
            sql_result = self.llm_model.call(query)
            print("=====The found equivalent SQL is=====")
            print(sql_result)
            self.primitive = sql_result
            result = parse_json_response(sql_result)
            if "NOT_REPLACEABLE" in result["result"]:
                return False
            else:
                self.eq_primitive.append(result["result"])
                return True
        else:
            return False


    async def llm_quick_sort(self, data, col, udf_name):
        def build_prompt(row_A, row_B):
            return binary_comparison.format(
                col=col,
                UDF_name=udf_name,
                row_A=row_A,
                row_B=row_B
            )

        async def quicksort(df_rows):
            if len(df_rows) <= 1:
                return df_rows

            pivot = df_rows[0]
            pivot_dict = pivot.to_dict()

            # Prepare prompts for pairwise comparison with pivot
            queries = [
                [{'role': 'user', 'content': build_prompt(row.to_dict(), pivot_dict)}]
                for row in df_rows[1:]
            ]

            # Call LLM in parallel
            results = await self.llm_model.call_batch_async(queries)
            
            print("results:")
            print(results)

            less, greater = [], []
            for row, result in zip(df_rows[1:], results):
                if "True" in result:
                    less.append(row)
                else:
                    greater.append(row)

            # Recurse
            sorted_less = await quicksort(less)
            sorted_greater = await quicksort(greater)
            return sorted_less + [pivot] + sorted_greater

        row_list = list(data.itertuples(index=False, name=None))
        col_names = list(data.columns)
        row_series_list = [pd.Series(row, index=col_names) for row in row_list]

        sorted_rows = await quicksort(row_series_list)

        result_df = pd.DataFrame([row.to_dict() for row in sorted_rows])
        return result_df

    async def execute_node(self, node):
        node_type = node["Node Type"]
        
        if node_type == "Table":
            return node
        elif node_type == "Join":
            node["Inputs"] = [await self.execute_node(input_node) for input_node in node["Inputs"]]
            if "UDF" in node:
                UDF_name = node["UDF"]["UDF Name"]
                print("Executing UDF: ", UDF_name, node["UDF"]["Input Columns"])
                new_table_name = (UDF_name + '_result').replace(" ", "")
                if self.check_temp_table_exists(new_table_name):
                    print(f"Temporary table {input_table_alias} already exists. Skipping creation.")
                    return
                if node["Inputs"][0]["Node Type"] == "Table":
                    join_table_left = node["Inputs"][0]["Table Name"]
                else:
                    join_table_left = f"{UDF_name}_left_table"
                    self.create_table_from_json(join_table_left, node["Inputs"][0])
                    node["Inputs"][0] = {"Node Type": "Table", "Table Name": join_table_left} # change input node to table node
                col_lst_l = parse_column_list(node["UDF"]["Input Columns"][0])
                columns_l_str = ','.join(f'[{col}]' for col in col_lst_l)
                data_left = self.obtain_distinct_val(join_table_left, columns_l_str)
                if node["Inputs"][1]["Node Type"] == "Table":
                    join_table_right = node["Inputs"][1]["Table Name"]
                else:
                    join_table_right = f"{UDF_name}_right_table"
                    self.create_table_from_json(join_table_right, node["Inputs"][1])
                    node["Inputs"][1] = {"Node Type": "Table", "Table Name": join_table_right} # change input node to table node
                col_lst_r = parse_column_list(node["UDF"]["Input Columns"][1])
                columns_r_str = ','.join(f'[{col}]' for col in col_lst_r)
                data_right = self.obtain_distinct_val(join_table_right, columns_r_str)

                if len(data_left) > 500 or len(data_right) > 500:
                    raise ValueError(f"The input data size {len(data_left)} or {len(data_right)} is too large. Program paused. Please check the data size.")

                print("data left is:")
                print(data_left)
                print("data right is:")
                print(data_right)
                
                # Select join algorithm implementation:
                # Option 1: Single LLM call (fastest, lower accuracy)
                # joined_data = await self.join_one_llm_call(data_left, data_right, col_lst_l, col_lst_r, UDF_name)
                
                # Option 2: Smart batching (recommended - adaptive batch sizing)
                joined_data = await self.join_smart_batching(data_left, data_right, col_lst_l, col_lst_r, UDF_name)
                
                # Option 3: Nested loop join (high accuracy, slower)
                # joined_data = await self.join_nested_loop(data_left, data_right, col_lst_l, col_lst_r, UDF_name, columns_l_str, columns_r_str)
                
                # Option 4: Projection-based join (efficient for small right table domains)
                # joined_data = await self.join_projection_based(data_left, data_right, col_lst_l, col_lst_r, UDF_name, columns_l_str, columns_r_str)

                input_table_schema = pd.concat([self.obtain_column_schema_from_sqlite(join_table_left), self.obtain_column_schema_from_sqlite(join_table_right)], axis=0)
                self.create_temp_table_from_dataframe(joined_data, new_table_name, input_table_schema)
                schema = self.obtain_column_schema_from_sqlite(new_table_name)
                Table_description = f'This table includes pairs selected from the columns ({columns_l_str}) and ({columns_r_str}) from tables {join_table_left} and {join_table_right} respectively, meeting the \'{UDF_name}\' condition. You can use these ({columns_l_str}) and ({columns_r_str}) pairs to perform join operations with Table {join_table_left} and Table {join_table_right}.'
                schema_str = "\n".join([",".join(map(str, row)) for row in schema.values])
                schema = f"\nTable: {new_table_name}\nTable Description: {Table_description}\n{schema_str}"
                self.extended_schema.append(schema)
                return node
            else:
                return node

        else:
            node["Input"] = await self.execute_node(node["Input"])
            if "UDF" in node:
                UDF_name = node["UDF"]["UDF Name"]
                col_lst = parse_column_list(node["UDF"]["Input Columns"])
                print("Executing UDF: ", UDF_name, node["UDF"]["Input Columns"])
                
                # if node type is not a table, create a temp table
                if node["Input"]["Node Type"] == "Table":
                    input_table_alias = node["Input"]["Table Name"]
                else:
                    try:
                        input_table_alias = f"{UDF_name}_input_table"
                        column_mappings = self.create_table_from_json(input_table_alias, node["Input"])
                        node["Input"] = {"Node Type": "Table", "Table Name": input_table_alias} 
                        self.json_tree = update_query_plan(self.json_tree, column_mappings)
                        if len(column_mappings):
                            col_lst = update_column_lst(node["UDF"]["Input Columns"], column_mappings)
                    except Exception as e:
                        print(e)
                if node["Node Type"] != 'Aggregation': # get distinct value for non-aggregation operators
                    columns_str = ','.join(f'[{col}]' for col in col_lst)
                    data = self.obtain_distinct_val(input_table_alias, columns_str)
                else: # for aggregation operators
                    if "GroupBy Columns" in node:
                        groupby_cols = parse_column_list(node["GroupBy Columns"])
                        col_lst = list(set(groupby_cols + col_lst))
                    columns_str = ','.join(f'[{col}]' for col in col_lst)
                    data = self.obtain_subset_val(input_table_alias, columns_str)
                value_list = parse_dataframe_to_str(data[col_lst])
                # UDF rewrite
                input_table_schema = self.obtain_column_schema_from_sqlite(input_table_alias)
                col_description_lst = obtain_column_description(self.schema_dict, col_lst)
                if self.optimization_udf_rewrite:
                    if self.UDF_rewrite(node_type, UDF_name, col_lst, col_description_lst, value_list):    
                        return node
                if node_type == "Selection":
                    new_table_name = (UDF_name + '_result_of_' + input_table_alias).replace(" ", "")
                    if self.check_temp_table_exists(new_table_name):
                        print(f"Temporary table {new_table_name} already exists. Skipping creation.")
                        return node
                    try:
                        print("The input data size is " + str(len(data)))
                        if len(data) > 1000:
                            raise ValueError(f"The input data size {len(data)} is too large. Program paused. Please check the data size.")
                        # 2- ROW EXECUTION
                        queries = []
                        for idx, row in data.iterrows():
                            formatted_row = ", ".join(f"{col}: {val}" for col, val in row.items())
                            impute = selection_imputation_row.format(col = col_lst, UDF_name = UDF_name, row = formatted_row)
                            query = [{'role': 'user', 'content': impute}]
                            queries.append(query)
                        results = await self.llm_model.call_batch_async(queries)
                        results = [int(str(x).strip('"').strip("'")) for x in results]
                        data[UDF_name] = results
                        self.imputation_result = data
                    except Exception as e:
                        print("Error:", str(e))
                    
                    # write the data into a temp table.
                    self.create_temp_table_from_dataframe(data, new_table_name, input_table_schema)
                    schema = self.obtain_column_schema_from_sqlite(new_table_name)
                    schema.loc[schema['column_name'] == UDF_name, 'column_description'] = (f'{UDF_name} is a predicate derived from the columns ({columns_str}) in Table {input_table_alias}. You can use {columns_str} to join with Table {input_table_alias}')
                    schema.loc[schema['column_name'] == UDF_name, 'data_type'] = 'TEXT'
                    schema.loc[schema['column_name'] == UDF_name, 'value_description'] = ('"1" means the selection predicate is satisfied and "0" means that the selection predicate is not satisfied')
                    schema_str = "\n".join([",".join(map(str, row)) for row in schema.values])
                    schema = f"\nTable: {new_table_name}\n{schema_str}"
                    self.extended_schema.append(schema)
                    return node

                elif node_type == "Projection":
                    new_table_name = (UDF_name + '_from_' + '_'.join(col_lst)).replace(" ", "")
                    if self.check_temp_table_exists(new_table_name):
                        print(f"Temporary table {input_table_alias} already exists. Skipping creation.")
                        return node
                    try:
                        print("The input data size is " + str(len(data)))
                        if len(data) > 1000:
                            raise ValueError(f"The input data size {len(data)} is too large. Program paused. Please check the data size.")
                        new_col_name = node['UDF']['Output Column']
                        queries = []
                        for idx, row in data.iterrows():
                            formatted_row = ", ".join(f"{col}: {val}" for col, val in row.items())
                            impute = projection_imputation_row.format(col = col_lst, UDF_name = UDF_name, new_col_name = new_col_name, row = formatted_row)
                            query = [{'role': 'user', 'content': impute}]
                            queries.append(query)
                        results = await self.llm_model.call_batch_async(queries)
                        data[new_col_name] = results
                        print(data)
                        self.imputation_result = data
                    except Exception as e:
                        print("Error:", str(e))     
                    self.create_temp_table_from_dataframe(data, new_table_name, input_table_schema)
                    schema = self.obtain_column_schema_from_sqlite(new_table_name)
                    schema.loc[schema['column_name'] == new_col_name, 'column_description'] = (f'{new_col_name} is extracted from the columns ({columns_str}) in Table {input_table_alias}. You can use ({columns_str}) to join this table with {input_table_alias}')
                    schema.loc[schema['column_name'] == new_col_name, 'value_description'] = (f'{UDF_name}')
                    schema_str = "\n".join([",".join(map(str, row)) for row in schema.values])
                    schema = f"\nTable: {new_table_name}\n{schema_str}"
                    self.extended_schema.append(schema)
                    return node

                elif node_type == "TopK":
                    k = node["k"]
                    new_table_name = (UDF_name + '_sort_by_' + '_'.join(col_lst)).replace(" ", "")
                    if self.check_temp_table_exists(new_table_name):
                        print(f"Temporary table {input_table_alias} already exists. Skipping creation.")
                        return node
                    try:
                        print("The input data size is " + str(len(data)))
                        if len(data) > 1000:
                            raise ValueError(f"The input data size {len(data)} is too large. Program paused. Please check the data size.")
                        print(data)
                        data = await self.llm_quick_sort(data, col_lst, UDF_name)    
                        data[f"{UDF_name}_rank"] = list(range(1, len(data) + 1))  
                        self.imputation_result = data
                        output_col = f"{UDF_name}_rank" 
                        column_description = f"{UDF_name}_rank is infered by the columns ({columns_str}) in Table {input_table_alias}. You can use ({columns_str}) to join this table with {input_table_alias}. "
                        value_description = f"{UDF_name}_rank is an integer rank indicating how well each item satisfies the criterion '{UDF_name}'. A smaller rank value (i.e., closer to 1) means better alignment. To retrieve top items based on this ranking, make sure to join with table {input_table_alias} and apply ORDER BY {UDF_name}_rank ASC."                     
                        if k != 'inf':
                            value_description += f" LIMIT {k}"
                    except Exception as e:
                        print("Error:", str(e)) 

                    self.create_temp_table_from_dataframe(data, new_table_name, input_table_schema)
                    schema = self.obtain_column_schema_from_sqlite(new_table_name)
                    schema.loc[schema['column_name'] == output_col, 'column_description'] = column_description
                    schema.loc[schema['column_name'] == output_col, 'value_description'] = value_description
                    schema_str = "\n".join([",".join(map(str, row)) for row in schema.values])
                    schema = f"\nTable: {new_table_name}\n{schema_str}"
                    self.extended_schema.append(schema)
                    return node
                    
                elif node_type == "Aggregation":
                    if "GroupBy Columns" in node:
                        groupby_val_list = parse_dataframe_to_str(data[groupby_cols])
                        new_table_name = (UDF_name + '_result_group_by_' + '_'.join(groupby_cols)).replace(" ", "")
                        if self.check_temp_table_exists(new_table_name):
                            print(f"Temporary table {input_table_alias} already exists. Skipping creation.")
                            return node
                        try:
                            impute = aggregation_imputation_with_groupby.format(col = col_lst, groupby_cols = groupby_cols, UDF_name = UDF_name, val_list = value_list, groupby_cols_val_list = groupby_val_list, Question = self.question)
                            query = [{'role': 'user', 'content': impute}]
                            result = self.llm_model.call(query)
                            data = parse_json_response_to_df(result)
                            if data is None or data.empty:
                                raise ValueError(f"Failed to parse aggregation result for {UDF_name}: got None or empty DataFrame")
                            self.imputation_result = data
                        except Exception as e:
                            print(f"Error in aggregation UDF execution with groupby: {str(e)}")
                            raise  # Re-raise to prevent continuing with None data
                        
                        self.create_temp_table_from_dataframe(data, new_table_name, input_table_schema)
                        schema = self.obtain_column_schema_from_sqlite(new_table_name)
                        schema.loc[schema['column_name'] == UDF_name, 'column_description'] = (f'The "{UDF_name}" column is derived by applying the "{UDF_name}" operation to the grouped data in Table {input_table_alias} based on the columns ({groupby_cols}). This aggregation is intended to help answer the question: "{self.question}".')
                        schema.loc[schema['column_name'] == UDF_name, 'value_description'] = (f'The values in the "{UDF_name}" column represent the result of the "{UDF_name}" aggregation operation. Each value summarizes the aggregation outcome for its respective group.')
                        schema_str = "\n".join([",".join(map(str, row)) for row in schema.values])
                        schema = f"\nTable: {new_table_name}\n{schema_str}"
                        self.extended_schema.append(schema)
                        return node
                    else:
                        new_table_name = (UDF_name + '_result').replace(" ", "")
                        if self.check_temp_table_exists(new_table_name):
                            print(f"Temporary table {input_table_alias} already exists. Skipping creation.")
                            return node                       
                        try:
                            impute = aggregation_imputation.format(col = col_lst, UDF_name = UDF_name, val_list = value_list, Question = self.question)
                            query = [{'role': 'user', 'content': impute}]
                            result = self.llm_model.call(query)
                            data = parse_json_response_to_df(result)
                            if data is None or data.empty:
                                raise ValueError(f"Failed to parse aggregation result for {UDF_name}: got None or empty DataFrame")
                        except Exception as e:
                            print(f"Error in aggregation UDF execution: {str(e)}")
                            raise  # Re-raise to prevent continuing with None data
                        
                        self.create_temp_table_from_dataframe(data, new_table_name, input_table_schema)
                        schema = self.obtain_column_schema_from_sqlite(new_table_name)
                        schema.loc[schema['column_name'] == 'agg_result', 'column_description'] = (f'The \"agg_result\" column contains only a single value, representing the outcome of the \"{UDF_name}\" operation applied on all values of \"{col_lst}\" in Table {input_table_alias}. This value is intended to help answer the question: \"{self.question}\".')
                        schema_str = "\n".join([",".join(map(str, row)) for row in schema.values])
                        schema = f"\nTable: {new_table_name}\n{schema_str}"
                        self.extended_schema.append(schema)
                        return node
            else:
                return node
            


    async def join_smart_batching(self, data_left, data_right, col_lst_l, col_lst_r, UDF_name):
        """
        Join implementation using adaptive batch sizing.
        Intelligently determines optimal batch sizes for each table based on data complexity.
        Recommended for most use cases.
        """
        print("="*8)
        print("SMART-BATCHING IMPLEMENTATION")
        start_time = time.perf_counter()

        # Step 1: Get sample data (first 3 rows from each)
        sample_left = data_left.head(3)
        sample_right = data_right.head(3)

        # Step 2: Prepare batch size determination prompt
        batch_size_prompt = f"""Given the following information about a join operation:

        UDF Name: {UDF_name}
        Total rows in left table: {len(data_left)}
        Total rows in right table: {len(data_right)}

        Sample data from left table (first 3 rows):
        {sample_left[col_lst_l].to_string()}

        Sample data from right table (first 3 rows):
        {sample_right[col_lst_r].to_string()}

        Based on:
        1. The complexity of the join operation (UDF: {UDF_name})
        2. The context length in each row for EACH table (they may differ)
        3. The total number of rows to process in EACH table

        Determine the optimal batch size for EACH table separately. Consider:
        - If contexts are short and the problem is simple: use larger batch sizes (e.g., 10-50)
        - If contexts are medium length or moderate complexity: use medium batch sizes (e.g., 5-10)
        - If contexts are very long or the problem is difficult: use batch size of 1
        - The two tables may have different optimal batch sizes based on their row complexity

        Respond with ONLY a JSON object in this format:
        {{"batch_size_left": <number>, "batch_size_right": <number>, "reasoning": "<brief explanation>"}}"""

        # Step 3: Call LLM to determine batch sizes
        query_batch = [{'role': 'user', 'content': batch_size_prompt}]
        batch_result = self.llm_model.call(query_batch)
        print(f"Batch size determination result: {batch_result}")

        # Parse batch sizes from response
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', batch_result, re.DOTALL)
            if json_match:
                batch_info = json.loads(json_match.group())
                batch_size_left = int(batch_info.get('batch_size_left', 5))
                batch_size_right = int(batch_info.get('batch_size_right', 5))
                reasoning = batch_info.get('reasoning', 'No reasoning provided')
                print(f"Determined batch size for left table: {batch_size_left}")
                print(f"Determined batch size for right table: {batch_size_right}")
                print(f"Reasoning: {reasoning}")
            else:
                batch_size_left = 5  # Default fallback
                batch_size_right = 5  # Default fallback
                print(f"Could not parse batch sizes, using defaults: left={batch_size_left}, right={batch_size_right}")
        except Exception as e:
            print(f"Error parsing batch sizes: {e}")
            batch_size_left = 5  # Default fallback
            batch_size_right = 5  # Default fallback
            print(f"Using default batch sizes: left={batch_size_left}, right={batch_size_right}")

        # Step 4: Split data into batches with different sizes
        def create_batches(df, batch_size):
            """Split dataframe into batches"""
            batches = []
            for i in range(0, len(df), batch_size):
                batches.append(df.iloc[i:i+batch_size])
            return batches

        left_batches = create_batches(data_left, batch_size_left)
        right_batches = create_batches(data_right, batch_size_right)

        print(f"Created {len(left_batches)} batches for left table (batch size: {batch_size_left})")
        print(f"Created {len(right_batches)} batches for right table (batch size: {batch_size_right})")

        # Step 5: Process each batch combination and collect results
        all_joined_data = []
        total_llm_calls = 1  # Count the batch size determination call

        for i, left_batch in enumerate(left_batches):
            for j, right_batch in enumerate(right_batches):
                print(f"Processing batch combination: left[{i}] x right[{j}]")
                
                # Prepare data for this batch
                value_list_l = parse_dataframe_to_str(left_batch[col_lst_l])
                value_list_r = parse_dataframe_to_str(right_batch[col_lst_r])
                
                # Create join prompt for this batch
                impute = join_imputation.format(
                    col_l=col_lst_l, 
                    col_r=col_lst_r, 
                    UDF_name=UDF_name, 
                    col_l_value_list=value_list_l, 
                    col_r_value_list=value_list_r
                )
                
                # Call LLM for this batch
                query = [{'role': 'user', 'content': impute}]
                result = self.llm_model.call(query)
                total_llm_calls += 1
                
                # Parse and collect results
                batch_joined_data = parse_json_response_to_df(result)
                all_joined_data.append(batch_joined_data)

        # Step 6: Combine all batch results
        joined_data = pd.concat(all_joined_data, ignore_index=True)

        end_time = time.perf_counter()
        latency = end_time - start_time

        print(f"# of LLM calls: {total_llm_calls}")
        print(f"Latency: {latency:.6f} seconds")
        print(f"Total batch combinations processed: {len(left_batches)} × {len(right_batches)} = {len(left_batches) * len(right_batches)}")
        print("Final joined data:")
        print(joined_data)
        
        return joined_data

    async def join_one_llm_call(self, data_left, data_right, col_lst_l, col_lst_r, UDF_name):
        """
        Join implementation using a single LLM call for all data.
        Lower accuracy but fastest for small datasets.
        """
        print("="*8)
        print("1. ONE LLM CALL IMPLEMENTATION")
        start_time = time.perf_counter()
        
        value_list_l = parse_dataframe_to_str(data_left[col_lst_l])
        value_list_r = parse_dataframe_to_str(data_right[col_lst_r])
        impute = join_imputation.format(
            col_l=col_lst_l, 
            col_r=col_lst_r, 
            UDF_name=UDF_name, 
            col_l_value_list=value_list_l, 
            col_r_value_list=value_list_r
        )
        query = [{'role': 'user', 'content': impute}]
        result = self.llm_model.call(query)
        print("# of LLM calls: 1")
        joined_data = parse_json_response_to_df(result)
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Latency: {latency:.6f} seconds")
        print(joined_data)
        return joined_data

    async def join_nested_loop(self, data_left, data_right, col_lst_l, col_lst_r, UDF_name, columns_l_str, columns_r_str):
        """
        Join implementation using nested loop with row-by-row comparison.
        High accuracy but slower for large datasets.
        """
        print("="*8)
        print("2. NESTED LOOP JOIN IMPLEMENTATION")
        start_time = time.perf_counter()
        
        queries = []
        row_pairs = list(itertools.product(data_left.iterrows(), data_right.iterrows()))
        
        for (idx_l, row_l), (idx_r, row_r) in row_pairs:
            formatted_row_left = ", ".join(f"{col}: {val}" for col, val in row_l.items())
            formatted_row_right = ", ".join(f"{col}: {val}" for col, val in row_r.items())
            impute = nested_loop_join_imputation_row.format(
                UDF_name=UDF_name,
                col_l=columns_l_str,
                col_r=columns_r_str,
                col_l_val=formatted_row_left,
                col_r_val=formatted_row_right
            )
            query = [{'role': 'user', 'content': impute}]
            queries.append(query)
        
        print(f"# of LLM calls: {len(queries)}")
        results = await self.llm_model.call_batch_async(queries)
        assert len(results) == len(row_pairs), "Mismatch between number of results and row pairs"
        
        joinable_rows = []
        cur_idx = 0
        for (idx_l, row_l), (idx_r, row_r) in row_pairs:
            if results[cur_idx] == '1':
                left_value = row_l[col_lst_l[0]] if col_lst_l[0] in row_l else row_l.iloc[0]
                right_value = row_r[col_lst_r[0]] if col_lst_r[0] in row_r else row_r.iloc[0]
                joinable_rows.append([left_value, right_value])
            cur_idx += 1
        
        if col_lst_l == col_lst_r:
            col_lst_l[0] = col_lst_l[0] + '_x'
            col_lst_r[0] = col_lst_r[0] + '_y'
        
        joined_data = pd.DataFrame(joinable_rows, columns=[col_lst_l[0], col_lst_r[0]])
        joined_data = joined_data[col_lst_l + col_lst_r]
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Latency: {latency:.6f} seconds")
        print(joined_data)
        return joined_data

    async def join_projection_based(self, data_left, data_right, col_lst_l, col_lst_r, UDF_name, columns_l_str, columns_r_str):
        """
        Join implementation using projection-based approach.
        Efficient when the smaller table domain is small.
        """
        print("="*8)
        print("3. PROJECTION-BASED JOIN IMPLEMENTATION")
        start_time = time.perf_counter()
        
        queries = []
        value_list_r = parse_dataframe_to_str(data_right[col_lst_r])
        
        for idx, row in data_left.iterrows():
            formatted_row = ", ".join(f"{col}: {val}" for col, val in row.items())
            impute = projection_join_imputation.format(
                row=formatted_row,
                col_l=columns_l_str,
                col_r=columns_r_str,
                val_list_r=value_list_r,
            )
            query = [{'role': 'user', 'content': impute}]
            queries.append(query)
        
        print(f"# of LLM calls: {len(queries)}")
        results = await self.llm_model.call_batch_async(queries)
        data_left['mapped_value'] = results
        joined_data = pd.merge(data_left, data_right, left_on=['mapped_value'], right_on=col_lst_r, how='inner')
        
        if col_lst_l == col_lst_r:
            col_lst_l[0] = col_lst_l[0] + '_x'
            col_lst_r[0] = col_lst_r[0] + '_y'
        
        joined_data = joined_data[col_lst_l + col_lst_r]
        
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"Latency: {latency:.6f} seconds")
        print(joined_data)
        return joined_data