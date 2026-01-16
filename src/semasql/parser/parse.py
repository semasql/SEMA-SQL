"""
Parser Module

Utilities for parsing LLM responses, SQL queries, and data structures.
"""

import re
import json
from typing import Dict, List

import pandas as pd
import tiktoken


# ============================================================================
# SQL Parsing Functions
# ============================================================================

def sql_parse(sql_result: str) -> str:
    """
    Extract SQL query from LLM response.
    """
    sql_result = sql_result.strip()
    sql_pattern = r"""(?ix)                    
        \b(                                    
            with\s+[\w\s,]+as\s*\(.*?\)\s*select 
            | select                             
        ) .*? ;                                  
    """
    match = re.search(sql_pattern, sql_result, re.DOTALL)
    return match.group(0).strip() if match else ''


def sql_create_parse(sql_result: str) -> str:
    """
    Extract CREATE TABLE statement from SQL response.
    """
    sql_pattern = r"CREATE\s.*?;"
    match = re.search(sql_pattern, sql_result, re.DOTALL | re.IGNORECASE)
    return match.group(0).strip() if match else ''


# ============================================================================
# JSON Parsing Functions
# ============================================================================

def parse_json_response(response: str) -> dict:
    """
    Extract and parse JSON from LLM response.
    """
    try:
        json_pattern = re.compile(r'{.*}', re.DOTALL)
        match = json_pattern.search(response)
        
        if not match:
            raise ValueError("Unable to find JSON portion in response")
        
        json_str = match.group(0)
        json_str = (
            json_str.replace("True", "true")
                    .replace("False", "false")
                    .replace("None", "null")
        )
        
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(response)
        raise ValueError("Failed to decode JSON string")
    except Exception as e:
        raise ValueError(f"Error: {e}")


def parse_json_response_to_df(response) -> pd.DataFrame:
    """
    Parse LLM response and convert to DataFrame.
    """
    try:
        if isinstance(response, dict):
            json_data = response
        elif isinstance(response, list):
            # If response is already a list, use it directly
            json_data = {'result': response}
        else:
            try:
                json_data = json.loads(response)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_data = json.loads(json_match.group(0))
                else:
                    # Try to parse as a list
                    list_match = re.search(r'\[.*\]', response, re.DOTALL)
                    if list_match:
                        json_data = {'result': json.loads(list_match.group(0))}
                    else:
                        raise ValueError("No valid JSON found in the response.")
        
        # Handle both dict and list responses
        if isinstance(json_data, list):
            result_list = json_data
        elif isinstance(json_data, dict):
            result_list = json_data.get('result', json_data)
        else:
            raise ValueError(f"Unexpected JSON data type: {type(json_data)}")
        
        # Ensure result_list is iterable
        if not isinstance(result_list, list):
            raise ValueError(f"Expected list but got {type(result_list)}")
        
        cleaned_data = []
        for item in result_list:
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict item but got {type(item)}")
            cleaned_item = {}
            for key, value in item.items():
                clean_key = key.strip("[]'\"")
                cleaned_item[clean_key] = value
            cleaned_data.append(cleaned_item)
        
        return pd.DataFrame(cleaned_data)
    except Exception as e:
        print(f"Error parsing response: {e}")
        raise  # Re-raise to allow caller to handle


# ============================================================================
# Column and Data Parsing Functions
# ============================================================================

def parse_column_list(column_list: List[str]) -> List[str]:
    """
    Extract column names without table prefixes.
    """
    return [col.split('.')[-1] for col in column_list]


def parse_dataframe_to_str(df: pd.DataFrame, threshold: int = 10000) -> str:
    """
    Convert DataFrame to string with token-based truncation.
    """
    df_cleaned = df.astype(str)
    encoding = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base for compatibility
    
    result_parts = []
    total_tokens = 0
    ellipsis = "..."
    
    for _, row in df_cleaned.iterrows():
        row_str = '{' + ','.join([f'"{val}"' for val in row]) + '}'
        row_str_with_comma = ("," if result_parts else "") + row_str
        row_tokens = len(encoding.encode(row_str_with_comma))
        
        if total_tokens + row_tokens + len(encoding.encode(ellipsis)) > threshold:
            break
        
        result_parts.append(row_str_with_comma)
        total_tokens += row_tokens
    
    result_str = "".join(result_parts)
    
    full_tokens = sum(
        len(encoding.encode('{' + ','.join([f'"{val}"' for val in row]) + '}')) + 1
        for _, row in df_cleaned.iterrows()
    ) - 1
    
    if total_tokens < full_tokens:
        result_str += ellipsis
    
    return result_str


# ============================================================================
# Column Mapping Functions
# ============================================================================

def extract_column_mappings_from_sql(sql: str) -> Dict[str, str]:
    """
    Extract column name mappings from CREATE TEMP TABLE statement.
    """
    column_mappings = {}
    
    pattern = r'(\w+)\."(\w+)"\s+AS\s+"(\w+)"'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    
    for table_alias, original_col, new_col in matches:
        column_mappings[f"{table_alias}.{original_col}"] = new_col
    
    alias_pattern = r'(?:FROM|JOIN)\s+(\w+)\s+AS\s+(\w+)'
    alias_matches = re.findall(alias_pattern, sql, re.IGNORECASE)
    table_aliases = {alias: table for table, alias in alias_matches}
    
    extended_mappings = {}
    for key, value in column_mappings.items():
        alias, col = key.split('.')
        if alias in table_aliases:
            full_table_name = table_aliases[alias]
            extended_mappings[f"{full_table_name}.{col}"] = value
        extended_mappings[key] = value
    
    return extended_mappings


def rewrite_column_reference(column_ref: str, mappings: Dict[str, str]) -> str:
    """
    Rewrite column reference using mapping dictionary.
    """
    return mappings.get(column_ref, column_ref)


def update_query_plan(plan: dict, mappings: Dict[str, str]) -> dict:
    """
    Recursively update column references in query plan.
    """
    updated_plan = {}
    
    for key, value in plan.items():
        if key == "Columns" and isinstance(value, list):
            updated_plan[key] = [
                rewrite_column_reference(col, mappings) for col in value
            ]
        elif key == "UDF" and isinstance(value, dict):
            updated_udf = value.copy()
            if "Input Columns" in updated_udf:
                updated_udf["Input Columns"] = [
                    rewrite_column_reference(col, mappings) 
                    for col in updated_udf["Input Columns"]
                ]
            updated_plan[key] = updated_udf
        elif key == "Condition" and isinstance(value, str):
            updated_condition = value
            for original, new in mappings.items():
                updated_condition = re.sub(
                    r'\b' + re.escape(original) + r'\b',
                    new,
                    updated_condition
                )
            updated_plan[key] = updated_condition
        elif isinstance(value, dict):
            updated_plan[key] = update_query_plan(value, mappings)
        elif isinstance(value, list):
            updated_plan[key] = [
                update_query_plan(item, mappings) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            updated_plan[key] = value
    
    return updated_plan


def update_column_lst(original_columns: List[str], mappings: Dict[str, str]) -> List[str]:
    """
    Update column list using mappings and remove table prefixes.
    """
    rewritten_columns = [mappings.get(col, col) for col in original_columns]
    return parse_column_list(rewritten_columns)
