"""
Schema Generator Module

Utilities for loading and generating database schema documents in YAML format.
Supports reading existing YAML schemas and generating new schemas using LLMs.
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional, Set

import pandas as pd
import yaml

from src.semasql.llm.LLMCaller import LLMCaller


# ============================================================================
# Schema Loading Functions
# ============================================================================

def obtain_db_schema(path: str, db: str) -> str:
    """
    Load database schema from YAML file.
    
    Args:
        path: Base path to database directory
        db: Database name (subdirectory name)
        
    Returns:
        YAML schema string
        
    Raises:
        FileNotFoundError: If schema.yaml does not exist
    """
    schema_path = os.path.join(path, db, 'schema.yaml')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        return f.read()


def is_valid_yaml(yaml_string: str) -> bool:
    """Validate YAML string format."""
    try:
        yaml.safe_load(yaml_string)
        return True
    except yaml.YAMLError as e:
        print(f"YAML Error: {e}")
        return False


def load_schema_dict(schema_string: str) -> Dict:
    """
    Load YAML schema string into dictionary.
    
    Args:
        schema_string: YAML-formatted schema string
        
    Returns:
        Dictionary representation of schema, or empty dict if invalid
    """
    if is_valid_yaml(schema_string):
        schema_dict = yaml.safe_load(schema_string)
        return schema_dict if schema_dict else {}
    return {}


# ============================================================================
# Schema Generation Functions
# ============================================================================

def _read_csv_with_encoding(file_path: str) -> pd.DataFrame:
    """Read CSV file with UTF-8 fallback to CP1252."""
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='cp1252')


def check_merge(columns_in_columns_df: Set[str], columns_in_table_info: Set[str]) -> None:
    """
    Validate column name matching between database schema and CSV descriptions.
    
    Args:
        columns_in_columns_df: Column names from database schema
        columns_in_table_info: Column names from CSV description
        
    Raises:
        ValueError: If columns exist in database but not in CSV
    """
    missing_in_table_info = columns_in_columns_df - columns_in_table_info
    missing_in_columns_df = columns_in_table_info - columns_in_columns_df

    if missing_in_table_info or missing_in_columns_df:
        print("Column mismatch detected:")
        if missing_in_table_info:
            print(f" - Missing in table_info: {sorted(missing_in_table_info)}")
            raise ValueError(
                "Column mismatch between table_info and columns_df. "
                "See printout above."
            )
        if missing_in_columns_df:
            print(f"Redundant columns in database_description: {sorted(missing_in_columns_df)}")


def generate_db_schema(path: str, db: str, model: str = 'gpt', llm_temperature: float = 0) -> str:
    """
    Generate YAML schema document from CSV descriptions and database using LLM.
    
    Args:
        path: Base path to database directory
        db: Database name (subdirectory name)
        model: LLM model ('gpt', 'qwen', 'claude', 'gemini', 'ds')
        llm_temperature: Temperature for LLM generation
        
    Returns:
        Generated YAML schema string
        
    Raises:
        FileNotFoundError: If required files not found
        ValueError: If generated YAML is invalid
    """
    db_path = os.path.join(path, db, f"{db}.sqlite")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    csv_path = os.path.join(path, db, 'database_description')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Database description directory not found: {csv_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema_data = {'database_name': db, 'tables': []}
    
    try:
        for filename in sorted(os.listdir(csv_path)):
            if not filename.endswith('.csv'):
                continue
                
            file_path = os.path.join(csv_path, filename)
            table = filename[:-4]
            
            try:
                table_info = _read_csv_with_encoding(file_path)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue
            
            if 'column_name' in table_info.columns:
                table_info = table_info.drop(columns=['column_name'])
            if 'original_column_name' in table_info.columns:
                table_info = table_info.rename(columns={'original_column_name': 'column_name'})
            
            required_cols = ['column_name', 'column_description', 'value_description']
            if not all(col in table_info.columns for col in required_cols):
                print(f"Warning: Missing required columns in {filename}")
                continue
            
            table_info = table_info[required_cols]
            table_info['column_name'] = table_info['column_name'].astype(str).str.strip()
            
            cursor.execute(f"PRAGMA table_info({table});")
            columns_info = cursor.fetchall()
            columns_df = pd.DataFrame(
                columns_info,
                columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk']
            )
            columns_df = columns_df[['name', 'pk', 'type']]
            columns_df = columns_df.rename(columns={'name': 'column_name', 'type': 'data_type'})
            columns_df['is_primary_key'] = columns_df['pk'].apply(lambda x: x == 1)
            
            check_merge(set(columns_df['column_name']), set(table_info['column_name']))
            table_info = pd.merge(columns_df, table_info, on='column_name', how='left')
            
            table_info['data_samples'] = None
            for column_name in table_info['column_name']:
                try:
                    cursor.execute(f"SELECT DISTINCT [{column_name}] FROM {table} LIMIT 3;")
                    distinct_values = [row[0] for row in cursor.fetchall()]
                    table_info.loc[table_info['column_name'] == column_name, 'data_samples'] = str(distinct_values)
                except Exception as e:
                    print(f"Warning: Could not get samples for {table}.{column_name}: {e}")
            
            table_schema = {'name': table, 'columns': []}
            for _, row in table_info.iterrows():
                column_schema = {
                    'name': row['column_name'],
                    'description': row.get('column_description', ''),
                    'data_type': row['data_type'],
                    'data_description': row.get('value_description', ''),
                    'is_primary_key': row['is_primary_key'],
                    'data_samples': eval(row['data_samples']) if row['data_samples'] else []
                }
                table_schema['columns'].append(column_schema)
            
            schema_data['tables'].append(table_schema)
        
        schema_json_str = json.dumps(schema_data, indent=2, ensure_ascii=False)
        
        prompt = f"""You are a database schema expert. Given the following database schema information in JSON format, generate a comprehensive YAML schema document.

The YAML schema should follow this structure:
- name: database name
- description: A clear, comprehensive description of what the database contains
- tables: A list of tables, each with:
  - name: table name
  - description: A detailed description of what the table contains
  - columns: A list of columns, each with:
    - name: column name
    - description: Brief description (should indicate if it's a Primary Key)
    - data type: The SQL data type
    - data description: Detailed description of the data
    - data samples: List of sample values (can include None)

Database Schema Information:
{schema_json_str}

Please generate a well-structured YAML document that:
1. Provides a comprehensive database description
2. Includes detailed table descriptions that explain the purpose and content of each table
3. Includes detailed column descriptions with clear explanations
4. Properly formats data samples (use None for null values, keep strings as strings, numbers as numbers)
5. Uses proper YAML indentation and formatting

Return ONLY the YAML document, no additional text or explanations."""

        llm = LLMCaller(model=model)
        query = [
            {"role": "system", "content": "You are an expert database schema documentation specialist."},
            {"role": "user", "content": prompt}
        ]
        
        yaml_schema = llm.call(query, temperature=llm_temperature)
        
        yaml_schema = yaml_schema.strip()
        if yaml_schema.startswith('```yaml'):
            yaml_schema = yaml_schema[7:]
        elif yaml_schema.startswith('```'):
            yaml_schema = yaml_schema[3:]
        if yaml_schema.endswith('```'):
            yaml_schema = yaml_schema[:-3]
        yaml_schema = yaml_schema.strip()
        
        if not is_valid_yaml(yaml_schema):
            raise ValueError("Generated YAML schema is invalid. Please check the LLM output.")
        
        schema_yaml_path = os.path.join(path, db, 'schema.yaml')
        with open(schema_yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_schema)
        
        print(f"Schema YAML generated successfully: {schema_yaml_path}")
        return yaml_schema
        
    finally:
        cursor.close()
        conn.close()


# ============================================================================
# Schema Utility Functions
# ============================================================================

def get_column_description_tc(tables: List[Dict], table_name: str, column_name: str) -> Optional[str]:
    """
    Get column description for specific table and column.
    
    Args:
        tables: List of table dictionaries from schema
        table_name: Name of the table
        column_name: Name of the column
        
    Returns:
        Column description or None if not found
    """
    for table in tables:
        if table['name'] == table_name:
            for column in table['columns']:
                if column['name'] == column_name:
                    return column['description']
    return None


def get_column_description(tables: List[Dict], column_name: str) -> Optional[str]:
    """
    Get column description by column name (returns first match).
    
    Args:
        tables: List of table dictionaries from schema
        column_name: Name of the column to find
        
    Returns:
        Column description or None if not found
    """
    for table in tables:
        for column in table['columns']:
            if column['name'] == column_name:
                return column['description']
    return None


def obtain_column_description(schema_dict: Dict, cols: List[str]) -> List[Optional[str]]:
    """
    Get column descriptions for a list of column names.
    
    Args:
        schema_dict: Schema dictionary loaded from YAML
        cols: List of column names
        
    Returns:
        List of column descriptions (may contain None for columns not found)
    """
    if 'tables' not in schema_dict:
        return [None] * len(cols)
        
    tables = schema_dict['tables']
    return [get_column_description(tables, col) for col in cols]
