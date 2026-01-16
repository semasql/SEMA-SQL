"""
Base Database Executor Module

Provides base class for SQLite database operations including query execution,
schema extraction, and temporary table management.
"""

import os
from typing import List, Optional, Union

import pandas as pd
import sqlite3


class BaseDBExecutor:
    """
    Base class for database operations on SQLite databases.
    
    Provides methods for connecting to databases, executing queries,
    extracting schema information, and managing temporary tables.
    """
    
    def __init__(self, db_name: str, path: str):
        """
        Initialize database executor.
        """
        self.db_name = db_name
        self.path = path
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def connect(self) -> None:
        """Connect to the SQLite database."""
        db_path = os.path.join(self.path, self.db_name, f"{self.db_name}.sqlite")
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        print(f"Connected to database: {self.db_name}")

    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
            print(f"Connection to database {self.db_name} closed")

    def execute_query(self, sql_query: str) -> List:
        """
        Execute SQL query and return results.
        
        Args:
            sql_query: SQL query string to execute
            
        Returns:
            List of query results (empty list for non-SELECT queries)
        """
        if not self.connection:
            self.connect()

        try:
            self.cursor.execute(sql_query)
        except Exception as e:
            print(f"SQL Error: {e}")
            print(f"Query: {sql_query}")
            raise

        if sql_query.strip().upper().startswith(("SELECT", "WITH")):
            query_result = self.cursor.fetchall()
        else:
            query_result = []
        
        self.connection.commit()
        return query_result

    def obtain_distinct_val(self, table: str, columns_str: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get distinct values for specified columns.
        
        Args:
            table: Table name
            columns_str: Column names (can be comma-separated)
            limit: Optional limit on number of results
            
        Returns:
            DataFrame with distinct values
            
        Raises:
            ValueError: If no distinct values found
        """
        if not self.connection:
            self.connect()
            
        limit_str = f" LIMIT {limit}" if limit is not None else ""
        
        try:
            data = pd.read_sql(
                f"SELECT DISTINCT {columns_str} FROM {table}{limit_str};",
                self.connection
            )
        except Exception as e:
            print(f"Error querying distinct values: {e}")
            raise
        
        if data.empty:
            raise ValueError(f"No distinct values found for columns {columns_str} in table {table}.")
        
        return data
    
    def obtain_subset_val(self, table: str, columns_str: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get subset of values (not necessarily distinct) for specified columns.
        

        Returns:
            DataFrame with values
            
        Raises:
            ValueError: If no values found
        """
        if not self.connection:
            self.connect()
            
        limit_str = f" LIMIT {limit}" if limit is not None else ""
        
        data = pd.read_sql(f"SELECT {columns_str} FROM {table}{limit_str};", self.connection)
        
        if data.empty:
            raise ValueError(f"No values found for columns {columns_str} in table {table}.")
        
        return data

    def obtain_column_schema_from_sqlite(self, table: str, col_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract column schema information from SQLite database.
        
        Returns:
            DataFrame with columns: column_name, is_PrimaryKey, data_type, 
            column_description, value_description, value_examples
        """
        obtain_table_info_sql = f"""SELECT 
            name AS column_name, 
            CASE WHEN pk = 1 THEN 'Primary Key' ELSE '' END AS is_PrimaryKey,
            type AS data_type
        FROM pragma_table_info('{table}')
        """
        
        table_info = self.execute_query(obtain_table_info_sql)
        table_info = pd.DataFrame(table_info, columns=['column_name', 'is_PrimaryKey', 'data_type'])
        
        if col_list is not None:
            table_info = table_info[table_info["column_name"].isin(col_list)]
        
        table_info['column_description'] = ''
        table_info['value_description'] = ''
        
        for column_name in table_info['column_name']:
            try:
                distinct_values = self.obtain_distinct_val(table, f"[{column_name}]", 3)
                values_list = distinct_values[column_name].values.flatten().tolist()
                table_info.loc[table_info['column_name'] == column_name, 'value_examples'] = str(values_list)
            except Exception as e:
                print(f"Warning: Could not get samples for {table}.{column_name}: {e}")
                table_info.loc[table_info['column_name'] == column_name, 'value_examples'] = '[]'
        
        return table_info

    @staticmethod
    def _map_dtype(dtype) -> str:
        """
        Map pandas dtype to SQLite type.
            
        Returns:
            SQLite type string ('INTEGER', 'REAL', or 'TEXT')
        """
        if pd.api.types.is_integer_dtype(dtype):
            return 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype):
            return 'REAL'
        elif pd.api.types.is_bool_dtype(dtype):
            return 'INTEGER'
        else:
            return 'TEXT'

    def determine_sqlite_type(
        self,
        column: pd.Series,
        col_name: str,
        reference_schema: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Determine SQLite type for a column.
        """
        if reference_schema is not None and col_name in reference_schema['column_name'].values:
            return reference_schema.loc[
                reference_schema['column_name'] == col_name,
                'data_type'
            ].iloc[0]
        
        try:
            converted = column.apply(pd.to_numeric, errors='coerce')
            if converted.notnull().all():
                if (converted % 1 != 0).any():
                    return 'REAL'
                else:
                    return 'INTEGER'
        except ValueError:
            pass
        
        return 'TEXT'

    def create_temp_table_from_dataframe(
        self,
        data: pd.DataFrame,
        temp_table_name: str,
        reference_schema: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Create temporary table from pandas DataFrame.
        """
        if not self.connection:
            self.connect()

        drop_table_sql = f"DROP TABLE IF EXISTS {temp_table_name};"
        self.execute_query(drop_table_sql)

        columns = []
        for column_name in data.columns:
            sqlite_type = self.determine_sqlite_type(data[column_name], column_name, reference_schema)
            columns.append(f'"{column_name}" {sqlite_type}')
        
        columns_str = ", ".join(columns)
        create_temp_table_sql = f'CREATE TEMP TABLE {temp_table_name} ({columns_str});'
        self.cursor.execute(create_temp_table_sql)

        placeholders = ', '.join(['?' for _ in data.columns])
        insert_sql = f'INSERT INTO {temp_table_name} VALUES ({placeholders});'
        self.cursor.executemany(insert_sql, data.values.tolist())
        self.connection.commit()

    def check_temp_table_exists(self, temp_table_name: str) -> bool:
        """
        Check if temporary table exists.
        Returns:
            True if table exists, False otherwise
        """
        if not self.connection:
            self.connect()

        try:
            check_table_sql = (
                f"SELECT name FROM sqlite_temp_master "
                f"WHERE type='table' AND name='{temp_table_name}';"
            )
            self.cursor.execute(check_table_sql)
            table_exists = self.cursor.fetchone()
            
            if not table_exists:
                check_table_sql_fallback = (
                    f"SELECT name FROM sqlite_master "
                    f"WHERE type='table' AND name='{temp_table_name}';"
                )
                self.cursor.execute(check_table_sql_fallback)
                table_exists = self.cursor.fetchone()

            return table_exists is not None
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return False
    
    def drop_all_temp_table(self) -> None:
        """Drop all temporary tables created during the session."""
        if not self.connection:
            self.connect()
            
        temp_tables_query = "SELECT name FROM sqlite_temp_master WHERE type='table';"
        self.cursor.execute(temp_tables_query)
        temp_tables = self.cursor.fetchall()

        for table in temp_tables:
            drop_table_query = f"DROP TABLE IF EXISTS {table[0]};"
            self.cursor.execute(drop_table_query)
            print(f"Dropped temporary table: {table[0]}")
        
        self.connection.commit()
