import sqlite3
import pandas as pd

def obtain_distinct_val(columns, path, db_name, table):
    """
    Obtain the distinct value of a given column.
    """
    columns_str = ','.join(f'[{col}]' for col in columns)
    conn = sqlite3.connect(f"{path}/{db_name}/{db_name}.sqlite")
    data = pd.read_sql(f"SELECT DISTINCT {columns_str} FROM {table} limit 3;", conn)
    conn.close()
    return data


def get_table_names(path, db_name):
    """
    Connect to the SQLite database and retrieve all table names.
    """
    # Connect to the SQLite database
    connection = sqlite3.connect(f"{path}/{db_name}/{db_name}.sqlite")
    
    try:
        cursor = connection.cursor()
        
        # Execute a query to retrieve all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='sqlite_sequence';")
        
        # Fetch all results
        table_names = cursor.fetchall()
        
        # Convert list of tuples to list of strings
        table_names = [name[0] for name in table_names]
        
        return table_names
    
    finally:
        # Close the connection
        connection.close()


def execute_sql(path, db_name, sql_query):
    """
    Connect to the SQLite database and execute the specified SQL query.
    """
    
    # Construct the database file path by joining path, db_name, and the sqlite file extension
    db_path = f"{path}/{db_name}/{db_name}.sqlite"
    
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    
    try:
        cursor = connection.cursor()
        
        # Execute the SQL query
        cursor.execute(sql_query)
        
        # If the SQL query is a SELECT statement, fetch the results
        if sql_query.strip().upper().startswith("SELECT"):
            query_result = cursor.fetchall()
        else:
            # For other types of SQL queries (like CREATE TABLE), return an empty list
            query_result = []
        
        # Commit any changes (like creating tables or inserting data)
        connection.commit()
        
        return query_result
    
    finally:
        # Close the connection
        connection.close()