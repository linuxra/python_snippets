import pyodbc
from typing import Dict, List, Tuple, Optional


class QueryManager:
    """
    A class to manage SQL queries in Teradata.

    :param dsn: The Data Source Name of the Teradata database.
    """

    def __init__(self, dsn: str):
        self.connection = pyodbc.connect(dsn)

    def generate_query(self, main_table: Tuple[str, str],
                       join_tables: Optional[List[Tuple[Tuple[str, str], str]]] = None,
                       columns: str = "*", case_columns: Optional[List[str]] = None,
                       conditions: Optional[Dict[str, str]] = None) -> str:
        """
        Generates a SQL query based on the provided parameters.

        :param main_table: A tuple with the main table name and its alias.
        :param join_tables: A list of tuples, each containing another tuple with the join table name and its alias, and the join condition.
        :param columns: A string with the columns to select. Defaults to "*".
        :param case_columns: A list of strings with the CASE statements to include in the SELECT clause.
        :param conditions: A dictionary with the WHERE conditions.
        :return: The generated SQL query.
        """
        main_table_name, main_table_alias = main_table
        query = f"SELECT {columns}"

        if case_columns:
            for case_column in case_columns:
                query += f", {case_column}"

        query += f" FROM {main_table_name} AS {main_table_alias}"

        if join_tables:
            for (table_name, table_alias), condition in join_tables:
                query += f" LEFT JOIN {table_name} AS {table_alias} ON {condition}"

        query += " WHERE CAST(date AS DATE FORMAT 'YYYY-MM-DD') >= ? AND CAST(date AS DATE FORMAT 'YYYY-MM-DD') < ?"

        if conditions:
            for field, value in conditions.items():
                if "?" in value:  # Parameterized value
                    query += f" AND {field} {value}"
                else:  # Static value
                    query += f" AND {field} = ?"

        return query

    def run_query(self, query: str, params: List):
        """
        Executes a SQL query and returns the results.

        :param query: The SQL query to execute.
        :param params: The parameters to pass to the query.
        :return: The results of the query.
        """
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()

    def create_volatile_table(self, table_name: str, query: str, params: List):
        """
        Creates a volatile table in the database from a given query.

        :param table_name: The name of the volatile table to create.
        :param query: The SQL query to use to populate the volatile table.
        :param params: The parameters to pass to the query.
        """
        create_table_query = f"""
        CREATE VOLATILE TABLE {table_name} AS (
            {query}
        ) WITH DATA ON COMMIT PRESERVE ROWS
        """
        cursor = self.connection.cursor()
        cursor.execute(create_table_query, params)
        self.connection.commit()

    def check_and_drop_table(self, table_name: str):
        """
        Checks if a table exists in the database and drops it if it does.

        :param table_name: The name of the table to check and drop    check and drop.
    """
    cursor = self.connection.cursor()
    cursor.execute(f"""
        SELECT * 
        FROM dbc.tablesV 
        WHERE TableName = '{table_name}'
    """)
    result = cursor.fetchone()
    if result:
        cursor.execute(f"DROP TABLE {table_name}")
        self.connection.commit()

    def create_index(self, table_name: str, columns: List[str]):
        """
        Creates an index on a table in the database.

        :param table_name: The name of the table to create an index on.
        :param columns: The list of columns to include in the index.
        """
        cursor = self.connection.cursor()
        columns_str = ", ".join(columns)
        cursor.execute(f"CREATE INDEX ({columns_str}) ON {table_name}")
        self.connection.commit()


query_manager = QueryManager(dsn="your_dsn_here")

# Define the main table and its alias
main_table = ("table1", "t1")

# Define the join tables and their aliases
join_tables = [
    (("table2", "t2"), "t1.id = t2.id"),
    (("table3", "t3"), "t2.id = t3.id")
]

# Define the columns to select
columns = "t1.id, t1.name, t2.value, t3.value"

# Define the CASE statements
case_columns = [
    "CASE WHEN t1.id IN (1,2,3) OR t2.value = 'oo' THEN 'P' ELSE 'O' END AS var"
]

# Define the WHERE conditions
conditions = {
    "t1.id": "?",
    "t2.value": "IN (?)"
}

# Generate the query
query = query_manager.generate_query(main_table, join_tables, columns, case_columns, conditions)

# Define the parameters for the query
params = ['2023-01-01', '2023-02-01', 1, ('oo', 'uu')]

# Execute the query
results = query_manager.run_query(query, params)

# Create a volatile table
query_manager.create_volatile_table("volatile_table", query, params)

# Check and drop a table
query_manager.check_and_drop_table("table_to_drop")

# Create an index
query_manager.create_index("table_with_index", ["column1", "column2"])

