import pyodbc
import pandas as pd
import concurrent.futures
from typing import List, Optional, Tuple, Union


class TeraUtils:
    def __init__(self, connection_string: str):
        """
        Initialize the TeraUtils class with a connection string.

        :param connection_string: The connection string for the Teradata database.
        """
        self.connection_string = connection_string
        self.connection = pyodbc.connect(connection_string)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the Teradata database.

        :param table_name: The name of the table to check.
        :return: True if the table exists, False otherwise.
        """
        cursor = self.get_cursor()
        cursor.execute("SHOW TABLE " + table_name)
        result = cursor.fetchone()
        cursor.close()
        return result is not None

    def get_cursor(self) -> pyodbc.Cursor:
        """
        Get a cursor for the Teradata database connection.

        :return: A pyodbc cursor object.
        """
        return self.connection.cursor()

    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Execute a query and return the result as a DataFrame.

        :param sql: The SQL query to execute.
        :param params: A tuple of parameters for the query, if any.
        :return: A pandas DataFrame containing the query result.
        """
        cursor = self.get_cursor()
        cursor.execute(sql, params)
        result = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        cursor.close()

        return pd.DataFrame(result, columns=column_names)

    def execute_query_cursor(self, cursor: pyodbc.Cursor, sql: str, params: Optional[Tuple] = None) -> None:
        """
        Execute a query using a given cursor.

        :param cursor: A pyodbc cursor object.
        :param sql: The SQL query to execute.
        :param params: A tuple of parameters for the query, if any.
        """
        cursor.execute(sql, params)
        self.connection.commit()

    def execute_query_dataframe(self, cursor: pyodbc.Cursor, sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Execute a query using a given cursor and return the result as a DataFrame.

        :param cursor: A pyodbc cursor object.
        :param sql: The SQL query to execute.
        :param params: A tuple of parameters for the query, if any.
        :return: A pandas DataFrame containing the query result.
        """
        cursor.execute(sql, params)
        result = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]

        return pd.DataFrame(result, columns=column_names)

    def execute_queries_transaction(self, queries: List[str], params: Optional[List[Tuple]] = None,
                                    return_data: bool = False, threading: bool = False) -> Optional[List[pd.DataFrame]]:
        """
        Execute multiple queries in a single transaction.

        :param queries: A list of SQL queries to execute.
        :param params: A list of tuples containing the parameters for each query, if any.
        :param return_data: If True, return a list of pandas DataFrames containing the query results.
        :param threading: If True, execute queries concurrently using multi-threading.
        :return: A list of pandas DataFrames containing the query results, if return_data is True. None otherwise.
        """
        cursor = self.get_cursor()
        dataframes = []

        def execute_query(query: str, query_params: Optional[Tuple] = None) -> pd.DataFrame:
            cursor.execute(query, query_params)
            if return_data:
                result = cursor.fetchall()
                column_names = [column[0] for column in cursor.description]
                return pd.DataFrame(result, columns=column_names)
            return None

        try:
            if threading:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(execute_query, query, params[i] if params else None) for i, query in
                               enumerate(queries)]
                    for future in concurrent.futures.as_completed(futures):
                        if return_data:
                            dataframes.append(future.result())
            else:
                for i, query in enumerate(queries):
                    query_params = params[i] if params else None
                    df = execute_query(query, query_params)
                    if return_data:
                        dataframes.append(df)

            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            print(f"Error while executing queries in a transaction: {e}")
        finally:
            cursor.close()

        return dataframes if return_data else None

    def close(self) -> None:
        """
        Close the Teradata database connection.
        """
        try:
            self.connection.close()
        except Exception as e:
            print(f"Error while closing the connection: {e}")

    import datetime

    # ...

    class TeraUtils:
        # ...

        def execute_queries_transaction1(self, queries: List[str], params: Optional[List[Tuple]] = None,
                                         return_data: bool = False, threading: bool = False) -> Optional[List[pd.DataFrame]]:
            """
            Execute multiple queries in a single transaction.

            :param queries: A list of SQL queries to execute.
            :param params: A list of tuples containing the parameters for each query, if any.
            :param return_data: If True, return a list of pandas DataFrames containing the query results.
            :param threading: If True, execute queries concurrently using multi-threading.
            :return: A list of pandas DataFrames containing the query results, if return_data is True. None otherwise.
            """
            cursor = self.get_cursor()
            dataframes = []

            def execute_query(query: str, query_params: Optional[Tuple] = None) -> pd.DataFrame:
                start_time = datetime.datetime.now()
                print(f"Executing query: {query} at {start_time}")

                cursor.execute(query, query_params)
                if return_data:
                    result = cursor.fetchall()
                    column_names = [column[0] for column in cursor.description]
                    df = pd.DataFrame(result, columns=column_names)
                else:
                    df = None

                end_time = datetime.datetime.now()
                duration = end_time - start_time
                print(f"Finished executing query: {query} at {end_time}. Duration: {duration}")
                return df

            try:
                if threading:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(execute_query, query, params[i] if params else None) for i, query in
                                   enumerate(queries)]
                        for future in concurrent.futures.as_completed(futures):
                            if return_data:
                                dataframes.append(future.result())
                else:
                    for i, query in enumerate(queries):
                        query_params = params[i] if params else None
                        df = execute_query(query, query_params)
                        if return_data:
                            dataframes.append(df)

                self.connection.commit()
            except Exception as e:
                self.connection.rollback()
                print(f"Error while executing queries in a transaction: {e}")
            finally:
                cursor.close()

            return dataframes if return_data else None
