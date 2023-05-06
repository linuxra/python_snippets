import pandas as pd
import numpy as np

# Sample input DataFrame with FICO scores
data = {'fico_score': [330, 375, 420, 460, 480, 510, 540, 570, 600, 630]}
input_df = pd.DataFrame(data)


def generate_case_statement_and_ranges(df, score_column, num_ranges, output_column_name='score_rank'):
    """
    Generate a SQL case statement and a DataFrame with low, high, and rank columns based on the distribution
    of the data in a DataFrame.

    :param df: Input DataFrame containing the scores
    :param score_column: Name of the column containing the scores
    :param num_ranges: Number of desired score ranges
    :param output_column_name: Name of the output column in the case statement (default is 'score_rank')
    :return: SQL case statement string and a DataFrame with low, high, and rank columns
    """
    quantiles = np.linspace(0, 1, num_ranges + 1)
    cutoff_points = df[score_column].quantile(quantiles).values

    case_statement = f"CASE"

    range_data = []

    for i in range(len(cutoff_points) - 1):
        low = int(np.ceil(cutoff_points[i]))
        high = int(np.floor(cutoff_points[i + 1]))
        case_statement += f" WHEN {low}-{high} THEN {i + 1}"

        range_data.append({'low': low, 'high': high, 'rank': i + 1})

    case_statement += f" ELSE NULL END AS {output_column_name}"

    ranges_df = pd.DataFrame(range_data)

    return case_statement, ranges_df


num_ranges = 4
score_column = 'fico_score'

sql_case_statement, ranges_df = generate_case_statement_and_ranges(input_df, score_column, num_ranges)

print(sql_case_statement)
print(ranges_df)
