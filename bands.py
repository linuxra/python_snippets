def generate_fico_bands(start_score, end_score, num_ranks):
    score_range = end_score - start_score
    step = score_range // num_ranks
    
    fico_bands = []
    for rank in range(1, num_ranks + 1):
        lower = start_score + (rank - 1) * step
        if rank == num_ranks:
            upper = end_score
        else:
            upper = lower + step - 1
        
        fico_bands.append({"range": (lower, upper), "rank": rank})
    
    return fico_bands
from datetime import datetime, timedelta

from datetime import datetime, timedelta

from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_month_list_backward(date_str, n):
    """
    Generate a list of YYYYMM for the previous N months from the given date.

    :param date_str: A string representing the starting date in the format 'YYYYMM'.
    :param n: An integer representing the number of previous months to generate.
    :return: A list of strings representing the previous N months in the format 'YYYYMM'.
    """
    # Convert the input date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y%m')

    # Create an empty list to store the month list
    month_list = []

    # Generate N months backward
    for i in range(1, n + 1):
        # Calculate the new month by subtracting i months from the input date
        new_month = date_obj - relativedelta(months=i)

        # Convert the new month back to a string in the format YYYYMM
        new_month_str = new_month.strftime('%Y%m')

        # Append the new month string to the month list
        month_list.append(new_month_str)

    return month_list

# Example usage:
date_str = '202201'
n = 6
month_list = generate_month_list_backward(date_str, n)
print(month_list)


# Example usage:
start_score = 300
end_score = 850
num_ranks = 10
fico_bands = generate_fico_bands(start_score, end_score, num_ranks)
print(fico_bands)
