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

def generate_month_list(date_str, n):
    # Convert the input date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y%m')

    # Create an empty list to store the month list
    month_list = []

    # Generate N months
    for i in range(n):
        # Calculate the new month by adding i months to the input date
        new_month = date_obj + timedelta(days=(i * 30))
        
        # Convert the new month back to a string in the format YYYYMM
        new_month_str = new_month.strftime('%Y%m')

        # Append the new month string to the month list
        month_list.append(new_month_str)

    return month_list

# Example usage:
date_str = '202201'
n = 6
month_list = generate_month_list(date_str, n)
print(month_list)

# Example usage:
start_score = 300
end_score = 850
num_ranks = 10
fico_bands = generate_fico_bands(start_score, end_score, num_ranks)
print(fico_bands)
