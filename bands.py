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

# Example usage:
start_score = 300
end_score = 850
num_ranks = 10
fico_bands = generate_fico_bands(start_score, end_score, num_ranks)
print(fico_bands)
