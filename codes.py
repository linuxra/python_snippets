df2 = df2.set_index('r')

# Mapping of cb_cd values to column names
cb_cd_mapping = {'ex': 'ex_cd', 'tu': 'tu_cd', 'ef': 'ef_cd'}

# Replace values in 'r1', 'r2', 'r3', 'r4' based on the merged values and cb_cd
for col in ['r1', 'r2', 'r3', 'r4']:
    # Merge with df2
    df1 = df1.merge(df2, left_on=col, right_index=True, suffixes=('', f'_{col}'))

    # Replace values
    df1[col] = df1.apply(lambda row: row[cb_cd_mapping[row['cb_cd']] + f'_{col}'], axis=1)

    # Drop the merged columns
    df1.drop([cb_cd_mapping[key] + f'_{col}' for key in cb_cd_mapping.keys()], axis=1, inplace=True)

# Display the modified DataFrame
print(df1)