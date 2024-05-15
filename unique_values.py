import pandas as pd

# Load your dataset into a DataFrame
diamonds =  pd.read_csv('./data/diamonds.csv')

print("Data loaded successfully.")

# Drop index column
diamonds.drop(diamonds.columns[0], axis=1, inplace=True)

# Create a dictionary to store unique values for each column
unique_values = {}

# Iterate over each column in the DataFrame
for column in diamonds.columns:
    # Get unique values for the column
    unique_values[column] = diamonds[column].unique()

# Define the path to the text file
output_file = 'unique_values.txt'

# Open the text file for writing
with open(output_file, 'w') as f:
    # Write unique values for each column to the text file
    for column, values in unique_values.items():
        f.write(f"Unique values for column '{column}':\n")
        for value in values:
            f.write(str(value) + '\n')
        f.write('\n')

print(f"Unique values for each column saved to '{output_file}'")
