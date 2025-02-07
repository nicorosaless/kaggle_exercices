import pandas as pd

# Load the CSV file
df = pd.read_csv('submission.csv')

# Convert the 'target' column to integers
df['target'] = df['target'].astype(int)

# Save the modified DataFrame back to a CSV file
df.to_csv('submission.csv', index=False)