import pandas as pd

# Read CSV file
file_path = 'nuforc_list.csv'
column_name = 'Text'
# Get column names

df = pd.read_csv(file_path)
print(df.columns.tolist())

# Extract column values
if column_name in df.columns:
    values = df[column_name].dropna().astype(str)  # Drop NaN values and ensure string format

    # Save values line by line
    values.to_csv("column_values.txt", index=False, header=False)

    # Save concatenated values in a single line
    concatenated_values = " ".join(values)
    with open("concatenated_values.txt", "w", encoding="utf-8") as f:
        f.write(concatenated_values)

    print("Files saved successfully!")
else:
    print(f"Column '{column_name}' not found in the CSV file.")