import pandas as pd

# Load the languages.csv file
languages_df = pd.read_csv("languages.csv")

# Preview the first few rows to inspect the columns
print(languages_df.head())

# Check column names
print(languages_df.columns)
# Count how many of each type
print(languages_df["level"].value_counts())
print(languages_df["Family_name"].value_counts())
unique_families = set(languages_df["Family_name"].dropna())  # remove NaNs if any
print("Unique families set:")
print(unique_families)

# Count the number of unique families
num_families = languages_df["Family_name"].nunique()
print("\nNumber of unique families:", num_families)

print(languages_df["Macroarea"].value_counts())
unique_m = set(languages_df["Macroarea"].dropna())  # remove NaNs if any
print("Unique Macroarea set:")
print(unique_m)

no_family = languages_df[languages_df["Family_name"].isna()]
print(f"Number of languages without a family: {len(no_family)}")

# Optionally, display them
print(no_family[["Name", "Glottocode", "Macroarea"]])

# Count the number of unique families
num_m = languages_df["Macroarea"].nunique()
print("\nNumber of unique Macroarea:", num_m)

