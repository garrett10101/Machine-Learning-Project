import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('output_data.csv')

# Drop columns with more than 75% NaN values
df = df.loc[:, df.isnull().mean() < 0.75]

# Convert numeric columns to numeric type and interpolate
df_numeric = df.apply(pd.to_numeric, errors='coerce').interpolate(method='linear', limit_direction='both')

# Handle non-numeric columns with forward and backward fill
df_non_numeric = df.select_dtypes(include=['object']).fillna(method='ffill').fillna(method='bfill')

# Convert non-numeric columns to numeric by encoding
df_encoded = pd.DataFrame()
for col in df_non_numeric.columns:
    df_encoded[col], _ = pd.factorize(df_non_numeric[col])

# Combine numeric and encoded non-numeric data
df_clean = pd.concat([df_numeric, df_encoded], axis=1)

# Final check for NaN values
remaining_nans = df_clean.isnull().sum().sum()
if remaining_nans > 0:
    original_rows = len(df_clean)
    df_clean.dropna(inplace=True)
    dropped_percentage = (1 - len(df_clean) / original_rows) * 100
    print(f"{dropped_percentage:.2f}% of data was dropped due to remaining NaN values.")
else:
    print("No NaN values remain after cleaning.")

# Export cleaned CSV
df_clean.to_csv('cleaned_output_data.csv', index=False)
print("Cleaned data saved to cleaned_output_data.csv.")
