import pandas as pd
import numpy as np

# Example table before imputation
data_before = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [25, np.nan, 30, 35],
    "Salary": [50000, 60000, np.nan, 80000]
}
df_before = pd.DataFrame(data_before)
display(df_before)
# Perform imputation (mean imputation for numeric columns)
df_after = df_before.copy()
df_after["Age"] = df_after["Age"].fillna(df_before["Age"].mean())
df_after["Salary"] = df_after["Salary"].fillna(df_before["Salary"]
                                       .mean().round(1))

# Display the before and after tables
display(df_after)