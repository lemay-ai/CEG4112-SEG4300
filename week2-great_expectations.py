# Import necessary libraries
import pandas as pd
import great_expectations as gx
from icecream import ic
# Create a sample pandas DataFrame
data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eggy", "Frank","Gerry","Hilda"],
    "age": [25, 30, 25, 30, 25, 30, 35, None],
    "salary": [70_000, 80_000, 120_000, 110_000,70_000, 80_000, 110_000, 10_000]
}
df = pd.DataFrame(data)
context = gx.get_context()
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

expectation1 = gx.expectations.ExpectColumnValuesToBeBetween(
    column="age", min_value=20, max_value=30)
expectation2 = gx.expectations.ExpectColumnValuesToBeBetween(
    column="salary", min_value=60_000, max_value=110_000)

for e in [expectation1,expectation2]:
  validation_result = batch.validate(e)
  ic(validation_result)