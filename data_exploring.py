import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('./data/archive/product_data.csv')

# Print all the current pairs of values found in the columns des_product_family and des_product_type
print(data[['des_product_family', 'des_product_type']].drop_duplicates())