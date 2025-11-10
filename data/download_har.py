import pandas as pd
# Load the dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
data = pd.read_csv(url, compression='zip', header=0, sep=',', quotechar='"')
# Display the first few rows of the dataset
print(data.head())