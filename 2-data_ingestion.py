######DATA INGESTION######
###ETL - EXTRACTION###
import requests

DATASET_ID = "nc67-uf89"    #Open Parking and Camera Violations
LIMIT = 1000                #Rows per page

url = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
params = {"$limit": LIMIT, "$offset":0}

response = requests.get(url, params=params)
data = response.json()                      
print(f"Fetched {len(data)} rows")

############################################################################################################

###ETL - TRANSFORMING###

import pandas as pd
df = pd.DataFrame(data)

"""
#Dict error - column to be dropped
for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, dict)).any():
        print("DICT COLUMN:", col)
"""

df_transformed = df.drop(columns='summons_image')







#Basic Structure 
print(df_transformed.info())
print(df_transformed.shape)
print(df_transformed.columns)
print(df_transformed.dtypes)
#Preview
print(df_transformed.head)
print(df_transformed.tail)
#Summary
print(df_transformed.describe)
#Missing
print(df_transformed.isna().sum())
#Memory
print(df_transformed.info(memory_usage="deep"))
############################################################################################################

###ETL - LOADING###

from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg2://postgres:admin@localhost:5432/df_db")
df_transformed.to_sql("violations_transf", engine, if_exists="append", index=False)
