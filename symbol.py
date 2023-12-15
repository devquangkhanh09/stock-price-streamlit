import psycopg2
import pandas as pd
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
import os
from datetime import datetime
from pyspark.sql.functions import *

engine = create_engine('postgresql://bd:bd@localhost:5432/stock')


df = pd.read_csv("/home/pm/Downloads/symbol.csv")
df = df.set_index("symbol")


df.to_sql('symbols', engine, if_exists='append')

