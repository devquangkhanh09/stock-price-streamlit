import psycopg2
import pandas as pd
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
import os
from datetime import datetime
from pyspark.sql.functions import *
import builtins

os.environ['PYSPARK_SUBMIT_ARGS'] ='''--packages io.delta:delta-core_2.12:2.4.0 pyspark-shell'''
spark_session = SparkSession.builder \
            .appName("read_hdfs") \
            .master("local") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
spark_session.conf.set("spark.sql.streaming.schemaInference", True)
engine = create_engine('postgresql://bd:bd@localhost:5432/stock')
year, month, day = datetime.now().year, datetime.now().month, datetime.now().day

conn = psycopg2.connect(
    host="localhost",
    database="stock",
    user="bd",
    password="bd",
    port=5432
)
cursor = conn.cursor()
query = "SELECT * FROM symbols"
cursor.execute(query)
rows = cursor.fetchall()
cursor.close()
conn.close()
    
for pair in rows:
    try:
        symbol = pair[0]
        path = f"/delta/stock2/symbol={symbol}/year={year}/month={month}/day={day}"
        df = spark_session.read.load(path)
        latestTradeTimeRow = df.orderBy(col("trade_time").desc()).first()
        close = latestTradeTimeRow['close_price']
        open = latestTradeTimeRow['open_price']
        vol = latestTradeTimeRow['traded_quantity']
        dayW = latestTradeTimeRow['trade_time'].date()
        maxP = latestTradeTimeRow['max_price']
        minP = latestTradeTimeRow['min_price']
        new_row = {"Date": [dayW], "Open": [open], "Close": [close], "Max": [maxP], "Min": [minP], "Vol": [vol]}   
        df = pd.DataFrame(new_row)
        df.to_sql(pair[1], engine, if_exists='append')
    except: 
        pass
spark_session.stop()