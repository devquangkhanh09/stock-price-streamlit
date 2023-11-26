import psycopg2
import pandas as pd
from pyspark.sql import SparkSession
from sqlalchemy import create_engine
import os
from datetime import datetime

os.environ['PYSPARK_SUBMIT_ARGS'] =''' --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.apache.spark:spark-streaming-kafka-0-10-assembly_2.12:3.4.1,org.apache.spark:spark-avro_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 --conf spark.sql.session.timeZone=GMT pyspark-shell'''

spark_session = SparkSession.builder \
            .appName("cachingPostgreSQL") \
            .master("local") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
symbol = "A32"
date_path = f"hdfs://localhost:9000/delta/stock1/symbol={symbol}/year={year}/month={month}/day={day}"
spark_session.conf.set("spark.sql.streaming.schemaInference", True)
df = spark_session.read.load(date_path)
spark_session.stop()