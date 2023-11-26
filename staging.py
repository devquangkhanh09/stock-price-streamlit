from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import split, length
from pyspark.sql.types import StringType
from delta import *
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import explode, from_json,size
from pyspark.sql.types import *
import json
from functools import reduce
import os
from pyspark.sql.avro.functions import from_avro, to_avro
from pyspark.sql.functions import year, month, dayofmonth, to_date

os.environ['PYSPARK_SUBMIT_ARGS'] ='''
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.apache.spark:spark-streaming-kafka-0-10-assembly_2.12:3.4.1,org.apache.spark:spark-avro_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 pyspark-shell'''

spark_session = SparkSession.builder \
            .appName("read_hdfs") \
            .master("local") \
            .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
            
binary_to_int = udf(lambda x: int.from_bytes(x, 'big'), IntegerType())

from confluent_kafka.schema_registry import SchemaRegistryClient
def get_schema_from_id(id: int):
    url = "http://localhost:8081"
    try:
        sr = SchemaRegistryClient({'url': url})
        version = sr.get_schema(id)
        return version.schema_str
    except Exception as e:
        raise e
spark_session.conf.set("spark.sql.session.timeZone", "UTC+7")

def batch_process(df, id):
    list_obj = df.select('valueSchemaId').collect()
    list_id = [id.asDict()['valueSchemaId'] for id in list_obj]
    set_id = set(list_id)
    mapping = {id:get_schema_from_id(id) for id in set_id}
    
    for id in mapping.keys():
        df.filter(col('valueSchemaId') == id) \
            .withColumn('result', from_avro(col('fixedValue'), mapping[id], {"datetimeRebaseMode": "CORRECTED", "mode": "PERMISSIVE"})) \
            .select('result.*') \
            .withColumn("timestamp", to_timestamp(col("trade_time"), 'yyyyMMddHHmm')) \
            .withColumn("year", date_format(col("timestamp"), "yyyy")) \
            .withColumn("month", date_format(col("timestamp"), "MM")) \
            .withColumn("day", date_format(col("timestamp"), "dd")) \
            .drop("timestamp") \
            .write \
            .option('checkpointLocation',"/delta/stock2/_checkpoints/")\
            .option("mergeSchema", "true") \
            .mode("append") \
            .format("delta") \
            .partitionBy("symbol","year", "month", "day") \
            .save("/delta/stock2")

df = spark_session\
        .readStream\
        .format("kafka")\
        .option("minPartitions", 1) \
        .option("startingOffsets", "latest") \
        .option("kafka.bootstrap.servers", "localhost:9092")\
        .option("failOnDataLoss", False)\
        .option("subscribe", "stock")\
        .load() \
        .withColumn('fixedValue', expr("substring(value, 6, length(value)-5)")) \
        .withColumn('valueSchemaId', expr("substring(value, 2, 4)")) \
        .withColumn('valueSchemaId', binary_to_int(col("valueSchemaId"))) \
        .writeStream \
        .foreachBatch(batch_process).start()
        
df.awaitTermination()