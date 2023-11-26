#!/bin/bash
crontab -e
0 18 * * * spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.apache.spark:spark-streaming-kafka-0-10-assembly_2.12:3.4.1,org.apache.spark:spark-avro_2.12:3.4.1,io.delta:delta-core_2.12:2.4.0 /home/pm/stock-price-streamlit/processing.py