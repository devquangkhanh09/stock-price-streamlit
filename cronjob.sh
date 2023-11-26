#!/bin/bash
crontab -e
0 18 * * * spark-submit --packages io.delta:delta-core_2.12:2.4.0 /home/pm/stock-price-streamlit/processing.py