import psycopg2
import pandas as pd

STOCK_INDICES = {
    "DHT": "dht",
    "HAT": "hat"
}

class PostgresqlService:
    def __init__(self):
        self.conn = psycopg2.connect(database = "historical", 
                        user = "bd", 
                        host= "localhost",
                        password = "bd",
                        port = 5433)
        self.cur = self.conn.cursor()
    
    def __del__(self):
        self.cur.close()
        self.conn.close()

    def get_data(self, query):
        self.cur.execute(query)
        return self.cur.fetchall()

    def get_stock_data_as_df(self, stock_index):
        self.cur.execute(f"SELECT * FROM {STOCK_INDICES[stock_index]}")
        data = self.cur.fetchall()
        df = pd.DataFrame(data, columns=["id", "Open", "High", "Low", "Close", "Volume", "Date"])
        return df
    
