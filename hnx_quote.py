#!/bin/python3

import json
import sys
import re
import urllib3.exceptions
import urllib3
import logging
import requests
from pytz import timezone
from itertools import islice
from datetime import datetime
from typing import Optional
from uuid import uuid4
from confluent_kafka import  Producer
from confluent_kafka.serialization import StringSerializer, SerializationContext, MessageField
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient

from pydantic_avro.base import AvroBase
from pydantic import Field

import requests
import urllib3
import ssl


class CustomHttpAdapter (requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def get_legacy_session():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    session = requests.session()
    session.mount('https://', CustomHttpAdapter(ctx))
    return session


class Quote(AvroBase):
    symbol: str = Field(default=...)
    exchange_name: str
    trade_time: datetime
    ref_price: Optional[float] = Field(default=None)
    ceil_price: Optional[float] = Field(default=None)
    floor_price: Optional[float] = Field(default=None)
    b_qtty_3: Optional[float] = Field(default=None)
    b_price_3: Optional[float] = Field(default=None)
    b_qtty_2: Optional[float] = Field(default=None)
    b_price_2: Optional[float] = Field(default=None)
    b_qtty_1: Optional[float] = Field(default=None)
    b_price_1: Optional[float] = Field(default=None)
    pct_change: Optional[float] = Field(default=None)
    last_price: Optional[float] = Field(default=None)
    last_qtty: Optional[float] = Field(default=None)
    traded_quantity: Optional[float] = Field(default=None)
    total_value_traded_qtt: Optional[float] = Field(default=None)
    s_qtty_3: Optional[float] = Field(default=None)
    s_price_3: Optional[float] = Field(default=None)
    s_qtty_2: Optional[float] = Field(default=None)
    s_price_2: Optional[float] = Field(default=None)
    s_qtty_1: Optional[float] = Field(default=None)
    s_price_1: Optional[float] = Field(default=None)
    open_price: Optional[float] = Field(default=None)
    mid_px: Optional[float] = Field(default=None)
    max_price: Optional[float] = Field(default=None)
    min_price: Optional[float] = Field(default=None)
    foreign_buy: Optional[float] = Field(default=None)
    foreign_sell: Optional[float] = Field(default=None)
    close_price: Optional[float] = Field(default=None)


# Echo
LOG = logging.getLogger(__name__)
logging.basicConfig(
    level = logging.INFO,
    datefmt = "%Y-%m-%d %H:%M:%S",
    format = "%(asctime)s | %(thread)-8d | %(levelname)-8s ]> %(message)s",
    filename = "hnx_quote.log",
    filemode = "w+",
    force=True
)

STDOUT_LOG = logging.StreamHandler(stream=sys.stdout)
STDOUT_LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(relativeCreated)6d | %(asctime)s - %(name)-12s ] - %(levelname)-8s - %(message)s')
STDOUT_LOG.setFormatter(formatter)
logging.getLogger().addHandler(STDOUT_LOG)

# Declare
DEFAULT_HEADER = {
    'Accept': '*/*',
    'Accept-Language': 'en,vi-VN;q=0.9,vi;q=0.8,fr-FR;q=0.7,fr;q=0.6,en-US;q=0.5',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Origin': 'https://banggia.hnx.vn',
    'Referer': 'https://banggia.hnx.vn/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Linux"'
}
BASE_URL = "https://banggia.hnx.vn/Home/GetStockInBasket"


def get_ohlcv(code: str, key: str):

    # Define payload
    payload = {
        "pFloorCode": code,
        "pKeyAreas": key,
    }

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _parse_content(content: str):

        output = {}
        metadata = []

        if re.search(r"\|", content) is None:
            return output, metadata
        else:

            element = content.split('|')
            assert element[0] == "", (
                "Based on the synponis of conent |NAME|NAME_<column>*<value>*<format>|. "
                f"The first value will empty. Got {element[0]}"
            )

            # The structure of the text in synponis: |NAME|NAME_<column>*<value>*<format>|...
            ticker = element[1]
            output["symbol"] = ticker

            # Loop
            for _elem in element[2:]:

                composite = _elem.split("*")
                if all([len(composite) == 1, composite[0] == _elem]):
                    metadata.append(_elem)
                else:
                    _k = re.sub(pattern=f"{ticker}_", repl="", string=composite[0]).lower()
                    _val = composite[1]
                    output[_k] = _val

            return output, metadata

    # Fetch API
    try:
        sess = get_legacy_session()
        res = sess.post(
            url=BASE_URL,
            params=payload,
            headers=DEFAULT_HEADER,
            verify=False,
        )
    except Exception as exc:
        mes = f"Cannot fetch data from HNX. Detail from {exc}"
        LOG.info(mes)
        raise Exception(mes)
    else:
        if res.status_code == 200:

            # Declare
            quote = []
            quote_attribute = []
            data = res.json()
            composition = data["obj"]

            if composition:

                for _composition in composition:
                    _quote, _attribute = _parse_content(_composition)
                    quote.append(_quote)
                    quote_attribute.append(_attribute)

            for obj in quote:

                for k, v in obj.items():
                    if v == '':
                        obj[k] = None
                    elif ',' in v:
                        obj[k] = ''.join(v.split(','))

            return quote, quote_attribute

    return [], []

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

if __name__ == '__main__':

    # Define const
    # Listing status
    EXCHANGE_CODE = ["LIS", "UPC"]

    # Key for fetch API
    KEY_AREA = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STUV", "WXYZ"]
    content = []

    for code in EXCHANGE_CODE:

        for key in KEY_AREA:

            LOG.info(f"Fetch data from code {code} and key {key}")
            _content, meta = get_ohlcv(code, key)

            content += _content


    LOG.info(f"Fetch data complete with total {len(content)} records")
    CUR_TIME = datetime.now(tz=timezone('Asia/SaiGon'))

    content = [
        {
            **i,
            "exchange_name": "HNX",
            "trade_time": CUR_TIME,
        }
        for i in content
    ]

    schema_registry_conf = {'url': "http://localhost:8081"}

    schema_registry_client = SchemaRegistryClient(schema_registry_conf)

    avro_serializer = AvroSerializer(
        schema_registry_client=schema_registry_client,
        schema_str=json.dumps(Quote.avro_schema()),
        to_dict=lambda x, _: x.__dict__
    )

    string_serializer = StringSerializer('utf_8')

    producer_conf = {'bootstrap.servers': 'localhost:9092'}

    producer: Producer = Producer(producer_conf)

    LOG.info("Start send message to broker")
    for idx, batch in enumerate(batched(content, 10), 1):

        _batch = list(batch)

        LOG.info(f"Send {idx}th batch to broker with total {len(_batch)} records")

        for mes in _batch:
            quote = Quote.model_validate(mes)

            producer.produce(
                topic="stock",
                key=string_serializer(str(uuid4())),
                value=avro_serializer(quote, SerializationContext("stock", MessageField.VALUE)),
            )
        LOG.info(f"Waiting for send {idx}th batch")
        producer.flush()
