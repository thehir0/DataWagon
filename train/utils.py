
import os
import random
import numpy as np
from datetime import datetime
import pytz


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def date_to_timestamp(date: str) -> int:
    utc = pytz.UTC
    date_utc = utc.localize(date)
    timestamp_seconds = int(date_utc.timestamp())
    return timestamp_seconds

def timestamp_to_date(timestamp: int) -> str:
    date = datetime.utcfromtimestamp(timestamp)
    date_string = date.strftime('%Y-%m-%d')
    return date_string