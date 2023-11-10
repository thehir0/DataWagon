from datetime import datetime
import pytz

def date_to_timestamp(date_string: str) -> int:
    date = datetime.strptime(date_string, '%Y-%m-%d')
    utc = pytz.UTC
    date_utc = utc.localize(date)
    timestamp_seconds = int(date_utc.timestamp())
    return timestamp_seconds

def timestamp_to_date(timestamp: int) -> str:
    date = datetime.utcfromtimestamp(timestamp)
    date_string = date.strftime('%Y-%m-%d')
    return date_string

if __name__ == '__main__':
    #TEST
    print(date_to_timestamp('2023-02-01'), timestamp_to_date(1675209600))