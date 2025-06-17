from datetime import datetime
import re
import requests
import json
import sys
import traceback

import pytz


def utc_to_local(utc_dt, tz):
    local_tz = pytz.timezone(tz)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_tz.normalize(local_dt)  # .normalize might be unnecessary


def get_datetime(tz=None):
    if tz == None:
        datetime_now = datetime.now()
        datetime_str = datetime_now.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        return {"dt": datetime_now, "dt_str": datetime_str}
    else:
        datetime_now = datetime.now()
        date_adj = utc_to_local(datetime_now, tz)
        datetime_str = date_adj.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        return {"dt": date_adj, "dt_str": datetime_str}


def apply_tz(datetime_obj, tz):
    date_adj = utc_to_local(datetime_obj, tz)
    datetime_str = date_adj.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
    return {"dt": date_adj, "dt_str": datetime_str}


