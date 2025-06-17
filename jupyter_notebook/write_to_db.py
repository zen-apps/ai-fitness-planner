import os
import requests
import numpy as np
import pandas as pd
import json
import csv
import psycopg2 
from io import StringIO
from datetime import datetime
import requests

from sqlalchemy import create_engine, text

import pandera as pa
from pandera.errors import SchemaError

def get_engine():

    DATABASE_URI = os.getenv("DATABASE_URI")
    engine = create_engine(DATABASE_URI)
    return engine

def query_bbdt_db(query):
    engine = get_engine()
    connection = engine.connect()
    df = pd.read_sql(query, connection)  # sql alchemy
    connection.close()  # not sure if I need this
    engine.dispose()  # not sure if I need this
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)
        

def show_table_cols(connection, table_name):
    query = f"SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
    result = connection.execute(query)
    for r in result:
        print(r)

def set_primiary_key(connection, table_name, primary_key_col):
    query = f"ALTER TABLE {table_name} ADD PRIMARY KEY ({primary_key_col});" 
    try:
        connection.execute(text(query))
    except Exception as e:
        print(e)
    
    query = f"ALTER TABLE {table_name} ADD COLUMN id SERIAL PRIMARY KEY;"
    try:
        connection.execute(text(query))
    except Exception as e:
        print(e)

def set_date_datatype(connection, table_name, date_key_col):
    query = f"ALTER TABLE {table_name} ALTER COLUMN {date_key_col} TYPE TIMESTAMP with time zone USING to_timestamp({date_key_col};"
    try:
        connection.execute(query)
    except Exception as e:
        print(e)
    show_table_cols(connection, table_name)