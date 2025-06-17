DROP TABLE IF EXISTS subscription_status; CREATE TABLE subscription_status (
id serial PRIMARY KEY,
attributes_id int,
datetime_create timestamp,
datetime_update timestamp,
product_id text,
username text,
status bool,
datetime_start timestamp,
datetime_end timestamp);