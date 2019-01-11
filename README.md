# sparkify_customer_churn

Currently working on this project

## Description
Predict customer churn with PySpark for an imaginary digital music service called Sparkify. Sparkify has a free-tier and a premium subscription plan. Users can upgrade, downgrade or cancel their service at any time. Churn means downgradding from premium to free tier or cancelling the service. If we can predict users who will churn, the company can offer them discounts and incentives.

* Step 1: Create a prototype of the model on local computer with PySpark using a smaller subset of the data
* Step 2: Hopefully, deploy the model onto AWS with the full 12gb dataset

## Motivation
To lean customer churn analysis, PySpark and AWS.

## Data
Data keeps track of timestamped events of the following actions:
1. play a song
2. logout
3. like
4. ad_heard
5. downgrade
6. ...

Data download links
* Link to small-sized subset of Sparkify data (125 mb): https://drive.google.com/open?id=1FwuyO5apNwy8q6BpG-_EIqqN-ED0X9tx
* Link to medium-sized subset of Sparkify data (237 mb): https://drive.google.com/open?id=17Lys6v7LOcAWFMHXwXwslUWj04LoNWp1
* Link to full Sparkify dataset on AWS (12 gb): s3n://dsnd-sparkify/sparkify_event_data.json

## Modelling Steps
1. Load data into spark
2. Explore & clean data
3. Create features
4. Build models
5. Predict churn
