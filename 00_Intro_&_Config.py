# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC Price optimization plays a pivotal role in driving revenue growth and maximizing profitability. Pricing analysts scrutinize historical data to determine how changes in price as well as the prior state of the market ahead of a pricing change affect consumer response.  This requires a careful balancing of exploratory, predictive and what-if analysis as well as good judgement about other factors at play before organizations are able to make effective changes to their prices. The purpose of the following notebooks is to examine some of the key forms of analysis at the heart of this solution.

# COMMAND ----------

# MAGIC %md ##Config

# COMMAND ----------

# DBTITLE 1,Initialize Config
#Databricks connect spark import required
from spark_utils import get_spark

spark = get_spark()

if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
# set catalog/database name
config['catalog name'] = 'manufacturing'
config['database name'] = 'dynamic_pricing'

# create database to house data
# _ = spark.sql('CREATE CATALOG  IF NOT EXISTS {0}'.format(config['catalog name']))
_ = spark.sql('USE CATALOG {0}'.format(config['catalog name']))
# _ = spark.sql('CREATE DATABASE IF NOT EXISTS {0}'.format(config['database name']))
_ = spark.sql('USE SCHEMA {0}'.format(config['database name']))

print('Catalog and database set')

# COMMAND ----------

# MAGIC %sql
# MAGIC select current_catalog(), current_schema();