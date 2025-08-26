# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC To examine price elasticity, we will train a model on an historical dataset within which various products are presented to consumers at various price points.  These price points may be the result of baseline pricing changes but more typically are associated with temporary, promotional discounts that shift consumer interest and product volumes.  The timing of these pricing changes along with their duration affect how consumers respond.  There are also cross-product effects whereby one product's pricing changes are counteracted by another product's changes.  As a result, our predictions of how pricing changes drive outcomes may not be as clean and smooth as often presented in economics textbooks. Still, this information can be useful in setting a pricing strategy.
# MAGIC
# MAGIC The basic approach we will take is to examine the average price change from the week prior as well as the prior week's volumes as predictors of a volume change in the current week. This is a simple regression exercise where the model produced can be useful in predicting how future changes will be responded to.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering==0.13.0a3
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_&_Config"

# COMMAND ----------

# DBTITLE 1,Import the Required Libraries
# Import external libraries required by the analysis.
import pandas as pd

import mlflow
from hyperopt import hp, fmin, tpe, SparkTrials, Trials, STATUS_OK, space_eval

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pyspark.sql.functions as fn
from spark_utils import get_spark

spark = get_spark()


# COMMAND ----------

# MAGIC %md ##Step 1: Feature Engineering
# MAGIC
# MAGIC Our feature engineering starts with the acquisition of historical sales data:

# COMMAND ----------

# DBTITLE 1,Retrieve Sales Data
sales = (
  spark
    .table(f'{config["catalog name"]}.{config['database name']}.enhanced_sales')
    .select(
      'productKey',
      'geographyKey',
      'dateKey',
      'storeCountWhereScanned',
      'avgSellingPrice',
      'baseVolumeSales',
      'incrementalVolumeSales',
      'volumeSales',
      'volumeSalesAnyTradePromotion',
      'volumeSalesNoPromotion'
    )
  )

# COMMAND ----------

# MAGIC %md So that the model is not tripped up by the sizeable differences in absolute values in sales volumes associated with these products, we convert these volumes into proportions (presented on a scale of 0 to 100 for ease of interpretation):

# COMMAND ----------

# DBTITLE 1,Convert Key Metrics to Proportions
# calculate proportions of sales under promotions, and relative to the baseline expected with no promotions
with_proportions = (
    sales.withColumn(
      'incrementalVolumeProportion',
      fn.expr("100 * (incrementalVolumeSales) / (incrementalVolumeSales + baseVolumeSales)")
    )
    .withColumn(
      'promotionVolumeProportion',
      fn.expr("100 * (volumeSalesAnyTradePromotion) / (volumeSalesAnyTradePromotion + volumeSalesNoPromotion)")
    )
  )

# COMMAND ----------

# MAGIC %md We then get the prior values for many of our metrics by looking back one week for each product and geo:

# COMMAND ----------

# DBTITLE 1,Get Prior Values
# List the columns to pass information to the next week
columns_to_lag = ['avgSellingPrice','volumeSales','incrementalVolumeProportion',
                  'promotionVolumeProportion','storeCountWhereScanned']

# Perform lagging on each column in term, adding a new lagged column to the dataset
with_priors = with_proportions
for c in columns_to_lag:
  with_priors = with_priors.withColumn(f"prior_{c}", fn.expr(f"LAG({c},1) OVER(PARTITION BY productKey, geographyKey ORDER BY dateKey)"))

# COMMAND ----------

# MAGIC %md With priors in place, we can calculate the percent change in price and volume.  The percent change in volume will later be used as our predicted (label) variable in our regression models:

# COMMAND ----------

# DBTITLE 1,Calculate Change from Prior
# Calculate percentage changes in price and volume of sales
with_changes = (
    with_priors.withColumn(
        'pricePercentChange', 
        fn.expr("100*(avgSellingPrice - prior_avgSellingPrice) / prior_avgSellingPrice")
    )
    .withColumn(
        'volumePercentChange',
        fn.expr("100*(volumesales - prior_volumeSales) / prior_volumeSales") 
    )
)


# COMMAND ----------

# MAGIC %md We'd like to use week and month information as an input to capture seasonality in our predictions.  But because the distance between months 11 and 12 and months 12 and 1 are numerically so different, we might apply a cyclical transformation to the data to ensure these values are equal distance from one another.  (We do the same thing with weeks which typically range from 1 to 52 within a given year):

# COMMAND ----------

# DBTITLE 1,Apply Cyclical Transformations to Weeks and Months
# Add month and week columns to the dataset by joining with our date table
with_rich_date = (
  with_changes
    .join(
      spark.table(f'{config["catalog name"]}.{config['database name']}.date')
      .selectExpr('dateKey','weekOfYear as week',"date_part('month',date) as month"),
      on='dateKey'
      )
)

# Create cyclic month and week columns with a cos transform, such that January and December are similar values.
with_cyclic_date = (
  with_rich_date.withColumn(
    "cyclicWeek",
    fn.expr("1-cos(week/53*2*pi())")
  )
  .withColumn(
    "cyclicMonth",
    fn.expr("1-cos(month/13*2*pi())")
  )
)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We are now able `to` limit our data to just those we will used during model training:

# COMMAND ----------

# DBTITLE 1,Limit Dataset to Just Relevant Features and Label
# Finally, select just the data to be used for model fitting
with_features = (
  with_cyclic_date
    .select(
      "productKey",
      "geographyKey",
      "cyclicWeek",
      "cyclicMonth",
      "prior_avgSellingPrice",
      "prior_volumeSales",
      "pricePercentChange",
      "volumePercentChange",
      "prior_incrementalVolumeProportion",
      "prior_promotionVolumeProportion",
      "prior_storeCountWhereScanned",
      "dateKey"
    )
) 


# COMMAND ----------

# MAGIC %md If you examine the dataset above, you will notice many NULL values.  There are many reasons for this but they don't dramatically affect our ability to perform price modeling.  That said, we still need to clean them up.
# MAGIC
# MAGIC We will start by examining the percentage of null values within each field:

# COMMAND ----------

# DBTITLE 1,Examine Missing Values
# get total rows in dataset
num_rows = with_features.count()

# for each column, identify rows with null values
results = []
for c in with_features.columns: # for each column
  num_nulls = with_features.select(c).filter(f"{c} is null").count() # count null values
  results += [[c, num_nulls, num_nulls/num_rows]] # assemble metrics for this column


# COMMAND ----------

# MAGIC %md For the prior average selling price field, a null value is typically an indicator of a bad record; records with a null value within this field will be removed.  For other fields, we will simply replace the null value with a zero:

# COMMAND ----------

# DBTITLE 1,Address Missing Values
# Drop rows with no previous sales information, and replace prior_promotionVolumeProportion where no promotion was active with zero.
without_nulls = (
  with_features
  .na.drop(
    subset=['prior_avgSellingPrice'] # addresses bulk of bad rows
    ) 
  .na.fill(
    0
  )
)


# COMMAND ----------

# MAGIC %md Lastly, we remove outlier records where there were extreme price changes that are not realistic.  Again, these reflect bad data in our input dataset:

# COMMAND ----------

# DBTITLE 1,Remove Price Change Outliers
# Remove rows with outliers in change columns from dataset, add datekey, and ensure unique (productKey, geographyKey, dateKey)
from pyspark.sql import functions as fn

sales_features = (
    without_nulls
    .filter('pricePercentChange < 300 and volumePercentChange < 600')
    .withColumn("dateKey", fn.to_date(fn.col("dateKey"), "yyyyMMdd"))
    .dropDuplicates(["productKey", "geographyKey", "dateKey"])
)

# COMMAND ----------

feature_table = f"{config['catalog name']}.{config['database name']}.features"
sales_features.write.mode("overwrite").saveAsTable(feature_table)
# spark.sql(f"ALTER TABLE {feature_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
# spark.sql(f"ALTER TABLE {feature_table} ALTER COLUMN productKey SET NOT NULL")
# spark.sql(f"ALTER TABLE {feature_table} ALTER COLUMN geographyKey SET NOT NULL")
# spark.sql(f"ALTER TABLE {feature_table} ALTER COLUMN dateKey SET NOT NULL")
# spark.sql(f"ALTER TABLE {feature_table} ADD CONSTRAINT primary_key PRIMARY KEY (productKey, geographyKey, dateKey)")

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk.errors import AlreadyExists
fe = FeatureEngineeringClient()

online_store_name = "field-demos"
try:
    online_store = fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"  # Valid options: "CU_1", "CU_2", "CU_4", "CU_8"
    )
except AlreadyExists:
    online_store = fe.get_online_store(name=online_store_name)

# COMMAND ----------

import time

# Get information about an existing online store
if online_store:
    print(f"online_store: {online_store.name}, State: {online_store.state}, Capacity: {online_store.capacity}")
    start_time = time.time()
    while online_store.state.value == "STARTING":
        if time.time() - start_time > 600:
            raise TimeoutError("Online store did not finish starting within 10 minutes.")
        time.sleep(10)
        online_store = fe.get_online_store(name=online_store_name)

# Publish the feature table to the online store
published_table = fe.publish_table(
    online_store=online_store,
    source_table_name=feature_table,
    online_table_name=feature_table+"_onlinetable"
)

# COMMAND ----------

# MAGIC %md ##Step 2: Model Fitting
# MAGIC
# MAGIC With our input dataset assembled, we can not proceed with the construction of our predictive models.  To ensure our different models are relatively consistent in how they use these data, we will record a few shared configurations now.  Please note that we are flagging our categorical features, *i.e.* productKey, as such to enable some simplified categorical feature handling in one of our models:

# COMMAND ----------

# DBTITLE 1,Shared Model Configurations
# set experiment path
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Users/{user_name}/dynamic-pricing")

# define feature columns
disc_feature_cols = ['productKey'] #,'geographyKey']

cont_feature_cols = [
        "prior_avgSellingPrice",
        "prior_volumeSales",
        "pricePercentChange",
        "prior_incrementalVolumeProportion",
        "prior_promotionVolumeProportion",
        "prior_storeCountWhereScanned",
        "cyclicWeek",
        "cyclicMonth"
        ]
        
label_col = 'volumePercentChange'

# set training fraction
train_fraction = 0.7

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

feature_lookups = [
    FeatureLookup(
        table_name=feature_table,
        feature_names=cont_feature_cols,
        lookup_key=["productKey", "geographyKey", "dateKey"]
    )
]

training_set = fe.create_training_set(
    df=sales_features.select("productKey", "geographyKey", "dateKey", "volumePercentChange"),
    feature_lookups=feature_lookups,
    label=label_col,
    exclude_columns=["geographyKey", "dateKey"],
    use_spark_native_join=True
)

training_df = training_set.load_df()

# COMMAND ----------

# MAGIC %md We will also split our data into training, validation and testing datasets.  The training dataset will be used during model training and the testing dataset will be set aside as a holdout for final model evaluation.  In the case that we will perform hyperparameter tuning, the validation set will be used for testing between hyperparameter tuning cycles (trials):

# COMMAND ----------

# DBTITLE 1,Split the Dataset
# get sales feature data to pandas dataframe
sales_features_pd = (
  sales_features
    #.sample(fraction=0.50) # randomly sample dataset if too large for pandas
    .toPandas()
  )

# identify categorical features as such
for c in disc_feature_cols:
  sales_features_pd[c].astype('category')

# split data into testing, validation & training sets
train_pd, val_test_pd = train_test_split(sales_features_pd, train_size=train_fraction)
val_pd, test_pd = train_test_split(val_test_pd, train_size=0.5)

# COMMAND ----------

# MAGIC %md ### A. Train a Linear Regression Model
# MAGIC
# MAGIC With our data assembled, we can now train our first model.  For this exercise, we'll use a linear regression model as this is a popular choice employed in price elasticity exercises.  For our categorical features to be used, we will convert these using a one-hot encoding.  There are no hyperparameter values associated with the linear regression algorithm, so this will be a simple matter of training and testing the model.  Please note that we are using [mlflow](https://www.databricks.com/product/managed-mlflow) to track metrics associated with this model and record the model itself:

# COMMAND ----------

# DBTITLE 1,Train a Linear Model for All Data
with mlflow.start_run(run_name='LR: All Products') as run:

  # define pipeline
  disc_feature_pipe = Pipeline( # discrete feature handling
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))]
      )
  preprocessor = ColumnTransformer( # all data prep
    transformers=[('disc_features', disc_feature_pipe, disc_feature_cols)], 
    remainder='passthrough' 
    )
  model_pipeline = Pipeline(steps=[ # combine data prep with model
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
    ])

  # train pipeline model
  feature_cols = disc_feature_cols + cont_feature_cols
  model = (
    model_pipeline
      .fit(
        train_pd[feature_cols],
        train_pd[label_col]
        )
      )
  fe.log_model( # record the trained model
    model=model,  # the sklearn model
    training_set=training_set,
    artifact_path='model',
    flavor=mlflow.sklearn,
    # input_example=test_pd[feature_cols].head(1) # a sample feature input record
  ) 

  # evaluate model on test holdout
  y_pred = model.predict(
    test_pd[feature_cols]
    )
  rmse = mean_squared_error(test_pd[label_col], y_pred, squared=False)
  mae = mean_absolute_error(test_pd[label_col], y_pred)
  mlflow.log_metrics({'rmse':rmse,'mae':mae}) # record eval metrics


# print key results
run_id_lrall = run.info.run_id
print(f"Run_Id: {run_id_lrall}")
print(f"RMSE:   {rmse}")
print(f"MAE:    {mae}")

# COMMAND ----------

# MAGIC %md ###B. Train an XGBR Model
# MAGIC
# MAGIC The linear regression model provides us a baseline result, but could we do better if we used a more sophisticated model such as gradient boosting algorithm?  To test this, we will now train an [XGBoostRegressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) on this same dataset.
# MAGIC
# MAGIC The XGBoostRegressor makes working with categorical features, such as our geography and product key values, much easier.  We could have incorporated these into the linear regression but this would have required a lot of one-hot encoding.  Here, we can just flag the appropriate fields as categorical variables within the pandas dataframe and set the model's *enable_categorical* hyperparameter to *True*.
# MAGIC
# MAGIC But while the XGBoostRegressor makes some things easier, it makes other things harder.  The algorithm is controlled by a large number of hyperparameters and the values we should assign to these are not always intuitive.  As a result, we often get our best results from this algorithm when we perform an extensive hyperparameter tuning exercise.
# MAGIC
# MAGIC To support this exercise, we will make use of [hyperopt](https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/index.html) which will allow us to perform an intelligent search of a hyperparameter search space.  If we configure it to perform it's trail runs using Spark, these evaluations will take place in a distributed manner across the Spark cluster, speeding up the overall process.  But to ensure this is efficient, we need to replicate copies of our test and validation datasets to each worker in the cluster:

# COMMAND ----------

# DBTITLE 1,Distribute Training & Validation Data to Cluster Workers
train_pd_broadcast = sc.broadcast(train_pd)
val_pd_broadcast = sc.broadcast(val_pd)

# COMMAND ----------

# MAGIC %md With our data distributed, we can now define our [search space](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/) as follows:

# COMMAND ----------

# DBTITLE 1,Define Hyperparameter Search Space
search_space = {
     'max_depth': hp.quniform('max_depth', 5, 15, 1)
    ,'n_estimators': hp.quniform('n_estimators', 5, 15, 1)                      
    ,'learning_rate' : hp.uniform('learning_rate', 0.01, 0.40) 
}

# COMMAND ----------

# MAGIC %md We can now define a function to which hyperopt will pass hyperparameter values selected from this search space.  This function will train the model and evaluate it.  An evaluation metric will be returned which hyperopt will attempt to minimize as it chooses subsequent values from the search space:

# COMMAND ----------

# DBTITLE 1,Define Function to Evaluate Model
def fmin_xgbr(hyperopt_params):

  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])

  # use both discrete and continuous features
  feature_cols = disc_feature_cols + cont_feature_cols
  
  # fit our model to the data
  model = (
    XGBRegressor(**params, enable_categorical=True, tree_method='approx')
      .fit(
        train_pd_broadcast.value[feature_cols],
        train_pd_broadcast.value[label_col]
        )
      )

  # predict the holdout
  y_pred = model.predict(
    val_pd_broadcast.value[feature_cols]
    )

  # score the predictions
  rmse = mean_squared_error(val_pd_broadcast.value[label_col], y_pred, squared=False)
  mae = mean_absolute_error(val_pd_broadcast.value[label_col], y_pred)
  mlflow.log_metrics({'rmse':rmse,'mae':mae})
  
  return {'loss':rmse, 'status':STATUS_OK}

# COMMAND ----------

# MAGIC %md With everything in place, we can now tune our model:

# COMMAND ----------

# DBTITLE 1,Perform Hyperparameter Tuning
with mlflow.start_run(run_name='XGBR-Tuning: All Products'):

  argmin = fmin(
      fmin_xgbr,
      space = search_space,
      algo = tpe.suggest,
      max_evals = sc.defaultParallelism * 10,
      trials = SparkTrials(parallelism = sc.defaultParallelism)
    )
  
best_params = space_eval(search_space, argmin)
print(
  best_params
)

# COMMAND ----------

# MAGIC %md With optimized hyperparameter values selected, we can now train and evaluate our model as follows:

# COMMAND ----------

# DBTITLE 1,Train Model on Optimized Hyperparameters
with mlflow.start_run(run_name='XGBR: All Products') as run:

  # configure model parameters
  params = best_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators'])
  mlflow.log_params(params)

  feature_cols = disc_feature_cols + cont_feature_cols

  # fit our model to the data
  model = (
    XGBRegressor(**params, enable_categorical=True, tree_method='approx')
      .fit(
        train_pd[feature_cols],
        train_pd[label_col]
        )
      )
  fe.log_model( # record the trained model
    model=model,  # the sklearn model
    training_set=training_set,
    artifact_path='model',
    flavor=mlflow.xgboost,
    # input_example=test_pd[feature_cols].head(1) # a sample feature input record
  ) 

  # predict the holdout
  y_pred = model.predict(
    test_pd[feature_cols]
    )

  # score the predictions
  rmse = mean_squared_error(test_pd[label_col], y_pred, squared=False)
  mae = mean_absolute_error(test_pd[label_col], y_pred)
  mlflow.log_metrics({'rmse':rmse,'mae':mae})

# print key results
run_id_xgbrall = run.info.run_id
print(f"Run_Id: {run_id_xgbrall}")
print(f"RMSE:   {rmse}")
print(f"MAE:    {mae}")

# COMMAND ----------

# MAGIC %md ##C. Other Approaches
# MAGIC
# MAGIC We could certainly consider generating other regression models to see if any of them gave us better results.  Ensemble techniques may also allow us to squeeze a bit more predictive capability from our model.  But what about building a model for each product individually?
# MAGIC
# MAGIC In an earlier version of this notebook, we did this using simple user-defined function in combination with the [*applyInPandas()*](https://spark.apache.org/docs/3.2.0/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html) method call.  The results were incredibly mixed given that many products did not have sufficient values for us to train a proper model and some products produced models with very low RMSE scores while others produced models with very high RSME scores.  The extreme variations in these scores makes it difficult to say that one approach is better than another though you could certainly argue that in scenarios where a per-product model outperforms the general model, you might have a preference to use that one. Still, our models are treating the products as categorical inputs which means that we are making product-specific predictions even in the *all products* models shown here which leaves us feeling like this approach is best for our purposes.

# COMMAND ----------

# Construct the model URI from the run ID
model_uri = f"runs:/{run.info.run_id}/model"

# The Unity Catalog model name (must include catalog and schema)
registered_model_name = f"{config['catalog name']}.{config['database name']}.dynamic_pricing"

# Register the model to Unity Catalog
registration = mlflow.register_model(model_uri, registered_model_name)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.errors.platform import ResourceAlreadyExists
serving_endpoint_name = "dynamic_pricing"
serving_config={
    "served_entities": [
        {
            "entity_name": registered_model_name,
            "entity_version": registration.version,
            "workload_size": "Small",  # or Medium/Large as needed
            "scale_to_zero_enabled": True
        }
    ]
}
core_config = EndpointCoreConfigInput.from_dict(serving_config)
w = WorkspaceClient()

try:
    print("Creating endpoint")
    w.serving_endpoints.create_and_wait(
        name=serving_endpoint_name,
        config=core_config
    )
except ResourceAlreadyExists:
    print("Endpoint already exists")
    served_entity = ServedEntityInput(
        entity_name=registered_model_name,
        entity_version=registration.version,
        workload_size="Small",
        scale_to_zero_enabled=True
    )
    # Update the endpoint to point to the new model version
    w.serving_endpoints.update_config(
        name=serving_endpoint_name,
        served_entities=[served_entity]
    )
    print(f"Updated endpoint {serving_endpoint_name} to model version {registration.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Step 3: Apply the Model
# MAGIC
# MAGIC We now have two models that predict how volumes change with changes in price as well as prior volumes. To make use of these models, we will typically generate a range of price changes we might apply to a future period and calculate volume changes from this.  
# MAGIC
# MAGIC To demonstrate this approach, we can grab data for a product sold within a given geo:

# COMMAND ----------

first_row = sales.limit(1).collect()[0]

# Prepare your input as a list of record dicts
records = [
    {"productKey": first_row["productKey"], "geographyKey": first_row["geographyKey"], 
     "dateKey": first_row["dateKey"]}
]

# Query the serving endpoint
response = w.serving_endpoints.query(
    name=serving_endpoint_name,           
    dataframe_records=records         
)
response

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# DBTITLE 1,Identify Sample Product & Geography
productKey = sales.select('productKey').collect()[0]['productKey']
geographyKey = sales.select('geographyKey').collect()[0]['geographyKey']

# retrieve historical sales for this product and geo
sample_sales = sales.filter(f"productKey={productKey} and geographyKey={geographyKey}")

# COMMAND ----------

# MAGIC %md We will now add a record to this dataset to hold information for the period against which we wish to make predictions:

# COMMAND ----------

# DBTITLE 1,Add Projection Row for "Next" Week
sample_projection = (
    sample_sales
        .orderBy('dateKey', ascending=False) # get last entry in set
        .limit(1)
        .join(spark.table(f'{config["catalog name"]}.{config['database name']}.date').select('dateKey','date'), on='dateKey') # join to date table
        .drop('dateKey')
        .withColumn('date', fn.expr("date_add(date,7)")) # push date ahead 7 days
        .join(spark.table(f'{config["catalog name"]}.{config['database name']}.date').select('dateKey','date'), on='date') # get key for this date
        .drop('date')
        .select('productKey','geographyKey', 'dateKey') # get keys for projection row
  )

with_projections =   sample_sales.unionByName(sample_projection, allowMissingColumns=True)


# COMMAND ----------

# MAGIC %md We can now perform all the proportional conversion, priors retrievals, and cyclical conversions previously performed against the historical dataset in order to provide feature values for this row:

# COMMAND ----------

# DBTITLE 1,Get Feature Values
# proportions
with_features = (
  with_projections
    .withColumn(
      'incrementalVolumeProportion',
      fn.expr("100 * (incrementalVolumeSales) / (incrementalVolumeSales + baseVolumeSales)")
    )
    .withColumn(
      'promotionVolumeProportion',
      fn.expr("100 * (volumeSalesAnyTradePromotion) / (volumeSalesAnyTradePromotion + volumeSalesNoPromotion)")
    )
  )

# priors
columns_to_lag = ['avgSellingPrice','volumeSales','incrementalVolumeProportion','promotionVolumeProportion','storeCountWhereScanned']
for c in columns_to_lag:
  with_features = (
      with_features
        .withColumn(f"prior_{c}", fn.expr(f"LAG({c},1) OVER(PARTITION BY productKey, geographyKey ORDER BY dateKey)"))
    )

# cyclic date features
with_features = (
  with_features
    .join(
      spark.table(f'{config["catalog name"]}.{config['database name']}.date').selectExpr('dateKey','weekOfYear as week',"date_part('month',date) as month"),
      on='dateKey'
      )
    .withColumn(
      "cyclicWeek",
      fn.expr("1-cos(week/53*2*pi())")
     )
    .withColumn(
      "cyclicMonth",
      fn.expr("1-cos(month/13*2*pi())")
      )
  )

# features only
sample_features = (
  with_features
    .select(
      'dateKey',
      "productKey",
      "geographyKey",
      "cyclicWeek",
      "cyclicMonth",
      "prior_avgSellingPrice",
      "prior_volumeSales",
      "prior_incrementalVolumeProportion",
      "prior_promotionVolumeProportion",
      "prior_storeCountWhereScanned",
      )
    .orderBy('dateKey', ascending=False) # get last record (projection record)
    .limit(1)
    # .drop('dateKey')
    )



# COMMAND ----------

# MAGIC %md We then create a range of pricing percentage changes we wish to explore during this period and cross it with the dataset.  This creates a set of possible future values against which we can make volume change predictions:

# COMMAND ----------

# DBTITLE 1,Estimate a Range of Price Changes
num_rows = 5000
start = -25
end = 30
step = (end - start) / (num_rows - 1)

price_change_percentages = (
    spark.range(0, num_rows)
    .withColumn(
        'pricePercentChange',
        fn.lit(start) + fn.col('id') * fn.lit(step)
    )
    .drop('id')
)

sample_to_score = (
  sample_features
    .crossJoin(price_change_percentages)
    .withColumn('dateKeyString',fn.col('dateKey').cast('string'))
    .withColumn('dateKey',fn.to_date('dateKeyString',"yyyyMMdd"))
    .drop('dateKeyString')
  )

# COMMAND ----------

# MAGIC %md Using our models (retrieved from mlflow), we can make various estimates of volume changes given the prior state of the market and the projected price change:

# COMMAND ----------

# batch_df has columns ['customer_id', 'account_creation_date']
predictions = fe.score_batch(
  model_uri=model_uri,
  df=sample_to_score
)

# COMMAND ----------

predictions.write.format("delta").mode("overwrite").saveAsTable(f"{config['catalog name']}.{config['database name']}.predictions_delta")

# COMMAND ----------

# MAGIC %md The two different models give us slightly different pictures of the impact of changes within these ranges.  The linear regression model is nice and smooth and easy to interpret but may be very simplistic.  The gradient boosted model is much more uneven, reflecting differences in signals found within different ranges of the available dataset.  The outputs of both models should be carefully scrutinized and may provide insights into the market dynamics affecting how consumers will respond to price changes, but neither is perfect.  Analysts will want to consider these outputs along with other information about the future market within which a pricing change may be made before making the leap one direction of another.

# COMMAND ----------

# DBTITLE 1,Update Function Path
import os
import re 

function_filepath = os.path.join(
    os.getcwd(), "04_Create_Genie_Function.sql"
)
with open(function_filepath, "r") as f:
    content = f.read()

pattern = re.compile(
    r"FUNCTION\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)",
    flags=re.IGNORECASE,
)

def _repl(_):
    return f"FUNCTION {config['catalog name']}.{config['database name']}._genie_query()"
new_content, n_subs = pattern.subn(_repl, content)

with open(function_filepath, "w") as f:
    f.write(new_content)

print(f"✔ Replaced {n_subs} occurrence(s) in {function_filepath}")

# COMMAND ----------

# DBTITLE 1,Update Dashboard Path
import os
import re 

dashboard_filepath = os.path.join(
    os.path.dirname(os.getcwd()), "Price Change Dashboard.lvdash.json"
)
with open(dashboard_filepath, "r") as f:
    content = f.read()

pattern = re.compile(
    r"FROM\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)",
    flags=re.IGNORECASE,
)

def _repl(_):
    return f"FROM {config['catalog name']}.{config['database name']}.predictions_delta"
new_content, n_subs = pattern.subn(_repl, content)

with open(dashboard_filepath, "w") as f:
    f.write(new_content)

print(f"✔ Replaced {n_subs} occurrence(s) in {dashboard_filepath}")