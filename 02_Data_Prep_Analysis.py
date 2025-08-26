# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC In order to examine price elasticicies, we need a set of historical sales data for products reflecting changes in volumes and revenue at various average price points.  The dataset we will be using presents anonymized weekly sales data from across multiple MFG organizations.
# MAGIC
# MAGIC In this notebook, we will load these data into our environment and perform some basic analysis to orient us to the data and patterns as they relate to price and sales volumes.

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_&_Config"

# COMMAND ----------

# DBTITLE 1,Import the Required Libraries
import pyspark.sql.functions as fn

# COMMAND ----------

# MAGIC %md ##Step 1: Access the Data
# MAGIC
# MAGIC To get started, we'll read the data from public storage location into tables residing in our Databricks database:

# COMMAND ----------

# MAGIC %md With our data loaded, we can examine the tables to understand their contents and structure:

# COMMAND ----------

# DBTITLE 1,Print Table Schemas
from spark_utils import get_spark

spark = get_spark()


table_names = ['brand_name_mapping','date','geography','product','retail_sales'] 

for table_name in table_names:

    # print table info
    row_count = spark.table(f"{config['catalog name']}.{config['database name']}.{table_name}").count()
    print(f"{table_name}: {row_count} rows") # print table name

    # print table schema
    tbl = spark.table(f"{config['catalog name']}.{config['database name']}.{table_name}") # access table within spark
    tbl.printSchema() # print table schema info

# COMMAND ----------

# MAGIC %md While most of the data is ready to use, there are a couple items in the *retail_sales* table we'd like to adjust. 
# MAGIC
# MAGIC First, we will rename the *dateInt* field to *dateKey*, aligning the name to the matching field name in the *date* table.  This isn't required but aligning the names will allow us to specify joins more easily later.
# MAGIC
# MAGIC Second, we'll calculate the average selling price for our products across each week of sales.  We will derive this by simply dividing the sales revenue generated (as captured in the *valueSales* column) by the number of units sold (as captured in the *volumeSales* column):

# COMMAND ----------

# DBTITLE 1,Adjust the Sales Data
_ = (
    spark
        .table(f'{config["catalog name"]}.{config['database name']}.retail_sales')
        .withColumnRenamed('dateInt','dateKey') # make column name consistent
        .withColumn('avgSellingPrice', fn.expr('valueSales/volumeSales'))
        .write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema','true')
        .saveAsTable(f'{config["catalog name"]}.{config['database name']}.enhanced_sales')
)


# COMMAND ----------

# MAGIC %md ##Step 2: Retrieve Data for Analysis
# MAGIC
# MAGIC With our data prepared for use, it would be helpful to examine how price and sales interact.  We'll limit this analysis to calendar years 2021 and 2022, the last two years for which we have complete data available to us:

# COMMAND ----------

# DBTITLE 1,Get Sales Data for Analysis
sales = (
   spark
      .table(f'{config["catalog name"]}.{config['database name']}.retail_sales')
      .join( # limit data to CY 2021 and 2022
        spark.table(f'{config["catalog name"]}.{config['database name']}.date').select('dateKey','date','year').filter('year between 2021 and 2022'),
        on='dateKey'
        )
      .join( # get product fields needed for analysis
        spark.table(f'{config["catalog name"]}.{config['database name']}.product').select('productKey','brandValue','packSizeValueUS'),
        on='productKey'
        )
      .join( # get brand fields needed for analysis
        spark.table(f'{config["catalog name"]}.{config['database name']}.brand_name_mapping').select('brandValue','brandName'),
        on='brandValue'
        )      
  )

# COMMAND ----------

# MAGIC %md To further constrain our analysis, we'll focus on the top performing brands, in terms of sales volume (revenue), in 2021 and 2022.  We will rank these brands and then focus our attention on just the top 5 performing brands in each year:
# MAGIC
# MAGIC **NOTE** Different brands may perform better in different years so that combined list of *top 5* brands from each year may contain more than 5 entries.

# COMMAND ----------

# DBTITLE 1,Calculate Annual Brand Rankings
brand_rankings = (
    sales
        .groupBy('year', 'brandValue') # calculate sales volume (revenue) by brand and year 
            .agg(
                fn.sum('volumesales').alias('volumesales')
                )
        .withColumn( # rank brands by annual revenues
            'rank', 
            fn.expr("RANK() OVER(PARTITION BY year ORDER BY volumesales DESC)")
            )
        .filter(fn.expr("rank <= 5"))
    )
brand_rankings.orderBy('year','rank')

# COMMAND ----------

# MAGIC %md ##Step 3: Examine Total Unit Sales vs. Average Unit Price
# MAGIC
# MAGIC For our first analysis, we will take a look at the influence of price on sales.  We'll leverage three key sales metrics for this:
# MAGIC </p>
# MAGIC
# MAGIC * unit sales - Number of Units sold
# MAGIC * value sales - Monetary Value of sales
# MAGIC * volume sales - Volumetric amount of goods sold in relation to fluid or ounce measurements
# MAGIC
# MAGIC We will combine these into two averages defined as follows:
# MAGIC </p>
# MAGIC
# MAGIC * average unit price - Total Value / # Units sold
# MAGIC * average volume price - Total Value / # Volume sold
# MAGIC
# MAGIC This should give us a nice starting point for understanding consumer willingness to pay for a particular brand:

# COMMAND ----------

# DBTITLE 1,Calculate Average Unit Price and Average Volume Price for Top Brands
aup_avp = (
  sales
    .join(
      brand_rankings.select('brandValue').distinct(), # top brands
      on=['brandValue']
      )
    .groupBy('brandName','date')
      .agg(
        fn.sum('unitSales').alias('unitSales'),
        fn.sum('valueSales').alias('valueSales'),
        fn.sum('volumeSales').alias('volumeSales')
      )
    .withColumn('avgUnitPrice', fn.expr('valueSales/unitSales'))
    .withColumn('avgVolumePrice', fn.expr('valueSales/volumeSales'))
    .orderBy('brandName','date')
    )



# COMMAND ----------

# MAGIC %md Examining unit sales across these brands, we can see that some brands experienced a unit sales drop at the beginning of January 2022.  This was most likely due to supply chain disruptions that limited the availability of products to consumers. Sales then begin to pickup shortly after but there's a dip in performance in some brands in July 2022 while other brands seem to pick up steam around this time:

# COMMAND ----------

# MAGIC %md If we take a look at price around July 2022, we can see that some brands saw effective price increases around that time.  What's an *effective price increase*?  It may be a literal per unit price increase or it may be the discontinuation of discounts that discounts on the items.  Either way, the consumer is paying more to acquire the product.
# MAGIC
# MAGIC If we look at FizzTreat which saw a noteable drop in sales around this time period, we can see that there's a late Summer price increase that may explain this trend.  At the same time, SparkleBlast and ElectroQuench both had similar price increases.  Both saw dips in sales performance, but not as much as FizzTreat, indicating these products may have consumer sensitivities to price (price elastiticies). Interestingly, JuicyJoy which has the highest unit price of the products observed here saw its own slight price increase.  However, there doesn't appear to be a corresponding dip in sales.  This might incidate that consumers see JuicyJoy as a premium product and are already willing to pay more for it and therefore are less sensitive to these price changes:

# COMMAND ----------

# MAGIC %md ##Step 4: Brand Sales Market Share
# MAGIC
# MAGIC As consumers shifted their purchasing patterns in response to pricing changes, it's interesting to examine the consequence of this on market share:

# COMMAND ----------

# DBTITLE 1,Calculate Market Share for Top Brands by Year

brand_sales = (
  sales
    .withColumn('totalAnnualValueSales', fn.expr("SUM(valueSales) OVER(PARTITION BY year)")) # get total sales across ALL BRANDS for each year
    .join( # limit visual to top brands
      brand_rankings.select('brandValue').distinct(), # top brands
      on='brandValue'
      )
    .groupBy('year','brandName')
      .agg(
        fn.first('totalAnnualValueSales').alias('totalAnnualValueSales'),
        fn.sum('valueSales').alias('totalValueSales')
      )
    .withColumn('brandSalesPercent', fn.expr('100 * totalValueSales/totalAnnualValueSales'))
)

brand_sales.orderBy('brandSalesPercent')

# COMMAND ----------

# MAGIC %md We can see the dip taken by FizzTreat clearly in the chart.  Sparkleblast which saw sales declines actually gained share which could indicate either an overall decline in the market that's larger than what these two particular brands experienced. JuicyJoy and ElectroQuench both held relatively stead within the market.

# COMMAND ----------

# MAGIC %md ##Step 5: Price Pack Architecture
# MAGIC This final section looks at strategy of pricing product at various different pack sizes. A pack size represents the size of the product the consumer purchases.  For beverages, we typically measure this in milliters (ml) or fluid ounces (fl oz).
# MAGIC
# MAGIC By visualizing unit price by pack size, we can see how different products perform within different size categories and where there may be opportunities for better aligning with customer expectations:

# COMMAND ----------

# DBTITLE 1,Calculate Price vs. Pack Size
price_pack = (
  sales
    .filter('year=2022 AND packSizeValueUS < 50') # limit to sales in 2022 and cost under $50 (not bulk bundles)
    .join(
      brand_rankings.select('brandValue').distinct(), # top brands
      on=['brandValue']
      )
    .groupBy('year','brandName','packSizeValueUS')
      .agg(
        fn.sum('unitsales').alias('unitSales'),
        fn.sum('valuesales').alias('valueSales'),
        fn.sum('volumesales').alias('volumeSales')
        )
    .withColumn('avgUnitPrice', fn.expr('valueSales/unitSales'))
    .withColumn('avgVolumePrice', fn.expr('valueSales/volumeSales'))
    .orderBy('brandName', 'year', 'packSizeValueUS')
  )

# COMMAND ----------

# MAGIC %md While this visualization is quite busy, we can see an overall linear increase in price as pack size increases.  We can see popular sizes of around 8 and 16 fluid ounces with many other products delivering products at other pack sizes.  The low sales volumes (as indicated by the size of the bubbles) in these *in between* sizes may point to an opportunity for SKU rationalization.  The dominance of some brands at some pack sizes, such as MixTaste at the 16 fluid ounce size, might indicate competitor niches to avoid.
# MAGIC
# MAGIC While this chart doesn't tell us too much about price elastiticies, it provides us a glimpse into how we might leverage knowledge of such elastiticies to adjust our product offerings.