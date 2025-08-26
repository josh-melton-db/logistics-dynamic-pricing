# Databricks notebook source
# MAGIC %run ./00_Intro_&_Config

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

from spark_utils import get_spark

spark = get_spark()

def generate_brand_name_mapping(n_brands=68, seed=42):
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    brand_values = []
    brand_names = []
    for i in range(n_brands):
        brand_values.append(f"BR{(i+1):04d}")
        brand_names.append(fake.unique.company())
    return pd.DataFrame({
        'brandValue': brand_values,
        'brandName': brand_names
    })

def generate_date_table(start_date='2021-01-01', periods=1461):
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    records = []
    for dt in dates:
        # Compute weekEndDate as the next Sunday
        days_to_sunday = (6 - dt.weekday()) % 7
        week_end = dt + timedelta(days=days_to_sunday)
        records.append({
            'monthOfYear': dt.month,
            'weekEndDate': week_end.date(),
            'date': dt.date(),
            'dateKey': int(dt.strftime('%Y%m%d')),
            'year': dt.year,
            'dayOfMonth': dt.day,
            'isWeekDay': 'Y' if dt.weekday() < 5 else 'N',
            'weekOfYear': int(dt.isocalendar()[1])
        })
    return pd.DataFrame(records)

def generate_geography(n_geo=10, seed=42):
    random.seed(seed)
    fake = Faker()
    sources = ['Nielsen', 'IRI', 'Internal']
    records = []
    for i in range(n_geo):
        header = fake.state()
        source = random.choice(sources)
        margin = round(random.uniform(5, 25), 2)
        margin_promo = round(margin * random.uniform(0.6, 0.9), 2)
        desc = f"{header} region coverage"
        records.append({
            'geographyKey': i + 1,
            'geographyHeader': header,
            'geographySource': source,
            'geographyAggregateRetailerMargin': margin,
            'geographyAggregateRetailerMarginPromo': margin_promo,
            'geographyDescription': desc
        })
    return pd.DataFrame(records)

def generate_product(n_products=5032, brand_df=None, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    adjectives = ['Heavy-Duty', 'Precision', 'High-Temp', 'Low-Viscosity', 'Reinforced',
                  'Alloyed', 'Anti-Corrosive', 'Multi-Grade', 'Sealed', 'OEM']
    nouns = ['Bearing', 'Sealant', 'Valve', 'Compressor', 'Sensor', 'Gasket',
             'Coupling', 'Fastener', 'Actuator', 'Module']
    categories = ['Powertrain', 'Chassis', 'Electrical', 'Hydraulics', 'Lubrication']
    need_states = ['Low Demand', 'Scheduled Replacement', 'Urgent Repair']
    segments = ['OEM Parts', 'Aftermarket', 'Maintenance', 'Tooling']
    subsegments = ['Automotive', 'Aerospace', 'Industrial Equipment']
    pack_units = ['pcs', 'L', 'kg']
    size_groups = ['Compact', 'Standard', 'Heavy', 'Industrial-Scale']
    formats = ['Boxed', 'Crated', 'Wrapped', 'Barreled', 'Bulk']
    specials = [None, 'Military Spec', 'ISO Certified', 'Export Only']

    brand_values = brand_df['brandValue'].tolist()
    records = []
    for i in range(n_products):
        bv = random.choice(brand_values)
        bk = brand_values.index(bv) + 1

        size_val = random.choice([50, 100, 200, 500, 1000])
        size_unit = random.choice(pack_units)

        # Simple US conversion for volume/weight
        if size_unit == 'L':
            us_val = round(size_val * 33.814, 6)  # fl oz
            us_unit = 'fl oz'
        elif size_unit == 'kg':
            us_val = round(size_val * 35.274, 6)  # oz
            us_unit = 'oz'
        else:  # 'pcs' – treat pack size as count of units
            us_val = float(size_val)
            us_unit = 'units'

        # Base price varies by category and pack size (bigger pack -> higher price, lower $/unit)
        cat = random.choice(categories)
        cat_price_anchor = {
            'Powertrain': 22.0, 'Chassis': 14.0, 'Electrical': 10.0, 'Hydraulics': 18.0, 'Lubrication': 8.0
        }[cat]
        # basePrice is per pack
        base_price = max(1.0, np.random.lognormal(mean=np.log(cat_price_anchor), sigma=0.35) * (size_val / 100.0) ** 0.7)

        # Own-price elasticity: most products in -0.7 .. -2.0 (more negative = more sensitive)
        # Urgent repair parts tend to be less elastic (closer to 0)
        need = random.choice(need_states)
        if need == 'Urgent Repair':
            elasticity = -0.6 + np.random.normal(0, 0.1)   # around -0.6
        elif need == 'Scheduled Replacement':
            elasticity = -1.0 + np.random.normal(0, 0.2)   # around -1.0
        else:  # Low Demand (discretionary)
            elasticity = -1.6 + np.random.normal(0, 0.3)   # around -1.6

        # Promo lift multiplier (units when on promo)
        promo_lift = np.clip(np.random.normal(1.35, 0.2), 1.05, 2.0)

        records.append({
            'productKey': i + 1,
            'productDescription': f"{random.choice(adjectives)} {random.choice(nouns)}",
            'barcodeValue': ''.join(random.choices('0123456789', k=12)),
            'categoryValue': cat,
            'needStateValue': need,
            'segmentValue': random.choice(segments),
            'subSegmentValue': random.choice(subsegments),
            'manufacturerValue': fake.company(),
            'brandValue': bv,
            'subBrandValue': random.choice(['Classic', 'Organic', 'Premium', 'Value']),
            'fragranceValue': random.choice(['Original', 'Lavender', 'Citrus', 'Unscented']),
            'packSize': f"{size_val} {size_unit}",
            'packSizeValue': size_val,
            'packSizeUnit': size_unit,
            'sizeGroupValue': random.choice(size_groups),
            'packFormatValue': random.choice(formats),
            'specialPackTypeValue': random.choice(specials),
            'brandKey': bk,
            'packSizeValueUS': us_val,
            'packSizeUnitUS': us_unit,

            # NEW FIELDS
            'basePrice': round(base_price, 2),
            'priceElasticity': float(elasticity),
            'promoLift': float(promo_lift)
        })
    return pd.DataFrame(records)



def generate_retail_sales(product_df, geography_df, date_df, n_records=852987, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Pre-index product attributes to avoid repeated .loc lookups
    prod = product_df.set_index('productKey')[['packSizeValueUS', 'basePrice', 'priceElasticity', 'promoLift']].to_dict('index')

    geos = geography_df['geographyKey'].tolist()
    dates = date_df['dateKey'].tolist()
    products = list(prod.keys())

    # Light seasonality by month and weekend effect
    date_idx = date_df.set_index('dateKey')
    month_map = date_idx['monthOfYear'].to_dict()
    weekday_map = date_idx['isWeekDay'].to_dict()

    # Geography demand scaler (e.g., market size); ~0.8–1.2
    geo_scale = {g: np.clip(np.random.normal(1.0, 0.1), 0.8, 1.2) for g in geos}

    records = []
    for _ in range(n_records):
        pk = random.choice(products)
        gk = random.choice(geos)
        dk = random.choice(dates)

        attrs = prod[pk]
        pack_size_us = float(attrs['packSizeValueUS'])
        base_price = float(attrs['basePrice'])
        elasticity = float(attrs['priceElasticity'])
        promo_lift = float(attrs['promoLift'])

        # Simple monthly seasonality and weekday/weekend effect
        m = month_map[dk]
        is_weekday = 1 if weekday_map[dk] == 'Y' else 0
        month_factor = {
            1: 0.95, 2: 0.97, 3: 1.00, 4: 1.02, 5: 1.05, 6: 1.07,
            7: 1.08, 8: 1.04, 9: 1.01, 10: 1.03, 11: 1.06, 12: 1.10
        }.get(m, 1.0)
        weekday_factor = 1.00 if is_weekday else 1.06  # slight weekend bump

        # Decide promotion first; promos lower price and (later) increase demand
        on_promo = (random.random() < 0.28)  # ~28% of days have promo
        # Daily price has noise; promos discount 10–25%
        day_price = base_price * np.clip(np.random.normal(1.0, 0.06), 0.85, 1.25)
        if on_promo:
            day_price *= random.uniform(0.75, 0.9)
        price = max(0.5, round(day_price, 2))

        # Expected demand: base ~ Poisson mean around 12 (scaled), adjusted by price elasticity etc.
        base_lambda = 12.0
        price_ratio = max(0.25, price / max(0.5, base_price))
        # Core demand model: lambda ∝ (price/basePrice)^{elasticity}
        lam = base_lambda * (price_ratio ** elasticity) * month_factor * weekday_factor * geo_scale[gk]
        if on_promo:
            lam *= promo_lift
        lam = max(0.1, lam)

        # Realized units (non-negative integer)
        unit_sales = int(np.random.poisson(lam))

        # Split units into promo/non-promo buckets for reporting
        if on_promo and unit_sales > 0:
            promo_units = int(round(unit_sales * random.uniform(0.6, 0.9)))
        else:
            promo_units = 0
        nonpromo_units = unit_sales - promo_units

        promo_price = price  # all promo units at the (discounted) price of the day
        regular_price = max(0.5, round(base_price * np.clip(np.random.normal(1.0, 0.03), 0.9, 1.2), 2))

        # Value
        val_any = promo_units * promo_price
        val_no = nonpromo_units * regular_price
        val = val_any + val_no
        base_val = unit_sales * base_price  # baseline at base price

        # Volume (using pack size)
        vol_any = promo_units * pack_size_us
        vol_no = nonpromo_units * pack_size_us
        vol = vol_any + vol_no
        base_vol = unit_sales * pack_size_us

        # Costs (gross margins vary)
        cost = val * random.uniform(0.4, 0.6)
        promo_cog = val_any * random.uniform(0.4, 0.6)
        nonpromo_cog = val_no * random.uniform(0.4, 0.6)

        # Incrementals (a chunk of promo is incremental)
        inc_unit = promo_units * random.uniform(0.25, 0.6)
        inc_vol = inc_unit * pack_size_us
        inc_val = inc_unit * promo_price

        records.append({
            'productKey': pk,
            'geographyKey': gk,
            'dateKey': dk,
            'unitSales': unit_sales,
            'valueSales': val,
            'volumeSales': vol,
            'baseUnitSales': float(unit_sales),
            'baseValueSales': base_val,
            'baseVolumeSales': base_vol,
            'storeCount': random.randint(50, 500),
            'storeCountWhereScanned': random.uniform(0.5, 1.0) * random.randint(50, 500),
            'cwdWeightedDistribution': random.uniform(0, 100),
            'valueSalesAnyTradePromotion': val_any,
            'unitSalesAnyTradePromotion': promo_units,
            'volumeSalesAnyTradePromotion': vol_any,
            'valueSalesNoPromotion': val_no,
            'baseValueSalesAnyTradePromotion': promo_units * base_price,
            'baseVolumeSalesNoPromotion': vol_no,
            'volumeSalesNoPromotion': vol_no,
            'unitSalesNoPromotion': nonpromo_units,
            'baseValueSalesNoPromotion': nonpromo_units * base_price,
            'baseVolumeSalesAnyTradePromotion': promo_units * pack_size_us,
            'incrementalVolumeSales': inc_vol,
            'incrementalUnitSales': inc_unit,
            'incrementalValueSales': inc_val,
            'costOfGoods': cost,
            'promoCostOfGoods': promo_cog,
            'noPromoCostofGoods': nonpromo_cog,
            'baseUnitsSalesNoPromotion': float(nonpromo_units),
            'baseUnitsSalesAnyTradePromotion': float(promo_units),

            # Helpful to keep around
            'effectivePrice': price,
            'regularPriceSample': regular_price,
            'onPromotion': 'Y' if on_promo else 'N'
        })
    return pd.DataFrame(records)


print("Generating dummy data...")
brand_df = generate_brand_name_mapping()
date_df = generate_date_table()
geo_df = generate_geography()
product_df = generate_product(brand_df=brand_df)
sales_df = generate_retail_sales(product_df, geo_df, date_df)

# COMMAND ----------

# Save to tables
print("Saving data to tables...")
spark.createDataFrame(brand_df).write.mode('overwrite').saveAsTable(f"{config['catalog name']}.{config['database name']}.brand_name_mapping")
spark.createDataFrame(date_df).write.mode('overwrite').saveAsTable(f"{config['catalog name']}.{config['database name']}.date")
spark.createDataFrame(geo_df).write.mode('overwrite').saveAsTable(f"{config['catalog name']}.{config['database name']}.geography")
spark.createDataFrame(product_df).write.mode('overwrite').saveAsTable(f"{config['catalog name']}.{config['database name']}.product")
spark.createDataFrame(sales_df).write.mode('overwrite').saveAsTable(f"{config['catalog name']}.{config['database name']}.retail_sales")