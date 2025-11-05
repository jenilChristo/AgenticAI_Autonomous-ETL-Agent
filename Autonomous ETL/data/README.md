# iPhone High-Intent Customer Analysis - Dataset Documentation

## Overview
This dataset collection is designed for analyzing high-intent iPhone customers for Amazon's customer acquisition team. The data simulates real-world e-commerce patterns with comprehensive customer journey tracking, purchase behavior, and marketing attribution.

## Dataset Files

### 1. CustomerOrders.csv (50 records)
**Purpose**: Transactional data for iPhone and related product purchases
**Key Metrics**: Order value, purchase frequency, product preferences, payment methods
**Time Range**: June 2025 - September 2025 (Last 4 months)
**Columns**: 13 fields including OrderID, CustomerID, ProductType, SKU, OrderValue, OrderTimestamp

### 2. RegionMapping.csv (30 records) 
**Purpose**: Geographic and demographic mapping of customers
**Key Metrics**: Regional distribution, customer tier, signup patterns, language preferences
**Coverage**: All major Indian states and metros
**Columns**: 10 fields including CustomerID, Region, State, City, CustomerTier, SignupDate

### 3. ProductCatalog.csv (39 records)
**Purpose**: Complete iPhone product catalog with metadata
**Key Metrics**: Product hierarchy, pricing, margins, specifications
**Coverage**: iPhone 13/14/15 series + accessories
**Columns**: 18 fields including SKU, ProductName, Category, Price, Storage, LaunchDate

### 4. CustomerDemographics.csv (30 records)
**Purpose**: Detailed customer profiles and segmentation data
**Key Metrics**: Age, income, education, tech-savviness, brand loyalty
**Segments**: Budget, Mid-Range, Premium, Luxury, Enterprise
**Columns**: 13 fields including Age, Gender, Income, Education, TechSavvy, PriceSegment

### 5. CustomerBehavior.csv (50 records)
**Purpose**: Digital touchpoint and engagement tracking
**Key Metrics**: Session duration, page views, conversion paths, device usage
**Channels**: Organic search, paid ads, social media, email, direct
**Columns**: 12 fields including ActivityType, Platform, SessionDuration, ConversionFlag

### 6. MarketingCampaigns.csv (10 records)
**Purpose**: Campaign performance and attribution data
**Key Metrics**: Budget allocation, ROI, channel effectiveness, audience targeting
**Campaign Types**: Product launch, seasonal sales, B2B, student offers
**Columns**: 14 fields including Budget, Spend, Impressions, Conversions, Revenue

## High-Intent Customer Definition (Business Logic)

### Primary Criteria:
1. **Frequency**: ≥3 iPhone-related purchases in last 90 days
2. **Recency**: Last purchase within 30 days  
3. **Basket Value**: Average order value > ₹50,000
4. **Engagement**: High digital engagement (multiple touchpoints)

### Secondary Indicators:
- Premium/VIP customer tier
- Apple brand loyalty
- High-income demographics (>₹10L annually)
- Expert tech-savviness level
- Multiple accessory purchases

## Geographic Distribution:
- **North Region**: 33% (Delhi, Gurgaon, Chandigarh, etc.)
- **West Region**: 27% (Mumbai, Pune, Ahmedabad, etc.)  
- **South Region**: 23% (Bangalore, Chennai, Hyderabad, etc.)
- **East Region**: 17% (Kolkata, Bhubaneswar, Guwahati, etc.)

## Customer Segmentation:
- **Enterprise**: 7% (B2B bulk purchasers)
- **Diamond/VIP**: 7% (Ultra-high-value customers)
- **Platinum**: 33% (High-value regular customers)
- **Gold**: 20% (Mid-high value customers)
- **Silver**: 20% (Regular customers)
- **Bronze**: 13% (Price-sensitive customers)

## Sample Analysis Opportunities:

### 1. Intent Scoring Model
- Combine recency, frequency, monetary (RFM) analysis
- Weight by engagement metrics and product affinity
- Factor in demographic and behavioral signals

### 2. Geographic Insights
- Regional preference analysis (Pro vs standard models)
- Price sensitivity by geography
- Campaign effectiveness by region

### 3. Customer Journey Analysis
- Multi-touchpoint attribution modeling
- Channel effectiveness measurement
- Conversion path optimization

### 4. Predictive Analytics
- Churn prediction for high-value customers  
- Next-best-product recommendations
- Lifetime value estimation

### 5. Marketing Optimization
- Budget allocation across channels
- Audience segmentation refinement
- Campaign timing and messaging

## Data Quality Notes:
- All timestamps in IST (Indian Standard Time)
- Currency values in Indian Rupees (₹)
- Customer IDs follow format: CUST0001-CUST0030
- Order IDs follow format: ORD001-ORD050
- Product SKUs follow Apple naming conventions

## Real-World Simulation Features:
- Realistic Indian pricing (GST included)
- Authentic product catalog (iPhone 13/14/15 series)
- Regional language preferences
- EMI payment options (common in India)
- B2B enterprise purchasing patterns
- Seasonal buying behavior
- Cross-selling and accessory purchases

## Usage Recommendations:
1. Upload to ADLS Gen2 in separate containers by data type
2. Use Delta Lake format for optimal performance
3. Implement data quality checks on load
4. Create star schema with CustomerOrders as fact table
5. Build incremental ETL pipelines for real-time updates

## Sample PySpark Code Structure:
```python
# Data ingestion from ADLS Gen2
customer_orders = spark.read.csv("abfss://data@storage.dfs.core.windows.net/CustomerOrders.csv", header=True)
region_mapping = spark.read.csv("abfss://data@storage.dfs.core.windows.net/RegionMapping.csv", header=True)

# High-intent customer identification
high_intent_customers = customer_orders.join(region_mapping, "CustomerID") \
    .filter(col("ProductType") == "iPhone") \
    .groupBy("CustomerID", "Region") \
    .agg(
        count("OrderID").alias("PurchaseFrequency"),
        max("OrderTimestamp").alias("LastPurchase"),
        avg("OrderValue").alias("AvgOrderValue")
    ) \
    .filter((col("PurchaseFrequency") >= 3) & 
            (datediff(current_date(), col("LastPurchase")) <= 30) &
            (col("AvgOrderValue") > 50000))
```

This dataset provides a comprehensive foundation for building sophisticated customer acquisition analytics and machine learning models for iPhone marketing at Amazon scale.