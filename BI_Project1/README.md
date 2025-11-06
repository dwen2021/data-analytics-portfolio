# 🧠 Insurance BI & Analytics Project  

### 👤 Author: Dongjing (James) Wen  
**University of Illinois Urbana-Champaign (UIUC)**  
**Program:** B.S. in Data Science · M.S. in Business Analytics (Gies College of Business)  
**Tools:** Python · Azure SQL · Power BI · Tableau  

---

## 🌟 Project Overview  

This end-to-end **Business Intelligence (BI)** project demonstrates a complete data pipeline — from raw data acquisition to interactive dashboard visualization.  
The goal is to analyze **insurance policy and claim behaviors** across U.S. regions, focusing on **risk patterns, customer demographics, and regional performance**.

---

## 🧭 Project Workflow  

Kaggle Dataset → Python ETL → Azure SQL → Power BI / Tableau Visualization


1️⃣ **Data Collection:**  
Downloaded the *Insurance Claims Dataset* from Kaggle.  

2️⃣ **Data Cleaning (Python ETL):**  
- Handled missing and outlier values  
- Encoded categorical variables  
- Created synthetic `customer_id` for multi-policy mapping  
- Generated cleaned CSV for database upload  

3️⃣ **Database Design (Azure SQL):**  
- Built **star schema** using 1 fact table + 5 dimension tables  
- Implemented data integrity via foreign keys  
- Uploaded cleaned data from Python using `pyodbc`  

4️⃣ **Visualization (Power BI & Tableau):**  
- Power BI → Interactive **U.S. Claim Rate Heatmap** with KPI cards  
- Tableau → Policy and Claim dashboards by safety rating, tier, and segment  

---

## 🧱 Database Schema  

**Star Schema:**
```bash
Fact_Policy
│
├── Dim_Customer
├── Dim_Vehicle
├── Dim_Region
├── Dim_Time
└── Dim_FuelType
```

**Fact_Policy Columns:**  
`policy_id`, `subscription_length`, `claim_status`, `customer_id`, `vehicle_id`, `region_id`, `start_time_id`, `end_time_id`

**Example Dimension Tables:**  
- `Dim_Customer`: `customer_age`, `age_group`  
- `Dim_Vehicle`: `segment`, `fuel_type`, `ncap_rating`  
- `Dim_Region`: `region_code`, `region_density`, `region_tier`  

SQL schema: [`2_Database/insurance_schema.sql`](2_Database/insurance_schema.sql)

---

## 🧹 ETL Pipeline  

Scripts in [`3_ETL_Pipeline`](3_ETL_Pipeline/) handle:
- Raw data cleaning → `data_clean_pipeline.py`  
- Azure SQL upload → `azure_upload.py`  

**Requirements:** 
```bash
pip install -r requirements.txt

Key Libraries:
pandas, numpy, pyodbc
```

## 📊 Visualization
### -Power BI Dashboard

Interactive analytics dashboard hosted on Power BI Cloud.

[Open in Power BI Online](https://app.powerbi.com/reportEmbed?reportId=72e30692-cd9e-4441-b376-e273209f4fc8&autoAuth=true&ctid=44467e6f-462c-4ea2-823f-7800de5434e3&actionBarEnabled=true&reportCopilotInEmbed=true)  


Highlights:

KPI Cards → Total Policies, Claim Rate, Avg Subscription Length

Heatmap → Claim Rate by U.S. State

Filters → Segment · Region Tier · Age Group


### -Tableau Dashboard

Visualizes relationships between vehicle safety rating, subscription length, and claim status.
Includes KPI summary and interactive region filtering.



## Folder Structure

```bash
Insurance_BI_Project/
│
├── 1_Data/                 # Raw & Clean data files
│   ├── raw/
│   ├── clean/
│   └── region_mapping.csv
│
├── 2_Database/             # Azure SQL schema
│   └── insurance_schema.sql
│
├── 3_ETL_Pipeline/         # Data processing & upload scripts
│   ├── data_clean_pipeline.py
│   ├── azure_upload.py
│   └── requirements.txt
│
├── 4_Visualization/        # Power BI & Tableau dashboards
│   ├── Heat_Map.pbix
│   └── Tableau_Dashboard.twb
│
└── 5_Documentation/        # READMEs, project notes, screenshots
```


## 📈 Insights & Findings

High-density regions show higher claim frequency and policy volume.

Older customer groups (46–55+) tend to maintain longer subscription lengths.

SUV and diesel segments exhibit moderate claim risks but high retention.

Region-tier segmentation effectively differentiates claim patterns by area.

## 🧰 Tech Stack Summary

```bash
Layer	             Tool	                           Purpose
Data Source	        Kaggle	                  Raw insurance claims dataset
ETL	           Python (Pandas, NumPy)	       Data cleaning, transformation
Database	       Azure SQL	              Star schema for BI analysis
Visualization	  Power BI · Tableau	  KPI dashboards interactive visuals
```


## 📜 Author Note

This project showcases practical BI workflow design and analytical storytelling,
combining data engineering, database modeling, and visualization for business insight generation.

© 2025 Dongjing Wen | University of Illinois Urbana-Champaign
For academic and professional portfolio use.