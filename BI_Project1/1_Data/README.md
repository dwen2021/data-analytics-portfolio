# 🗂️ Data — Insurance BI Project  

**Author:** Dongjing (James) Wen  
**Tools:** Python (Pandas) · Azure SQL  

---

## 📘 Overview  
Contains all datasets used for the Insurance Policy Analytics project.  
Raw CSVs are sourced from Kaggle, cleaned using Python, and prepared for Azure SQL loading.  

---

## 📁 Structure  
1_Data/
├── raw/ # Original Kaggle data
├── clean/ # Cleaned output after ETL
└── region_mapping.csv # Lookup for region_code → state_name

## 🧹 ETL Summary  
- Handled missing values & outliers  
- Filled categorical nulls with mode  
- Removed duplicates by `policy_id`  
- Simulated `customer_id` for multi-policy modeling  

Cleaned file: `Insurance_claims_data_cleaned.csv`  
ETL script: `3_ETL_Pipeline/data_clean_pipeline.py`

---

© 2025 Dongjing Wen | UIUC  