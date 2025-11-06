# 🧩 Database — Insurance BI Project  

**Author:** Dongjing (James) Wen  
**Tools:** Azure SQL · ADS  

---

## 📘 Overview  
Defines the Azure SQL database used to store cleaned insurance data.  
Implements a **star schema** for efficient BI analysis and visualization.

---

## 🧱 Tables  
- **Fact_Policy** — Core fact table (claims, subscription length, region, vehicle, customer)  
- **Dim_Customer** — Customer age and group  
- **Dim_Vehicle** — Vehicle specs and safety rating  
- **Dim_Region** — Region density and tier  
- **Dim_Time** — Date hierarchy for reports  
- **Dim_FuelType** — Fuel category reference  

---

## ⚙️ Setup  
Run `insurance_schema.sql` in **Azure Data Studio** to create all tables.  
Data upload handled via `3_ETL_Pipeline/azure_upload.py`.

---

© 2025 Dongjing Wen | UIUC  
