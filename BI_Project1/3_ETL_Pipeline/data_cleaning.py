"""
====================================================
📊 Insurance Claim Data Cleaning Script
Author: Dongjing Wen
Project: BI_PROJECT
Purpose:
    Clean raw insurance claim data from Kaggle,
    handle missing values, outliers, and simulate customer_id.
Output:
    ../1_Data/clean/Insurance_claims_data_cleaned.csv
====================================================
"""

# ========== 1️⃣ Import Libraries ==========
import pandas as pd
import numpy as np
import os
import shutil

# ========== 2️⃣ Define Directory Paths ==========
RAW_DIR = "../1_Data/raw"
CLEAN_DIR = "../1_Data/clean"

# Create folders if they don’t exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# ========== 3️⃣ Load Raw Data ==========
original_path = os.path.join(RAW_DIR, "Insurance_claims_data.csv")

# If the file isn’t in the raw folder, copy from root if found
if not os.path.exists(original_path):
    if os.path.exists("Insurance claims data.csv"):
        shutil.copy("Insurance claims data.csv", original_path)
        print("📦 Copied original data into 1_Data/raw folder.")
    else:
        raise FileNotFoundError(
            "❌ Could not find Insurance_claims_data.csv. Please place it in 1_Data/raw or project root."
        )

# Read the CSV
df = pd.read_csv(original_path)
print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ========== 4️⃣ Identify Variable Types ==========
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"🔢 Numeric columns: {len(numeric_cols)}")
print(f"🔤 Categorical columns: {len(categorical_cols)}")

# ========== 5️⃣ Clean Numeric Columns ==========
for col in numeric_cols:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)
    
    # Handle outliers via IQR
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df.loc[(df[col] < lower) | (df[col] > upper), col] = mean_val

print("✅ Numeric columns cleaned (missing values + outliers handled).")

# ========== 6️⃣ Clean Categorical Columns ==========
for col in categorical_cols:
    mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
    df[col].fillna(mode_val, inplace=True)

print("✅ Categorical columns cleaned (missing values filled).")

# ========== 7️⃣ Remove Duplicates + Simulate Customer IDs ==========
if "policy_id" in df.columns:
    df = df.drop_duplicates(subset=["policy_id"]).reset_index(drop=True)

policy_count = len(df)
target_customer_count = int(policy_count / 1.5)  # ~1.5 policies per customer

np.random.seed(42)
df["customer_id"] = np.random.randint(1, target_customer_count + 1, size=policy_count)

unique_customers = df["customer_id"].nunique()
print(f"🧍‍♂️ Simulated {unique_customers} unique customers (~{policy_count/unique_customers:.2f} policies per customer)")

# ========== 8️⃣ Save Cleaned File ==========
clean_path = os.path.join(CLEAN_DIR, "Insurance_claims_data_cleaned.csv")
df.to_csv(clean_path, index=False)

print(f"💾 Cleaned data saved to: {clean_path}")
print("🎉 Data cleaning completed successfully!")
