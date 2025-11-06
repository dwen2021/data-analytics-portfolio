"""
====================================================
🚀 Insurance BI Data Upload Script (Azure SQL)
Author: Dongjing Wen
Project: BI_PROJECT
Purpose:
    Upload cleaned insurance dataset into Azure SQL
    following the dimensional schema (Dim_*, Fact_Policy)
====================================================
"""

# ========== 1️⃣ 导入库 ==========
import pandas as pd
import pyodbc
from math import ceil
import os

# ========== 2️⃣ 路径设置 ==========
CLEAN_PATH = "../1_Data/clean/Insurance_claims_data_cleaned.csv"

# ========== 3️⃣ Azure SQL 连接信息 ==========
server = "dongjing-sql-server.database.windows.net"
database = "insurance_db"
username = "sqladmin"
password = "YourStrongPassword123!"

conn_str = (
    "Driver={ODBC Driver 18 for SQL Server};"
    f"Server=tcp:{server},1433;"
    f"Database={database};"
    f"Uid={username};"
    f"Pwd={password};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)

conn = pyodbc.connect(conn_str)
cursor = conn.cursor()
cursor.fast_executemany = True
print("✅ Connected to Azure SQL Database")

# ========== 4️⃣ 读取清洗数据 ==========
df = pd.read_csv(CLEAN_PATH)
print(f"✅ Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ========== 5️⃣ 辅助函数 ==========
def batched(lst, size=1000):
    """Batch generator"""
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


# ==============================
# 6️⃣ 维度表：Customer
# ==============================
customers = df[['customer_id', 'customer_age']].drop_duplicates('customer_id').reset_index(drop=True)
customers['age_group'] = pd.cut(customers['customer_age'],
                                bins=[0,25,35,45,55,100],
                                labels=['<25','26-35','36-45','46-55','55+'])

params = [(int(r.customer_id), float(r.customer_age), r.age_group) for _, r in customers.iterrows()]

for chunk in batched(params, 1000):
    cursor.executemany("""
        INSERT INTO Dim_Customer (customer_id, customer_age, age_group)
        VALUES (?, ?, ?)
    """, chunk)
    conn.commit()
print(f"✅ Inserted {len(params)} rows into Dim_Customer")


# ==============================
# 7️⃣ 维度表：FuelType
# ==============================
fueltypes = df[['fuel_type']].drop_duplicates().reset_index(drop=True)
fueltypes['fuel_category'] = fueltypes['fuel_type'].apply(
    lambda x: 'EV' if str(x).lower() in ['electric', 'hybrid'] else 'Combustion'
)
params = [(r.fuel_type, r.fuel_category) for _, r in fueltypes.iterrows()]

for chunk in batched(params, 1000):
    cursor.executemany("""
        INSERT INTO Dim_FuelType (fuel_type_name, fuel_category)
        VALUES (?, ?)
    """, chunk)
    conn.commit()
print(f"✅ Inserted {len(params)} rows into Dim_FuelType")


# ==============================
# 8️⃣ 维度表：Time
# ==============================
dates = pd.date_range(start="2020-01-01", end="2025-12-31")
time_df = pd.DataFrame({
    'full_date': dates,
    'year': dates.year,
    'quarter': dates.quarter,
    'month': dates.month,
    'week': dates.isocalendar().week.astype(int),
    'day': dates.day
})
params = [(r.full_date, int(r.year), int(r.quarter), int(r.month), int(r.week), int(r.day))
          for _, r in time_df.iterrows()]

for chunk in batched(params, 2000):
    cursor.executemany("""
        INSERT INTO Dim_Time (full_date, year, quarter, month, week, day)
        VALUES (?, ?, ?, ?, ?, ?)
    """, chunk)
    conn.commit()
print(f"✅ Inserted {len(params)} rows into Dim_Time")


# ==============================
# 9️⃣ 维度表：Vehicle
# ==============================
vehicles = df[['vehicle_age','segment','model','fuel_type','engine_type',
               'max_torque','max_power','displacement','cylinder',
               'transmission_type','steering_type','turning_radius',
               'length','width','gross_weight','airbags','is_esc',
               'is_parking_camera','is_brake_assist','ncap_rating']].drop_duplicates().reset_index(drop=True)

# 读取 fuel_type 对应 ID
fuel_map = pd.read_sql("SELECT fuel_type_id, fuel_type_name FROM Dim_FuelType", conn)
fuel_dict = dict(zip(fuel_map['fuel_type_name'], fuel_map['fuel_type_id']))
vehicles['fuel_type_id'] = vehicles['fuel_type'].map(fuel_dict)
vehicles = vehicles.dropna(subset=['fuel_type_id'])

def b(x): return 1 if str(x).lower() == 'yes' else 0

params = [(
    float(r.vehicle_age), r.segment, r.model, int(r.fuel_type_id), r.engine_type, r.max_torque, r.max_power,
    int(r.displacement), int(r.cylinder), r.transmission_type, r.steering_type, float(r.turning_radius),
    int(r.length), int(r.width), int(r.gross_weight), int(r.airbags), b(r.is_esc),
    b(r.is_parking_camera), b(r.is_brake_assist), int(r.ncap_rating)
) for _, r in vehicles.iterrows()]

for chunk in batched(params, 1000):
    cursor.executemany("""
        INSERT INTO Dim_Vehicle (
            vehicle_age, segment, model, fuel_type_id, engine_type, max_torque, max_power,
            displacement, cylinder, transmission_type, steering_type, turning_radius,
            length, width, gross_weight, airbags, is_esc, is_parking_camera, is_brake_assist, ncap_rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, chunk)
    conn.commit()
print(f"✅ Inserted {len(params)} rows into Dim_Vehicle")


# ==============================
# 🔟 维度表：Region
# ==============================
regions = df[['region_code','region_density']].drop_duplicates().reset_index(drop=True)
regions['region_tier'] = pd.cut(regions['region_density'],
                                bins=[0,10000,20000,40000],
                                labels=['Low','Medium','High'])

params = [(r.region_code, int(r.region_density), r.region_tier) for _, r in regions.iterrows()]

for chunk in batched(params, 1000):
    cursor.executemany("""
        INSERT INTO Dim_Region (region_code, region_density, region_tier)
        VALUES (?, ?, ?)
    """, chunk)
    conn.commit()
print(f"✅ Inserted {len(params)} rows into Dim_Region")


# ==============================
# 1️⃣1️⃣ 事实表：Fact_Policy
# ==============================
region_map = pd.read_sql("SELECT region_id, region_code FROM Dim_Region;", conn)
vehicle_map = pd.read_sql("""
    SELECT MIN(vehicle_id) AS vehicle_id, model
    FROM Dim_Vehicle
    GROUP BY model
""", conn)

df_fact = (
    df.drop_duplicates(subset=['policy_id'])
      .merge(region_map, on='region_code', how='left')
      .merge(vehicle_map, on='model', how='left')
)

bad_rows = df_fact[df_fact['region_id'].isna() | df_fact['vehicle_id'].isna()]
print(f"⚠️ Unmatched foreign key rows: {len(bad_rows)}")
df_fact = df_fact.dropna(subset=['region_id', 'vehicle_id'])

params = [
    (r.policy_id, float(r.subscription_length), int(r.claim_status),
     int(r.customer_id), int(r.vehicle_id), int(r.region_id), r.policy_id)
    for _, r in df_fact.iterrows()
]

for chunk in batched(params, 1000):
    cursor.executemany("""
        INSERT INTO Fact_Policy (
            policy_id, subscription_length, claim_status,
            customer_id, vehicle_id, region_id
        )
        SELECT ?, ?, ?, ?, ?, ?
        WHERE NOT EXISTS (SELECT 1 FROM Fact_Policy WHERE policy_id = ?)
    """, chunk)
    conn.commit()

cursor.execute("ALTER TABLE Fact_Policy WITH CHECK CHECK CONSTRAINT ALL;")
conn.commit()
print(f"✅ Inserted {len(params)} rows into Fact_Policy")

# ==============================
# ✅ 结束连接
# ==============================
cursor.close()
conn.close()
print("🎯 All data successfully uploaded & Azure SQL connection closed cleanly!")
