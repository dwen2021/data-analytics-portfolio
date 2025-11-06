-- ========================================
--   Database: InsuranceBI
--   Schema: dbo
--   Author: Dongjing Wen
--   Purpose: Final BI schema for Azure SQL
-- ========================================

-- Drop existing tables if re-running
DROP TABLE IF EXISTS Fact_Policy;
DROP TABLE IF EXISTS Dim_Customer;
DROP TABLE IF EXISTS Dim_Vehicle;
DROP TABLE IF EXISTS Dim_Region;
DROP TABLE IF EXISTS Dim_Time;
DROP TABLE IF EXISTS Dim_FuelType;

-- ==============================
-- 1️⃣ Customer Dimension
-- ==============================
CREATE TABLE Dim_Customer (
    customer_id INT IDENTITY(1,1) PRIMARY KEY,
    customer_age INT,
    age_group VARCHAR(20)
);

-- ==============================
-- 2️⃣ Region Dimension
-- ==============================
CREATE TABLE Dim_Region (
    region_id INT IDENTITY(1,1) PRIMARY KEY,
    region_code VARCHAR(10) UNIQUE,
    region_density INT,
    region_tier VARCHAR(10)
);

-- ==============================
-- 3️⃣ Vehicle Dimension
-- ==============================
CREATE TABLE Dim_Vehicle (
    vehicle_id INT IDENTITY(1,1) PRIMARY KEY,
    vehicle_age FLOAT,
    segment VARCHAR(10),
    model VARCHAR(10),
    fuel_type_id INT,           -- 外键改为 fuel_type_id
    engine_type VARCHAR(50),
    max_torque VARCHAR(50),
    max_power VARCHAR(50),
    displacement INT,
    cylinder INT,
    transmission_type VARCHAR(20),
    steering_type VARCHAR(20),
    turning_radius FLOAT,
    length INT,
    width INT,
    gross_weight INT,
    airbags INT,
    is_esc BIT,
    is_parking_camera BIT,
    is_brake_assist BIT,
    ncap_rating INT
);

-- ==============================
-- 4️⃣ Time Dimension
-- ==============================
CREATE TABLE Dim_Time (
    time_id INT IDENTITY(1,1) PRIMARY KEY,
    full_date DATE UNIQUE,
    year INT,
    quarter INT,
    month INT,
    week INT,
    day INT
);

-- ==============================
-- 5️⃣ Fuel Type Dimension
-- ==============================
CREATE TABLE Dim_FuelType (
    fuel_type_id INT IDENTITY(1,1) PRIMARY KEY,
    fuel_type_name VARCHAR(20),
    fuel_category VARCHAR(20)   -- e.g. 'Petrol', 'Diesel', 'Electric'
);

-- ==============================
-- 6️⃣ Fact Table
-- ==============================
CREATE TABLE Fact_Policy (
    policy_id VARCHAR(20) PRIMARY KEY,
    subscription_length FLOAT,
    claim_status INT,
    start_date DATE,
    end_date DATE,
    customer_id INT,
    vehicle_id INT,
    region_id INT,
    start_time_id INT,
    end_time_id INT,
    FOREIGN KEY (customer_id) REFERENCES Dim_Customer(customer_id),
    FOREIGN KEY (vehicle_id) REFERENCES Dim_Vehicle(vehicle_id),
    FOREIGN KEY (region_id) REFERENCES Dim_Region(region_id),
    FOREIGN KEY (start_time_id) REFERENCES Dim_Time(time_id),
    FOREIGN KEY (end_time_id) REFERENCES Dim_Time(time_id)
);

-- Add foreign key for Vehicle -> FuelType
ALTER TABLE Dim_Vehicle
ADD CONSTRAINT FK_Vehicle_FuelType FOREIGN KEY (fuel_type_id)
REFERENCES Dim_FuelType(fuel_type_id);
