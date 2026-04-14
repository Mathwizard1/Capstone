import pandas as pd
import numpy as np

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("daily_food_delivery_orders.csv")

# -----------------------------
# 2. Keep only required columns
# -----------------------------
df = df[[
    "order_id",
    "order_date",
    "restaurant_type",
    "delivery_distance_km",
    "delivery_time_minutes"
]]

# -----------------------------
# 3. Convert date → timestamp
# -----------------------------
df["order_date"] = pd.to_datetime(df["order_date"])

# Add random minutes (simulate real arrival time)
df["order_time"] = df["order_date"] + pd.to_timedelta(
    np.random.randint(0, 1440, size=len(df)), unit="m"
)

# -----------------------------
# 4. Create customer nodes
# -----------------------------
df["customer_id"] = df["order_id"]

# -----------------------------
# 5. Create restaurant IDs
# -----------------------------
# Map restaurant_type → unique IDs
df["restaurant_id"] = df["restaurant_type"].astype("category").cat.codes

# -----------------------------
# 6. Generate driver nodes (synthetic)
# -----------------------------
num_drivers = 50  # change as needed

drivers = pd.DataFrame({
    "driver_id": range(num_drivers),
    "driver_rating": np.random.uniform(3.0, 5.0, num_drivers),
    "driver_location": np.random.uniform(0, 10, num_drivers)  # dummy location
})

# -----------------------------
# 7. Assign driver availability (optional)
# -----------------------------
df["assigned_driver_id"] = np.random.choice(drivers["driver_id"], size=len(df))

# -----------------------------
# 8. Create batching window (Δ)
# -----------------------------
# Example: 5-minute batching
df = df.sort_values("order_time")

delta_minutes = 5

df["batch_id"] = (
    (df["order_time"] - df["order_time"].min())
    .dt.total_seconds() // (delta_minutes * 60)
).astype(int)

# -----------------------------
# 9. Final dataset (tripartite-ready)
# -----------------------------
final_df = df[[
    "customer_id",
    "restaurant_id",
    "assigned_driver_id",
    "order_time",
    "batch_id",
    "delivery_distance_km",
    "delivery_time_minutes"
]]

# -----------------------------
# 10. Save dataset
# -----------------------------
final_df.to_csv("tripartite_batch_dataset.csv", index=False)

print("Dataset ready!")
print(final_df.head())