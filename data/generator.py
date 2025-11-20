import csv
import random

# Column names
columns = ["CustomerID", "NumberPurchases", "AvgPurchaseValue", "TotalSpend", "SpendingScore", "Age", "Income"]

# Generate random data
rows = []
for i in range(1, 101):  # Generate 100 rows
    rows.append({
        "CustomerID": i,
        "NumberPurchases": random.randint(1, 50),
        "AvgPurchaseValue": round(random.uniform(20, 500), 2),
        "TotalSpend": round(random.uniform(100, 10000), 2),
        "SpendingScore": random.randint(1, 100),
        "Age": random.randint(18, 80),
        "Income": random.randint(20000, 150000)
    })

# Write to CSV
with open("/Users/zayed/Desktop/learning/Customer-Segmenter/data/processed/customers.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("Random data generated successfully!")