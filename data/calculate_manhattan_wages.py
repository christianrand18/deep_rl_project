import pandas as pd
import json

# Manhattan zip codes in order (zone 0-11)
MANHATTAN_ZIP_CODES = ['10002', '10003', '10005', '10006', '10007', '10009', 
                       '10010', '10011', '10012', '10013', '10014', '10038']

# Income bracket definitions (lower, upper, midpoint)
INCOME_BRACKETS = [
    (0, 9999, 5000),           # Less than $10,000
    (10000, 14999, 12500),     # $10,000 to $14,999
    (15000, 24999, 20000),     # $15,000 to $24,999
    (25000, 34999, 30000),     # $25,000 to $34,999
    (35000, 49999, 42500),     # $35,000 to $49,999
    (50000, 74999, 62500),     # $50,000 to $74,999
    (75000, 99999, 87500),     # $75,000 to $99,999
    (100000, 149999, 125000),  # $100,000 to $149,999
    (150000, 199999, 175000),  # $150,000 to $199,999
    (200000, None, 200000)     # $200,000 or more
]

# Annual work hours (52 weeks × 40 hours)
ANNUAL_HOURS = 52 * 40

# Load S1901 data (income distribution percentages)
# Skip row 1 (description row) and use row 0 as header
s1901_df = pd.read_csv("Trip Data/ACSST5Y2013.S1901-Data.csv", skiprows=[1])

# Filter for Manhattan zip codes
manhattan_geo_ids = [f"8600000US{zip_code}" for zip_code in MANHATTAN_ZIP_CODES]
s1901_mh = s1901_df[s1901_df['GEO_ID'].isin(manhattan_geo_ids)].copy()

# Extract relevant columns
s1901_mh['zip_code'] = s1901_mh['GEO_ID'].str.replace('8600000US', '')
households_col = 'S1901_C01_001E'
income_pct_cols = [f'S1901_C01_{str(i).zfill(3)}E' for i in range(2, 12)]  # 002E through 011E

# Load S1902 data (population)
# Skip row 1 (description row) and use row 0 as header
s1902_df = pd.read_csv("Trip Data/ACSST5Y2013.S1902-Data.csv", skiprows=[1])
s1902_mh = s1902_df[s1902_df['GEO_ID'].isin(manhattan_geo_ids)].copy()
s1902_mh['zip_code'] = s1902_mh['GEO_ID'].str.replace('8600000US', '')

# Merge to get population data
merged_df = s1901_mh.merge(
    s1902_mh[['zip_code', 'S1902_C01_015E']], 
    on='zip_code', 
    how='left'
)

# Calculate household size
merged_df['population'] = pd.to_numeric(merged_df['S1902_C01_015E'], errors='coerce')
merged_df['households'] = pd.to_numeric(merged_df[households_col], errors='coerce')
merged_df['household_size'] = merged_df['population'] / merged_df['households']

# Sort by zip code order
merged_df['zone_index'] = merged_df['zip_code'].apply(lambda x: MANHATTAN_ZIP_CODES.index(x))
merged_df = merged_df.sort_values('zone_index').reset_index(drop=True)

# Initialize output dictionaries
wage_distribution = {}
average_salaries = {}

# Process each zone
for idx, row in merged_df.iterrows():
    zone_idx = row['zone_index']
    household_size = row['household_size']
    
    # Extract income percentages
    percentages = []
    for col in income_pct_cols:
        pct = pd.to_numeric(row[col], errors='coerce')
        percentages.append(pct if pd.notna(pct) else 0.0)
    
    # Calculate hourly wages per person
    hourly_wages = []
    for midpoint in [bracket[2] for bracket in INCOME_BRACKETS]:
        # Annual income -> hourly wage -> per person
        hourly_wage = (midpoint / ANNUAL_HOURS) / household_size
        hourly_wages.append(round(hourly_wage, 2))
    
    # Calculate weighted average hourly salary per person
    weighted_income = sum(
        INCOME_BRACKETS[i][2] * (percentages[i] / 100) 
        for i in range(10)
    )
    avg_hourly_salary_per_person = (weighted_income / ANNUAL_HOURS) / household_size
    
    # Store results
    wage_distribution[zone_idx] = {
        "hourly_wages": hourly_wages,
        "probabilities": percentages
    }
    average_salaries[zone_idx] = round(avg_hourly_salary_per_person, 2)

# Create final output structure
output_data = {
    "wage_distribution": wage_distribution,
    "average_salaries": average_salaries
}

# Save to JSON file with pretty printing
with open("manhattan_wage_data.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("✓ manhattan_wage_data.json created successfully")

# Calculate overall weighted average hourly wage
print("\n" + "="*70)
print("WEIGHTED AVERAGE HOURLY WAGE ANALYSIS")
print("="*70)
print(f"{'Zone':<6} {'Zip Code':<10} {'Households':<12} {'Avg Wage/hr':<12} {'Contribution':<15}")
print("-"*70)

total_weighted_wage = 0
total_households = 0

for idx, row in merged_df.iterrows():
    zone_idx = row['zone_index']
    zip_code = row['zip_code']
    households = row['households']
    avg_wage = average_salaries[zone_idx]
    weighted_contribution = households * avg_wage
    
    total_weighted_wage += weighted_contribution
    total_households += households
    
    print(f"{zone_idx:<6} {zip_code:<10} {households:>11,.0f} ${avg_wage:>10.2f} ${weighted_contribution:>14,.2f}")

overall_avg_wage = total_weighted_wage / total_households

print("-"*70)
print(f"{'Total':<17} {total_households:>11,.0f} {'':<12} ${total_weighted_wage:>14,.2f}")
print("="*70)
print(f"\nOverall average hourly wage (weighted by households): ${overall_avg_wage:.2f}/hour")
print("="*70)
