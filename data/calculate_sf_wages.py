"""
Calculate hourly average wage per individual for San Francisco downtown zip codes
using census income distribution data (S1901) and population data (S1902).
"""

import pandas as pd
import json

# San Francisco downtown zip codes
SF_ZIP_CODES = [
    '94102',  # Civic Center / Tenderloin
    '94103',  # South of Market (SoMa)
    '94104',  # Financial District
    '94105',  # Rincon Hill / South Beach
    '94107',  # Potrero Hill / Mission Bay
    '94108',  # Chinatown / Nob Hill
    '94109',  # Polk Gulch / Nob Hill
    '94110',  # Mission District
    '94111',  # Financial District (northern part)
]

# Inflation adjustment: deflate 2011 values to 2008
# US inflation rates: 2008 (3.8%), 2009 (-0.4%), 2010 (1.6%), 2011 (3.2%)
# Cumulative: (1.038 × 0.996 × 1.016 × 1.032) ≈ 1.0876
# To convert 2011 → 2008: divide by 1.0876
DEFLATION_FACTOR = 1.0 / 1.0876  # ≈ 0.9194

# Income brackets and their midpoints (2011 values)
INCOME_BRACKETS_2011 = [
    (5000, 'S1901_C01_002E'),      # Less than $10,000
    (12500, 'S1901_C01_003E'),     # $10,000 to $14,999
    (20000, 'S1901_C01_004E'),     # $15,000 to $24,999
    (30000, 'S1901_C01_005E'),     # $25,000 to $34,999
    (42500, 'S1901_C01_006E'),     # $35,000 to $49,999
    (62500, 'S1901_C01_007E'),     # $50,000 to $74,999
    (87500, 'S1901_C01_008E'),     # $75,000 to $99,999
    (125000, 'S1901_C01_009E'),    # $100,000 to $149,999
    (175000, 'S1901_C01_010E'),    # $150,000 to $199,999
    (200000, 'S1901_C01_011E'),    # $200,000 or more
]

# Apply deflation to get 2008 equivalent values
INCOME_BRACKETS = [(midpoint * DEFLATION_FACTOR, column) for midpoint, column in INCOME_BRACKETS_2011]

# Work hours per year (40 hours/week * 52 weeks)
ANNUAL_HOURS = 2080

# Load census data (2011)
print("Loading 2011 census data...")
s1901_df = pd.read_csv("Trip Data/ACSST5Y2011.S1901-Data.csv", skiprows=[1])
s1902_df = pd.read_csv("Trip Data/ACSST5Y2011.S1902-Data.csv", skiprows=[1])

# Filter for San Francisco zip codes
sf_geo_ids = [f"8600000US{zip_code}" for zip_code in SF_ZIP_CODES]
s1901_sf = s1901_df[s1901_df['GEO_ID'].isin(sf_geo_ids)].copy()
s1902_sf = s1902_df[s1902_df['GEO_ID'].isin(sf_geo_ids)].copy()

# Merge the dataframes (2011 uses S1902_C01_015E for population)
merged_df = pd.merge(s1901_sf, s1902_sf[['GEO_ID', 'S1902_C01_015E', 'S1902_C01_001E']], 
                     on='GEO_ID', how='inner')

# Add zone index
merged_df['zone_index'] = merged_df['GEO_ID'].apply(
    lambda x: SF_ZIP_CODES.index(x.replace('8600000US', ''))
)
merged_df = merged_df.sort_values('zone_index')

# Calculate household size
merged_df['households'] = pd.to_numeric(merged_df['S1901_C01_001E'], errors='coerce')
merged_df['population'] = pd.to_numeric(merged_df['S1902_C01_015E'], errors='coerce')  # 2011 format
merged_df['household_size'] = merged_df['population'] / merged_df['households']

print(f"\nProcessing {len(merged_df)} San Francisco zip codes...")
print(f"Total households: {merged_df['households'].sum():,.0f}")
print(f"Total population: {merged_df['population'].sum():,.0f}")

# Initialize output dictionaries
wage_distribution = {}
average_salaries = {}

# Process each zip code
for idx, row in merged_df.iterrows():
    zone_idx = row['zone_index']
    household_size = row['household_size']
    households = row['households']
    
    # Extract income distribution probabilities and calculate wages
    hourly_wages = []
    probabilities = []
    weighted_income = 0
    
    for midpoint, column_name in INCOME_BRACKETS:
        # Get percentage for this income bracket
        percentage = pd.to_numeric(row[column_name], errors='coerce')
        if pd.isna(percentage):
            percentage = 0
        
        # Calculate hourly wage per person for this bracket
        hourly_wage = (midpoint / ANNUAL_HOURS) / household_size
        
        hourly_wages.append(round(hourly_wage, 2))
        probabilities.append(percentage)
        
        # Accumulate weighted income for average calculation
        weighted_income += midpoint * (percentage / 100)
    
    # Store wage distribution for this zone
    wage_distribution[zone_idx] = {
        "hourly_wages": hourly_wages,
        "probabilities": probabilities
    }
    
    # Calculate average hourly salary per person
    avg_hourly_salary_per_person = (weighted_income / ANNUAL_HOURS) / household_size
    average_salaries[zone_idx] = round(avg_hourly_salary_per_person, 2)

# Save to JSON file
output_data = {
    "wage_distribution": wage_distribution,
    "average_salaries": average_salaries
}

with open('sf_wage_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ sf_wage_data.json created successfully")

# Calculate overall weighted average hourly wage
print("\n" + "="*80)
print("WEIGHTED AVERAGE HOURLY WAGE ANALYSIS - SAN FRANCISCO")
print("(Using 2011 census data deflated to 2008 values)")
print("="*80)
print(f"{'Zone':<6} {'Zip Code':<10} {'Households':>12} {'Avg Wage/hr':>13} {'Contribution':>15}")
print("-"*80)

total_weighted_wage = 0
total_households = 0

for idx, row in merged_df.iterrows():
    zone_idx = row['zone_index']
    zip_code = SF_ZIP_CODES[zone_idx]
    households = row['households']
    avg_wage = average_salaries[zone_idx]
    
    weighted_contribution = households * avg_wage
    total_weighted_wage += weighted_contribution
    total_households += households
    
    print(f"{zone_idx:<6} {zip_code:<10} {households:>12,.0f} ${avg_wage:>12.2f} ${weighted_contribution:>14,.2f}")

print("-"*80)
print(f"{'Total':<17} {total_households:>12,.0f} {'':>14} ${total_weighted_wage:>14,.2f}")
print("="*80)

overall_avg_wage = total_weighted_wage / total_households
print(f"\nOverall average hourly wage (weighted by households): ${overall_avg_wage:.2f}/hour")
print("="*80)
