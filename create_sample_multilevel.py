import pandas as pd
from pathlib import Path

# Create sample data with multi-level headers similar to the image
# Level 1: Year, Unit value index (2 cols), Quantum index (2 cols), Terms of trade (3 cols)
# Level 2: blank, Exports, Imports, Exports, Imports, Gross, Net, Income

# Create multi-level column index
columns = pd.MultiIndex.from_tuples([
    ('Year', ''),
    ('Unit value index', 'Exports'),
    ('Unit value index', 'Imports'),
    ('Quantum index', 'Exports'),
    ('Quantum index', 'Imports'),
    ('Terms of trade', 'Gross'),
    ('Terms of trade', 'Net'),
    ('Terms of trade', 'Income')
])

# Sample data
data = [
    [2020, 105.2, 98.5, 110.3, 102.1, 106.8, 104.5, 112.3],
    [2021, 108.5, 101.2, 115.6, 105.8, 107.2, 106.1, 115.8],
    [2022, 112.3, 105.8, 120.5, 110.2, 106.1, 108.5, 118.2],
    [2023, 115.8, 108.2, 125.3, 114.5, 107.0, 110.2, 120.5],
    [2024, 118.2, 110.5, 128.9, 118.1, 106.9, 111.8, 122.8],
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to Excel
output_path = Path("tests/sample_data/trade_indices_multilevel.xlsx")
output_path.parent.mkdir(parents=True, exist_ok=True)

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df.to_excel(writer, index=False, sheet_name='Trade Indices')

print(f"Created multi-level header Excel file: {output_path}")
print("\nColumn structure:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
