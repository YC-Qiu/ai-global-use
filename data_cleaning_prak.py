import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/prakeerthprasad/Desktop/NU/MSAI 339 Data Science/Project/Project Github/ai-global-use/gpo-ai-data.csv')

print("Original dataset shape:", df.shape)
print("\n" + "="*80)

# ============================================================================
# 1. SUBJECTIVE_INCOME - Clean and encode
# ============================================================================
print("\n1. CLEANING: subjective_income")
print("-" * 80)
print("Original values:")
print(df['subjective_income'].value_counts())
print(f"\nMissing values: {df['subjective_income'].isna().sum()}")

# Map subjective income to ordinal scale (1-4)
income_mapping = {
    'Living comfortably on present income': 4,
    'Coping on present income': 3,
    'Finding it difficult on present income': 2,
    'Finding it very difficult on present income': 1
}

df['subjective_income'] = df['subjective_income'].map(income_mapping)

print("\nAfter cleaning:")
print(df['subjective_income'].value_counts().sort_index())
print(f"Missing values: {df['subjective_income'].isna().sum()}")

# ============================================================================
# 2. GPT_USE - Clean and encode
# ============================================================================
print("\n2. CLEANING: GPT_use")
print("-" * 80)
print("Original values:")
print(df['GPT_use'].value_counts())
print(f"\nMissing values: {df['GPT_use'].isna().sum()}")

# Map Yes/No to binary (1/0), treat "Unsure" as missing
gpt_use_mapping = {
    'Yes': 1,
    'No': 0
}

df['GPT_use'] = df['GPT_use'].map(gpt_use_mapping)

print("\nAfter cleaning:")
print(df['GPT_use'].value_counts().sort_index())
print(f"Missing values (including 'Unsure'): {df['GPT_use'].isna().sum()}")

# ============================================================================
# 3. REPLACE_FUTURE - Clean and encode
# ============================================================================
print("\n3. CLEANING: replace_future")
print("-" * 80)
print("Original values:")
print(df['replace_future'].value_counts())
print(f"\nMissing values: {df['replace_future'].isna().sum()}")

# Map job replacement concern to ordinal scale (0-3)
replace_mapping = {
    'Definitely not': 0,
    'Probably not': 1,
    'Probably yes': 2,
    'Definitely yes': 3
}

df['replace_future'] = df['replace_future'].map(replace_mapping)

print("\nAfter cleaning:")
print(df['replace_future'].value_counts().sort_index())
print(f"Missing values: {df['replace_future'].isna().sum()}")

# ============================================================================
# 4. CLOTHES_LIKELY - Clean and encode to 0-1 scale
# ============================================================================
print("\n4. CLEANING: Clothes_likely")
print("-" * 80)
print("Original values:")
print(df['Clothes_likely'].value_counts())
print(f"\nMissing values: {df['Clothes_likely'].isna().sum()}")

# Map likelihood to 0-1 scale
likelihood_mapping = {
    'Extremely unlikely': 0.0,
    'Somewhat unlikely': 0.25,
    'Neither likely nor unlikely': 0.5,
    'Somewhat likely': 0.75,
    'Extremely likely': 1.0
}

df['Clothes_likely'] = df['Clothes_likely'].map(likelihood_mapping)

print("\nAfter cleaning:")
print(df['Clothes_likely'].value_counts().sort_index())
print(f"Missing values: {df['Clothes_likely'].isna().sum()}")

# ============================================================================
# 5. GPT_USE_FUTURE - Clean and encode to 0-1 scale
# ============================================================================
print("\n5. CLEANING: GPT_use_future")
print("-" * 80)
print("Original values:")
print(df['GPT_use_future'].value_counts())
print(f"\nMissing values: {df['GPT_use_future'].isna().sum()}")

# Map likelihood to 0-1 scale based on actual values in data
# "Don't know" will be treated as missing (NaN)
future_use_mapping = {
    'Not likely at all': 0.0,
    'Not very likely': 0.25,
    'Somewhat likely': 0.75,
    'Very likely': 1.0
}

df['GPT_use_future'] = df['GPT_use_future'].map(future_use_mapping)

print("\nAfter cleaning:")
print(df['GPT_use_future'].value_counts().sort_index())
print(f"Missing values (including 'Don't know'): {df['GPT_use_future'].isna().sum()}")

# ============================================================================
# SAVE CLEANED DATASET
# ============================================================================
print("\n" + "="*80)
print("FINAL DATASET SUMMARY")
print("="*80)
print(f"Final dataset shape: {df.shape}")
print(f"\nTotal missing values in cleaned columns:")
print(f"  subjective_income: {df['subjective_income'].isna().sum()}")
print(f"  GPT_use: {df['GPT_use'].isna().sum()}")
print(f"  replace_future: {df['replace_future'].isna().sum()}")
print(f"  Clothes_likely: {df['Clothes_likely'].isna().sum()}")
print(f"  GPT_use_future: {df['GPT_use_future'].isna().sum()}")

# Save the cleaned dataset
output_path = 'df_model_input_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ Cleaned dataset saved to: {output_path}")

# Display sample of cleaned columns
print("\n" + "="*80)
print("SAMPLE OF CLEANED COLUMNS (first 10 rows):")
print("="*80)
cleaned_cols = ['subjective_income', 'GPT_use', 'replace_future', 'Clothes_likely', 'GPT_use_future']
print(df[cleaned_cols].head(10))

# ============================================================================
# ENCODING SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ENCODING REFERENCE")
print("="*80)
print("\n1. subjective_income (ordinal 1-4):")
print("   1 = Finding it very difficult on present income")
print("   2 = Finding it difficult on present income")
print("   3 = Coping on present income")
print("   4 = Living comfortably on present income")

print("\n2. GPT_use (binary):")
print("   0 = No")
print("   1 = Yes")
print("   NaN = Unsure")

print("\n3. replace_future (ordinal 0-3):")
print("   0 = Definitely not")
print("   1 = Probably not")
print("   2 = Probably yes")
print("   3 = Definitely yes")

print("\n4. Clothes_likely (continuous 0-1):")
print("   0.00 = Extremely unlikely")
print("   0.25 = Somewhat unlikely")
print("   0.50 = Neither likely nor unlikely")
print("   0.75 = Somewhat likely")
print("   1.00 = Extremely likely")

print("\n5. GPT_use_future (continuous 0-1):")
print("   0.00 = Not likely at all")
print("   0.25 = Not very likely")
print("   0.75 = Somewhat likely")
print("   1.00 = Very likely")
print("   NaN = Don't know")