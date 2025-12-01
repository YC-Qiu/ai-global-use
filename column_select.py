import pandas as pd
from pandas import read_csv
cols_to_keep = [
    # demographics
    'respondent_yob',
    'respondent_gender',
    'respondent_education',
    'subjective_income',
    'respondent_employment',
    'respondent_industry',
    'urban',
    # AI knowledge & exposure
    'understand',
    'algorithm',
    'GPT_know',
    'GPT_use',
    'risk_daily',
    # attitudes toward AI
    'view_AI',
    'split_why',
    'replace_future',
    'trust_AI_1',
    'trust_AI_2',
    'trust_AI_3',
    'trust_AI_4',
    'trust_AI_5',
    'trust_AI_6',
    'trust_AI_7',
    'trust_AI_8',
    'concern',
    # country
    'respondent_country',
    # target variable
    'Clothes_likely',
    'Travel_use',
    'Dating_use',
    'Grocery_use',
    'GPT_use_future'
]

df = read_csv('gpo-ai-data.csv')[cols_to_keep]

df.to_csv("gpo-ai-data.csv", index=False)
print(f"Saved cleaned dataset to 'gpo-ai-data.csv' with shape: {df.shape}")