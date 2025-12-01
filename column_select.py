import pandas as pd
import numpy as np
cols_to_keep = [
    # demographics
    'age',
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

df = pd.read_csv('gpo-ai-data.csv')[cols_to_keep]
print("Dataset shape after column selection:", df.shape)

df.to_csv("gpo-ai-data.csv", index=False)
print(f"Saved cleaned dataset to gpo-ai-data.csv")