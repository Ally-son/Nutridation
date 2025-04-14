import pandas as pd
import pyarrow
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Load the data
data = pd.read_parquet(r'C:\Users\65868\OneDrive - Nanyang Technological University\VS code\sc1304 proj\recipes.parquet')
data_copy = data.copy()

columns = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'RecipeCategory', 'RecipeIngredientParts',
           'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
           'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
           'ProteinContent', 'RecipeInstructions']

# Filter the dataset to include only the selected columns
data_copy = data_copy[columns]

# Count how many foods are in each category
category_counts = data_copy['RecipeCategory'].value_counts()

print("Category counts:\n", category_counts)

dataset_size = data_copy.shape
print(f"\nSize of the dataset: {dataset_size[0]} rows and {dataset_size[1]} columns")

# Filter out categories with less than 100 recipes
threshold = 100
low_freq_categories = category_counts[category_counts < threshold].index
data_copy = data_copy[~data_copy['RecipeCategory'].isin(low_freq_categories)]

# List of unwanted categories
unwanted_categories = ['Dessert','Frozen Desserts','Sauces', 'Condiments', 'Household Cleaner',
                      'Cheesecake', 'Bar Cookie', 'Jellies' ,'Candy', 'Bath/Beauty',
                      'Homeopathy/Remedies', 'Ice Cream', 'Chocolate Chip Cookies']

# Remove unwanted categories
filtered_data = data_copy[~data_copy['RecipeCategory'].isin(unwanted_categories)]

category_counts = filtered_data['RecipeCategory'].value_counts()
print("\nCategory counts after filtering:\n", category_counts)

# Deleting duplicates and rows with missing values
filtered_data = filtered_data.dropna()
filtered_data['RecipeInstructions'] = filtered_data['RecipeInstructions'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else str(x))
filtered_data['RecipeIngredientParts'] = filtered_data['RecipeIngredientParts'].apply(lambda x: ' '.join(eval(x)) if isinstance(x, str) else str(x))
filtered_data = filtered_data.drop_duplicates()

# Print some statistics before standardization
print("\nNutritional value ranges before processing:")
nutrition_cols = ['Calories', 'FatContent', 'SodiumContent', 'CarbohydrateContent', 
                 'FiberContent', 'SugarContent', 'ProteinContent']

for col in nutrition_cols:
    print(f"\n{col}:")
    print(f"Mean: {filtered_data[col].mean():.2f}")
    print(f"Median: {filtered_data[col].median():.2f}")
    print(f"Min: {filtered_data[col].min():.2f}")
    print(f"Max: {filtered_data[col].max():.2f}")

# Remove extreme outliers (values beyond 3 standard deviations)
for col in nutrition_cols:
    mean = filtered_data[col].mean()
    std = filtered_data[col].std()
    filtered_data = filtered_data[
        (filtered_data[col] >= mean - 3*std) & 
        (filtered_data[col] <= mean + 3*std)
    ]

# Create a copy of the original nutritional values
original_nutrition = filtered_data[nutrition_cols].copy()

# Standardize the nutritional values
scaler = StandardScaler()
filtered_data[nutrition_cols] = scaler.fit_transform(filtered_data[nutrition_cols])

# Add back the original values with a suffix
for col in nutrition_cols:
    filtered_data[f'Original_{col}'] = original_nutrition[col]

# Print some statistics after standardization
print("\nNutritional value ranges after processing:")
for col in nutrition_cols:
    print(f"\nOriginal_{col}:")
    print(f"Mean: {filtered_data[f'Original_{col}'].mean():.2f}")
    print(f"Median: {filtered_data[f'Original_{col}'].median():.2f}")
    print(f"Min: {filtered_data[f'Original_{col}'].min():.2f}")
    print(f"Max: {filtered_data[f'Original_{col}'].max():.2f}")

# Save the scaler for later use
joblib.dump(scaler, 'scaler.joblib')

# Save the DataFrame to Parquet
filtered_data.to_parquet(r"C:\Users\65868\OneDrive - Nanyang Technological University\VS code\sc1304 proj\cleaned_recipes.parquet",
                        engine='pyarrow', index=False)
