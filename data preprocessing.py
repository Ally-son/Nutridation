import pandas as pd
import pyarrow  


data = pd.read_parquet(r"C:\Users\65868\OneDrive - Nanyang Technological University\VS code\sc1304 proj\recipes.parquet")

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


# Since there are 311 unique categories, with some only having 1 recipie, we will remove categories that have less than 100 recipies
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


dataset_size = filtered_data.shape
print(f"\nSize of the dataset after filtering: {dataset_size[0]} rows and {dataset_size[1]} columns")


# Standardising the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
prep_data=scaler.fit_transform(filtered_data.iloc[:,7:16].to_numpy())



'''
1. not sure if we will have to encode our data (eg. one-hot encoding, label encoding, etc.)

2. have to use classification model to sort data into food categories (eg. vegetarian, vegan, etc.) 
    (if can sort into these, do you think we can also use a classification model to sort out the 'normal' recipies?)

3. consider if we need text data preprocessing (eg. tokenization, stemming, lemmatization, etc.)
    for the recipe instructions and ingredients

4. since standardising will change the values of the data, if we want to set nutritional ranges 
    (eg. low, medium, high), we have to set the range and categorise them, then scale the data
'''