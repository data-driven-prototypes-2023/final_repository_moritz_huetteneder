import pandas as pd

# Function to clean a dataframe
def clean_dataframe(df):
    # Step 1: Remove columns with more than 50% missing values
    missing_values_percentage = df.isnull().mean() * 100
    columns_to_drop = missing_values_percentage[missing_values_percentage > 50].index
    df = df.drop(columns=columns_to_drop)
    
    # Step 2: Remove duplicate rows
    df = df.drop_duplicates()
    
    # Step 3: Fill missing values
    numeric_cols = df.select_dtypes(include='number').columns
    non_numeric_cols = df.select_dtypes(exclude='number').columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[non_numeric_cols] = df[non_numeric_cols].fillna('Unknown')
    
    return df

# Load the CSV files
austria_df = pd.read_csv('Assignment 2/data/beverages_austria.csv', delimiter='\t', on_bad_lines='skip', low_memory=False)
germany_df = pd.read_csv('Assignment 2/data/beverages_germany.csv', delimiter='\t', on_bad_lines='skip', low_memory=False)
spain_df = pd.read_csv('Assignment 2/data/beverages_spain.csv', delimiter='\t', on_bad_lines='skip', low_memory=False)

# Clean each dataframe
austria_cleaned_df = clean_dataframe(austria_df)
germany_cleaned_df = clean_dataframe(germany_df)
spain_cleaned_df = clean_dataframe(spain_df)

# Overwrite the 'countries' column with the respective country name
austria_cleaned_df['countries'] = 'Austria'
germany_cleaned_df['countries'] = 'Germany'
spain_cleaned_df['countries'] = 'Spain'

# Get the intersection of columns in all dataframes to ensure they can be joined
common_columns = list(set(austria_cleaned_df.columns).intersection(set(germany_cleaned_df.columns)).intersection(set(spain_cleaned_df.columns)))

# Keep only the common columns
austria_cleaned_df = austria_cleaned_df[common_columns]
germany_cleaned_df = germany_cleaned_df[common_columns]
spain_cleaned_df = spain_cleaned_df[common_columns]

# Display the number of rows and columns in the cleaned dataframes
print(austria_cleaned_df.shape)
print(germany_cleaned_df.shape)
print(spain_cleaned_df.shape)

# Save the cleaned dataframes to new CSV files
austria_cleaned_df.to_csv('beverages_austria_cleaned.csv', index=False)
germany_cleaned_df.to_csv('beverages_germany_cleaned.csv', index=False)
spain_cleaned_df.to_csv('beverages_spain_cleaned.csv', index=False)

# Join the cleaned dataframes and keep all rows from each dataframe
beverages_df = pd.concat([austria_cleaned_df, germany_cleaned_df, spain_cleaned_df], ignore_index=True)
print(beverages_df.shape)
print(beverages_df.info())

# Check for any remaining missing values
missing_values = beverages_df.isnull().sum()

# Display the missing values and data types to confirm
print(missing_values)
print(beverages_df.dtypes)

# Dictionary to rename columns
new_column_names = {
    'fat_unit': 'Fat Unit',
    'categories': 'Categories',
    'off:food_groups_tags': 'Food Groups Tags',
    'nutrition_data_prepared_per': 'Nutrition Data Prepared Per',
    'countries': 'Countries',
    'categories_tags': 'Categories Tags',
    'sugars_value': 'Sugars (g)',
    'proteins_value': 'Proteins (g)',
    'proteins_unit': 'Proteins Unit',
    'off:nutriscore_grade': 'Nutriscore Grade',
    'off:nutriscore_score': 'Nutriscore Score',
    'off:ecoscore_data.adjustments.origins_of_ingredients.value': 'Ingredients Origin Score',
    'saturated-fat_value': 'Saturated Fat (g)',
    'off:ecoscore_score': 'Ecoscore',
    'salt_value': 'Salt (g)',
    'carbohydrates_unit': 'Carbohydrates Unit',
    'off:food_groups': 'Food Groups',
    'energy-kcal_value': 'Energy (kcal)',
    'brands': 'Brands',
    'data_sources': 'Data Sources',
    'nutrition_data_per': 'Nutrition Data Per',
    'off:ecoscore_data.agribalyse.code': 'Agribalyse Code',
    'off:ecoscore_data.adjustments.packaging.non_recyclable_and_non_biodegradable_materials': 'Non-Recyclable Packaging',
    'obsolete': 'Obsolete',
    'sugars_unit': 'Sugars Unit',
    'off:ecoscore_grade': 'Ecoscore Grade',
    'carbohydrates_value': 'Carbohydrates (g)',
    'energy_unit': 'Energy Unit',
    'off:ecoscore_data.adjustments.production_system.value': 'Production System Score',
    'off:nova_groups_tags': 'Nova Groups Tags',
    'countries_tags': 'Countries Tags',
    'energy-kcal_unit': 'Energy Unit (kcal)',
    'sodium_unit': 'Sodium Unit',
    'code': 'Product Code',
    'fat_value': 'Fat (g)',
    'saturated-fat_unit': 'Saturated Fat Unit',
    'energy_value': 'Energy Value',
    'lc': 'Language Code',
    'sodium_value': 'Sodium (mg)',
    'brands_tags': 'Brands Tags',
    'salt_unit': 'Salt Unit',
    'off:ecoscore_data.adjustments.packaging.value': 'Packaging Score'
}

# Rename the columns
beverages_df.rename(columns=new_column_names, inplace=True)

# Ensure consistency in data types
beverages_df['Nutriscore Grade'] = beverages_df['Nutriscore Grade'].astype('category')
beverages_df['Ecoscore Grade'] = beverages_df['Ecoscore Grade'].astype('category')

# Delete all rows where the 'Nutriscore Grade' is not 'a', 'b', 'c', 'd', or 'e'
beverages_df = beverages_df[beverages_df['Nutriscore Grade'].isin(['a', 'b', 'c', 'd', 'e'])]

# Display the number of rows and columns in the final dataframe
print(beverages_df.shape)

# Save the combined dataframe to a new CSV file
beverages_df.to_csv('beverages_combined.csv', index=False)

print("The final combined dataframe with renamed columns has been saved to 'beverages_combined.csv'.")
