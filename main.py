#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from black_friday_eda.utils import load_train, load_test, check_df, concat_df_on_y_axis, grab_col_names, cat_summary, num_summary_enhanced, one_hot_encoder, RandomForestRegressor, plot_importance

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#Loading datasets
df_train = load_train()
df_test = load_test()

#Concatenating train-test sets to preprocess as a whole and to prevent data leak.

df_train_test = concat_df_on_y_axis(df_train, df_test)

# Dropping User_ID
df_train_test.drop("User_ID", axis=1, inplace=True)


df_train_test['Product_Category_2'] = df_train_test['Product_Category_2'].fillna(0)
df_train_test['Product_Category_3'] = df_train_test['Product_Category_3'].fillna(0)

# type fixing of product categories as int
df_train_test['Product_Category_2'] = df_train_test['Product_Category_2'].astype("int64")
df_train_test['Product_Category_3'] = df_train_test['Product_Category_3'].astype("int64")



#### Identifying Categorical and Numerical Features
cat_cols, num_cols, cat_but_car = grab_col_names(df_train_test)
print(f"Categorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")
print(f"Categorical but Cardinal Columns: {cat_but_car}")


# Product_ID frequency df
product_id_freq = df_train_test['Product_ID'].value_counts().reset_index()
product_id_freq.columns = ['Product_ID', 'Frequency']

product_purchase_sum = df_train_test.groupby('Product_ID')['Purchase'].sum().reset_index()
product_purchase_sum = product_purchase_sum.sort_values(by='Purchase', ascending=False)

# merging product_id_freq and product_purchase_sum
product_mean_purchase_freq = pd.merge(product_id_freq, product_purchase_sum, on='Product_ID')
product_mean_purchase_freq = product_mean_purchase_freq.sort_values(by='Purchase', ascending=False)

# removing 'P' from Product_ID
product_mean_purchase_freq['Product_ID'] = product_mean_purchase_freq['Product_ID'].str.replace('P', '', regex=False)

# extracting integers from Product_ID
product_mean_purchase_freq['Product_ID'] = product_mean_purchase_freq['Product_ID'].str.extract('(\d+)')


# Feature Engineering idea: Product_ID last 2 characters
#### Product_ID Last Two Digits Based Feature
product_mean_purchase_freq['Last_Two_Digits'] = product_mean_purchase_freq['Product_ID'].str[-2:].astype(int)
product_mean_purchase_freq['Last_Two_Digits'].value_counts()
last_two_freq = (
    product_mean_purchase_freq.groupby('Last_Two_Digits')
    .size()
    .reset_index(name='Frequency')
    .sort_values(by='Frequency', ascending=False)
)

last2_purchase_means = (
    product_mean_purchase_freq
    .groupby('Last_Two_Digits')['Purchase']
    .mean()
    .sort_values()
)
last2_purchase_means.sort_values(ascending=False).head(10)
product_id_last2_label_map = {
    '42': 6,
    '44': 5,
    '45': 4,
    '36': 3,
    '53': 2,
    '93': 1,
}

#### Product_ID Length Based Feature
# product_id_lenght label map feature engineering
product_mean_purchase_freq['Product_ID_Length'] = product_mean_purchase_freq['Product_ID'].astype(int).astype(str).str.len()

# product id length mean purchase
product_id_length_means = (
    product_mean_purchase_freq
    .groupby('Product_ID_Length')['Purchase']
    .mean()
    .sort_values()
)


product_id_length_label_map = {
    3: 4,
    4: 3,
    6: 2,
    5: 1,
}


#### Strategy
#- Product_ID is a cardinal feature. Model cant be based on product id no matter how related it is with purchase, But extracting impactfull features on purchase feature from Product_ID would do no harm.
#- Product categories has to play a crucial role in generalizing the data. Frequent and valuable (threshold < mean Purchase) category number combinations will be turned into features and fed to the model.

### Feature Engineering

df_train_test['NEW_Category_Combo'] = list(zip(
    df_train_test['Product_Category_1'],
    df_train_test['Product_Category_2'],
    df_train_test['Product_Category_3']
))

combo_mean_purchase = df_train_test.groupby('NEW_Category_Combo')['Purchase'].mean().reset_index()
combo_mean_purchase.columns = ['Category_Combo', 'Combo_Mean_Purchase']

# sample count per category combo
combo_counts = df_train_test['NEW_Category_Combo'].value_counts().reset_index()
combo_counts.columns = ['Category_Combo', 'Count']

# Merging mean purchase and counts
combo_stats = combo_mean_purchase.merge(combo_counts, on='Category_Combo')



# Implementing combo means into the main dataframe
df_train_test['NEW_Category_Combo_Mean_Purchase'] = df_train_test['NEW_Category_Combo'].map(
    combo_mean_purchase.set_index('Category_Combo')['Combo_Mean_Purchase']
)
selected_combos = combo_stats[
    (combo_stats['Count'] >= 399) & (combo_stats['Count'] <= 2439) &
    (combo_stats['Combo_Mean_Purchase'] >= 6399.331) & (combo_stats['Combo_Mean_Purchase'] <= 13430.779)
]['Category_Combo'].tolist()

#############
# Encoding
#############

#### One-Hot Encoding on Selected Combos
# Implementing the selected combos into the main dataframe
#df_train_test['NEW_Category_Combo'] = df_train_test['NEW_Category_Combo'].apply(
#    lambda x: x if x in selected_combos else 'Other'
#)



#### Ordinal Label Encoding All Category Combos
# Label encoding the NEW_Category_Combo based on their NEW_Category_Combo_Mean_Purchase

# Ranking respect to Combo_Mean_Purchase
combo_mean_sorted = (
    combo_mean_purchase
    .sort_values('Combo_Mean_Purchase')
    .reset_index(drop=True)
)

# Label for each category combo
combo_mean_sorted['Category_Combo_Label'] = range(len(combo_mean_sorted))

# Şimdi df_train_test'e bu label'ları eşliyoruz
label_mapping = combo_mean_sorted.set_index('Category_Combo')['Category_Combo_Label']

# Mapping 
df_train_test['NEW_Category_Combo_Label'] = df_train_test['NEW_Category_Combo'].map(label_mapping)

#### Adding Product Category Means as Feature
# Examining product category mean purchases

# Product_Category_1 için ortalama
category1_mean = df_train_test.groupby('Product_Category_1')['Purchase'].mean().reset_index()
category1_mean.columns = ['Product_Category_1', 'NEW_Product_Category_1_Mean_Purchase']

# Product_Category_2 için ortalama
category2_mean = df_train_test.groupby('Product_Category_2')['Purchase'].mean().reset_index()
category2_mean.columns = ['Product_Category_2', 'NEW_Product_Category_2_Mean_Purchase']

# Product_Category_3 için ortalama
category3_mean = df_train_test.groupby('Product_Category_3')['Purchase'].mean().reset_index()
category3_mean.columns = ['Product_Category_3', 'NEW_Product_Category_3_Mean_Purchase']

df_train_test = df_train_test.merge(category1_mean, on='Product_Category_1', how='left')
df_train_test = df_train_test.merge(category2_mean, on='Product_Category_2', how='left')
df_train_test = df_train_test.merge(category3_mean, on='Product_Category_3', how='left')

#### Feature Extraction from Cardinal Variable: Product_ID
# removing 'P' from Product_ID
df_train_test['Product_ID'] = df_train_test['Product_ID'].str.replace('P', '', regex=False)

# extracting integers from Product_ID
df_train_test['Product_ID'] = df_train_test['Product_ID'].str.extract('(\d+)').astype(int)
df_train_test['NEW_Product_ID_Length'] = df_train_test['Product_ID'].astype(str).str.len().astype(int).map(
    product_id_length_label_map
)

df_train_test['NEW_Product_ID_Last_Two_Digits'] = df_train_test['Product_ID'].astype(str).str[-2:].map(
    product_id_last2_label_map
)

cat_cols, num_cols, cat_but_car = grab_col_names(df_train_test)
print(f"Categorical Columns: {cat_cols}")
print(f"Numerical Columns: {num_cols}")
print(f"Categorical but Cardinal Columns: {cat_but_car}")

columns_to_drop = ['NEW_Category_Combo', 'Product_ID', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
df_train_test.drop(columns=columns_to_drop, inplace=True)

columns_to_encode = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Occupation', 'Marital_Status']
df_train_test = one_hot_encoder(
    df_train_test,
    categorical_cols=columns_to_encode,
    drop_first=True
)


# Labeling bool columns
bool_cols = df_train_test.select_dtypes(include=['bool']).columns.tolist()
for col in bool_cols:
    df_train_test[col] = df_train_test[col].astype(int)


### Feature Importance Examination

# Splitting the data back into train and test sets
df_train_preprocessed = df_train_test[df_train_test['Purchase'].notna()]
df_test_preprocessed = df_train_test[df_train_test['Purchase'].isna()]
df_test_preprocessed.drop(columns=['Purchase'], inplace=True)

# Training a simple tree-based model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = df_train_preprocessed.drop(['Purchase'], axis=1)
y = df_train_preprocessed['Purchase']
X = pd.DataFrame(scaler.fit_transform(X), columns=df_train_test.columns.drop('Purchase'))
X_test = pd.DataFrame(scaler.transform(df_test_preprocessed), columns=df_test_preprocessed.columns)

rfmodel = RandomForestRegressor(random_state=42, n_jobs=-1)
rfmodel.fit(X, y)

plot_importance(rfmodel, X, num=20)

## Model Training
