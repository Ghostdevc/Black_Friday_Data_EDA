# Black Friday Data EDA and Feature Engineering
### dataset link: https://www.kaggle.com/datasets/sdolezel/black-friday?resource=download
### Problem Statement: A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
### Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

## Cleaning and Preprocessing

### Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv('black_friday_eda/data/train.csv')
df_test = pd.read_csv('black_friday_eda/data/test.csv')

df_train.head()