#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


train = pd.read_csv("desktop/data/train.csv")
test  = pd.read_csv("desktop/data/test.csv")

centers = pd.read_csv("desktop/data/fulfilment_center_info.csv")
meals   = pd.read_csv("desktop/data/meal_info.csv")


# In[7]:


train = train.merge(centers, on="center_id", how="left")
train = train.merge(meals, on="meal_id", how="left")

test = test.merge(centers, on="center_id", how="left")
test = test.merge(meals, on="meal_id", how="left")


# In[8]:


cm_mean = (
    train.groupby(['center_id', 'meal_id'])['num_orders']
    .mean()
    .reset_index()
    .rename(columns={'num_orders': 'cm_mean'})
)

train = train.merge(cm_mean, on=['center_id','meal_id'], how='left')
test  = test.merge(cm_mean, on=['center_id','meal_id'], how='left')


# In[9]:


full = pd.concat([train, test], sort=False)
full = full.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)

full['lag_1'] = (
    full.groupby(['center_id','meal_id'])['num_orders']
    .shift(1)
)


# In[10]:


train = full[full['week'] <= 145].copy()
test  = full[full['week'] >= 146].copy()


# In[11]:


train['lag_1'] = train['lag_1'].fillna(train['cm_mean'])
test['lag_1']  = test['lag_1'].fillna(test['cm_mean'])


# In[12]:


for df in [train, test]:
    df['discount'] = df['base_price'] - df['checkout_price']
    df['price_ratio'] = df['checkout_price'] / df['base_price']


# In[13]:


for df in [train, test]:
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)


# In[14]:


from sklearn.preprocessing import LabelEncoder

cat_cols = ['center_type','category','cuisine','city_code','region_code']

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])


# In[15]:


features = [
    'cm_mean',
    'lag_1',
    'checkout_price',
    'discount',
    'price_ratio',
    'emailer_for_promotion',
    'homepage_featured',
    'op_area',
    'center_type',
    'city_code',
    'region_code',
    'category',
    'cuisine',
    'week_sin',
    'week_cos'
]


# In[16]:


import xgboost as xgb
from sklearn.model_selection import train_test_split


# In[19]:


X = train[features]
y = np.log1p(train['num_orders'])


# In[20]:


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[21]:


model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

model.fit(X_train, y_train)


# In[22]:


from sklearn.metrics import mean_squared_log_error

val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_pred = np.maximum(0, val_pred)

rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), val_pred))
rmsle


# In[23]:


test_pred_log = model.predict(test[features])
test_pred_log = np.clip(test_pred_log, 0, 10)

test['num_orders'] = np.expm1(test_pred_log)
test['num_orders'] = np.maximum(0, test['num_orders'])


# In[24]:


submission = test[['id','num_orders']]
submission.to_csv("submission.csv", index=False)


# In[ ]:




