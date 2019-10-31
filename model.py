import pandas as pd
import pickle
import requests
import category_encoders as ce
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
import json# Importing the dataset

df1 = pd.read_csv('/Users/ericchiyembekeza/Desktop/Lambda School/BW-1/DS/')
df2 = pd.read_csv('/Users/ericchiyembekeza/Desktop/Lambda School/BW-1/DS/')
df = pd.concat([df1,df2], ignore_index=True)
features = ['basementsqft','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet','decktypeid','storytypeid', 'poolcnt','garagecarcnt', 'taxvaluedollarcnt']
df = df[['basementsqft','bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet','decktypeid','storytypeid', 'poolcnt','garagecarcnt','taxvaluedollarcnt']

for i in features:
    df[i] = df[i].fillna((df[i].mean()))

X = df
y = X['taxvaluedollarcnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

target = 'taxvaluedollarcnt'
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
y_val_log = np.log1p(y_val)

encoder = ce.OrdinalEncoder()
xtre = encoder.fit_transform(X_train)
xve = encoder.transform(X_val)

X_test = encoder.transform(X_test)


eval_set = [(xtre, y_train_log), (xve, y_val_log)]

model = XGBRegressor(n_estimators=50, n_jobs=-1)
model.fit(xtre, y_train_log, eval_set=eval_set, eval_metric='rmse', early_stopping_rounds=50)

pickle.dump(regressor, open('model.pkl','wb'))# Loading model to compare the results

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1.8]]))
