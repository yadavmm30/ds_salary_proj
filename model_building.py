import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('eda_data.csv')
print(df.head(10))

# choose relevant columns
print(df.columns)
df_model = df[
    ['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'num_comp', 'hourly',
     'employer_provided',
     'job_state', 'same_state', 'age', 'python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'job_simp', 'seniority',
     'desc_len']]

# get dummy data
df_dum = pd.get_dummies(df_model)

# train test split
target = 'avg_salary'
X = df_dum.drop('avg_salary', axis=1)
y = df.avg_salary.values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# multiple Linear reg
import statsmodels.api as sm

X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
print(model.fit().summary())

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

print('-----------------------------------------------------------------------------------')
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_pred = lr.predict(x_test)
print(np.mean(cross_val_score(lr, x_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
print('-----------------------------------------------------------------------------------')
# lasso Regression

alp = []
error = []
for i in range(1, 101):
    alp.append(i / 100)
    lml = Lasso(alpha=i / 100)
    error.append(np.mean(cross_val_score(lml, x_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# plt.plot(alp, error)
# plt.show()

err = tuple(zip(alp, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
print(df_err[df_err.error == max(df_err.error)])  # 0.13

lm_l = Lasso(alpha=0.13)
lm_l.fit(x_train, y_train)
lm_y_pred = lm_l.predict(x_test)
print(np.mean(cross_val_score(lm_l, x_train, y_train, scoring='neg_mean_absolute_error', cv=3)))

# random forest
print('-----------------------------------------------------------------------------------')
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
print('cross val val for Random forest : ',np.mean(cross_val_score(rf,x_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3)))

# Tune rf model using grid search CV
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('squared_error', 'absolute_error'), 'max_features': (1.0, 'sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3, error_score='raise')
gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_estimator_)
rf_y_pred = gs.best_estimator_.predict(x_test)

# test ensembles
print('-----------------------------------------------------------------------------------')
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, lr_y_pred))
print(mean_absolute_error(y_test, lm_y_pred))
print(mean_absolute_error(y_test, rf_y_pred))
print(mean_absolute_error(y_test, (lm_y_pred + rf_y_pred) / 2))

print('-----------------------------------------------------------------------------------')

# saving the best model

import pickle
pkl = {'model': gs.best_estimator_}
pickle.dump(pkl, open('model_file' + ".p", "wb"))

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(x_test.iloc[1, :])).reshape(1, -1))[0]

list(x_test.iloc[1, :])




print('----------')
