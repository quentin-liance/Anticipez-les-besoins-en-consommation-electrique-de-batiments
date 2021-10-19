# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:33:16 2021

@author: Quentin
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from category_encoders import CountEncoder, LeaveOneOutEncoder, TargetEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('dataset.csv', index_col=0)
df = df.drop('PropertyName', axis=1)

cat_attribs = ['BuildingType', 'PrimaryPropertyType','CouncilDistrictCode',
               'Neighborhood', 'ListOfAllPropertyUseTypes',
               'LargestPropertyUseType']

num_attribs = ['YearBuilt', 'NumberofBuildings', 'NumberofFloors',
               'PropertyGFAParking', 'PropertyGFABuilding(s)',
               'LargestPropertyUseTypeGFA', 'Latitude', 'Longitude']

y = 'TotalGHGEmissions'

# Création d'un jeu d'entraînement et d'un jeu de test
train, test = train_test_split(df, test_size=0.3, random_state=42)
X_train = train[cat_attribs + num_attribs]
y_train = train[y].values

#### 1. Baseline - Régression linéaire multiple.
#### RMSE moyen sur les 5 jeux de validation : 103,182

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder()),
    ])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
    ])

X_train_prepared = full_pipeline.fit_transform(X_train)

model = LinearRegression()

stats = cross_validate(model, X_train_prepared, y_train, groups=None,
                       scoring='neg_mean_squared_error', cv=5, n_jobs=2,
                       return_train_score=True)

stats = pd.DataFrame(stats)
stats['train_score'] = np.sqrt(-stats['train_score'])
stats['test_score'] = np.sqrt(-stats['test_score'])
stats = stats.describe().transpose()

#### 2. Encodage supervisé
#### RMSE moyen sur les 5 jeux de validation : 33,82 
### Enorme gain sur le RMSE en supprimant la variable PropertyName.

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('loo', LeaveOneOutEncoder()),
    ])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs),
    ])

X_train_prepared = full_pipeline.fit_transform(X_train, y_train)

model = RandomForestRegressor(random_state=42)

stats = cross_validate(model, X_train_prepared, y_train, groups=None,
                       scoring='neg_mean_squared_error', cv=5, n_jobs=2,
                       return_train_score=True)

stats = pd.DataFrame(stats)
stats['train_score'] = np.sqrt(-stats['train_score'])
stats['test_score'] = np.sqrt(-stats['test_score'])
stats = stats.describe().transpose()