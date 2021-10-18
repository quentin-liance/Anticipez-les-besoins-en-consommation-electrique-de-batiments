import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#  from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv', index_col=0)
df = df.drop('ENERGYSTARScore', axis=1)
df = df.dropna(axis=0).reset_index(drop=True)

cat_attribs = ['BuildingType', 'PrimaryPropertyType', 'PropertyName',
               'CouncilDistrictCode', 'Neighborhood',
               'ListOfAllPropertyUseTypes','LargestPropertyUseType']

num_attribs = ['YearBuilt', 'NumberofBuildings', 'NumberofFloors',
               'PropertyGFAParking', 'PropertyGFABuilding(s)',
               'LargestPropertyUseTypeGFA', 'Latitude', 'Longitude']

y_1, y_2 = 'TotalGHGEmissions', 'SiteEnergyUse(kBtu)'

train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

X_train = train_set[cat_attribs + num_attribs]
y_train_1 = train_set[y_1]  # TotalGHGEmissions
y_train_2 = train_set[y_2]  # SiteEnergyUse(kBtu)

loo_encoder = ce.LeaveOneOutEncoder(cols=cat_attribs)
loo_encoder.fit(X_train, y_train_1)
X_train = loo_encoder.transform(X_train)

# train set for linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled

def display_scores(scores):
    print("Scores:", scores)
    print("Moyenne:", scores.mean())
    print("Ecart-type:", scores.std())
    
### 1. Regression linéaire

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_1)    
y_pred_1 = lin_reg.predict(X_train_scaled)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)

# using cross-validation
scores = cross_val_score(lin_reg, X_train_scaled, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
lin_reg_rmse_scores = np.sqrt(-scores)
display_scores(lin_reg_rmse_scores)

### 2. Lasso

lasso = Lasso()
lasso.fit(X_train_scaled, y_train_1)
y_pred_1 = lasso.predict(X_train_scaled)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)

# using cross-validation
scores = cross_val_score(lasso, X_train_scaled, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
lasso_rmse_scores = np.sqrt(-scores)
display_scores(lasso_rmse_scores)

### 3. Ridge

ridge = Ridge()
ridge.fit(X_train_scaled, y_train_1)
y_pred_1 = ridge.predict(X_train_scaled)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)

scores = cross_val_score(ridge, X_train_scaled, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
ridge_rmse_scores = np.sqrt(-scores)
display_scores(ridge_rmse_scores)

### 4. SVR

svr = SVR()
svr.fit(X_train_scaled, y_train_1)
y_pred_1 = svr.predict(X_train_scaled)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)
# using cross-validation
scores = cross_val_score(svr, X_train_scaled, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
svr_rmse_scores = np.sqrt(-scores)
display_scores(svr_rmse_scores)

### 5. KNN

knn_reg = KNeighborsRegressor()
knn_reg.fit(X_train_scaled, y_train_1)
y_pred_1 = knn_reg.predict(X_train_scaled)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)
# using cross-validation
scores = cross_val_score(knn_reg, X_train_scaled, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
knn_rmse_scores = np.sqrt(-scores)
display_scores(knn_rmse_scores)

### 6. Random Forest

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train_1)
y_pred_1 = rf.predict(X_train)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)
# using cross-validation
scores = cross_val_score(rf, X_train, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
rf_rmse_scores = np.sqrt(-scores)
display_scores(rf_rmse_scores)

### 7. Gradient Boosting

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train_1)
y_pred_1 = gb.predict(X_train)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)
# using cross-validation
scores = cross_val_score(gb, X_train, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
gb_rmse_scores = np.sqrt(-scores)
display_scores(gb_rmse_scores)

### 8. Hist Gradient Boosting

hist_gb = HistGradientBoostingRegressor(random_state=42)
hist_gb.fit(X_train, y_train_1)
y_pred_1 = hist_gb.predict(X_train)
mse = mean_squared_error(y_train_1, y_pred_1)
rmse = np.sqrt(mse)
print('RMSE on train set :', rmse)
# using cross-validation
scores = cross_val_score(hist_gb, X_train, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
hist_gb_rmse_scores = np.sqrt(-scores)
display_scores(hist_gb_rmse_scores)

### Recherche par quadrillage des hyperparamètres du random forest.

param_grid = {
    'n_estimators': [5, 10, 50, 100, 500],
    'max_features': [2, 4, 6, 8],
    'bootstrap':[True, False]
}

forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, verbose=1)

grid_search.fit(X_train, y_train_1)

cvres = grid_search.cv_results_
cvres_sample = pd.DataFrame({
    'mean_test_score': np.sqrt(-cvres['mean_test_score']),
    'params': cvres['params']})
cvres_sample = cvres_sample.sort_values('mean_test_score',
                                        ascending=True).reset_index(drop=True)

feature_importances = grid_search.best_estimator_.feature_importances_
pd.DataFrame(
    sorted(zip(feature_importances,list(X_train.columns)), reverse=True)
    )

best_model = grid_search.best_estimator_
y_pred_1 = cross_val_predict(best_model, X_train, y_train_1, cv=5)
y_train_1 = y_train_1.values
plt.scatter(y_train_1, y_pred_1, alpha=0.5)
plt.plot(np.linspace(0, 1_400), np.linspace(0, 1_400))
plt.xlabel("valeurs réelles", fontsize=15)
plt.ylabel("valeurs prédites", fontsize=15)
plt.show()