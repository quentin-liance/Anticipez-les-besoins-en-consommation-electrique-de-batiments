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
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
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

# Création d'un jeu d'entraînement et d'un jeu de test
train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

X_train = train_set[cat_attribs + num_attribs]
y_train_1 = train_set[y_1]  # TotalGHGEmissions
y_train_2 = train_set[y_2]  # SiteEnergyUse(kBtu)

loo_encoder = ce.LeaveOneOutEncoder(cols=cat_attribs)
loo_encoder.fit(X_train, y_train_1)
X_train = loo_encoder.transform(X_train)

# Standardisation pour les modèles linéaires
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
    
def get_rmse_on_train_set(model, training_set):
    model.fit(training_set, y_train_1)    
    y_pred_1 = model.predict(training_set)
    mse = mean_squared_error(y_train_1, y_pred_1)
    rmse = np.sqrt(mse)
    return rmse
    
def get_mean_rmse_on_validation_sets(model, training_set):
    scores = cross_val_score(model, training_set, y_train_1,
                         scoring="neg_mean_squared_error", cv=5)
    lin_reg_rmse_scores = np.sqrt(-scores)
    return lin_reg_rmse_scores.mean()
    
non_tree_based_models = [LinearRegression(), Lasso(), SVR(), 
                         KNeighborsRegressor()]

tree_based_models = [RandomForestRegressor(random_state=42), 
                     GradientBoostingRegressor(random_state=42),
                     HistGradientBoostingRegressor(random_state=42)]

models = non_tree_based_models + tree_based_models

rmse_train_set, rmse_validation_sets = [], []

for model in models:    
    if model in non_tree_based_models:
        rmse_train_set.append(get_rmse_on_train_set(model, X_train_scaled))
        rmse_validation_sets.append(
            get_mean_rmse_on_validation_sets(model, X_train_scaled))
    else: 
        rmse_train_set.append(get_rmse_on_train_set(model, X_train))
        rmse_validation_sets.append(
            get_mean_rmse_on_validation_sets(model, X_train))
        
results = pd.DataFrame(
    list(zip(models, rmse_train_set, rmse_validation_sets)),
    columns=['modèles', 'rmse train', 'rmse validation (mean)']
    ).sort_values('rmse validation (mean)', ascending=True)        

    
### Recherche par quadrillage des hyperparamètres du random forest.

param_grid = {
    'n_estimators': [50, 100, 150, 500],
    'max_features': [3, 4, 5, 6],
    'bootstrap':[False]
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

# n_estimators = 100, max_features = 4, bootstrap = False.

feature_importances = grid_search.best_estimator_.feature_importances_
feature_imp_df = pd.DataFrame(
    sorted(zip(feature_importances,list(X_train.columns)), reverse=True),
    columns=['MDI', 'variable']
    )

best_model = grid_search.best_estimator_
y_pred_1 = cross_val_predict(best_model, X_train, y_train_1, cv=5)
y_train_1 = y_train_1.values
plt.scatter(y_train_1, y_pred_1, alpha=0.5)
plt.plot(np.linspace(0, 1_400), np.linspace(0, 1_400), c='red')
plt.xlabel("valeurs réelles", fontsize=15)
plt.ylabel("valeurs prédites", fontsize=15)
plt.show()

# Sauvegarde du meilleur modèle
joblib.dump(best_model, 'best_model.pkl')