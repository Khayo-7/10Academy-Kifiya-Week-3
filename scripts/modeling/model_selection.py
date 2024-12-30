from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_linear_regression(X_train, y_train):
    """
    Train a simple linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
    
    Returns:
        model: Trained Linear Regression model.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def train_random_forest(X_train, y_train, params=None):
    """
    Train a Random Forest model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
        params (dict): Hyperparameters for the RandomForestRegressor.
    
    Returns:
        model: Trained Random Forest model.
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    return model

def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost Regressor.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
        params (dict): Hyperparameters for the XGBRegressor.
    
    Returns:
        model: Trained XGBoost model.
    """
    if params is None:
        params = {
            'learning_rate': 0.1,
            'n_estimators': 100,
            'verbosity': 0
        }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    return model

def hyperparameter_tuning(model, param_grid, X_train, y_train):
    """
    Perform Grid Search CV for hyperparameter tuning.

    Args:
        model: Estimator to tune.
        param_grid (dict): Hyperparameter grid.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Target values.
    
    Returns:
        best_model: Best model after hyperparameter tuning.
        best_params: Optimal parameters.
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_