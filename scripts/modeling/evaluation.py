import shap
import pandas as pd
from lime import lime_tabular
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a regression model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True test target values.

    Returns:
        dict: Evaluation metrics (MAE, MSE, R2).
    """
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "Median AE": median_absolute_error(y_test, y_pred),
        "Explained Variance": explained_variance_score(y_test, y_pred),
    }

def feature_importance(model, feature_names):
    """
    Retrieve feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.

    Returns:
        list: Sorted feature importance [(feature, importance), ...].
    """
    if hasattr(model, 'feature_importances_'):
        return sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    else:
        raise ValueError("Model does not support feature importance analysis.")

# SHAP integration
def feature_importance_shap(model, X_train, feature_names=None):
    """
    Analyze feature impact using SHAP values.

    Args:
        model: Trained model.
        X_train: Training features (DataFrame or numpy array).
        feature_names: Names of the features (optional).

    Returns:
        None
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Global importance
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)

    # Example: Local explanation for the first instance
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        X_train.iloc[0] if feature_names else None
    )

def feature_importance_shap_tree(model, X_train, feature_names=None):
    """
    Analyze feature impact for tree-based models using SHAP values.

    Args:
        model: Trained tree-based model.
        X_train: Training features (DataFrame or numpy array).
        feature_names: Names of the features (optional).

    Returns:
        None
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Global importance
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)

    # Example: Local explanation for the first instance
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        X_train.iloc[0] if feature_names else None
    )

# LIME integration
def feature_importance_lime(model, X_train, feature_names, sample_idx=0):
    """
    Use LIME for feature importance analysis.

    Args:
        model: Trained model.
        X_train: Training features (DataFrame or numpy array).
        feature_names: Names of the features.
        sample_idx: Index of the instance to explain.

    Returns:
        None
    """
    # Convert to numpy array if needed
    if isinstance(X_train, pd.DataFrame):
        X_train_array = X_train.values
    else:
        X_train_array = X_train

    # Initialize the LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_array,
        feature_names=feature_names,
        class_names=['Target'],
        verbose=True,
        mode='regression'
    )

    # Explain the prediction for a specific instance
    explanation = explainer.explain_instance(
        X_train_array[sample_idx],
        model.predict,
        num_features=10  # Top 10 features contributing to the prediction
    )

    # Save the explanation
    explanation.show_in_notebook(show_all=False)
    explanation.save_to_file('lime_explanation_sample_{}.html'.format(sample_idx))

