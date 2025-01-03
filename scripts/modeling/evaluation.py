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
    Calculate and retrieve feature importance for tree-based models/models that support the feature_importances_ attribute.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.

    Returns:
        list: Sorted feature importance [(feature, importance), ...].
    """    
    if hasattr(model, 'feature_importances_'):
        return sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    elif hasattr(model, 'coef_'):
        # For linear models
        importance = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
        return sorted(zip(feature_names, importance), key=lambda x: abs(x[1]), reverse=True)
    else:
        raise ValueError("Model does not support feature importance analysis.")

def get_shap_values(model, data, method='shap'):
    """
    Calculate SHAP values for a given model and data.

    Args:
        model: Trained model.
        data: Data used for SHAP explanation (DataFrame or numpy array).
        method (str): Method to use for SHAP explanation. Defaults to 'shap'.

    Returns:
        array: SHAP values for the given data.
    """
    if method == 'shap':
        explainer = shap.Explainer(model, data)
        shap_values = explainer.shap_values(data)
    elif method == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
    else:
        raise ValueError("Invalid method for SHAP explanation.")
    return shap_values, explainer

# SHAP visualization utilities
def global_feature_importance(shap_values, data, feature_names=None):
    """Generate SHAP global feature importance plot."""
    shap.summary_plot(shap_values, data, feature_names=feature_names)

def feature_dependence_plot(feature_name, shap_values, data):
    """Generate SHAP dependence plot for a specific feature."""
    shap.dependence_plot(feature_name, shap_values, data)

def local_feature_explanation(explainer, shap_values, data, instance_idx):
    """Generate SHAP local explanation for a specific instance."""
    shap.initjs()
    shap.force_plot(
        explainer.expected_value[0] if hasattr(explainer.expected_value, '__iter__') else explainer.expected_value,
        shap_values[0][instance_idx] if isinstance(shap_values, list) else shap_values[instance_idx],
        data.iloc[instance_idx]
    )

# SHAP integration
def feature_importance_shap(model, data, feature_name, feature_names=None, instance_idx=0):
    """
    Analyze feature impact using SHAP values for any model and provide both global and local interpretability.

    Args:
        model: Trained model.
        data: Data used for SHAP explanation (DataFrame or numpy array).
        feature_names: Names of the features (optional).
        instance_idx: Index of the instance to explain for local interpretability (force plot).

    Returns:
        None
    """
    shap_values, explainer = get_shap_values(model, data, method='shap')

    # Global importance
    global_feature_importance(shap_values, data, feature_names)

    # Feature dependence plot
    feature_dependence_plot(feature_name, shap_values, data)

    # Local explanation for an instance if instance_idx is provided
    if instance_idx is None or instance_idx >= len(data):
        raise ValueError(f"Invalid instance index. Must be between 0 and {len(data) - 1}.")
    
    # Local explanation
    local_feature_explanation(explainer, shap_values, data, instance_idx)

def feature_importance_shap_tree(model, data, feature_name, feature_names=None, instance_idx=0):
    """
    Analyze feature impact for tree-based models using SHAP values with TreeExplainer.

    Args:
        model: Trained tree-based model.
        data: Data used for SHAP explanation (DataFrame or numpy array).
        feature_names: Names of the features (optional).
        instance_idx: Index of the instance to explain for local interpretability (force plot).

    Returns:
        None
    """
    shap_values, explainer = get_shap_values(model, data, method='shap_tree')

    # Global importance
    global_feature_importance(shap_values, data, feature_names)

    # Feature dependence plot
    feature_dependence_plot(feature_name, shap_values, data)

    # Local explanation for an instance if instance_idx is provided
    if instance_idx is None or instance_idx >= len(data):
        raise ValueError(f"Invalid instance index. Must be between 0 and {len(data) - 1}.")

    # Local explanation
    local_feature_explanation(explainer, shap_values, data, instance_idx)

# LIME integration
def feature_importance_lime(model, data, feature_names, instance_idx=0, output_file=None):
    """
    Use LIME for feature importance analysis.

    Args:
        model: Trained model.
        data: Data used for SHAP explanation (DataFrame or numpy array).
        feature_names: Names of the features.
        instance_idx: Index of the instance to explain.
        output_file: Path to save the explanation as an HTML file.

    Returns:
        None
    """
    # Convert to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = data

    # Initialize the LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=data_array,
        feature_names=feature_names,
        class_names=['Target'],
        verbose=True,
        mode='regression' if not hasattr(model, 'predict_proba') else 'classification'
    )

    # Explain the prediction for a specific instance
    explanation = explainer.explain_instance(
        data_array[instance_idx],
        # data_row=data.iloc[instance_idx].values,
        predict_fn=model.predict,
        num_features=min(10, len(feature_names))  # Top 10 features contributing to the prediction
    )

    # Save the explanation
    explanation.show_in_notebook()
    # explanation.show_in_notebook(show_all=False)

    if output_file:
        explanation.save_to_file(output_file)
    else:
        explanation.save_to_file(f'lime_explanation_sample_{instance_idx}.html')


# Wrapper function to select explanation technique dynamically
def explain_model(model, data, feature_names=None, method='shap', instance_idx=None, output_file=None):
    """
    Wrapper to dynamically select the explanation method based on model and preferences.
    """
    if method == 'shap':
        feature_importance_shap(model, data, feature_names, instance_idx)
    elif method == 'shap_tree' and hasattr(model, 'tree_'):  # Check for tree-based models
        feature_importance_shap_tree(model, data, feature_names, instance_idx)
    elif method == 'lime':
        feature_importance_lime(model, data, feature_names, instance_idx=instance_idx or 0, output_file=output_file)
    else:
        raise ValueError("Unsupported method or model type.")