# import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import xgboost as xgb
from sklearn.tree import plot_tree

def plot_distribution(data, column: str, save_path=None):
    """Plots a histogram with KDE for a given column."""

    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_counts(data, column: str, save_path=None):
    """Plots counts for a given column."""

    sns.countplot(data[column])
    plt.title(f'Distribution of {column}')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_boxplot(data, x_column, y_column=None, title=None, save_path=None):
    """Plots a boxplot for a given dataset or column."""
    if y_column:
        sns.boxplot(x=x_column, y=y_column, data=data)
        plt.title(title)
    else:
        sns.boxplot(data[x_column])
        plt.title(f'Outliers in {x_column}')
    plt.xticks(rotation=90)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def correlation_matrix(data, columns=None, save_path=None):
    """
    Generates a correlation heatmap for specified columns.

    Args:
        data (pd.DataFrame): Input dataset.
        columns (list): List of columns for correlation analysis.
    """

    if columns:
        numeric_columns = data[columns]
    else:
        # Select only numeric columns
        numeric_columns = data.select_dtypes(include='number')

    plt.figure(figsize=(10, 8))
    corr = numeric_columns.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def plot_missing_values(data, save_path=None):
    """
    Visualize missing values with a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    if save_path:
        plt.savefig(save_path)
    plt.show()

def bar_plot(data, group_col, value_col, title, ylabel="", xlabel="", save_path=None):
    """Plots a bar chart for averages of a specified grouping."""

    group_means = data.groupby(group_col)[value_col].mean()
    group_means.plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel or value_col)
    plt.xlabel(xlabel or group_col)
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_bar(data=None, x_column=None, y_column=None, title="", ylabel="", xlabel="", orientation="v", save_path=None):
    """
    Bar plot of a numeric column grouped by a column.

    Args:
        data (pd.DataFrame): Input dataset.
        x_column (str): Column name for grouping.
        y_column (str): Column to calculate averages.
        orientation (str): Orientation of the bar plot. Can be 'vertical' or 'horizontal'.
    """
    plt.figure(figsize=(10, 6))
    
    sns.barplot(data=data, x=x_column, y=y_column, hue=x_column, palette="viridis", legend=False, orient=orientation)

    plt.title(title)
    plt.ylabel(ylabel or y_column)
    plt.xlabel(xlabel or x_column)
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def bubble_chart(data, x, y, size, hue, title, save_path=None):
    """
    Generate a bubble chart.

    Args:
        data (pd.DataFrame): Input dataset.
        x (str): X-axis variable.
        y (str): Y-axis variable.
        size (str): Column defining bubble size.
        hue (str): Color code bubbles by a categorical column.
        title (str): Chart title.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x=x, y=y, size=size, hue=hue, alpha=0.6, sizes=(50, 500))
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Visualize a single tree with limited depth
def visualize_rf_tree(forest_model, feature_names, tree_index=0, max_depth=4):
    """
    Visualize a single tree with a limited depth from the Random Forest model.

    Args:
    - forest_model: The trained Random Forest model.
    - feature_names: List of feature names.
    - tree_index: Index of the tree to visualize.
    - max_depth: Maximum depth of the tree to display.
    """
    tree = forest_model.estimators_[tree_index]
    plt.figure(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_names, max_depth=max_depth, filled=True, rounded=True, fontsize=10)
    plt.title(f"Tree {tree_index} from the Random Forest (Max Depth: {max_depth})")
    plt.show()

def visualize_xgb_tree(xgb_model, tree_index=0):
    """
    Visualize a specific tree from the XGBoost model.

    Args:
        xgb_model: The trained XGBoost model.
        tree_index (int): Index of the tree to visualize.
    """
    plt.figure(figsize=(20, 10))
    xgb.plot_tree(xgb_model, num_trees=tree_index)
    plt.show()

def visualize_missing_data(data):
    """
    Visualize missing data in a dataset.

    Args:
        data (pd.DataFrame): Input dataset.
    """

    msno.matrix(data, figsize=(10, 6), color=(0.25, 0.5, 0.9))  # Adjust size and color
    plt.title("Missing Data Visualization", fontsize=16)

    plt.xticks(
        range(data.shape[1]),
        data.columns,
        rotation=90, fontsize=10
    )
    plt.show()