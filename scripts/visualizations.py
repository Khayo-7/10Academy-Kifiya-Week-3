import matplotlib as plt
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(data, column: str):
    """Plots a histogram with KDE for a given column."""

    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()
    
def plot_counts(data, column: str):
    """Plots counts for a given column."""

    sns.countplot(data[column])
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_outliers(data, column: str):
    """Plots a boxplot for detecting outliers in a given column."""

    sns.boxplot(x=data[column])
    plt.title(f'Outliers in {column}')
    plt.show()

def correlation_matrix(data, columns=None):
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
    plt.show()

def bar_plot(data, group_col, value_col, title, ylabel, xlabel=""):
    """Plots a bar chart for averages of a specified grouping."""

    group_means = data.groupby(group_col)[value_col].mean()
    group_means.plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    
def plot_missing_values(data):
    """
    Visualize missing values with a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

def geographic_analysis(df, group_col, value_col):
    """
    Bar plot of a numeric column grouped by a geographical column.

    Args:
        df (pd.DataFrame): Input dataset.
        group_col (str): Column name for grouping (e.g., 'Province').
        value_col (str): Column to calculate averages (e.g., 'TotalPremium').
    """
    plt.figure(figsize=(10, 6))
    mean_values = df.groupby(group_col)[value_col].mean().sort_values()
    sns.barplot(x=mean_values.values, y=mean_values.index, palette="viridis")
    plt.title(f"Average {value_col} by {group_col}")
    plt.xlabel(value_col)
    plt.ylabel(group_col)
    plt.show()

def bubble_chart(df, x, y, size, hue, title):
    """
    Generate a bubble chart.

    Args:
        df (pd.DataFrame): Input dataset.
        x (str): X-axis variable.
        y (str): Y-axis variable.
        size (str): Column defining bubble size.
        hue (str): Color code bubbles by a categorical column.
        title (str): Chart title.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x=x, y=y, size=size, hue=hue, alpha=0.6, sizes=(50, 500))
    plt.title(title)
    plt.show()

