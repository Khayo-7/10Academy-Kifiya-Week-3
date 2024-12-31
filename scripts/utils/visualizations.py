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

def plot_boxplot(data, x_column, y_column=None, title=None):
    """Plots a boxplot for a given dataset or column."""
    if y_column:
        sns.boxplot(x=x_column, y=y_column, data=data)
        plt.title(title)
    else:
        sns.boxplot(data[x_column])
        plt.title(f'Outliers in {x_column}')
    plt.xticks(rotation=90)
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
    
def plot_missing_values(data):
    """
    Visualize missing values with a heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

def bar_plot(data, group_col, value_col, title, ylabel="", xlabel=""):
    """Plots a bar chart for averages of a specified grouping."""

    group_means = data.groupby(group_col)[value_col].mean()
    group_means.plot(kind="bar", color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel or value_col)
    plt.xlabel(xlabel or group_col)
    plt.xticks(rotation=45)
    plt.show()

def plot_bar(data=None, x_column=None, y_column=None, title="", ylabel="", xlabel="", orientation="v"):
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
    plt.show()

def bubble_chart(data, x, y, size, hue, title):
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
    plt.show()
