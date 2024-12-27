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

def correlation_matrix(data):
    """Displays a heatmap of the correlation matrix."""
    
    # Select only numeric columns
    numeric_data = data.select_dtypes(include='number')
    
    corr = numeric_data.corr()
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
    