import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, mannwhitneyu

def perform_t_test(group_a, group_b, metric):
    """
    Perform Welch's two-sample t-test between two numerical data groups.
    """
    stat, p_value = ttest_ind(group_a[metric], group_b[metric], equal_var=False, nan_policy="omit")
    return {"stat": stat, "p_value": p_value, "significant": p_value < 0.05}
 
def perform_chi2_test(contingency_table):
    """
    Perform Chi-Squared test for independence on categorical data/variables.
    """
    stat, p_value, _, _ = chi2_contingency(contingency_table)
    return {"stat": stat, "p_value": p_value, "significant": p_value < 0.05}

def perform_anova_test(*groups):
    """
    Perform a one-way ANOVA test across multiple numerical groups.
    """
    stat, p_value = f_oneway(*groups)
    # stat, p_value = f_oneway(*[group for group in groups])
    return {"stat": stat, "p_value": p_value, "significant": p_value < 0.05}

def perform_mannwhitneyu_test(group_a, group_b, metric):
    """
    Perform a Mann-Whitney U test on non-normal numerical data or non-parametric data.
    """
    stat, p_value = mannwhitneyu(group_a[metric], group_b[metric], alternative='two-sided')
    return {"stat": stat, "p_value": p_value, "significant": p_value < 0.05}

def generate_contingency_table(df, column1, column2, threshold=0):
    """
    Generates a contingency table for two columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column1 (str): The name of the first column.
    - column2 (str): The name of the second column.

    Returns:
    - pd.DataFrame: A contingency table showing the frequency of each combination of values in column1 and column2.
    """
    return pd.crosstab(df[column1], df[column2] > threshold)

def segment_data(data, column, values_group_a, values_group_b):
    """
    Segment data into control (Group A) and test (Group B) for comparison.
    """

    # group_a = data[data[column] == value_a]
    # group_b = data[data[column] == value_b]
    group_a = data[data[column].isin(values_group_a)]
    group_b = data[data[column].isin(values_group_b)]

    return group_a, group_b

def aggregate_by_group(data, group_column, metrics):
    """
    Aggregate data by group and calculate the provided metrics.
    """

    data = data.copy()
    return data.groupby(group_column)[metrics].agg(["sum", "mean"]).reset_index()

def compute_metric(data, column, condition, metric_col, operation='mean'):
    """
    Compute a metric for a subset of data based on a condition.
    """
    subset = data[data[column] == condition]
    if operation == 'mean':
        return subset[metric_col].mean()
    elif operation == 'sum':
        return subset[metric_col].sum()