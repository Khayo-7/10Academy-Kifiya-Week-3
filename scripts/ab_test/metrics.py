import pandas as pd

def compute_average_risk(data, column, condition):
    """
    Calculate the average risk for a group based on condition.
    """
    group = data[data[column] == condition]
    return (group["TotalClaims"] / group["TotalPremium"]).mean()

def compute_average_claims(data, column, condition):
    """
    Calculate the average TotalClaims for a subgroup based on the condition.
    """
    group = data[data[column] == condition]
    return group["TotalClaims"].mean()

def compute_margin(data, column, condition):
    """
    Calculate the average margin for a group.
    """
    group = data[data[column] == condition]
    return (group["TotalPremium"] - group["TotalClaims"]).mean()

def compute_risk_metrics(data):
    """
    Add derived risk metrics to the dataset, handling invalid or zero TotalPremium values.
    """
    
    data["ProfitMargin"] = data["TotalPremium"] - data["TotalClaims"]
    data["Risk"] = data["TotalClaims"] / data["TotalPremium"].replace(0, pd.NA)
    data["Risk"] = pd.to_numeric(data["Risk"], errors='coerce')
    # data["Risk"] = data["Risk"].astype(float)  # Ensure 'Risk' column is of float type
    data.dropna(subset=["Risk"], inplace=True)  # Drop rows with invalid Risk
    
    return data

def calculate_cost_savings(data, group_col, cost_col, claim_col):
    """
    Compute potential cost savings by group.

    Args:
        data (pd.DataFrame): Input dataset
        group_col (str): Column for grouping (e.g., 'Province', 'PostalCode')
        cost_col (str): Column for cost incurred (e.g., 'TotalPremium')
        claim_col (str): Column for claim losses (e.g., 'TotalClaims')
    
    Returns:
        pd.DataFrame: Cost savings summary by group.
    """
    group_data = data.groupby(group_col).agg(
        TotalCost=(cost_col, "sum"),
        TotalClaims=(claim_col, "sum"),
    )
    group_data["SavingsPotential"] = group_data["TotalCost"] - group_data["TotalClaims"]
    group_data["SavingsPercentage"] = (group_data["SavingsPotential"] / group_data["TotalCost"]) * 100
    return group_data.reset_index()

def rank_savings_opportunities(savings_summary, threshold=0):
    """
    Rank savings opportunities by potential percentage.

    Args:
        savings_summary (pd.DataFrame): Savings data returned by calculate_cost_savings
        threshold (float): Minimum savings percentage to filter results.

    Returns:
        pd.DataFrame: Ranked groups with high savings potential.
    """
    filtered = savings_summary[savings_summary["SavingsPercentage"] > threshold]
    return filtered.sort_values(by="SavingsPercentage", ascending=False).reset_index(drop=True)
