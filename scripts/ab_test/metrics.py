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
