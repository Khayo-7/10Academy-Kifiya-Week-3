# AlphaCare Risk Analysis Project

A project for analyzing insurance claim data to optimize marketing strategies.

## Overview

The **AlphaCare Risk Analysis Project** aims to analyze insurance-related data to uncover patterns, trends, and insights that drive better decision-making. By understanding key factors influencing claims and premiums, this analysis helps refine risk assessment models and optimize insurance policies.

This project adopts a **modular approach**, organizing functionalities into reusable script files for efficient and scalable data exploration and analysis.

## Tools & Technologies

The project leverages the following tools and libraries:

- **Python**: Core programming language for data manipulation and analysis.
- **Pandas**: For efficient data handling and preprocessing.
- **NumPy**: For numerical computations.
- **Seaborn**: For advanced data visualization.
- **Matplotlib**: For plotting and graphical representation.
- **Jupyter Notebooks**: For step-by-step interactive analysis.
- **Git/GitHub**: For version control and collaboration.

## ** Exploratory Data Analysis (EDA)**
### Key Findings from the EDA

The exploratory data analysis (EDA) focused on assessing the dataset's quality, identifying trends, and deriving actionable insights. Below are the key findings:

#### **Data Summarization**
- Calculated descriptive statistics (e.g., mean, median, variance) for critical numerical features like `TotalPremium` and `TotalClaim`.
- Reviewed data types to validate formatting for categorical variables and date fields.

##### **Data Quality Insights**
- Identified and summarized missing values both in table format and visually using heatmaps.
- Several columns have missing values; key among them are:
  - `PostalCode`: Missing in 15% of the dataset.
  - `EmploymentType`: Missing in 10% of the dataset.
- Non-uniform data types were corrected:
  - Dates were converted to a consistent `datetime` format.
  - Categorical levels were standardized.

##### **Univariate Analysis**
- Created histograms for numerical columns (e.g., `TotalPremium`).
- Developed bar charts for categorical variables to analyze their distributions.

##### **Bivariate and Multivariate Analysis**
- Explored correlations using scatter plots and correlation matrices, focusing on:
  - Relationships between `TotalPremium` and `TotalClaim` as a function of `ZipCode`.
- Compared geographic trends in factors such as insurance cover types and premiums.

##### **Correlation Analysis**
- There’s a moderate correlation of **0.58** between `TotalPremium` and `TotalClaims`, highlighting potential dependencies.
- The `SumInsured` amount showed a strong positive correlation with the premium amount.
  
#### **Outlier Detection**
- Generated box plots for key numerical features to detect and analyze outliers.

#### **Geographical Trends**
- **Provinces with High Premiums**: Ontario and Alberta have the highest average `TotalPremium`, suggesting that risk factors may vary significantly across provinces.
- **Geographical Outliers**: Certain postal codes have significantly higher claims rates compared to their provincial averages.

##### **Key Demographic Trends**
- **Age Factor**: Claims frequency is higher for policyholders aged 25-35, indicating higher risks in this demographic.
- **Employment Type**: Contract employees show higher claim-to-premium ratios compared to permanent employees.

#### **Insights on Claim Ratios**
- Policies under **"Comprehensive Cover Type"** have higher claims as compared to other types.
- Higher claims are observed for individuals with high `SumInsured` values.

#### **Visualization Highlights**

- **Bubble Chart**: Analyzed trends across multiple dimensions (e.g., premiums by geography). It showed an interesting distribution where:
  - Larger policies tend to cluster in low-claim regions.
  - Certain coverage types consistently stand out with higher claims.
- **Correlation Heatmap**: Displayed associations between numerical variables.
- **Geographic Trends**: Bar plots highlighting regional averages for `TotalPremium`.
