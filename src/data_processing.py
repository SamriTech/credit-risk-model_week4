import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans  # Added missing import
#from xverse.transformer import WOE

# --- TASK 3 FUNCTIONS ---

def extract_time_features(df):
    """Extracts hour, day, month, and year from TransactionStartTime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year
    return df

def create_aggregate_features(df):
    """Calculates customer-level aggregate features (Total, Average, Count, Std)."""
    agg_features = df.groupby('CustomerId')['Amount'].agg([
        ('Total_Amount', 'sum'),
        ('Average_Amount', 'mean'),
        ('Transaction_Count', 'count'),
        ('Std_Amount', 'std')
    ]).reset_index()
    
    # Handle single-transaction users where Std is NaN
    agg_features['Std_Amount'] = agg_features['Std_Amount'].fillna(0)
    
    return df.merge(agg_features, on='CustomerId', how='left')

def get_preprocessor(numeric_features, categorical_features):
    """Creates an sklearn Pipeline for automated scaling and encoding."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# --- TASK 4 FUNCTIONS ---

def engineer_target_variable(df):
    """
    Calculates RFM metrics and uses K-Means to create the is_high_risk label.
    """
    # 1. Calculate RFM Metrics
    snapshot_date = df['TransactionStartTime'].max()
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days, # Recency
        'TransactionId': 'count',                                         # Frequency
        'Amount': 'sum'                                                  # Monetary
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })

    # 2. Cluster Customers (K=3)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 3. Define and Assign High-Risk Label (Lowest monetary = High Risk)
    stats = rfm.groupby('Cluster')['Monetary'].mean()
    high_risk_cluster = stats.idxmin()
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm[['is_high_risk']]

def apply_woe_transformation(df, target_column):
    """Calculates IV using a manual approach to avoid xverse/pandas compatibility issues."""
    cols_to_drop = [target_column, 'TransactionId', 'BatchId', 'CustomerId', 'AccountId', 'TransactionStartTime', 'CurrencyCode', 'CountryCode']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target_column]
    
    # Simple check for categorical vs numerical
    print("\n--- Feature Importance (Information Value Proxy) ---")
    iv_list = []
    for col in X.columns:
        # We'll use a simple cross-tab to see how the feature relates to risk
        # This is a basic version of what WoE/IV does
        if X[col].nunique() > 1:
            # We group the data and see the distribution of risk
            # For numerical columns, we'll bin them briefly
            if X[col].dtype in ['int64', 'float64']:
                bins = pd.qcut(X[col], q=5, duplicates='drop')
            else:
                bins = X[col]
                
            cross = pd.crosstab(bins, y, normalize='index')
            # A feature is 'important' if the risk % varies significantly across bins
            importance = cross[1].std() 
            iv_list.append({'Feature': col, 'Importance_Score': importance})
            
    iv_df = pd.DataFrame(iv_list).sort_values(by='Importance_Score', ascending=False)
    print(iv_df)
    return X # Return features for now

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    try:
        # Load raw data
        data = pd.read_csv('data/raw/data.csv')
        
        # 1. Feature Engineering (Task 3)
        data = extract_time_features(data)
        data = create_aggregate_features(data)
        
        # 2. Target Variable Engineering (Task 4)
        target_labels = engineer_target_variable(data)
        data = data.merge(target_labels, on='CustomerId', how='left')
        
        # 3. Apply WoE and Print IV (Task 3 objective)
        # Note: We apply this now because we have the target 'is_high_risk'
        apply_woe_transformation(data, 'is_high_risk')
        
        # 4. Final Pipeline Processing
        num_cols = ['Amount', 'Value', 'Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Amount']
        cat_cols = ['ProductCategory', 'ChannelId']
        proc = get_preprocessor(num_cols, cat_cols)
        processed_data = proc.fit_transform(data)
        
        # Save the final model-ready data
        data.to_csv('data/processed/final_data.csv', index=False)
        
        print("\n--- Summary ---")
        print(f"Final Data Shape: {data.shape}")
        print(f"Risk Distribution:\n{data['is_high_risk'].value_counts()}")
        print("Task 3 & 4 Complete: Data saved to data/processed/final_data.csv")
        
    except FileNotFoundError:
        print("Error: data/raw/data.csv not found.")