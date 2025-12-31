import pytest
import pandas as pd
from src.data_processing import extract_time_features

def test_extract_time_features_logic():
    """Verify that the hour is correctly extracted from the timestamp."""
    sample_df = pd.DataFrame({'TransactionStartTime': ['2025-12-30 15:45:00']})
    processed = extract_time_features(sample_df)
    
    # Assert that the hour is 15
    assert processed['TransactionHour'].iloc[0] == 15
    # Assert that all 4 time columns were created
    for col in ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']:
        assert col in processed.columns

def test_aggregate_features_columns():
    """Check if the aggregate function creates the expected feature names."""
    from src.data_processing import create_aggregate_features
    sample_df = pd.DataFrame({
        'CustomerId': ['User1', 'User1'],
        'Amount': [100, 200]
    })
    processed = create_aggregate_features(sample_df)
    
    expected_features = ['Total_Amount', 'Average_Amount', 'Transaction_Count', 'Std_Amount']
    for feature in expected_features:
        assert feature in processed.columns