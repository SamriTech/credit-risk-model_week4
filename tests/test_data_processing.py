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


def test_build_customer_features_columns_and_one_row():
    """Verify build_customer_features returns expected columns and fills Std_Amount for one-row customer."""
    from src.train import build_customer_features

    sample_df = pd.DataFrame({
        'CustomerId': ['U1', 'U1', 'U2'],
        'Amount': [100, 200, 50],
        'TransactionStartTime': ['2025-12-30 10:00:00', '2025-12-31 12:00:00', '2025-12-29 09:00:00']
    })
    processed = build_customer_features(sample_df)

    for col in ['CustomerId', 'Transaction_Count', 'Total_Amount', 'Average_Amount', 'Std_Amount', 'RecencyDays']:
        assert col in processed.columns

    # One-row customer should have Std_Amount == 0.0 after fillna
    sample_single = pd.DataFrame({
        'CustomerId': ['S1'],
        'Amount': [123.45],
        'TransactionStartTime': ['2025-12-30 15:45:00']
    })
    processed_single = build_customer_features(sample_single)
    assert processed_single['Std_Amount'].iloc[0] == 0.0


def test_build_customer_features_empty_df():
    """Verify build_customer_features handles empty DataFrame without crashing."""
    from src.train import build_customer_features

    empty_df = pd.DataFrame({'CustomerId': [], 'Amount': [], 'TransactionStartTime': []})
    processed_empty = build_customer_features(empty_df)
    assert processed_empty.empty
    for col in ['CustomerId', 'Transaction_Count', 'Total_Amount', 'Average_Amount', 'Std_Amount', 'RecencyDays']:
        assert col in processed_empty.columns