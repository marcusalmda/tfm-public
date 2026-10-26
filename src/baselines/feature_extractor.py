import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Optional

def aggregate_features_for_documents(
    sequences_data: Dict[Any, List[Dict[str, Any]]],
    document_ids: List[Any],
    feature_keys_ordered: List[str], # Not directly used in current aggregation, but good for context
    feature_info: Dict[str, Dict[str, Any]],
    max_event_type_id: Optional[int] = None,
    specific_event_max_idx: Optional[int] = None,
    counterparty_max_idx: Optional[int] = None,
) -> pd.DataFrame:
    """
    Aggregates event features for a list of document_ids from sequence data.
    This function computes various summary statistics for numerical features and
    frequency encodings for categorical features associated with each document.

    Args:
        sequences_data: A dictionary where keys are document_ids and values are lists of event dictionaries.
                        Each event dictionary should contain features like 'value_processed', 'event_type_id', etc.
        document_ids: A list of document_ids for which to aggregate features. These are the target documents.
        feature_keys_ordered: Ordered list of feature keys as they appear in the raw event data structure.
                              (Currently not directly used in aggregation logic but useful for understanding data structure).
        feature_info: Dictionary providing metadata about features, including their type ('continuous', 'categorical')
                      and 'vocab_size' for categorical features. This is used to determine how to treat each feature
                      and to define the range for frequency encoding of categorical features.
        max_event_type_id: If provided, creates frequency columns for 'event_type_id' from 0 up to this value (exclusive).
                           This ensures a fixed number of features. If None, columns are created based on observed unique values.
        specific_event_max_idx: Similar to max_event_type_id, but for 'specific_event_idx'. Used for frequency encoding.
        counterparty_max_idx: Similar to max_event_type_id, but for 'counterparty_idx'. Used for frequency encoding.
                              Use with caution if cardinality is very high, as it can lead to many features.

    Returns:
        A pandas DataFrame where each row corresponds to a document_id (from input `document_ids` that were found
        in `sequences_data`) and columns are the aggregated features.
    """
    aggregated_data = []  # List to store feature dictionaries for each document
    processed_doc_ids = []  # List to store document_ids that are actually processed

    # Define common feature column names for clarity and consistency
    # These names are expected to be keys in the event dictionaries within sequences_data
    value_col = 'value_processed'
    event_type_col = 'event_type_id'
    delta_t_col = 'delta_t_seconds'
    specific_event_col = 'specific_event_idx'
    counterparty_col = 'counterparty_idx'
    time_hour_col = 'time_hour'
    time_weekday_col = 'time_weekday'

    # Iterate over the specified document IDs to perform aggregation
    for doc_id in tqdm(document_ids, desc="Aggregating features for non-sequential models"):
        event_list = sequences_data.get(doc_id) # Get the list of events for the current document

        # Initialize a dictionary to store features for the current document
        features: Dict[str, Any] = {}

        if event_list is None:
            # This document_id was in the requested list but not found in sequences_data.
            # It will be skipped, and not included in the output DataFrame.
            # Alternatively, one could add a row of NaNs, but skipping is cleaner if inputs are expected to align.
            continue

        processed_doc_ids.append(doc_id) # Add doc_id to list of processed documents

        if not event_list:  # Document ID was in sequences_data but had an empty list of events
            # For documents with no events, create features with default zero/neutral values
            features['event_count'] = 0
            features[f'{value_col}_sum'] = 0.0
            features[f'{value_col}_mean'] = 0.0
            features[f'{value_col}_std'] = 0.0
            features[f'{value_col}_min'] = 0.0
            features[f'{value_col}_max'] = 0.0
            features[f'{delta_t_col}_sum'] = 0.0
            features[f'{delta_t_col}_mean'] = 0.0
            # Categorical frequency features will be implicitly zero if not set
        else:
            # Convert list of event dictionaries to a pandas DataFrame for easier manipulation
            df_events = pd.DataFrame(event_list)
            features['event_count'] = len(df_events)

            # Aggregate 'value_processed' (monetary value)
            if value_col in df_events.columns:
                numeric_values = pd.to_numeric(df_events[value_col], errors='coerce').fillna(0)
                features[f'{value_col}_sum'] = numeric_values.sum()
                features[f'{value_col}_mean'] = numeric_values.mean()
                features[f'{value_col}_std'] = numeric_values.std()
                features[f'{value_col}_min'] = numeric_values.min()
                features[f'{value_col}_max'] = numeric_values.max()
            else: # Handle case where value_col might be missing for some reason
                features[f'{value_col}_sum'] = 0.0
                features[f'{value_col}_mean'] = 0.0
                features[f'{value_col}_std'] = 0.0
                features[f'{value_col}_min'] = 0.0
                features[f'{value_col}_max'] = 0.0


            # Aggregate 'delta_t_seconds' (time difference between events)
            if delta_t_col in df_events.columns:
                numeric_delta_t = pd.to_numeric(df_events[delta_t_col], errors='coerce').fillna(0)
                features[f'{delta_t_col}_sum'] = numeric_delta_t.sum()
                features[f'{delta_t_col}_mean'] = numeric_delta_t.mean()
            else:
                features[f'{delta_t_col}_sum'] = 0.0
                features[f'{delta_t_col}_mean'] = 0.0

            # Frequency encoding for 'event_type_id'
            if event_type_col in df_events.columns:
                event_type_counts = df_events[event_type_col].value_counts(normalize=True)
                if max_event_type_id is not None: # Fixed-range frequency encoding
                    for i in range(max_event_type_id):
                        features[f'{event_type_col}_freq_{i}'] = event_type_counts.get(i, 0.0)
                else: # Data-driven frequency encoding (creates columns based on observed values)
                    for type_id, freq in event_type_counts.items():
                        features[f'{event_type_col}_freq_{type_id}'] = freq

            # Frequency encoding for 'specific_event_idx'
            if specific_event_col in df_events.columns and specific_event_max_idx is not None:
                specific_event_counts = df_events[specific_event_col].value_counts(normalize=True)
                for i in range(specific_event_max_idx):
                     features[f'{specific_event_col}_freq_{i}'] = specific_event_counts.get(i, 0.0)

            # Frequency encoding for 'counterparty_idx'
            if counterparty_col in df_events.columns and counterparty_max_idx is not None:
                counterparty_counts = df_events[counterparty_col].value_counts(normalize=True)
                for i in range(counterparty_max_idx):
                     features[f'{counterparty_col}_freq_{i}'] = counterparty_counts.get(i, 0.0)

            # Frequency encoding for time features: 'time_hour' and 'time_weekday'
            if time_hour_col in df_events.columns:
                hour_counts = df_events[time_hour_col].value_counts(normalize=True)
                # Use vocab_size from feature_info if available, otherwise default to 24
                num_hours = feature_info.get(time_hour_col, {}).get('vocab_size', 24)
                for hr in range(num_hours):
                    features[f'{time_hour_col}_freq_{hr}'] = hour_counts.get(hr, 0.0)

            if time_weekday_col in df_events.columns:
                weekday_counts = df_events[time_weekday_col].value_counts(normalize=True)
                # Use vocab_size from feature_info if available, otherwise default to 7
                num_weekdays = feature_info.get(time_weekday_col, {}).get('vocab_size', 7)
                for wd in range(num_weekdays):
                    features[f'{time_weekday_col}_freq_{wd}'] = weekday_counts.get(wd, 0.0)

        aggregated_data.append(features) # Add the features of the current document to the list

    # Create a pandas DataFrame from the list of aggregated feature dictionaries
    df_aggregated = pd.DataFrame(aggregated_data)
    if not processed_doc_ids and document_ids: # No documents were processed (e.g. all missing from sequences_data)
        # Return an empty DataFrame with expected columns if possible, or just empty.
        # This depends on how downstream code handles empty DataFrames.
        # For now, if df_aggregated is empty, it will be returned as such.
        pass

    if processed_doc_ids: # Only set index if there's data
        df_aggregated.index = processed_doc_ids  # Set document_ids as the DataFrame index

    # Fill NaNs that might have occurred (e.g., std for a single event, or if a column was all NaNs)
    df_aggregated.fillna(0.0, inplace=True)

    # Ensure all expected frequency columns exist, even if a particular category was not observed in any document.
    # This is important for consistency if fixed-range frequency encoding (e.g., max_event_type_id) was used.
    # These columns will be filled with 0.0 if they don't exist.
    if max_event_type_id is not None and event_type_col in feature_info : # Check if event_type_col is relevant
        for i in range(max_event_type_id):
            col_name = f'{event_type_col}_freq_{i}'
            if col_name not in df_aggregated.columns:
                df_aggregated[col_name] = 0.0

    if specific_event_max_idx is not None and specific_event_col in feature_info:
        for i in range(specific_event_max_idx):
            col_name = f'{specific_event_col}_freq_{i}'
            if col_name not in df_aggregated.columns:
                df_aggregated[col_name] = 0.0

    if counterparty_max_idx is not None and counterparty_col in feature_info:
        for i in range(counterparty_max_idx):
            col_name = f'{counterparty_col}_freq_{i}'
            if col_name not in df_aggregated.columns:
                df_aggregated[col_name] = 0.0

    if time_hour_col in feature_info:
        num_hours = feature_info.get(time_hour_col, {}).get('vocab_size', 24)
        for hr in range(num_hours):
            col_name = f'{time_hour_col}_freq_{hr}'
            if col_name not in df_aggregated.columns:
                df_aggregated[col_name] = 0.0

    if time_weekday_col in feature_info:
        num_weekdays = feature_info.get(time_weekday_col, {}).get('vocab_size', 7)
        for wd in range(num_weekdays):
            col_name = f'{time_weekday_col}_freq_{wd}'
            if col_name not in df_aggregated.columns:
                df_aggregated[col_name] = 0.0

    # Sort columns alphabetically for consistent feature order
    df_aggregated = df_aggregated.sort_index(axis=1)

    return df_aggregated

if __name__ == '__main__':
    # Example Usage (illustrative and for basic verification)
    print("Running mock feature aggregation in feature_extractor.py...")

    # Define mock data simulating the structure of sequences_data
    mock_sequences_data = {
        'doc1': [
            {'event_type_id': 0, 'value_processed': 100, 'delta_t_seconds': 10, 'specific_event_idx': 1, 'counterparty_idx': 10, 'time_hour': 9, 'time_weekday': 0},
            {'event_type_id': 1, 'value_processed': 200, 'delta_t_seconds': 20, 'specific_event_idx': 2, 'counterparty_idx': 11, 'time_hour': 10, 'time_weekday': 0},
        ],
        'doc2': [
            {'event_type_id': 0, 'value_processed': 50, 'delta_t_seconds': 5, 'specific_event_idx': 1, 'counterparty_idx': 10, 'time_hour': 14, 'time_weekday': 2},
        ],
        'doc3': [], # Document with no events
        'doc4_with_events': [ # Document with only one event type, to test frequency padding
            {'event_type_id': 2, 'value_processed': 75, 'delta_t_seconds': 0, 'specific_event_idx': 3, 'counterparty_idx': 12, 'time_hour': 15, 'time_weekday': 3},
        ]
    }
    # Define list of document IDs to process; 'doc4_missing' is intentionally not in mock_sequences_data
    mock_document_ids = ['doc1', 'doc2', 'doc3', 'doc4_with_events', 'doc4_missing']

    # Simplified feature_info for the mock example
    # 'feature_keys_ordered' is not strictly used by this aggregation function but good for context
    mock_feature_keys_ordered = ['event_type_id', 'value_processed', 'delta_t_seconds', 'specific_event_idx', 'counterparty_idx', 'time_hour', 'time_weekday']
    mock_feature_info = {
        'event_type_id': {'type': 'categorical', 'vocab_size': 3}, # Max ID is 2 (categories 0, 1, 2)
        'value_processed': {'type': 'continuous'},
        'delta_t_seconds': {'type': 'continuous'},
        'specific_event_idx': {'type': 'categorical', 'vocab_size': 5}, # Max ID 4
        'counterparty_idx': {'type': 'categorical', 'vocab_size': 15}, # Max ID 14
        'time_hour': {'type': 'categorical', 'vocab_size': 24}, # Max ID 23
        'time_weekday': {'type': 'categorical', 'vocab_size': 7}  # Max ID 6
    }

    df_aggregated_mock = aggregate_features_for_documents(
        mock_sequences_data,
        mock_document_ids,
        mock_feature_keys_ordered,
        mock_feature_info,
        max_event_type_id=mock_feature_info['event_type_id']['vocab_size'],
        specific_event_max_idx=mock_feature_info['specific_event_idx']['vocab_size'],
        # Not enabling counterparty_max_idx for cleaner example output due to high feature count
        # counterparty_max_idx=mock_feature_info['counterparty_idx']['vocab_size']
    )

    print("\nAggregated Features DataFrame:")
    print(df_aggregated_mock)
    print(f"\nShape: {df_aggregated_mock.shape}") # Expected rows: 4 (doc1, doc2, doc3, doc4_with_events)
    print(f"\nColumns: {df_aggregated_mock.columns.tolist()}")

    # --- Verification Checks ---
    # Check that 'doc4_missing' is not in the index
    assert 'doc4_missing' not in df_aggregated_mock.index
    assert len(df_aggregated_mock) == 4 # doc1, doc2, doc3, doc4_with_events

    # Check 'doc3' (no events)
    assert 'doc3' in df_aggregated_mock.index
    assert df_aggregated_mock.loc['doc3', 'event_count'] == 0
    assert df_aggregated_mock.loc['doc3', 'value_processed_sum'] == 0.0
    assert df_aggregated_mock.loc['doc3', f"event_type_id_freq_0"] == 0.0 # Freq columns should exist and be 0

    # Check 'doc1' features
    assert df_aggregated_mock.loc['doc1', 'event_count'] == 2
    assert df_aggregated_mock.loc['doc1', 'value_processed_sum'] == 300.0
    assert df_aggregated_mock.loc['doc1', 'value_processed_mean'] == 150.0
    assert df_aggregated_mock.loc['doc1', f"event_type_id_freq_0"] == 0.5 # 1 of 2 events is type 0
    assert df_aggregated_mock.loc['doc1', f"event_type_id_freq_1"] == 0.5 # 1 of 2 events is type 1
    assert df_aggregated_mock.loc['doc1', f"event_type_id_freq_2"] == 0.0 # No event type 2 for doc1

    # Check 'doc4_with_events' to ensure frequency padding works
    assert df_aggregated_mock.loc['doc4_with_events', f"event_type_id_freq_0"] == 0.0
    assert df_aggregated_mock.loc['doc4_with_events', f"event_type_id_freq_1"] == 0.0
    assert df_aggregated_mock.loc['doc4_with_events', f"event_type_id_freq_2"] == 1.0

    # Check existence of all frequency columns for time_hour and time_weekday
    num_hours_expected = mock_feature_info['time_hour']['vocab_size']
    num_weekdays_expected = mock_feature_info['time_weekday']['vocab_size']
    hour_freq_cols = [col for col in df_aggregated_mock.columns if 'time_hour_freq' in col]
    weekday_freq_cols = [col for col in df_aggregated_mock.columns if 'time_weekday_freq' in col]
    assert len(hour_freq_cols) == num_hours_expected
    assert len(weekday_freq_cols) == num_weekdays_expected

    # Check specific_event_idx frequency columns
    num_specific_event_expected = mock_feature_info['specific_event_idx']['vocab_size']
    specific_event_freq_cols = [col for col in df_aggregated_mock.columns if 'specific_event_idx_freq' in col]
    assert len(specific_event_freq_cols) == num_specific_event_expected
    assert df_aggregated_mock.loc['doc1', f"specific_event_idx_freq_1"] == 0.5
    assert df_aggregated_mock.loc['doc1', f"specific_event_idx_freq_2"] == 0.5
    assert df_aggregated_mock.loc['doc1', f"specific_event_idx_freq_0"] == 0.0


    # Test with a single event for std calculation (should be 0.0 after fillna)
    mock_sequences_data_single_event = {
        'doc_single': [
            {'event_type_id': 0, 'value_processed': 100, 'delta_t_seconds': 10, 'time_hour': 9, 'time_weekday': 0},
        ]
    }
    df_single = aggregate_features_for_documents(
        mock_sequences_data_single_event, ['doc_single'], mock_feature_keys_ordered, mock_feature_info,
        max_event_type_id=mock_feature_info['event_type_id']['vocab_size']
    )
    print("\nSingle event document features (value_processed_std should be 0):")
    print(df_single[['event_count', 'value_processed_sum', 'value_processed_mean', 'value_processed_std']])
    assert df_single.loc['doc_single', 'value_processed_std'] == 0.0 # std of a single number is NaN, then filled to 0.0

    print("\nMock feature aggregation example and checks in feature_extractor.py completed successfully.")
