"""
Data Preprocessing Module
AI Ethics Assignment - COMPAS dataset preparation and cleaning

Functions for loading, exploring, and cleaning COMPAS recidivism data
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# DATA LOADING
# ============================================================================

def load_compas_from_csv(filepath):
    """
    Load COMPAS dataset from CSV file
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        pd.DataFrame: COMPAS dataset
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded COMPAS dataset")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def load_compas_from_aif360():
    """
    Load COMPAS dataset from AI Fairness 360 library (auto-download)
    
    More reliable than CSV - handles data formatting automatically
    
    Returns:
        pd.DataFrame: COMPAS dataset
    """
    try:
        from aif360.datasets import CompasDataset
        
        print("Loading COMPAS dataset from AI Fairness 360...")
        dataset = CompasDataset()
        df = dataset.convert_to_dataframe()[0]
        
        print(f"✓ Loaded COMPAS dataset from AIF360")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    
    except ImportError:
        raise ImportError("aif360 library not installed. Run: pip install aif360")
    except Exception as e:
        raise Exception(f"Error loading COMPAS from AIF360: {str(e)}")

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def explore_dataset(df):
    """
    Print comprehensive dataset exploration
    
    Args:
        df: DataFrame to explore
    """
    print("\n" + "="*70)
    print("DATASET EXPLORATION")
    print("="*70)
    
    print("\nBasic Info:")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nData Types:")
    print(f"  Numeric: {df.select_dtypes(include=['number']).shape[1]}")
    print(f"  Categorical: {df.select_dtypes(include=['object']).shape[1]}")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  Total missing: {missing.sum()}")
        for col in missing[missing > 0].index:
            print(f"    {col}: {missing[col]} ({missing[col]/len(df)*100:.1f}%)")
    else:
        print("  None")
    
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n" + "="*70 + "\n")


def get_column_info(df):
    """
    Get detailed information about each column
    
    Args:
        df: DataFrame
    
    Returns:
        pd.DataFrame: Column information
    """
    info = {
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null': df.count(),
        'Missing': df.isnull().sum(),
        'Unique': [df[col].nunique() for col in df.columns]
    }
    return pd.DataFrame(info)


# ============================================================================
# DATA CLEANING
# ============================================================================

def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in dataset
    
    Args:
        df: DataFrame
        strategy: 'drop' or 'fill'
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    initial_rows = len(df)
    
    if strategy == 'drop':
        df = df.dropna()
        rows_removed = initial_rows - len(df)
        print(f"✓ Removed {rows_removed} rows with missing values")
    
    elif strategy == 'fill':
        # Fill numeric with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        print("✓ Filled missing values (numeric: median, categorical: mode)")
    
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows
    
    Args:
        df: DataFrame
    
    Returns:
        pd.DataFrame: DataFrame without duplicates
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    rows_removed = initial_rows - len(df)
    
    if rows_removed > 0:
        print(f"✓ Removed {rows_removed} duplicate rows")
    else:
        print("✓ No duplicates found")
    
    return df


def convert_data_types(df, type_mapping=None):
    """
    Convert column data types for optimal processing
    
    Args:
        df: DataFrame
        type_mapping: Dict of {column: dtype}
    
    Returns:
        pd.DataFrame: DataFrame with converted types
    """
    if type_mapping is None:
        type_mapping = {}
    
    for col, dtype in type_mapping.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
                print(f"✓ Converted {col} to {dtype}")
            except Exception as e:
                print(f"⚠ Could not convert {col} to {dtype}: {str(e)}")
    
    return df

# ============================================================================
# FEATURE ENGINEERING FOR COMPAS
# ============================================================================

def prepare_compas_features(df):
    """
    Prepare COMPAS-specific features for analysis
    
    Handles:
    - Race/demographic categories
    - Recidivism target variable
    - Risk score binning
    
    Args:
        df: Raw COMPAS dataframe
    
    Returns:
        pd.DataFrame: Prepared dataframe
    """
    df = df.copy()
    
    # Ensure race is categorical
    if 'race' in df.columns:
        df['race'] = df['race'].astype('category')
        print(f"✓ Race categories: {df['race'].unique().tolist()}")
    
    # Ensure target variable is binary
    if 'two_year_recidivism' in df.columns:
        df['two_year_recidivism'] = df['two_year_recidivism'].astype(int)
    
    # Bin COMPAS scores if present
    if 'decile_score' in df.columns:
        df['high_risk'] = (df['decile_score'] >= 5).astype(int)
        print("✓ Created high_risk feature (decile_score >= 5)")
    
    return df


# ============================================================================
# DEMOGRAPHIC ANALYSIS
# ============================================================================

def get_demographic_distribution(df, demographic_col='race'):
    """
    Get distribution of demographic groups
    
    Args:
        df: DataFrame
        demographic_col: Column name for demographic
    
    Returns:
        pd.Series: Distribution
    """
    return df[demographic_col].value_counts()


def get_outcome_by_demographic(df, demographic_col='race', outcome_col='two_year_recidivism'):
    """
    Get outcome rates by demographic group
    
    Args:
        df: DataFrame
        demographic_col: Column name for demographic
        outcome_col: Column name for outcome
    
    Returns:
        pd.DataFrame: Outcome rates by group
    """
    outcome_by_group = df.groupby(demographic_col)[outcome_col].agg(['sum', 'count', 'mean'])
    outcome_by_group.columns = ['positive_outcomes', 'total', 'rate']
    return outcome_by_group


def demographic_parity_analysis(df, demographic_col='race', pred_col='high_risk'):
    """
    Analyze demographic parity in predictions
    
    Args:
        df: DataFrame with predictions
        demographic_col: Demographic column
        pred_col: Prediction column
    
    Returns:
        dict: Parity analysis
    """
    analysis = {}
    
    for group in df[demographic_col].unique():
        group_df = df[df[demographic_col] == group]
        analysis[str(group)] = {
            'n': len(group_df),
            'selection_rate': group_df[pred_col].mean(),
            'positive_pred': (group_df[pred_col] == 1).sum()
        }
    
    return analysis

# ============================================================================
# DATA EXPORT/SAVE
# ============================================================================

def save_processed_data(df, filepath, format='csv'):
    """
    Save processed data to file
    
    Args:
        df: DataFrame
        filepath: Path to save
        format: 'csv', 'parquet', or 'excel'
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✓ Saved processed data to {filepath}")


# ============================================================================
# PREPROCESSING PIPELINE
# ============================================================================

def full_preprocessing_pipeline(df, remove_dup=True, handle_missing=True, missing_strategy='drop'):
    """
    Execute full preprocessing pipeline
    
    Args:
        df: Raw dataframe
        remove_dup: Remove duplicates
        handle_missing: Handle missing values
        missing_strategy: 'drop' or 'fill'
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE")
    print("="*70)
    
    initial_shape = df.shape
    print(f"\nInitial shape: {initial_shape[0]} rows × {initial_shape[1]} columns")
    
    # Remove duplicates
    if remove_dup:
        df = remove_duplicates(df)
    
    # Handle missing
    if handle_missing:
        df = handle_missing_values(df, strategy=missing_strategy)
    
    # Prepare features
    df = prepare_compas_features(df)
    
    # Summary
    print(f"\nFinal shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Rows removed: {initial_shape[0] - df.shape[0]}")
    print("="*70 + "\n")
    
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dataset_summary(df):
    """
    Get quick summary of dataset
    
    Args:
        df: DataFrame
    
    Returns:
        dict: Summary statistics
    """
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'n_numeric': len(df.select_dtypes(include=['number']).columns),
        'n_categorical': len(df.select_dtypes(include=['object']).columns),
        'missing': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }


def print_dataset_summary(df):
    """
    Pretty-print dataset summary
    
    Args:
        df: DataFrame
    """
    summary = get_dataset_summary(df)
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Rows: {summary['n_rows']:,}")
    print(f"Columns: {summary['n_cols']}")
    print(f"Numeric: {summary['n_numeric']}")
    print(f"Categorical: {summary['n_categorical']}")
    print(f"Missing values: {summary['missing']}")
    print(f"Duplicates: {summary['duplicates']}")
    print("="*50 + "\n")