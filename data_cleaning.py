"""
Data Cleaning and Formatting Module for Insurance Customer Dataset.

This module provides a comprehensive suite of functions for cleaning, standardizing,
and preparing insurance customer data for analysis. Functions are designed to handle
common data quality issues including inconsistent naming conventions, invalid categorical
values, incorrect data types, missing values, and duplicate records.

Author: Data Cleaning Pipeline
Version: 1.0.0
"""

import pandas as pd


def clean_column_names(df):
    """
    Standardize column names to follow naming conventions.

    Converts all column names to lowercase, replaces spaces with underscores,
    and renames 'ST' to 'state' for consistency.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with column names to standardize.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with standardized column names.

    Examples
    --------
    >>> df_clean = clean_column_names(df)
    """
    df = df.copy()
    # Convert to lowercase and replace spaces with underscores
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # Rename 'st' to 'state'
    df.rename(columns={'st': 'state'}, inplace=True)
    return df


def clean_invalid_values(df):
    """
    Clean and standardize invalid categorical values across the dataset.

    Handles inconsistencies in the following columns:
    - Gender: Standardizes to 'M' or 'F'
    - State: Converts abbreviations to full state names
    - Education: Standardizes 'Bachelors' to 'Bachelor'
    - Vehicle Class: Maps luxury vehicle categories to 'Luxury'
    - Customer Lifetime Value: Removes '%' character for numeric conversion

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with invalid categorical values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with cleaned categorical values.

    Examples
    --------
    >>> df_clean = clean_invalid_values(df)
    """
    df = df.copy()

    # Clean gender column
    gender_map = {
        "Male": "M",
        "male": "M",
        "female": "F",
        "Femal": "F",
        "F": "F",
        "M": "M"
    }
    df['gender'] = df['gender'].replace(gender_map)

    # Clean state column
    state_map = {
        "Cali": "California",
        "WA": "Washington",
        "AZ": "Arizona"
    }
    df['state'] = df['state'].replace(state_map)

    # Clean education column
    df['education'] = df['education'].replace("Bachelors", "Bachelor")

    # Clean vehicle class column
    luxury_map = {
        "Sports Car": "Luxury",
        "Luxury SUV": "Luxury",
        "Luxury Car": "Luxury"
    }
    df['vehicle_class'] = df['vehicle_class'].replace(luxury_map)

    # Clean customer lifetime value - remove '%' character
    df['customer_lifetime_value'] = df['customer_lifetime_value'].str.replace(
        '%', '', regex=False
    )

    return df


def format_data_types(df):
    """
    Format and convert data types to appropriate types for analysis.

    Handles the following conversions:
    - customer_lifetime_value: Converts to numeric (float)
    - number_of_open_complaints: Extracts middle value from 'X/Y/Z' format
      and converts to numeric (int)

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with incorrectly formatted data types.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with properly formatted data types.

    Examples
    --------
    >>> df_formatted = format_data_types(df)
    """
    df = df.copy()

    # Convert customer_lifetime_value to numeric
    df['customer_lifetime_value'] = pd.to_numeric(
        df['customer_lifetime_value'], errors='coerce'
    )

    # Extract middle value from number_of_open_complaints (format: X/Y/Z -> Y)
    df['number_of_open_complaints'] = df['number_of_open_complaints'].apply(
        lambda x: x.split('/')[1] if isinstance(x, str) else x
    )
    df['number_of_open_complaints'] = pd.to_numeric(
        df['number_of_open_complaints'], errors='coerce'
    )

    return df


def handle_null_and_duplicates(df):
    """
    Handle missing values and duplicate rows in the dataset.

    Processing steps:
    1. Drop rows where 'customer' column is null (no customer info)
    2. Fill numeric columns with their mean values
    3. Fill categorical columns with 'Unknown'
    4. Convert all numeric columns to integers
    5. Remove duplicate rows (keeping first occurrence)
    6. Reset index for consistency

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with potential null values and duplicates.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with all null values handled and duplicates removed.

    Examples
    --------
    >>> df_clean = handle_null_and_duplicates(df)
    """
    df = df.copy()

    # Drop rows where customer is null
    df = df.dropna(subset=['customer'])

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill numeric columns with mean
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns with 'Unknown'
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')

    # Convert numeric columns to integers
    for col in numeric_cols:
        df[col] = df[col].astype(int)

    # Remove duplicate rows (keep first occurrence)
    df = df.drop_duplicates(keep='first')

    # Reset index
    df = df.reset_index(drop=True)

    return df


def run_pipeline(url):
    """
    Execute the complete data cleaning and formatting pipeline.

    Orchestrates the entire workflow: loads data from URL, applies all cleaning
    transformations in the correct sequence, and returns the cleaned DataFrame.

    Parameters
    ----------
    url : str
        The URL to the CSV file containing the raw insurance customer data.

    Returns
    -------
    pd.DataFrame
        The fully cleaned and formatted DataFrame ready for analysis.

    Raises
    ------
    Exception
        If the URL is invalid or the CSV cannot be read.
        If any processing step encounters an error.

    Examples
    --------
    >>> url = "https://raw.githubusercontent.com/data-bootcamp-v4/data/main/file1.csv"
    >>> df_cleaned = run_pipeline(url)
    """
    # Load data from URL
    df = pd.read_csv(url)

    # Apply cleaning functions in sequence
    df = clean_column_names(df)
    df = clean_invalid_values(df)
    df = format_data_types(df)
    df = handle_null_and_duplicates(df)

    return df
