"""Data validation and schema enforcement for market trend forecasting."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Enumeration of supported data types."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    DATETIME = "datetime"
    BOOLEAN = "boolean"

@dataclass
class FieldSchema:
    """Schema definition for a single field."""
    name: str
    data_type: DataType
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    description: Optional[str] = None

class DataSchema:
    """Data schema validator for structured data validation."""
    
    def __init__(self, fields: List[FieldSchema]):
        """Initialize data schema.
        
        Args:
            fields: List of field schema definitions
        """
        self.fields = {field.name: field for field in fields}
        self.required_fields = [field.name for field in fields if field.required]
    
    def validate(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary of validation errors by field
        """
        errors = {}
        
        # Check for missing required fields
        missing_fields = set(self.required_fields) - set(df.columns)
        if missing_fields:
            errors['schema'] = [f"Missing required fields: {missing_fields}"]
        
        # Validate each field
        for field_name, field_schema in self.fields.items():
            if field_name in df.columns:
                field_errors = self._validate_field(df[field_name], field_schema)
                if field_errors:
                    errors[field_name] = field_errors
        
        return errors
    
    def _validate_field(self, series: pd.Series, schema: FieldSchema) -> List[str]:
        """Validate a single field against its schema.
        
        Args:
            series: Pandas series to validate
            schema: Field schema definition
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check data type
        if schema.data_type == DataType.INTEGER:
            if not pd.api.types.is_integer_dtype(series):
                errors.append(f"Expected integer type, got {series.dtype}")
        elif schema.data_type == DataType.FLOAT:
            if not pd.api.types.is_numeric_dtype(series):
                errors.append(f"Expected numeric type, got {series.dtype}")
        elif schema.data_type == DataType.STRING:
            if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
                errors.append(f"Expected string/object type, got {series.dtype}")
        elif schema.data_type == DataType.DATETIME:
            if not pd.api.types.is_datetime64_any_dtype(series):
                errors.append(f"Expected datetime type, got {series.dtype}")
        elif schema.data_type == DataType.BOOLEAN:
            if not pd.api.types.is_bool_dtype(series):
                errors.append(f"Expected boolean type, got {series.dtype}")
        
        # Check value ranges for numeric fields
        if schema.data_type in [DataType.INTEGER, DataType.FLOAT]:
            if schema.min_value is not None:
                below_min = series < schema.min_value
                if below_min.any():
                    count = below_min.sum()
                    errors.append(f"{count} values below minimum {schema.min_value}")
            
            if schema.max_value is not None:
                above_max = series > schema.max_value
                if above_max.any():
                    count = above_max.sum()
                    errors.append(f"{count} values above maximum {schema.max_value}")
        
        # Check allowed values
        if schema.allowed_values is not None:
            invalid_values = ~series.isin(schema.allowed_values)
            if invalid_values.any():
                unique_invalid = series[invalid_values].unique()
                errors.append(f"Invalid values found: {unique_invalid.tolist()}")
        
        # Check for null values in required fields
        if schema.required and series.isnull().any():
            null_count = series.isnull().sum()
            errors.append(f"{null_count} null values in required field")
        
        return errors

# Predefined schemas for different data types
TENANT_SCHEMA = DataSchema([
    FieldSchema("tenant_id", DataType.INTEGER, required=True, min_value=1),
    FieldSchema("property_id", DataType.STRING, required=True),
    FieldSchema("lease_start_date", DataType.DATETIME, required=True),
    FieldSchema("lease_end_date", DataType.DATETIME, required=True),
    FieldSchema("monthly_rent", DataType.FLOAT, required=True, min_value=0),
    FieldSchema("payment_delays_last_6_months", DataType.INTEGER, required=True, min_value=0),
    FieldSchema("maintenance_requests_last_year", DataType.INTEGER, required=True, min_value=0),
    FieldSchema("feedback_score", DataType.FLOAT, required=False, min_value=0, max_value=5),
    FieldSchema("status", DataType.STRING, required=True, allowed_values=["Active", "Inactive", "Pending"]),
    FieldSchema("lease_renewal", DataType.INTEGER, required=False, allowed_values=[0, 1])
])

PROPERTY_SCHEMA = DataSchema([
    FieldSchema("property_id", DataType.STRING, required=True),
    FieldSchema("address", DataType.STRING, required=True),
    FieldSchema("beds", DataType.INTEGER, required=True, min_value=0, max_value=20),
    FieldSchema("baths", DataType.FLOAT, required=True, min_value=0, max_value=20),
    FieldSchema("sqft", DataType.INTEGER, required=True, min_value=100, max_value=50000),
    FieldSchema("price", DataType.FLOAT, required=True, min_value=0),
    FieldSchema("property_type", DataType.STRING, required=True, 
               allowed_values=["Single Family", "Condo", "Townhouse", "Multi-Family", "Apartment"]),
    FieldSchema("last_sold_date", DataType.DATETIME, required=False),
    FieldSchema("year_built", DataType.INTEGER, required=False, min_value=1800, max_value=2030)
])

class DataQualityReport:
    """Generate comprehensive data quality reports."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize data quality analyzer.
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.report = {}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report.
        
        Returns:
            Dictionary containing quality metrics and issues
        """
        self.report = {
            'overview': self._get_overview(),
            'missing_data': self._analyze_missing_data(),
            'duplicates': self._analyze_duplicates(),
            'outliers': self._detect_outliers(),
            'data_types': self._analyze_data_types(),
            'uniqueness': self._analyze_uniqueness(),
            'correlations': self._analyze_correlations()
        }
        
        return self.report
    
    def _get_overview(self) -> Dict[str, Any]:
        """Get basic overview of the dataset."""
        return {
            'shape': self.df.shape,
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'total_cells': self.df.shape[0] * self.df.shape[1],
            'columns': list(self.df.columns)
        }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_stats = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            
            missing_stats[col] = {
                'count': int(missing_count),
                'percentage': round(missing_percentage, 2)
            }
        
        return missing_stats
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate rows."""
        duplicate_count = self.df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(self.df)) * 100
        
        return {
            'total_duplicates': int(duplicate_count),
            'percentage': round(duplicate_percentage, 2),
            'unique_rows': len(self.df) - duplicate_count
        }
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers in numeric columns using IQR method."""
        outliers = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round((outlier_count / len(self.df)) * 100, 2),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        
        return outliers
    
    def _analyze_data_types(self) -> Dict[str, str]:
        """Analyze data types of columns."""
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}
    
    def _analyze_uniqueness(self) -> Dict[str, Any]:
        """Analyze uniqueness of values in each column."""
        uniqueness = {}
        
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            uniqueness_percentage = (unique_count / len(self.df)) * 100
            
            uniqueness[col] = {
                'unique_count': int(unique_count),
                'uniqueness_percentage': round(uniqueness_percentage, 2),
                'is_likely_id': uniqueness_percentage > 95
            }
        
        return uniqueness
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {}
        
        correlation_matrix = numeric_df.corr()
        
        # Find high correlations (> 0.8)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_correlations.append({
                        'column1': correlation_matrix.columns[i],
                        'column2': correlation_matrix.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.round(3).to_dict(),
            'high_correlations': high_correlations
        }
    
    def print_summary(self):
        """Print a human-readable summary of the data quality report."""
        if not self.report:
            self.generate_report()
        
        print("=" * 50)
        print("DATA QUALITY REPORT")
        print("=" * 50)
        
        # Overview
        overview = self.report['overview']
        print(f"\nDATASET OVERVIEW:")
        print(f"  Shape: {overview['shape']}")
        print(f"  Memory Usage: {overview['memory_usage']:,} bytes")
        print(f"  Total Cells: {overview['total_cells']:,}")
        
        # Missing Data
        print(f"\nMISSING DATA:")
        missing = self.report['missing_data']
        high_missing = {k: v for k, v in missing.items() if v['percentage'] > 10}
        if high_missing:
            for col, stats in high_missing.items():
                print(f"  {col}: {stats['count']} ({stats['percentage']}%)")
        else:
            print("  No significant missing data found")
        
        # Duplicates
        duplicates = self.report['duplicates']
        print(f"\nDUPLICATE ROWS:")
        print(f"  Total: {duplicates['total_duplicates']} ({duplicates['percentage']}%)")
        
        # Outliers
        print(f"\nOUTLIERS:")
        outliers = self.report['outliers']
        high_outliers = {k: v for k, v in outliers.items() if v['percentage'] > 5}
        if high_outliers:
            for col, stats in high_outliers.items():
                print(f"  {col}: {stats['count']} ({stats['percentage']}%)")
        else:
            print("  No significant outliers found")
        
        # High Correlations
        print(f"\nHIGH CORRELATIONS (>0.8):")
        correlations = self.report['correlations'].get('high_correlations', [])
        if correlations:
            for corr in correlations:
                print(f"  {corr['column1']} <-> {corr['column2']}: {corr['correlation']}")
        else:
            print("  No high correlations found")

def validate_and_clean_data(df: pd.DataFrame, 
                          schema: DataSchema,
                          remove_duplicates: bool = True,
                          handle_outliers: bool = True) -> pd.DataFrame:
    """Validate and clean data according to schema.
    
    Args:
        df: Input DataFrame
        schema: Data schema for validation
        remove_duplicates: Whether to remove duplicate rows
        handle_outliers: Whether to handle outliers
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data validation and cleaning...")
    
    # Generate quality report
    quality_report = DataQualityReport(df)
    quality_report.generate_report()
    quality_report.print_summary()
    
    # Validate against schema
    validation_errors = schema.validate(df)
    if validation_errors:
        logger.warning(f"Validation errors found: {validation_errors}")
        # Log errors but continue with cleaning
        for field, errors in validation_errors.items():
            for error in errors:
                logger.warning(f"Field '{field}': {error}")
    
    cleaned_df = df.copy()
    
    # Remove duplicates if requested
    if remove_duplicates:
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
    
    # Handle outliers if requested
    if handle_outliers:
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            initial_outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            if initial_outliers > 0:
                logger.info(f"Capped {initial_outliers} outliers in column '{col}'")
    
    logger.info(f"Data cleaning completed. Shape: {cleaned_df.shape}")
    return cleaned_df