"""Unit tests for data validation module."""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.validation import (
    DataSchema, FieldSchema, DataType, DataQualityReport,
    TENANT_SCHEMA, PROPERTY_SCHEMA, validate_and_clean_data
)

class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_tenant_data = pd.DataFrame({
            'tenant_id': [1, 2, 3],
            'property_id': ['P001', 'P002', 'P003'],
            'lease_start_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'lease_end_date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'monthly_rent': [1500.0, 2000.0, 1200.0],
            'payment_delays_last_6_months': [0, 1, 2],
            'maintenance_requests_last_year': [1, 0, 3],
            'feedback_score': [4.5, 3.8, 2.1],
            'status': ['Active', 'Active', 'Inactive'],
            'lease_renewal': [1, 1, 0]
        })
        
        self.invalid_tenant_data = pd.DataFrame({
            'tenant_id': [-1, 2, 3],  # Invalid: negative ID
            'property_id': ['P001', '', 'P003'],  # Invalid: empty string
            'lease_start_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'lease_end_date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01']),
            'monthly_rent': [-100.0, 2000.0, 1200.0],  # Invalid: negative rent
            'payment_delays_last_6_months': [0, 1, -1],  # Invalid: negative delays
            'maintenance_requests_last_year': [1, 0, 3],
            'feedback_score': [6.0, 3.8, 2.1],  # Invalid: score > 5
            'status': ['Active', 'Unknown', 'Inactive'],  # Invalid: 'Unknown' status
            'lease_renewal': [1, 2, 0]  # Invalid: value 2
        })
    
    def test_field_schema_creation(self):
        """Test FieldSchema creation."""
        field = FieldSchema(
            name="test_field",
            data_type=DataType.INTEGER,
            required=True,
            min_value=0,
            max_value=100
        )
        
        self.assertEqual(field.name, "test_field")
        self.assertEqual(field.data_type, DataType.INTEGER)
        self.assertTrue(field.required)
        self.assertEqual(field.min_value, 0)
        self.assertEqual(field.max_value, 100)
    
    def test_data_schema_validation_valid_data(self):
        """Test schema validation with valid data."""
        errors = TENANT_SCHEMA.validate(self.valid_tenant_data)
        self.assertEqual(len(errors), 0)
    
    def test_data_schema_validation_invalid_data(self):
        """Test schema validation with invalid data."""
        errors = TENANT_SCHEMA.validate(self.invalid_tenant_data)
        
        # Should have errors for multiple fields
        self.assertGreater(len(errors), 0)
        
        # Check specific errors
        self.assertIn('tenant_id', errors)
        self.assertIn('monthly_rent', errors)
        self.assertIn('payment_delays_last_6_months', errors)
        self.assertIn('feedback_score', errors)
        self.assertIn('status', errors)
        self.assertIn('lease_renewal', errors)
    
    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        incomplete_data = self.valid_tenant_data.drop(columns=['tenant_id', 'monthly_rent'])
        errors = TENANT_SCHEMA.validate(incomplete_data)
        
        self.assertIn('schema', errors)
        self.assertIn('tenant_id', str(errors['schema']))
        self.assertIn('monthly_rent', str(errors['schema']))
    
    def test_data_quality_report_generation(self):
        """Test data quality report generation."""
        # Create data with quality issues
        problematic_data = pd.DataFrame({
            'col1': [1, 2, 2, 4, 5],  # Has duplicates
            'col2': [1.0, 2.0, np.nan, 4.0, 100.0],  # Has missing and outlier
            'col3': ['a', 'b', 'c', 'd', 'e'],
            'col4': [1, 1, 1, 1, 1]  # Low uniqueness
        })
        
        quality_report = DataQualityReport(problematic_data)
        report = quality_report.generate_report()
        
        # Check report structure
        self.assertIn('overview', report)
        self.assertIn('missing_data', report)
        self.assertIn('duplicates', report)
        self.assertIn('outliers', report)
        self.assertIn('data_types', report)
        self.assertIn('uniqueness', report)
        
        # Check specific findings
        self.assertEqual(report['overview']['shape'], (5, 4))
        self.assertGreater(report['missing_data']['col2']['count'], 0)
        self.assertGreater(report['duplicates']['total_duplicates'], 0)
        self.assertGreater(report['outliers']['col2']['count'], 0)
    
    def test_validate_and_clean_data(self):
        """Test data validation and cleaning function."""
        # Add some duplicates and outliers
        dirty_data = pd.concat([
            self.valid_tenant_data,
            self.valid_tenant_data.iloc[[0]],  # Duplicate row
        ]).reset_index(drop=True)
        
        # Add outlier
        dirty_data.loc[0, 'monthly_rent'] = 50000.0
        
        cleaned_data = validate_and_clean_data(
            dirty_data,
            TENANT_SCHEMA,
            remove_duplicates=True,
            handle_outliers=True
        )
        
        # Should have fewer rows (duplicates removed)
        self.assertLess(len(cleaned_data), len(dirty_data))
        
        # Should have no extreme outliers
        self.assertLess(cleaned_data['monthly_rent'].max(), 10000.0)
    
    def test_property_schema_validation(self):
        """Test property schema validation."""
        valid_property_data = pd.DataFrame({
            'property_id': ['P001', 'P002'],
            'address': ['123 Main St', '456 Oak Ave'],
            'beds': [2, 3],
            'baths': [2.0, 2.5],
            'sqft': [1200, 1500],
            'price': [300000.0, 450000.0],
            'property_type': ['Condo', 'Single Family'],
            'last_sold_date': pd.to_datetime(['2022-01-01', '2021-06-15']),
            'year_built': [1990, 2005]
        })
        
        errors = PROPERTY_SCHEMA.validate(valid_property_data)
        self.assertEqual(len(errors), 0)
        
        # Test with invalid property type
        invalid_property_data = valid_property_data.copy()
        invalid_property_data.loc[0, 'property_type'] = 'Invalid Type'
        
        errors = PROPERTY_SCHEMA.validate(invalid_property_data)
        self.assertIn('property_type', errors)
    
    def test_data_type_validation(self):
        """Test data type validation."""
        # Create test schema
        test_schema = DataSchema([
            FieldSchema("int_field", DataType.INTEGER),
            FieldSchema("float_field", DataType.FLOAT),
            FieldSchema("string_field", DataType.STRING),
            FieldSchema("datetime_field", DataType.DATETIME),
            FieldSchema("bool_field", DataType.BOOLEAN),
        ])
        
        # Valid data
        valid_data = pd.DataFrame({
            'int_field': [1, 2, 3],
            'float_field': [1.1, 2.2, 3.3],
            'string_field': ['a', 'b', 'c'],
            'datetime_field': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'bool_field': [True, False, True]
        })
        
        errors = test_schema.validate(valid_data)
        self.assertEqual(len(errors), 0)
        
        # Invalid data types
        invalid_data = pd.DataFrame({
            'int_field': [1.1, 2.2, 3.3],  # Float instead of int
            'float_field': ['a', 'b', 'c'],  # String instead of float
            'string_field': [1, 2, 3],  # Int instead of string
            'datetime_field': ['not a date', '2023-01-02', '2023-01-03'],  # Invalid datetime
            'bool_field': [1, 0, 1]  # Int instead of bool
        })
        
        # Convert datetime column properly for this test
        invalid_data['datetime_field'] = pd.to_datetime(invalid_data['datetime_field'], errors='coerce')
        
        errors = test_schema.validate(invalid_data)
        
        # Should have type errors for most fields
        self.assertGreater(len(errors), 0)

class TestDataQualityReport(unittest.TestCase):
    """Test cases for DataQualityReport class."""
    
    def setUp(self):
        """Set up test data with known quality issues."""
        self.test_data = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'missing_col': [1, np.nan, 3, np.nan, 5],
            'outlier_col': [1, 2, 3, 4, 1000],  # 1000 is an outlier
            'duplicate_col': [1, 1, 2, 2, 3],
            'unique_col': [1, 2, 3, 4, 5],
            'low_unique_col': [1, 1, 1, 1, 2],
            'corr_col1': [1, 2, 3, 4, 5],
            'corr_col2': [2, 4, 6, 8, 10]  # Perfectly correlated with corr_col1
        })
    
    def test_overview_analysis(self):
        """Test overview analysis."""
        report = DataQualityReport(self.test_data)
        overview = report._get_overview()
        
        self.assertEqual(overview['shape'], (5, 8))
        self.assertIn('memory_usage', overview)
        self.assertIn('total_cells', overview)
        self.assertEqual(overview['total_cells'], 40)
    
    def test_missing_data_analysis(self):
        """Test missing data analysis."""
        report = DataQualityReport(self.test_data)
        missing = report._analyze_missing_data()
        
        # Should detect missing values in missing_col
        self.assertEqual(missing['missing_col']['count'], 2)
        self.assertEqual(missing['missing_col']['percentage'], 40.0)
        
        # Should show no missing values in normal_col
        self.assertEqual(missing['normal_col']['count'], 0)
        self.assertEqual(missing['normal_col']['percentage'], 0.0)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        report = DataQualityReport(self.test_data)
        outliers = report._detect_outliers()
        
        # Should detect outlier in outlier_col
        self.assertGreater(outliers['outlier_col']['count'], 0)
        self.assertIn('lower_bound', outliers['outlier_col'])
        self.assertIn('upper_bound', outliers['outlier_col'])
    
    def test_correlation_analysis(self):
        """Test correlation analysis."""
        report = DataQualityReport(self.test_data)
        correlations = report._analyze_correlations()
        
        # Should detect high correlation between corr_col1 and corr_col2
        high_corrs = correlations.get('high_correlations', [])
        self.assertGreater(len(high_corrs), 0)
        
        # Find the correlation between our test columns
        corr_found = False
        for corr in high_corrs:
            if ('corr_col1' in [corr['column1'], corr['column2']] and 
                'corr_col2' in [corr['column1'], corr['column2']]):
                corr_found = True
                self.assertGreater(abs(corr['correlation']), 0.8)
        
        self.assertTrue(corr_found, "High correlation not detected between test columns")
    
    def test_uniqueness_analysis(self):
        """Test uniqueness analysis."""
        report = DataQualityReport(self.test_data)
        uniqueness = report._analyze_uniqueness()
        
        # unique_col should have 100% uniqueness
        self.assertEqual(uniqueness['unique_col']['uniqueness_percentage'], 100.0)
        
        # low_unique_col should have low uniqueness
        self.assertLess(uniqueness['low_unique_col']['uniqueness_percentage'], 50.0)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)