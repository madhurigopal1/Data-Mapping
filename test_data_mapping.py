# -*- coding: utf-8 -*-
"""
Unit Tests for Data Mapping Solution
Tests for config, utils, and core functionality
"""

import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from io import StringIO
import sys

# Import modules to test
from config import (
    get_accuracy_level, get_config, ACCURACY_LEVELS, 
    PDF_POLICIES, HYBRID_MAPPER_CONFIG
)
from utils import (
    preprocess_text, clean_text, lexical_overlap_score,
    calculate_similarity_metrics, validate_text_input,
    validate_input_dataframe, format_confidence_score,
    create_output_record, calculate_mapping_statistics,
    read_csv_safe, write_csv_safe
)


class TestConfigModule(unittest.TestCase):
    """Test cases for config.py module"""
    
    def test_accuracy_level_high(self):
        """Test accuracy level classification for high scores"""
        self.assertEqual(get_accuracy_level(85), 'High')
        self.assertEqual(get_accuracy_level(70), 'High')
        self.assertEqual(get_accuracy_level(100), 'High')
    
    def test_accuracy_level_medium(self):
        """Test accuracy level classification for medium scores"""
        self.assertEqual(get_accuracy_level(50), 'Medium')
        self.assertEqual(get_accuracy_level(25), 'Medium')
        self.assertEqual(get_accuracy_level(69), 'Medium')
    
    def test_accuracy_level_low(self):
        """Test accuracy level classification for low scores"""
        self.assertEqual(get_accuracy_level(0), 'Low')
        self.assertEqual(get_accuracy_level(10), 'Low')
        self.assertEqual(get_accuracy_level(24), 'Low')
    
    def test_get_config_hybrid(self):
        """Test retrieval of hybrid mapper configuration"""
        config = get_config('hybrid')
        self.assertIsNotNone(config)
        self.assertIn('input_file', config)
        self.assertIn('semantic_weight', config)
        self.assertEqual(config['semantic_weight'], 0.8)
    
    def test_get_config_bert(self):
        """Test retrieval of BERT configuration"""
        config = get_config('bert')
        self.assertIsNotNone(config)
        self.assertIn('model_name', config)
        self.assertEqual(config['model_name'], 'bert-base-uncased')
    
    def test_get_config_api(self):
        """Test retrieval of API configuration"""
        config = get_config('api')
        self.assertIsNotNone(config)
        self.assertIn('host', config)
        self.assertIn('port', config)
        self.assertEqual(config['port'], 8000)
    
    def test_get_config_all(self):
        """Test retrieval of all configurations"""
        config = get_config('all')
        self.assertIn('hybrid', config)
        self.assertIn('bert', config)
        self.assertIn('api', config)
    
    def test_pdf_policies_exist(self):
        """Test that PDF policies are defined"""
        self.assertGreater(len(PDF_POLICIES), 0)
        self.assertIn('1.0 Overview', PDF_POLICIES)
        self.assertIn('4.7 Identity and Access Management', PDF_POLICIES)


class TestUtilsPreprocessing(unittest.TestCase):
    """Test cases for text preprocessing functions"""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing"""
        text = "HELLO WORLD"
        result = preprocess_text(text)
        self.assertEqual(result, "hello world")
    
    def test_preprocess_text_special_chars(self):
        """Test preprocessing with special characters"""
        text = "Hello, World! @#$"
        result = preprocess_text(text)
        self.assertEqual(result, "hello world")
    
    def test_preprocess_text_whitespace(self):
        """Test preprocessing with extra whitespace"""
        text = "Hello    World"
        result = preprocess_text(text)
        self.assertEqual(result, "hello world")
    
    def test_preprocess_text_empty(self):
        """Test preprocessing with empty string"""
        result = preprocess_text("")
        self.assertEqual(result, "")
    
    def test_preprocess_text_non_string(self):
        """Test preprocessing with non-string input"""
        result = preprocess_text(None)
        self.assertEqual(result, "")
        result = preprocess_text(123)
        self.assertEqual(result, "")
    
    def test_clean_text_basic(self):
        """Test basic text cleaning"""
        text = "Hello, World!"
        result = clean_text(text)
        self.assertIn("Hello", result)
        self.assertIn("World", result)
    
    def test_clean_text_special_chars(self):
        """Test cleaning with special characters"""
        text = "test@example.com"
        result = clean_text(text)
        self.assertNotIn("@", result)
        self.assertNotIn(".", result)


class TestUtilsSimilarity(unittest.TestCase):
    """Test cases for similarity scoring functions"""
    
    def test_lexical_overlap_identical(self):
        """Test lexical overlap with identical texts"""
        text = "hello world"
        score = lexical_overlap_score(text, text)
        self.assertEqual(score, 1.0)
    
    def test_lexical_overlap_different(self):
        """Test lexical overlap with completely different texts"""
        score = lexical_overlap_score("hello world", "foo bar")
        self.assertEqual(score, 0.0)
    
    def test_lexical_overlap_partial(self):
        """Test lexical overlap with partial overlap"""
        score = lexical_overlap_score("hello world", "hello foo")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
    
    def test_lexical_overlap_empty(self):
        """Test lexical overlap with empty strings"""
        score = lexical_overlap_score("", "hello")
        self.assertEqual(score, 0.0)
    
    def test_calculate_similarity_metrics(self):
        """Test calculation of similarity metrics"""
        text1 = "Employee must authenticate"
        text2 = "Multi-factor authentication required"
        metrics = calculate_similarity_metrics(text1, text2)
        
        self.assertIn('lexical', metrics)
        self.assertIn('text1_length', metrics)
        self.assertIn('text2_length', metrics)
        self.assertGreaterEqual(metrics['lexical'], 0.0)
        self.assertLessEqual(metrics['lexical'], 1.0)


class TestUtilsValidation(unittest.TestCase):
    """Test cases for validation functions"""
    
    def test_validate_text_input_valid(self):
        """Test validation with valid text"""
        is_valid, msg = validate_text_input("hello world")
        self.assertTrue(is_valid)
    
    def test_validate_text_input_empty(self):
        """Test validation with empty string"""
        is_valid, msg = validate_text_input("")
        self.assertFalse(is_valid)
    
    def test_validate_text_input_non_string(self):
        """Test validation with non-string input"""
        is_valid, msg = validate_text_input(123)
        self.assertFalse(is_valid)
    
    def test_validate_input_dataframe_valid(self):
        """Test DataFrame validation with valid data"""
        df = pd.DataFrame({
            'Policy Name': ['Policy A', 'Policy B'],
            'Description': ['Desc A', 'Desc B']
        })
        is_valid, msg = validate_input_dataframe(
            df, 
            ['Policy Name', 'Description']
        )
        self.assertTrue(is_valid)
    
    def test_validate_input_dataframe_missing_column(self):
        """Test DataFrame validation with missing column"""
        df = pd.DataFrame({'Policy Name': ['Policy A']})
        is_valid, msg = validate_input_dataframe(
            df,
            ['Policy Name', 'Missing']
        )
        self.assertFalse(is_valid)
        self.assertIn('Missing', msg)
    
    def test_validate_input_dataframe_empty(self):
        """Test DataFrame validation with empty DataFrame"""
        df = pd.DataFrame()
        is_valid, msg = validate_input_dataframe(df, ['any_column'])
        self.assertFalse(is_valid)


class TestUtilsFormatting(unittest.TestCase):
    """Test cases for output formatting functions"""
    
    def test_format_confidence_score_percent(self):
        """Test formatting confidence score as percentage"""
        result = format_confidence_score(0.85, as_percent=True)
        self.assertEqual(result, "85.0%")
    
    def test_format_confidence_score_absolute(self):
        """Test formatting confidence score as absolute value"""
        result = format_confidence_score(85, as_percent=False)
        self.assertEqual(result, "85.0%")
    
    def test_create_output_record(self):
        """Test creation of standardized output record"""
        record = create_output_record(
            policy_id='P-001',
            policy_name='Data Privacy',
            mapped_section='Section 4.7',
            confidence_score=85,
            rationale='High semantic match'
        )
        
        self.assertEqual(record['Policy ID'], 'P-001')
        self.assertEqual(record['Policy Name'], 'Data Privacy')
        self.assertEqual(record['Mapped Section'], 'Section 4.7')
        self.assertEqual(record['Confidence Score'], 85)
        self.assertEqual(record['Accuracy'], 'High')
        self.assertEqual(record['Rationale'], 'High semantic match')


class TestUtilsStatistics(unittest.TestCase):
    """Test cases for statistics calculation"""
    
    def test_calculate_mapping_statistics(self):
        """Test calculation of mapping statistics"""
        df = pd.DataFrame({
            'Policy Name': ['P1', 'P2', 'P3', 'P4'],
            'Confidence Score': [85, 50, 30, 75],
            'Accuracy': ['High', 'Medium', 'Low', 'High']
        })
        
        stats = calculate_mapping_statistics(df)
        
        self.assertEqual(stats['total_records'], 4)
        self.assertAlmostEqual(stats['average_confidence'], 60.0)
        self.assertEqual(stats['min_confidence'], 30)
        self.assertEqual(stats['max_confidence'], 85)
        self.assertEqual(stats['high_accuracy'], 2)
        self.assertEqual(stats['medium_accuracy'], 1)
        self.assertEqual(stats['low_accuracy'], 1)
    
    def test_calculate_mapping_statistics_empty(self):
        """Test statistics calculation with empty DataFrame"""
        df = pd.DataFrame()
        stats = calculate_mapping_statistics(df)
        self.assertEqual(stats, {})


class TestUtilsFileOperations(unittest.TestCase):
    """Test cases for file operations"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_write_csv_safe(self):
        """Test safe CSV writing"""
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob'],
            'Score': [85, 92]
        })
        
        filepath = os.path.join(self.test_dir, 'test.csv')
        result = write_csv_safe(df, filepath)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(filepath))
    
    def test_read_csv_safe(self):
        """Test safe CSV reading"""
        # Create a test CSV file
        filepath = os.path.join(self.test_dir, 'test.csv')
        df_original = pd.DataFrame({
            'Name': ['Alice', 'Bob'],
            'Score': [85, 92]
        })
        df_original.to_csv(filepath, index=False)
        
        # Read it back
        df_read = read_csv_safe(filepath)
        
        self.assertIsNotNone(df_read)
        self.assertEqual(len(df_read), 2)
        self.assertListEqual(list(df_read['Name']), ['Alice', 'Bob'])
    
    def test_read_csv_safe_nonexistent(self):
        """Test reading non-existent CSV file"""
        df = read_csv_safe('/nonexistent/file.csv')
        self.assertIsNone(df)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components"""
    
    def test_end_to_end_mapping_record(self):
        """Test creating a complete mapping record"""
        text1 = "Employee authentication requirement"
        text2 = "Multi-factor authentication mandatory"
        
        metrics = calculate_similarity_metrics(text1, text2)
        confidence = int(metrics['lexical'] * 100)
        
        record = create_output_record(
            policy_id='P-001',
            policy_name='Authentication',
            mapped_section='4.8 Security Principals',
            confidence_score=confidence,
            rationale=f"Lexical similarity: {metrics['lexical']:.2f}"
        )
        
        self.assertIsNotNone(record)
        self.assertEqual(record['Policy ID'], 'P-001')
        self.assertGreaterEqual(record['Confidence Score'], 0)
        self.assertLessEqual(record['Confidence Score'], 100)
        self.assertIn(record['Accuracy'], ['High', 'Medium', 'Low'])


def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigModule))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsSimilarity))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsFormatting))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsFileOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
