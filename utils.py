# -*- coding: utf-8 -*-
"""
Utility Functions for Data Mapping Solution
Common helper functions used across modules
"""

import logging
import re
import os
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from config import LOGGING_CONFIG, get_accuracy_level

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str) -> logging.Logger:
    """
    Setup and return a configured logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(LOGGING_CONFIG['level'])
        
        # File handler
        file_handler = logging.FileHandler(LOGGING_CONFIG['log_file'])
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(LOGGING_CONFIG['format'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


logger = setup_logger(__name__)

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text: str, lowercase: bool = True, remove_special: bool = True) -> str:
    """
    Preprocess text by cleaning and normalizing.
    
    Args:
        text: Input text to preprocess
        lowercase: Convert to lowercase
        remove_special: Remove special characters
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    if lowercase:
        text = text.lower()
    
    if remove_special:
        text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and extra whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove special characters except spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================================
# SIMILARITY SCORING
# ============================================================================

def lexical_overlap_score(text1: str, text2: str) -> float:
    """
    Calculate lexical overlap (Jaccard similarity) between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    set1 = set(text1.split())
    set2 = set(text2.split())
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def calculate_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """
    Calculate multiple similarity metrics between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Dictionary with different similarity metrics
    """
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    lexical_score = lexical_overlap_score(processed_text1, processed_text2)
    
    return {
        'lexical': lexical_score,
        'text1_length': len(processed_text1.split()),
        'text2_length': len(processed_text2.split()),
    }


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def read_csv_safe(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Safely read CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional pandas read_csv arguments
        
    Returns:
        DataFrame or None if error
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath, **kwargs)
        logger.info(f"Successfully read {len(df)} rows from {filepath}")
        return df
    
    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        return None


def write_csv_safe(df: pd.DataFrame, filepath: str, **kwargs) -> bool:
    """
    Safely write DataFrame to CSV file.
    
    Args:
        df: DataFrame to write
        filepath: Output file path
        **kwargs: Additional pandas to_csv arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False, **kwargs)
        logger.info(f"Successfully wrote {len(df)} rows to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {str(e)}")
        return False


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_input_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, str]:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, "Validation passed"


def validate_text_input(text: str, min_length: int = 1) -> Tuple[bool, str]:
    """
    Validate text input.
    
    Args:
        text: Text to validate
        min_length: Minimum required length
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not isinstance(text, str):
        return False, "Input must be a string"
    
    if len(text.strip()) < min_length:
        return False, f"Text must be at least {min_length} characters"
    
    return True, "Validation passed"


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_confidence_score(score: float, as_percent: bool = True) -> str:
    """
    Format confidence score for display.
    
    Args:
        score: Score between 0 and 1 (or 0-100 if as_percent=False)
        as_percent: If True, score is 0-1; if False, score is 0-100
        
    Returns:
        Formatted string
    """
    if as_percent:
        percentage = score * 100
    else:
        percentage = score
    
    return f"{percentage:.1f}%"


def create_output_record(
    policy_id: str,
    policy_name: str,
    mapped_section: str,
    confidence_score: float,
    rationale: str = ""
) -> Dict[str, Any]:
    """
    Create a standardized output record.
    
    Args:
        policy_id: Unique policy identifier
        policy_name: Name of the policy
        mapped_section: Target section mapped to
        confidence_score: Confidence score (0-100)
        rationale: Explanation of the mapping
        
    Returns:
        Dictionary with formatted output record
    """
    return {
        'Policy ID': policy_id,
        'Policy Name': policy_name,
        'Mapped Section': mapped_section,
        'Confidence Score': confidence_score,
        'Accuracy': get_accuracy_level(confidence_score),
        'Rationale': rationale,
    }


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

def calculate_mapping_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics from mapping results.
    
    Args:
        df: DataFrame with mapping results containing 'Confidence Score' column
        
    Returns:
        Dictionary with statistics
    """
    if df is None or df.empty or 'Confidence Score' not in df.columns:
        return {}
    
    scores = df['Confidence Score'].astype(float)
    
    return {
        'total_records': len(df),
        'average_confidence': scores.mean(),
        'min_confidence': scores.min(),
        'max_confidence': scores.max(),
        'median_confidence': scores.median(),
        'std_dev': scores.std(),
        'high_accuracy': len(df[df['Accuracy'] == 'High']),
        'medium_accuracy': len(df[df['Accuracy'] == 'Medium']),
        'low_accuracy': len(df[df['Accuracy'] == 'Low']),
    }


def print_statistics(stats: Dict[str, Any]) -> None:
    """
    Pretty print mapping statistics.
    
    Args:
        stats: Statistics dictionary
    """
    if not stats:
        logger.warning("No statistics to display")
        return
    
    logger.info("=" * 60)
    logger.info("MAPPING STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total Records: {stats.get('total_records', 'N/A')}")
    logger.info(f"Average Confidence: {stats.get('average_confidence', 0):.1f}%")
    logger.info(f"Min Confidence: {stats.get('min_confidence', 0):.1f}%")
    logger.info(f"Max Confidence: {stats.get('max_confidence', 0):.1f}%")
    logger.info(f"Median Confidence: {stats.get('median_confidence', 0):.1f}%")
    logger.info(f"Standard Deviation: {stats.get('std_dev', 0):.1f}%")
    logger.info(f"High Accuracy: {stats.get('high_accuracy', 0)}")
    logger.info(f"Medium Accuracy: {stats.get('medium_accuracy', 0)}")
    logger.info(f"Low Accuracy: {stats.get('low_accuracy', 0)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    # Test utilities
    test_text1 = "Employee must authenticate with MFA"
    test_text2 = "Multi-factor authentication is required for all employees"
    
    print(f"Text 1: {test_text1}")
    print(f"Text 2: {test_text2}")
    print(f"Lexical Overlap: {lexical_overlap_score(test_text1, test_text2):.2f}")
    print(f"Similarity Metrics: {calculate_similarity_metrics(test_text1, test_text2)}")