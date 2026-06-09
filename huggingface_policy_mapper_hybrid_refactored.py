# -*- coding: utf-8 -*-
"""
Refactored Hybrid Policy Mapper
Uses centralized config and utilities for better maintainability
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from config import HYBRID_MAPPER_CONFIG, PDF_POLICIES, get_accuracy_level
from utils import (
    setup_logger, preprocess_text, lexical_overlap_score,
    validate_input_dataframe, write_csv_safe,
    calculate_mapping_statistics, print_statistics
)

# Setup logger
logger = setup_logger(__name__)


def get_config_values():
    """Get configuration values from centralized config module"""
    config = HYBRID_MAPPER_CONFIG
    return {
        'input_file': config['input_file'],
        'output_file': config['output_file'],
        'confidence_threshold': config['confidence_threshold'],
        'model_name': config['model_name'],
        'semantic_weight': config['semantic_weight'],
        'lexical_weight': config['lexical_weight'],
        'device': config['device'],
    }


def load_models(model_name: str, device: str):
    """Load BERT tokenizer and model
    
    Args:
        model_name: Name of the BERT model
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Tuple of (tokenizer, model)
    """
    try:
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def perform_mapping():
    """Perform policy mapping using hybrid approach
    
    Combines semantic matching (BERT embeddings) with lexical matching
    (keyword overlap) to map policies from input CSV to PDF sections.
    """
    # Get configuration
    config = get_config_values()
    
    logger.info("="*60)
    logger.info("HYBRID POLICY MAPPING - STARTING")
    logger.info("="*60)
    logger.info(f"Input file: {config['input_file']}")
    logger.info(f"Output file: {config['output_file']}")
    
    # Read input CSV
    logger.info("Reading input CSV...")
    df = pd.read_csv(config['input_file'], low_memory=False)
    
    # Validate input
    is_valid, msg = validate_input_dataframe(df, ['Policy Name'])
    if not is_valid:
        logger.error(f"Input validation failed: {msg}")
        return
    
    logger.info(f"Read {len(df)} policies from input file")
    
    # Rename columns if needed
    if 'Policy Id' in df.columns:
        df.rename(columns={'Policy Id': 'Policy ID'}, inplace=True)
    if 'Severity' in df.columns:
        df.rename(columns={'Severity': 'Severity level'}, inplace=True)
    
    policy_names = df['Policy Name'].fillna('').tolist()
    
    # Load models
    logger.info("Loading BERT models...")
    tokenizer, model = load_models(
        config['model_name'],
        config['device']
    )
    
    # Prepare PDF descriptions
    logger.info("Preparing PDF policy embeddings...")
    pdf_descriptions = list(PDF_POLICIES.values())
    pdf_titles = list(PDF_POLICIES.keys())
    processed_pdf = [preprocess_text(desc) for desc in pdf_descriptions]
    
    # Generate PDF embeddings
    with torch.no_grad():
        pdf_inputs = tokenizer(
            pdf_descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        pdf_outputs = model(**pdf_inputs)
        pdf_embeddings = pdf_outputs.last_hidden_state[:, 0, :]
    
    logger.info(f"Generated embeddings for {len(pdf_titles)} PDF sections")
    
    # Process each policy
    results = []
    logger.info("Processing policies...")
    
    for idx, policy_text in enumerate(policy_names):
        if (idx + 1) % max(1, len(policy_names) // 10) == 0:
            logger.info(f"Progress: {idx + 1}/{len(policy_names)}")
        
        preprocessed = preprocess_text(policy_text)
        
        # Generate policy embedding
        with torch.no_grad():
            inputs = tokenizer(
                preprocessed,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            outputs = model(**inputs)
            policy_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Calculate similarity scores
        semantic_scores = torch.nn.functional.cosine_similarity(
            policy_embedding,
            pdf_embeddings
        ).numpy()
        
        lexical_scores = np.array([
            lexical_overlap_score(preprocessed, p)
            for p in processed_pdf
        ])
        
        # Combine scores
        combined_scores = (
            (semantic_scores * config['semantic_weight']) +
            (lexical_scores * config['lexical_weight'])
        )
        
        # Find best match
        best_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_idx]
        confidence_score = int(best_score * 100)
        
        # Create result record
        if confidence_score >= config['confidence_threshold']:
            mapped_section = pdf_titles[best_idx]
            accuracy = get_accuracy_level(confidence_score)
            rationale = (
                f"Matched with '{mapped_section}' using hybrid "
                f"semantic ({semantic_scores[best_idx]:.2f}) and "
                f"lexical ({lexical_scores[best_idx]:.2f}) similarity."
            )
        else:
            mapped_section = "No confident match"
            accuracy = "Low"
            rationale = "Confidence score below threshold"
        
        policy_id = (
            df.loc[idx, 'Policy ID']
            if 'Policy ID' in df.columns
            else f"P-{idx+1}"
        )
        
        results.append({
            'Policy ID': policy_id,
            'Policy Name': policy_text,
            'Mapped Section': mapped_section,
            'Confidence Score': confidence_score,
            'Accuracy': accuracy,
            'Rationale': rationale
        })
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Write output
    logger.info("Writing output file...")
    success = write_csv_safe(output_df, config['output_file'])
    
    if success:
        logger.info("Mapping completed successfully!")
        logger.info(f"Output saved to: {config['output_file']}")
        
        # Print statistics
        stats = calculate_mapping_statistics(output_df)
        print_statistics(stats)
    else:
        logger.error("Failed to write output file")
    
    logger.info("="*60)
    logger.info("MAPPING COMPLETE")
    logger.info("="*60)


if __name__ == '__main__':
    try:
        perform_mapping()
    except Exception as e:
        logger.error(f"Mapping failed with error: {str(e)}", exc_info=True)
        raise
