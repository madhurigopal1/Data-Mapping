# -*- coding: utf-8 -*-
"""
Configuration Management for Data Mapping Solution
Centralized settings for all mapping modules
"""

import os
from typing import Dict, Any

# ============================================================================
# FILE PATHS
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_CSV_FILE = os.path.join(DATA_DIR, 'Policies-2025-08.csv')
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, 'mapped_policies_output.csv')

# ============================================================================
# HYBRID MAPPER CONFIGURATION
# ============================================================================

HYBRID_MAPPER_CONFIG = {
    'input_file': INPUT_CSV_FILE,
    'output_file': OUTPUT_CSV_FILE,
    'confidence_threshold': 25,  # Minimum confidence to report a match (0-100)
    'model_name': 'bert-base-uncased',
    'semantic_weight': 0.8,  # Weight for semantic similarity (0-1)
    'lexical_weight': 0.2,   # Weight for lexical similarity (0-1)
    'device': 'cpu',  # 'cpu' or 'cuda' for GPU
}

# ============================================================================
# BERT MODEL CONFIGURATION
# ============================================================================

BERT_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 512,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 3,
    'device': 'cpu',
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8000,
    'reload': True,
    'workers': 1,
    'log_level': 'info',
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

ACCURACY_LEVELS = {
    'high': {'min': 70, 'label': 'High'},
    'medium': {'min': 25, 'label': 'Medium'},
    'low': {'min': 0, 'label': 'Low'},
}

# ============================================================================
# PDF POLICIES (Reference Document)
# ============================================================================

PDF_POLICIES = {
    '1.0 Overview': 'General overview of the cloud security standard.',
    '2.0 Scope': 'Scope of the standard, including IaaS, PaaS, and development stages.',
    '2.1 Shared Responsibility Model': 'Division of responsibilities between CSP and consumer.',
    '3.0 Purpose': 'Purpose and vendor-neutral security requirements.',
    '4.0 Requirements': 'Overall security requirements.',
    '4.1 Roles and Responsibilities': 'Defines roles like CSO, Security Governance, and Business Units.',
    '4.2 Compliance Considerations': 'External regulations and client contracts.',
    '4.3 Oversight and Accountability': 'Proactive engagement with security teams and conformance.',
    '4.4 Approved CSPs': 'Rules for using Cognizant-approved Cloud Service Providers.',
    '4.5 Non-Production Environments': 'Criteria and controls for non-production environments like sandbox and dev.',
    '4.6 Production Environments': 'Criteria and controls for production networks containing customer data.',
    '4.7 Identity and Access Management': 'IAM practices, compliance with Authentication, Authorization, and Assurance standards.',
    '4.8 Security Principals for CSP Administration': 'Authentication federation, break-glass accounts, and MFA.',
    '4.9 Management Plane Access': 'Controls for web portals, consoles, APIs, and administrative functions.',
    '4.10 Resource Access': 'Logical access controls for cloud resources.',
    '4.11 Privileged Identity Management': 'PIM solutions for managing and monitoring privileged accounts.',
    '4.12 Production Access Management': 'Break-glass processes for production access, no persistent access.',
    '4.13 Technology Approval and Third-Party Components': 'Approval for third-party and marketplace components.',
    '4.14 Logging': 'Requirements for logging, encryption of logs, and SIEM availability.',
    '4.15 SIEM Integration': 'Ingestion of mandatory logging events into the corporate SIEM.',
    '4.16 Auditing': 'Periodic recertification of IAM accounts and identities.',
    '4.17 Asset Tags': 'Requirements for tagging all cloud resources for identification.',
    '4.18 Architecture': 'Network architecture, external connections, on-premises connectivity, and segmentation.',
    '4.19 Cloud Work Planning': 'Design and planning for solutions deployed to CSPs.',
    '4.20 Deployment': 'Change control, automated deployment, and response/recovery planning.',
    '5.0 Enforcement': 'Enforcement of the cloud security strategy.',
    '6.0 Exceptions': 'Process for handling exceptions to the standard.'
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': os.path.join(BASE_DIR, 'data_mapping.log'),
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_accuracy_level(confidence_score: float) -> str:
    """
    Determine accuracy level based on confidence score.
    
    Args:
        confidence_score: Confidence score between 0 and 100
        
    Returns:
        Accuracy level string ('High', 'Medium', 'Low')
    """
    for level in ['high', 'medium', 'low']:
        if confidence_score >= ACCURACY_LEVELS[level]['min']:
            return ACCURACY_LEVELS[level]['label']
    return 'Low'


def get_config(section: str = None) -> Any:
    """
    Retrieve configuration section or all configurations.
    
    Args:
        section: Section name ('hybrid', 'bert', 'api') or None for all
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        'hybrid': HYBRID_MAPPER_CONFIG,
        'bert': BERT_CONFIG,
        'api': API_CONFIG,
        'all': {
            'hybrid': HYBRID_MAPPER_CONFIG,
            'bert': BERT_CONFIG,
            'api': API_CONFIG,
        }
    }
    
    if section is None or section == 'all':
        return config_map.get('all', {})
    
    return config_map.get(section, {})


if __name__ == '__main__':
    # Print all configurations
    import json
    print(json.dumps(get_config(), indent=2))