
import pandas as pd
import fitz  # PyMuPDF
import re
import numpy as np
import textdistance
from sentence_transformers import SentenceTransformer, util

CSV_INPUT_FILE = 'Policies-2025-08.csv'
PDF_INPUT_FILE = 'Cloud Security Standard.pdf'
CSV_OUTPUT_FILE = 'mapped_policies_output_huggingface_lexical.csv'
CONFIDENCE_THRESHOLD = 25
SEMANTIC_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(csv_path):
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        rename_map = {
            'Policy Id': 'Policy ID',
            'Severity': 'Severity level'
        }
        df.rename(columns=rename_map, inplace=True)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def perform_mapping():
    print("Starting mapping using HuggingFace + Lexical similarity...")

    policies_df = load_and_prepare_data(CSV_INPUT_FILE)
    if policies_df is None:
        return

    pdf_policies = {
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

    policy_names = policies_df['Policy Name'].fillna('').tolist()
    pdf_titles = list(pdf_policies.keys())
    pdf_descriptions = list(pdf_policies.values())

    model = SentenceTransformer('all-MiniLM-L6-v2')
    pdf_embeddings = model.encode(pdf_descriptions, convert_to_tensor=True)

    results = []
    print(f"Mapping {len(policy_names)} policies...")

    for idx, policy_text in enumerate(policy_names):
        preprocessed = preprocess_text(policy_text)
        policy_embedding = model.encode(preprocessed, convert_to_tensor=True)
        semantic_scores = util.cos_sim(policy_embedding, pdf_embeddings)[0].cpu().numpy()
        lexical_scores = np.array([
            textdistance.jaro_winkler(preprocessed, preprocess_text(desc)) for desc in pdf_descriptions
        ])
        combined_scores = (semantic_scores * SEMANTIC_WEIGHT) + (lexical_scores * LEXICAL_WEIGHT)
        best_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_idx]
        confidence_score = round(float(best_score) * 100)

        if confidence_score >= CONFIDENCE_THRESHOLD:
            mapped_section = pdf_titles[best_idx]
            rationale = f"Matched with '{mapped_section}' using combined semantic and lexical similarity."
        else:
            mapped_section = "No confident match"
            rationale = "Low confidence match."

        results.append({
            'Policy ID': policies_df.loc[idx, 'Policy ID'],
            'Policy Name': policy_text,
            'Mapped Section': mapped_section,
            'Confidence Score': confidence_score,
            'Rationale': rationale
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"Mapping completed. Results saved to {CSV_OUTPUT_FILE}.")

perform_mapping()
