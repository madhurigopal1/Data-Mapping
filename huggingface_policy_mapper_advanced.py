
import pandas as pd
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

CSV_INPUT_FILE = 'Policies-2025-08.csv'
CSV_OUTPUT_FILE = 'mapped_policies_output_advanced.csv'
CONFIDENCE_THRESHOLD = 25
MODEL_NAME = 'bert-base-uncased'

SEMANTIC_WEIGHT = 0.3
LEXICAL_WEIGHT = 0.2
EMBEDDING_WEIGHT = 0.2
FEEDBACK_WEIGHTS = {
    'accuracy': 0.1,
    'confidence': 0.05,
    'correlation': 0.05,
    'positivity': 0.05,
    'negativity': 0.05
}

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

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lexical_overlap_score(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def dummy_feedback_scores():
    return {
        'accuracy': 0.9,
        'confidence': 0.85,
        'correlation': 0.8,
        'positivity': 0.75,
        'negativity': 0.2
    }

def perform_mapping():
    df = pd.read_csv(CSV_INPUT_FILE, low_memory=False)
    df.rename(columns={'Policy Id': 'Policy ID', 'Severity': 'Severity level'}, inplace=True)
    policy_names = df['Policy Name'].fillna('').tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    pdf_descriptions = list(pdf_policies.values())
    pdf_titles = list(pdf_policies.keys())
    processed_pdf = [preprocess_text(desc) for desc in pdf_descriptions]

    with torch.no_grad():
        pdf_inputs = tokenizer(pdf_descriptions, padding=True, truncation=True, return_tensors="pt")
        pdf_outputs = model(**pdf_inputs)
        pdf_embeddings = pdf_outputs.last_hidden_state[:, 0, :]

    results = []

    for idx, policy_text in enumerate(policy_names):
        preprocessed = preprocess_text(policy_text)
        with torch.no_grad():
            inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            policy_embedding = outputs.last_hidden_state[:, 0, :]

        semantic_scores = torch.nn.functional.cosine_similarity(policy_embedding, pdf_embeddings).numpy()
        lexical_scores = np.array([lexical_overlap_score(preprocessed, p) for p in processed_pdf])
        embedding_scores = semantic_scores.copy()

        feedback = dummy_feedback_scores()
        feedback_score = sum([feedback[k] * FEEDBACK_WEIGHTS[k] for k in FEEDBACK_WEIGHTS])

        combined_scores = (
            semantic_scores * SEMANTIC_WEIGHT +
            lexical_scores * LEXICAL_WEIGHT +
            embedding_scores * EMBEDDING_WEIGHT +
            feedback_score
        )

        best_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_idx]
        confidence_score = round(best_score * 100)

        if confidence_score >= CONFIDENCE_THRESHOLD:
            mapped_section = pdf_titles[best_idx]
            rationale = f"Matched with '{mapped_section}' using combined semantic, lexical, embedding, and feedback scores."
        else:
            mapped_section = "No confident match"
            rationale = "Low confidence match."

        results.append({
            'Policy ID': df.loc[idx, 'Policy ID'],
            'Policy Name': policy_text,
            'Mapped Section': mapped_section,
            'Confidence Score': confidence_score,
            'Rationale': rationale
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"Mapping completed. Results saved to {CSV_OUTPUT_FILE}.")

if __name__ == '__main__':
    perform_mapping()
