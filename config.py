# config.py
from dataclasses import dataclass


@dataclass
class Config:
    # Hugging Face dataset
    HF_DATASET_NAME: str = "bigbio/biored"
    HF_CONFIG_NAME: str = "biored_bigbio_kb"

    # Baseline models
    BIOBERT_MODEL: str = "dmis-lab/biobert-base-cased-v1.1"
    PUBMEDBERT_MODEL: str = (
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )

    # Output dirs
    OUTPUT_DIR_BIOBERT: str = "./outputs/biobert"
    OUTPUT_DIR_PUBMEDBERT: str = "./outputs/pubmedbert"

    # For negative sampling 
    MAX_NEGATIVE_PAIRS_PER_DOC: int = 5

    # Random seed
    SEED: int = 42


cfg = Config()
