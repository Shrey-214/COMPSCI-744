# COMPSCI-744
Final Project for Text Retrieval and Its Applications in Biomedicine.

# BioLinkSense: Biomedical Relation Extraction & Gemini Summarization

This project builds a biomedical relation extraction system on the BioRED (BigBio) dataset, compares two transformer backbones (PubMedBERT and BioBERT), and then uses Google Gemini to generate abstractive summaries of biomedical documents.

The pipeline is:

Load BioRED documents (either a small demo subset or the full dataset).
Convert each document into entity–entity relation instances (positive and sampled negative pairs).
Train a supervised text classification model (PubMedBERT or BioBERT) to predict relation types.
Evaluate models and report macro precision/recall/F1 and a detailed classification report.
Use Gemini 2.5 Flash to generate document summaries and compute ROUGE-L scores.

Repository Structure

At the top level of the repo you should see:



├── biolinksense.py          # Main entry script with CLI (train / eval / summarize)
|
├── config.py                # Configuration (model names, dataset ids, training hyperparams)

├── data_utils.py            # Loading BioRED + building relation-level dataset

├── model_utils.py           # Training & evaluation helpers for transformers

├── llm_utils.py             # Gemini API client + ROUGE evaluation

├── utils.py                 # General utilities (seeding, logging helpers, etc.)

├── requirements.txt         # Python dependencies

├── demo_data/               # SMALL sampled subset of BioRED 

│   ├── biored_docs_train.jsonl

│   ├── biored_docs_validation.jsonl

│   └── biored_docs_test.jsonl

├── outputs/                 # Model checkpoints & metrics (created after running)

│   ├── pubmedbert/          # PubMedBERT checkpoints + metrics

│   ├── biobert/             # BioBERT checkpoints + metrics

│   └── gemini_demo_summaries.jsonl  # Example Gemini summaries (demo run)

└── README.md                # This file



Demo data: the three JSONL files in demo_data/ are a small random sample of the full BioRED dataset (≈20 docs per split) 

Full dataset: when demo_data/ is not present, the code falls back to downloading bigbio/biored (config: biored_bigbio_kb) from HuggingFace.

Task & Expected Output
Task

Input: a pair of biomedical entities (e.g., gene–disease, drug–drug) plus their abstract context.

Output: a relation type for this entity pair (e.g., Association, Positive_Correlation, Drug_Interaction, or no_relation).

Each document in BioRED is converted to many training instances, including both annotated positive relations and sampled negative pairs.

Model Outputs

Running the code produces:

Trained model checkpoints in ./outputs/<backbone>/ (e.g. pubmedbert, biobert).

Validation and test metrics printed to the console and logged to ./outputs/<backbone>/eval_tmp/.

A classification report on the test set (precision/recall/F1 per label + overall accuracy).

Gemini summaries for a subset of documents in ./outputs/gemini_demo_summaries.jsonl.

On the full BioRED relation dataset, the best results were:

PubMedBERT: macro F1 ≈ 0.53, accuracy ≈ 0.68. 

Complete data outputs

BioBERT: macro F1 ≈ 0.49, accuracy ≈ 0.65. 

Installation

# Install dependencies
pip install -r requirements.txt

How to Run the Code on the Demo Dataset 

1. Train PubMedBERT on the demo data
python biolinksense.py --mode train --backbone pubmedbert --device cpu

This will:

Load ./demo_data/*.jsonl.
Build a relation-level dataset.
Fine-tune PubMedBERT on the demo train/validation splits.
Save the model and tokenizer to ./outputs/pubmedbert/.

2. Evaluate PubMedBERT on the demo test split
python biolinksense.py --mode eval --backbone pubmedbert --device cpu


Expected behavior:

Loads the trained model from ./outputs/pubmedbert/.
Evaluates on the demo test split.
Prints test macro precision / recall / F1 and a full classification report.
These printed metrics are the main expected output for the project.

3. Train BioBERT on the demo data:

python biolinksense.py --mode train --backbone biobert --device cpu

This will:

Load the demo BioRED subset from ./demo_data/*.jsonl.
Build the relation-level dataset (entity–entity pairs + context).
Fine-tune BioBERT on the demo train/validation splits.
Save the trained checkpoint and tokenizer to:
./outputs/biobert/

4. Evaluate BioBERT on the demo test split
python biolinksense.py --mode eval --backbone biobert --device cpu

This will:

Load the trained model from ./outputs/biobert/.
Evaluate on the demo test split.
Print:
Macro precision, recall, F1
Full classification report (per-label precision/recall/F1 + accuracy)


Running on the Full BioRED Dataset
If you want to reproduce the full-dataset numbers:

Temporarily move or rename the demo_data/ folder so that it is not found:

mv demo_data demo_data_backup

Run training and evaluation:

# PubMedBERT full BioRED
python biolinksense.py --mode train --backbone pubmedbert --device cuda   # or cpu
python biolinksense.py --mode eval  --backbone pubmedbert --device cuda

# BioBERT full BioRED
python biolinksense.py --mode train --backbone biobert --device cuda
python biolinksense.py --mode eval  --backbone biobert --device cuda

The script will automatically:

Download bigbio/biored (biored_bigbio_kb) via datasets.load_dataset.
Build the full relation-level dataset.
Train and evaluate on the full splits.

Gemini Summarization

To test the Gemini-based summarization on the demo documents:

Set your Gemini API key as an environment variable:

export GOOGLE_API_KEY="YOUR_REAL_API_KEY_HERE"

Run the summarization mode:

python biolinksense.py --mode summarize --device cpu

This will:

Load a small number of demo documents
Call Gemini 2.5 Flash via google-genai to generate abstractive summaries.
Compute ROUGE-L scores against reference abstracts.
Save the generated summaries and scores to:
./outputs/gemini_demo_summaries.jsonl


Notes on Code Structure & Modularity

data_utils.py: handles dataset loading, negative sampling, and conversion from document-level to relation-level instances.
model_utils.py: wraps HuggingFace’s Trainer for training and evaluation, computing macro metrics, and printing classification reports.
llm_utils.py: encapsulates Gemini API calls and ROUGE scoring (keeps all LLM-specific code in one place).
biolinksense.py: provides a clean command-line interface (--mode, --backbone, --device, etc.), tying together data loading, model training, evaluation, and summarization.

