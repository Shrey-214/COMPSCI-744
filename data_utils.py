# data_utils.py
from typing import Dict, List, Tuple

import os
import random
from datasets import load_dataset, DatasetDict

from config import cfg
from utils import set_seed


def load_biored_bigbio(data_dir: str = "./demo_data") -> DatasetDict:
    
    demo_files = {
        "train": os.path.join(data_dir, "biored_docs_train.jsonl"),
        "validation": os.path.join(data_dir, "biored_docs_validation.jsonl"),
        "test": os.path.join(data_dir, "biored_docs_test.jsonl"),
    }

    demo_available = all(os.path.isfile(p) for p in demo_files.values())

    if demo_available:
        print(
            f"DEBUG: demo_data found in {data_dir}, "
            "loading local BioRED subset from JSONL ...",
            flush=True,
        )
        ds_docs = load_dataset(
            "json",
            data_files=demo_files,  # keys become splits: train/validation/test
        )
    else:
        print(
            "DEBUG: demo_data not found, loading dataset bigbio/biored "
            "(config=biored_bigbio_kb) from Hugging Face ...",
            flush=True,
        )
        ds_docs = load_dataset(
            cfg.HF_DATASET_NAME,
            name=cfg.HF_CONFIG_NAME,
            trust_remote_code=True,
        )

    print("DEBUG: dataset loaded:", ds_docs, flush=True)
    return ds_docs


def build_abstract_text(passages: List[dict]) -> str:
    pieces: List[str] = []
    for p in passages:
        if "text" in p and isinstance(p["text"], list):
            pieces.extend(p["text"])
    return " ".join(pieces).strip()


def prepare_biored_relation_dataset(
    max_negative_per_doc: int = 5,
) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str], DatasetDict]:
    
    ds_docs = load_biored_bigbio()

    # Collect all relation types
    relation_types = set()
    for split in ["train", "validation", "test"]:
        for ex in ds_docs[split]:
            for rel in ex["relations"]:
                relation_types.add(rel["type"])
    relation_types = sorted(list(relation_types))

    # Add no_relation
    if "no_relation" not in relation_types:
        relation_types.append("no_relation")

    label2id: Dict[str, int] = {label: i for i, label in enumerate(relation_types)}
    id2label: Dict[int, str] = {i: label for label, i in label2id.items()}
    print("DEBUG: label mapping:", label2id, flush=True)

    all_splits: Dict[str, Dict[str, List]] = {}
    set_seed(cfg.SEED)

    for split in ["train", "validation", "test"]:
        texts: List[str] = []
        labels: List[int] = []
        relation_type_names: List[str] = []
        doc_ids: List[str] = []
        e1_ids: List[str] = []
        e2_ids: List[str] = []

        for doc in ds_docs[split]:
            doc_id = doc["document_id"]
            abstract_text = build_abstract_text(doc["passages"])

            # Build entity map
            ent_map: Dict[str, Tuple[str, str]] = {}
            for ent in doc["entities"]:
                ent_id = ent["id"]
                ent_text = (
                    " ".join(ent["text"])
                    if isinstance(ent["text"], list)
                    else str(ent["text"])
                )
                ent_type = ent["type"]
                ent_map[ent_id] = (ent_text, ent_type)

            # Positive relations
            positive_pairs = set()
            for rel in doc["relations"]:
                r_type = rel["type"]
                arg1_id = rel["arg1_id"]
                arg2_id = rel["arg2_id"]
                if arg1_id not in ent_map or arg2_id not in ent_map:
                    continue
                positive_pairs.add((arg1_id, arg2_id))

                e1_text, e1_type = ent_map[arg1_id]
                e2_text, e2_type = ent_map[arg2_id]

                text = (
                    f"Entity1: {e1_text} (type: {e1_type}); "
                    f"Entity2: {e2_text} (type: {e2_type}); "
                    f"Context: {abstract_text}"
                )

                texts.append(text)
                labels.append(label2id[r_type])
                relation_type_names.append(r_type)
                doc_ids.append(doc_id)
                e1_ids.append(arg1_id)
                e2_ids.append(arg2_id)

            # Negative sampling: pairs with no annotated relation
            ent_ids = list(ent_map.keys())
            candidate_neg = []
            for i in range(len(ent_ids)):
                for j in range(len(ent_ids)):
                    if i == j:
                        continue
                    pair = (ent_ids[i], ent_ids[j])
                    if pair not in positive_pairs:
                        candidate_neg.append(pair)

            random.shuffle(candidate_neg)
            candidate_neg = candidate_neg[:max_negative_per_doc]

            for arg1_id, arg2_id in candidate_neg:
                e1_text, e1_type = ent_map[arg1_id]
                e2_text, e2_type = ent_map[arg2_id]
                text = (
                    f"Entity1: {e1_text} (type: {e1_type}); "
                    f"Entity2: {e2_text} (type: {e2_type}); "
                    f"Context: {abstract_text}"
                )
                texts.append(text)
                labels.append(label2id["no_relation"])
                relation_type_names.append("no_relation")
                doc_ids.append(doc_id)
                e1_ids.append(arg1_id)
                e2_ids.append(arg2_id)

        all_splits[split] = {
            "text": texts,
            "label": labels,
            "relation_type": relation_type_names,
            "doc_id": doc_ids,
            "e1_id": e1_ids,
            "e2_id": e2_ids,
        }

    ds_rel = DatasetDict(
        {
            split: ds_docs[split].from_dict(all_splits[split])
            for split in ["train", "validation", "test"]
        }
    )

    print("DEBUG: built relation-level dataset:", ds_rel, flush=True)
    return ds_rel, label2id, id2label, ds_docs
