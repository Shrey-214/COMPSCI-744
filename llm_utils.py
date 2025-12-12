
import json
import os
from typing import Dict, List

import numpy as np
from datasets import DatasetDict
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
)

from utils import ensure_dir
from data_utils import build_abstract_text


try:
    import ollama  
except ImportError:
    ollama = None

try:
    from google import genai 
except ImportError:
    genai = None


# --------- dependency checks ---------


def ollama_required() -> None:
    if ollama is None:
        raise RuntimeError(
            "The 'ollama' Python package is not installed or not available. "
            "Install it with 'pip install ollama' and make sure the Ollama daemon "
            "is running on your local machine."
        )


def gemini_required() -> None:
    if genai is None:
        raise RuntimeError(
            "The 'google-genai' package is not installed.\n"
            "Install it with: pip install -U google-genai\n"
            "Also set your API key in an environment variable, e.g.:\n"
            "  export GEMINI_API_KEY='YOUR_KEY_HERE'"
        )


# --------- Ollama relation classification baseline ---------


def ollama_classify_relation(
    text: str,
    candidate_labels: List[str],
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
) -> str:
    
    ollama_required()

    if hasattr(ollama, "Client"):
        client = ollama.Client(host=base_url)
    else:
        client = ollama

    labels_str = ", ".join(candidate_labels)
    prompt = (
        "You are a biomedical NLP model.\n"
        "Read the following text describing two biomedical entities and their context.\n"
        "Decide which relation label best describes the relation between the two entities.\n\n"
        f"Possible labels: {labels_str}\n\n"
        f"Text:\n{text}\n\n"
        "Answer with ONLY the label name (exactly as in the list)."
    )

    resp = client.generate(
        model=model,
        prompt=prompt,
        stream=False,
    )
    if isinstance(resp, dict):
        content = resp.get("response", "").strip()
    else:
        content = str(resp).strip()

    content_lower = content.lower()
    for lab in candidate_labels:
        if lab.lower() in content_lower:
            return lab

    if "no_relation" in candidate_labels:
        return "no_relation"
    return candidate_labels[0]


def ollama_relation_eval(
    ds_rel: DatasetDict,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    ollama_model: str = "llama2",
    base_url: str = "http://localhost:11434",
    num_examples: int = 200,
) -> None:
   
    ollama_required()

    test_ds = ds_rel["test"]
    if num_examples < len(test_ds):
        test_ds = test_ds.select(range(num_examples))

    y_true: List[str] = []
    y_pred: List[str] = []
    candidate_labels = list(label2id.keys())

    print(
        f"DEBUG: Running Ollama relation eval with model={ollama_model}, examples={len(test_ds)}",
        flush=True,
    )

    for i, ex in enumerate(test_ds):
        text = ex["text"]
        gold_label_id = ex["label"]
        gold_label = id2label[gold_label_id]

        pred_label = ollama_classify_relation(
            text=text,
            candidate_labels=candidate_labels,
            model=ollama_model,
            base_url=base_url,
        )

        y_true.append(gold_label)
        y_pred.append(pred_label)

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1} / {len(test_ds)} examples", flush=True)

    y_true_ids = [label2id[l] for l in y_true]
    y_pred_ids = [label2id[l] for l in y_pred]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_ids, y_pred_ids, average="macro", zero_division=0
    )
    print(
        f"Ollama (model={ollama_model}) macro P/R/F1: "
        f"{precision:.4f}, {recall:.4f}, {f1:.4f}"
    )

    all_labels = sorted(set(y_true_ids) | set(y_pred_ids))
    label_names = [id2label[i] for i in all_labels]

    print(
        classification_report(
            y_true_ids,
            y_pred_ids,
            labels=all_labels,
            target_names=label_names,
            zero_division=0,
        )
    )


# --------- ROUGE-L  ---------


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def rouge_l(candidate: str, reference: str) -> float:
    cand_tokens = candidate.split()
    ref_tokens = reference.split()
    if not cand_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_length(cand_tokens, ref_tokens)
    prec = lcs / len(cand_tokens)
    rec = lcs / len(ref_tokens)
    if prec == 0 and rec == 0:
        return 0.0
    beta = 1.0
    score = (1 + beta**2) * prec * rec / (rec + beta**2 * prec)
    return score


def build_docid_to_abstract(ds_docs: DatasetDict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split in ["train", "validation", "test"]:
        for doc in ds_docs[split]:
            doc_id = doc["document_id"]
            if doc_id not in mapping:
                mapping[doc_id] = build_abstract_text(doc["passages"])
    return mapping


# --------- Ollama summarization ---------


def summarize_relations_with_ollama(
    ds_rel: DatasetDict,
    ds_docs: DatasetDict,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    ollama_model: str = "llama2",
    base_url: str = "http://localhost:11434",
    num_summary_docs: int = 10,
    output_path: str = "./outputs/ollama_summaries.jsonl",
) -> None:
   
    ollama_required()
    ensure_dir(os.path.dirname(output_path))

    if hasattr(ollama, "Client"):
        client = ollama.Client(host=base_url)
    else:
        client = ollama

    docid_to_abstract = build_docid_to_abstract(ds_docs)

    
    test_ds = ds_rel["test"]
    doc_to_examples: Dict[str, List[dict]] = {}
    for ex in test_ds:
        doc_id = ex["doc_id"]
        doc_to_examples.setdefault(doc_id, []).append(ex)

    doc_ids = sorted(doc_to_examples.keys())
    if num_summary_docs < len(doc_ids):
        doc_ids = doc_ids[:num_summary_docs]

    all_scores: List[float] = []

    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, doc_id in enumerate(doc_ids):
            abstract = docid_to_abstract.get(doc_id, "").strip()
            examples = doc_to_examples[doc_id]

            positive_examples = [
                ex for ex in examples if id2label[ex["label"]] != "no_relation"
            ]
            if not positive_examples:
                selected_examples = examples[:5]
            else:
                selected_examples = positive_examples[:10]

            relation_sentences: List[str] = []
            for i, ex in enumerate(selected_examples):
                rel_label = id2label[ex["label"]]
                relation_sentences.append(f"{i+1}. [{rel_label}] {ex['text']}")

            if not relation_sentences:
                continue

            prompt = (
                "You are a biomedical NLP assistant.\n"
                "You are given several sentences describing relations among biomedical entities\n"
                "extracted from a PubMed abstract. Write a concise 3–4 sentence summary focusing\n"
                "on the key biomedical relationships (e.g., gene–disease, drug–adverse effect,\n"
                "chemical–gene, etc.). Use clear scientific language.\n\n"
                "Sentences:\n"
                + "\n".join(relation_sentences)
                + "\n\nSummary:"
            )

            resp = client.generate(
                model=ollama_model,
                prompt=prompt,
                stream=False,
            )
            if isinstance(resp, dict):
                summary = resp.get("response", "").strip()
            else:
                summary = str(resp).strip()

            score = rouge_l(summary, abstract)
            all_scores.append(score)

            record = {
                "doc_id": doc_id,
                "summary_model": ollama_model,
                "summary": summary,
                "reference_abstract": abstract,
                "rouge_l": score,
            }
            f_out.write(json.dumps(record) + "\n")

            print(
                f"[{idx+1}/{len(doc_ids)}] doc_id={doc_id}, ROUGE-L={score:.4f}",
                flush=True,
            )

    avg_rouge = float(np.mean(all_scores)) if all_scores else 0.0
    print(
        f"Average ROUGE-L over {len(all_scores)} documents (model={ollama_model}): {avg_rouge:.4f}"
    )
    print(f"Summaries saved to: {output_path}")


# --------- Gemini summarization ---------


def summarize_relations_with_gemini(
    ds_rel: DatasetDict,
    ds_docs: DatasetDict,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    gemini_model: str = "gemini-2.5-flash",
    num_summary_docs: int = 10,
    output_path: str = "./outputs/gemini_summaries.jsonl",
    api_key_env: str = "GEMINI_API_KEY",
) -> None:
    gemini_required()
    ensure_dir(os.path.dirname(output_path))

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"No Gemini API key found in environment variable '{api_key_env}'.\n"
            f"Set it before running, e.g.:\n"
            f"  export {api_key_env}='YOUR_KEY_HERE'"
        )

    client = genai.Client(api_key=api_key)

    docid_to_abstract = build_docid_to_abstract(ds_docs)

    test_ds = ds_rel["test"]
    doc_to_examples: Dict[str, List[dict]] = {}
    for ex in test_ds:
        doc_id = ex["doc_id"]
        doc_to_examples.setdefault(doc_id, []).append(ex)

    doc_ids = sorted(doc_to_examples.keys())
    if num_summary_docs < len(doc_ids):
        doc_ids = doc_ids[:num_summary_docs]

    all_scores: List[float] = []

    with open(output_path, "w", encoding="utf-8") as f_out:
        for idx, doc_id in enumerate(doc_ids):
            abstract = docid_to_abstract.get(doc_id, "").strip()
            examples = doc_to_examples[doc_id]

            positive_examples = [
                ex for ex in examples if id2label[ex["label"]] != "no_relation"
            ]
            if not positive_examples:
                selected_examples = examples[:5]
            else:
                selected_examples = positive_examples[:10]

            relation_sentences: List[str] = []
            for i, ex in enumerate(selected_examples):
                rel_label = id2label[ex["label"]]
                relation_sentences.append(f"{i+1}. [{rel_label}] {ex['text']}")

            if not relation_sentences:
                continue

            prompt = (
                "You are a biomedical NLP assistant.\n"
                "You are given several sentences describing relations among biomedical entities\n"
                "extracted from a PubMed abstract. Write a concise 3–4 sentence summary focusing\n"
                "on the key biomedical relationships (e.g., gene–disease, drug–adverse effect,\n"
                "chemical–gene, etc.). Use clear scientific language.\n\n"
                "Sentences:\n"
                + "\n".join(relation_sentences)
                + "\n\nSummary:"
            )

            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
            )

            summary = (getattr(response, "text", None) or "").strip()

            score = rouge_l(summary, abstract)
            all_scores.append(score)

            record = {
                "doc_id": doc_id,
                "summary_model": gemini_model,
                "summary": summary,
                "reference_abstract": abstract,
                "rouge_l": score,
            }
            f_out.write(json.dumps(record) + "\n")

            print(
                f"[{idx+1}/{len(doc_ids)}] doc_id={doc_id}, ROUGE-L={score:.4f}",
                flush=True,
            )

    avg_rouge = float(np.mean(all_scores)) if all_scores else 0.0
    print(
        f"Average ROUGE-L over {len(all_scores)} documents (model={gemini_model}): {avg_rouge:.4f}"
    )
    print(f"Summaries saved to: {output_path}")
