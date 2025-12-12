

"""
BioLinkSense: Biomedical Relation Extraction and Summarization
"""

import argparse

from config import cfg
from utils import set_seed, get_device
from data_utils import prepare_biored_relation_dataset
from model_utils import train_supervised_model, evaluate_supervised_model
from llm_utils import (
    ollama_relation_eval,
    summarize_relations_with_ollama,
    summarize_relations_with_gemini,
)


def main():
    print("DEBUG: BioLinkSense script started", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prepare_data", "train", "eval", "ollama_eval", "summarize"],
    )
    parser.add_argument(
        "--backbone",
        choices=["biobert", "pubmedbert"],
        default="biobert",
        help="Which supervised backbone to train/eval.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=20000,
        help="Max train examples (subset for speed).",
    )
    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=5000,
        help="Max validation examples (subset for speed).",
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default="llama2",
        help="Ollama model name (e.g., llama2, llama3).",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for Ollama server.",
    )
    parser.add_argument(
        "--num_ollama_eval",
        type=int,
        default=200,
        help="Number of test examples for Ollama relation eval.",
    )
    parser.add_argument(
        "--num_summary_docs",
        type=int,
        default=10,
        help="Number of test documents to summarize.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="./outputs/summaries.jsonl",
        help="Where to save summaries (Gemini or Ollama).",
    )
    parser.add_argument(
        "--summarizer",
        choices=["ollama", "gemini"],
        default="ollama",
        help="Which backend to use for summarization.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name for summarization.",
    )

    args = parser.parse_args()
    print("DEBUG: entering main()", flush=True)
    set_seed(cfg.SEED)

    if args.mode == "prepare_data":
        print("DEBUG: inside main() -> prepare_data", flush=True)
        _ds_rel, _label2id, _id2label, _ds_docs = prepare_biored_relation_dataset(
            max_negative_per_doc=cfg.MAX_NEGATIVE_PAIRS_PER_DOC
        )
        print("Data prepared successfully.")
        print("DEBUG: main() finished", flush=True)
        return

    
    print("DEBUG: inside main()", flush=True)
    ds_rel, label2id, id2label, ds_docs = prepare_biored_relation_dataset(
        max_negative_per_doc=cfg.MAX_NEGATIVE_PAIRS_PER_DOC
    )

    device = get_device(args.device)

    if args.mode == "train":
        train_supervised_model(
            ds_rel=ds_rel,
            label2id=label2id,
            id2label=id2label,
            backbone=args.backbone,
            device=device,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
        )

    elif args.mode == "eval":
        evaluate_supervised_model(
            ds_rel=ds_rel,
            label2id=label2id,
            id2label=id2label,
            backbone=args.backbone,
            device=device,
        )

    elif args.mode == "ollama_eval":
        ollama_relation_eval(
            ds_rel=ds_rel,
            label2id=label2id,
            id2label=id2label,
            ollama_model=args.ollama_model,
            base_url=args.ollama_url,
            num_examples=args.num_ollama_eval,
        )

    elif args.mode == "summarize":
        if args.summarizer == "ollama":
            summarize_relations_with_ollama(
                ds_rel=ds_rel,
                ds_docs=ds_docs,
                label2id=label2id,
                id2label=id2label,
                ollama_model=args.ollama_model,
                base_url=args.ollama_url,
                num_summary_docs=args.num_summary_docs,
                output_path=args.summary_output,
            )
        else:
            summarize_relations_with_gemini(
                ds_rel=ds_rel,
                ds_docs=ds_docs,
                label2id=label2id,
                id2label=id2label,
                gemini_model=args.gemini_model,
                num_summary_docs=args.num_summary_docs,
                output_path=args.summary_output,
            )

    print("DEBUG: main() finished", flush=True)


if __name__ == "__main__":
    main()
