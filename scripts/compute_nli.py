# Pour obtenir le df, lancer depuis un terminal la commande suite :
# python scripts/compute_nli.py data/01_251021_Notations200.csv data/01_251021_Notations200_with_nli.csv

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

#. DEFAULT_MODEL = "huggingface/distilbert-base-uncased-finetuned-mnli"
# DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_MODEL = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

def compute_nli_columns(
    df: pd.DataFrame,
    premise_col: str = "contribution",
    hypothesis_col: str = "ideas_text",
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
    max_length: int = 512, # Limité à 512 tokens
    batch_size: int = 1,
    progress_every: int = 50,
) -> pd.DataFrame:
    device_t = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device_t)
    model.eval()

    cfg = AutoConfig.from_pretrained(model_name)
    id2label = {int(k): v for k, v in cfg.id2label.items()} if isinstance(cfg.id2label, dict) else cfg.id2label
    entail_idx = next((k for k, v in id2label.items() if "entail" in str(v).lower()), 2)
    contra_idx = next((k for k, v in id2label.items() if "contrad" in str(v).lower()), 0)

    n = len(df)
    entail = np.zeros(n, dtype=np.float32)
    contra = np.zeros(n, dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            premises = df[premise_col].iloc[start:end].fillna("").astype(str).tolist()
            hypos = df[hypothesis_col].iloc[start:end].fillna("").astype(str).tolist()

            enc = tokenizer(
                premises,
                hypos,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device_t) for k, v in enc.items()}

            probs = torch.softmax(model(**enc).logits, dim=1).detach().cpu().numpy()
            entail[start:end] = probs[:, entail_idx]
            contra[start:end] = probs[:, contra_idx]

            if progress_every and (end % progress_every == 0):
                print(f"{end} / {n} calculés...", flush=True)

    df_nli = df.copy()
    df_nli["score_humain"] = df_nli[["Garance", "Matthias", "Yannis"]].mean(axis=1)
    df_nli["nli_entailment"] = entail.astype(float)
    df_nli["nli_contradiction"] = contra.astype(float)
    df_nli["nli_1_minus_contradiction"] = (1.0 - df_nli["nli_contradiction"]).astype(float)

    return df_nli


def main():
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  python scripts/compute_nli.py <input.csv> <output.csv>\n\n"
            "Exemple:\n"
            "  python scripts/compute_nli.py data/01_251021_Notations200.csv data/01_251021_Notations200_with_nli.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    df = pd.read_csv(in_path)

    df_nli = compute_nli_columns(
        df,
        premise_col="contribution",
        hypothesis_col="ideas_text",
        model_name=DEFAULT_MODEL,
        device="cpu",
        max_length=128,
        batch_size=1,
        progress_every=20,
    )

    df_nli.to_csv(out_path, index=False)
    print(f"OK - écrit : {out_path}", flush=True)


if __name__ == "__main__":
    main()