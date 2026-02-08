"""Pipeline d'extraction d'idées via LLM avec filtre post-extraction."""

# Librairies requises
import re
import pandas as pd
from io import StringIO
from tqdm import tqdm
import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer


# Fonctions nécessaires à la pipeline d'extraction
class LLMBadCSV(Exception):
    """Exception levée quand le CSV retourné par le LLM est invalide."""
    pass


def extraction_pipeline(
        df: pd.DataFrame, 
        system_prompt: str,
        user_template: str,
        extract_model: str = "llama3:8b-instruct-q4_K_M",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        return_intermediate: bool = False,
        error_filter: str = "Oui",
        rouge_filter: float = 0.0,
        qualit_filter: float = 0.0
    ) -> pd.DataFrame:
    """Pipeline d'extraction d'idées via LLM avec filtres post-extraction.
    Arguments :
        df : DataFrame avec colonnes 'author_id' et 'contribution'.
        system_prompt : Prompt système pour le LLM.
        user_template : Template utilisateur pour le LLM.
        extract_model : Modèle Ollama pour l'extraction (défaut = "llama3:8b-instruct-q4_K_M").
        embed_model : Modèle de sentence-transformers pour les embeddings (défaut = "sentence-transformers/all-MiniLM-L6-v2").
        device : mode de calcul des embeddings (défaut = "cpu").
        return_intermediate : retourne aussi le DataFrame intermédiaire des extractions (défaut = False).
        error_filter : "Oui" pour filtrer les erreurs de parsing (défaut = "Oui").
        rouge_filter : Seuil minimal pour le score ROUGE (défaute = 0 : pas de filtre).
        qualit_filter : Seuil minimal pour le score QualIT (défaut = 0 : pas de filtre).
    Returns :
        DataFrame avec les extractions et leurs scores.
        Si return_intermediate=True, retourne un tuple (data_extracted, result).
    """
    
    # Vérification du dataframe
    required_columns = {"author_id", "contribution"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Le dataframe doit contenir les colonnes suivantes : {required_columns}")
    
    # Extraction des idées via LLM
    rows = []
    type_ok = {"statement", "proposition"}
    syntax_ok = {"positive", "negative"}
    semantic_ok = {"positive", "negative", "neutral"}
    for i, row in tqdm(df.iterrows(), total=len(df), desc="LLM extraction"):
        text = str(row["contribution"]).strip()
        author_id = row["author_id"]
        if not text:
            continue
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(input=text)},
            ]
            resp = ollama.chat(
                model=extract_model,
                messages=messages,
                options={
                    "num_ctx": 2048,
                    "num_batch": 4,
                    "temperature": 0,
                    "top_p": 0.95,
                    "seed": 42,
                }
            )
            raw = resp["message"]["content"]
            # Nettoyage des balises de code et préfixes
            cleaned = raw.strip()
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            idx = cleaned.find("CSV:")
            if idx != -1:
                cleaned = cleaned[idx:]
            # Extraction du bloc CSV
            if cleaned.startswith("CSV:"):
                csv_block = cleaned[len("CSV:"):]
            else:
                m = re.search(r"(?mi)^description,type,syntax,semantic\s*$", cleaned)
                csv_block = cleaned[m.start():] if m else cleaned
            # Parsing du CSV
            csv_text = csv_block.strip()
            if not csv_text.lower().startswith("description,type,syntax,semantic"):
                lines = csv_text.splitlines()
                if lines:
                    header = lines[0].replace(" ", "")
                    if header.lower() == "description,type,syntax,semantic":
                        lines[0] = "description,type,syntax,semantic"
                        csv_text = "\n".join(lines)
            try:
                ideas_df = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)
            except Exception:
                # Fallback robuste: tolère entête manquante et virgules dans la description
                lines = [l for l in csv_text.splitlines() if l.strip()]
                if not lines:
                    raise LLMBadCSV("CSV vide après nettoyage")
                # Détecte le délimiteur principal
                sample = lines[:5]
                comma_count = sum(l.count(",") for l in sample)
                semi_count = sum(l.count(";") for l in sample)
                delim = ";" if semi_count > comma_count else ","
                parsed_rows = []
                for line in lines:
                    cleaned_line = line.strip()
                    if cleaned_line.lower().startswith("csv:"):
                        cleaned_line = cleaned_line[4:].strip()
                    normalized = cleaned_line.replace(" ", "").lower()
                    # Ignore les entêtes ou lignes bruitées
                    if normalized in {"description,type,syntax,semantic", "description;type;syntax;semantic"}:
                        continue
                    if normalized.startswith("description") and "type" in normalized and "syntax" in normalized and "semantic" in normalized:
                        continue
                    parts = [p.strip() for p in cleaned_line.rsplit(delim, 3)]
                    if len(parts) != 4:
                        continue
                    parsed_rows.append(parts)
                if not parsed_rows:
                    raise LLMBadCSV("Entête CSV invalide")
                ideas_df = pd.DataFrame(parsed_rows, columns=["description", "type", "syntax", "semantic"])
            expected_cols = ["description", "type", "syntax", "semantic"]
            missing = [c for c in expected_cols if c not in ideas_df.columns]
            if missing:
                raise LLMBadCSV(f"Colonnes manquantes: {missing}")
            ideas_df = ideas_df[expected_cols].copy()
            # Normalisation des valeurs
            ideas_df["type"] = ideas_df["type"].astype(str).str.strip().str.lower()
            ideas_df["syntax"] = ideas_df["syntax"].astype(str).str.strip().str.lower()
            ideas_df["semantic"] = ideas_df["semantic"].astype(str).str.strip().str.lower()
            ideas_df.loc[~ideas_df["type"].isin(type_ok), "type"] = "statement"
            ideas_df.loc[~ideas_df["syntax"].isin(syntax_ok), "syntax"] = "positive"
            ideas_df.loc[~ideas_df["semantic"].isin(semantic_ok), "semantic"] = "neutral"
        except Exception as e:
            # On enregistre une ligne "échec" minimale pour traçabilité
            ideas_df = pd.DataFrame([{
                "description": f"[PARSE_FAIL] {str(e)[:200]}",
                "type": "statement",
                "syntax": "positive",
                "semantic": "neutral"
            }])
        ideas_df = ideas_df.copy()
        ideas_df.insert(0, "author_id", author_id)
        ideas_df.insert(1, "contribution_index", i)
        rows.append(ideas_df)
    if not rows:
        empty = pd.DataFrame(columns=[
            "author_id", "contribution_index", "description", "type", "syntax", "semantic"
        ])
        return (empty, empty) if return_intermediate else empty
    result = pd.concat(rows, ignore_index=True)

    # Agrégation et calcul QualIT
    ideas_grouped = (
        result
        .groupby("contribution_index", as_index=False)
        .agg(
            n_ideas=("description", "size"),
            ideas_text=("description", lambda s: " || ".join([str(x).strip() for x in s if str(x).strip()]))
        )
    )
    dfc = df.reset_index(drop=False).rename(columns={"index": "contribution_index"})
    merged = ideas_grouped.merge(
        dfc[["contribution_index", "author_id", "contribution"]],
        on="contribution_index", how="left"
    ).dropna(subset=["contribution"]).copy()
    model = SentenceTransformer(embed_model)
    emb_contrib = model.encode(
        merged["contribution"].tolist(),
        device=device,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    emb_ideas = model.encode(
        merged["ideas_text"].fillna("").tolist(),
        device=device,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    qualit_scores = np.sum(emb_contrib * emb_ideas, axis=1)
    data_extracted = pd.DataFrame({
        "author_id": merged["author_id"].values,
        "contribution": merged["contribution"].values,
        "contribution_index": merged["contribution_index"].values,
        "contribution_length": merged["contribution"].map(len).values,
        "extraction": merged["ideas_text"].fillna("").values,
        "n_ideas": merged["n_ideas"].values,
        "extraction_length": merged["ideas_text"].fillna("").map(len).values,
        "qualit_score": qualit_scores,
    }).sort_values("contribution_index").reset_index(drop=True)

    # Calcul de la métrique ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    def calc_rouge(row):
        try:
            scores = scorer.score(str(row["extraction"]), str(row["contribution"]))
            return pd.Series({
                "rouge_score_1gram": scores["rouge1"].fmeasure,
                "rouge_score_L": scores["rougeL"].fmeasure
            })
        except Exception:
            return pd.Series({"rouge_score_1gram": 0.0, "rouge_score_L": 0.0})
    data_extracted[["rouge_score_1gram", "rouge_score_L"]] = data_extracted.apply(calc_rouge, axis=1)

    # Filtre des extractions échouées : présence de "[PARSE_FAIL]"
    if error_filter == "Oui":
        data_extracted = data_extracted[~data_extracted["extraction"].str.contains("[PARSE_FAIL]", na=False, regex=False)]
    
    # Filtre ROUGE (hallucinations)
    if rouge_filter > 0:
        data_extracted = data_extracted[data_extracted["rouge_score_1gram"] >= rouge_filter]
    
    # Filtre QualIT (qualité faible)
    if qualit_filter > 0:
        data_extracted = data_extracted[data_extracted["qualit_score"] >= qualit_filter]

    # Résultats finaux
    final_df = data_extracted.reset_index(drop=True)
    return {"final": final_df, "intermediate": result} if return_intermediate else final_df