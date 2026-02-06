"""Pipeline d'extraction d'idées via LLM avec filtre post-extraction."""

import re
from typing import Dict, Any
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


def strip_code_fences(s: str) -> str:
    """Nettoie les balises de code Markdown et préfixes indésirables."""
    # Supprime les balises de code Markdown
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # Supprime tout préfixe avant "CSV:" si ça arrive
    idx = s.find("CSV:")
    if idx != -1:
        s = s[idx:]
    return s


def extract_csv_block(s: str) -> str:
    """Extrait le bloc CSV du texte retourné par le LLM."""
    s = strip_code_fences(s)
    # On vérifie si la sortie commence par "CSV:" comme on s'y attend d'après le prompt
    if s.startswith("CSV:"):
        # On retourne tout ce qui suit "CSV:"
        return s[len("CSV:"):]
    # Sinon, on tente de récupérer les lignes à partir de l'entête demandée dans le prompt
    m = re.search(r"(?mi)^description,type,syntax,semantic\s*$", s)
    if m:
        # On retourne tout à partir de l'entête
        return s[m.start():]
    # Sinon on retourne tout le texte
    return s


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise une ligne du CSV si les contraintes ne sont pas respectées."""
    # Normalise les valeurs attendues
    row["type"] = str(row.get("type","")).strip().lower()
    row["syntax"] = str(row.get("syntax","")).strip().lower()
    row["semantic"] = str(row.get("semantic","")).strip().lower()
    # Contraintes
    type_ok = {"statement", "proposition"}
    syntax_ok = {"positive", "negative"}
    semantic_ok = {"positive", "negative", "neutral"}
    # Normalisation si besoin
    if row["type"] not in type_ok:
        row["type"] = "statement" 
    if row["syntax"] not in syntax_ok:
        row["syntax"] = "positive"
    if row["semantic"] not in semantic_ok:
        row["semantic"] = "neutral"
    return row


def parse_llm_csv(csv_text: str) -> pd.DataFrame:
    """Parse le CSV retourné par le LLM en DataFrame pandas."""
    csv_text = csv_text.strip()
    if not csv_text.lower().startswith("description,type,syntax,semantic"):
        # Parfois le modèle met des espaces, on nettoie la première ligne
        lines = csv_text.splitlines()
        if lines:
            header = lines[0].replace(" ", "")
            if header.lower() == "description,type,syntax,semantic":
                lines[0] = "description,type,syntax,semantic"
                csv_text = "\n".join(lines)
    try:
        df = pd.read_csv(StringIO(csv_text), dtype=str, keep_default_na=False)
    except Exception as e:
        raise LLMBadCSV(f"CSV illisible: {e}")
    # Colonnes minimales
    expected_cols = ["description", "type", "syntax", "semantic"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise LLMBadCSV(f"Colonnes manquantes: {missing}")
    # Normalisation
    df = df[expected_cols].copy()
    df = df.apply(lambda r: pd.Series(normalize_row(r.to_dict())), axis=1)
    return df


def call_llm_return_df(text: str, system_prompt: str, user_template: str) -> pd.DataFrame:
    """Appelle le LLM et retourne un DataFrame des idées extraites."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(input=text)},
    ]
    resp = ollama.chat(
        model="llama3:8b-instruct-q4_K_M",
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
    csv_block = extract_csv_block(raw)
    return parse_llm_csv(csv_block)


def extract_ideas_from_df(
        df_contrib: pd.DataFrame,
        system_prompt: str,
        user_template: str,
        text_col: str = "contribution",
        id_col: str = "author_id"
    ) -> pd.DataFrame:
    """Extrait les idées de chaque contribution via LLM."""
    rows = []

    for i, row in tqdm(df_contrib.iterrows(), total=len(df_contrib), desc="LLM extraction"):
        text = str(row[text_col]).strip()
        author_id = row[id_col]
        if not text:
            continue

        try:
            ideas_df = call_llm_return_df(text, system_prompt, user_template)
        except Exception as e:
            # On enregistre une ligne "échec" minimale pour traçabilité
            ideas_df = pd.DataFrame([{
                "description": f"[PARSE_FAIL] {str(e)[:200]}",
                "type": "statement",
                "syntax": "positive",
                "semantic": "neutral"
            }])

        # Ajoute le contexte
        ideas_df = ideas_df.copy()
        ideas_df.insert(0, "author_id", author_id)
        ideas_df.insert(1, "contribution_index", i)
        rows.append(ideas_df)

    if not rows:
        return pd.DataFrame(columns=["author_id", "contribution_index", "description", "type", "syntax", "semantic"])
    out = pd.concat(rows, ignore_index=True)
    return out


def compute_qualit_scores(
        df: pd.DataFrame,
        result: pd.DataFrame,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        text_col: str = "contribution",
        index_col: str = "contribution_index",
        sep: str = " || "
    ) -> pd.DataFrame:
    """Calcule les scores QualIT pour chaque contribution."""
    # Agrégation des idées extraites
    ideas_grouped = (
        result
        .groupby(index_col, as_index=False)
        .agg(
            n_ideas=("description", "size"),
            ideas_text=("description", lambda s: sep.join([str(x).strip() for x in s if str(x).strip()]))
        )
    )
    
    # Aligne avec df pour récupérer les informations
    dfc = df.reset_index(drop=False).rename(columns={"index": index_col})
    merged = ideas_grouped.merge(
        dfc[[index_col, "author_id", text_col]],
        on=index_col, how="left"
    ).dropna(subset=[text_col]).copy()
    
    # Calcul des embeddings
    model = SentenceTransformer(embed_model)
    emb_contrib = model.encode(
        merged[text_col].tolist(),
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
    
    # Calcul de la similarité cosinus (dot product sur vecteurs normalisés)
    qualit_scores = np.sum(emb_contrib * emb_ideas, axis=1)
    
    # Construit la sortie
    return pd.DataFrame({
        "author_id": merged["author_id"].values,
        "contribution": merged[text_col].values,
        "contribution_index": merged[index_col].values,
        "contribution_length": merged[text_col].map(len).values,
        "extraction": merged["ideas_text"].fillna("").values,
        "n_ideas": merged["n_ideas"].values,
        "extraction_length": merged["ideas_text"].fillna("").map(len).values,
        "qualit_score": qualit_scores,
    }).sort_values("contribution_index").reset_index(drop=True)


def extraction_pipeline(
        df: pd.DataFrame, 
        system_prompt: str,
        user_template: str,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        error_filtre: str = "Oui",
        rouge_filter: float = 0.0,
        qualit_filter: float = 0.0
    ) -> pd.DataFrame:
    """Pipeline d'extraction d'idées via LLM avec filtres post-extraction.

    Args:
        df: DataFrame avec colonnes 'author_id' et 'contribution'
        system_prompt: Prompt système pour le LLM
        user_template: Template utilisateur pour le LLM
        error_filtre: "Oui" pour filtrer les erreurs de parsing
        rouge_filter: Seuil minimal pour le score ROUGE (0 = pas de filtre)
        qualit_filter: Seuil minimal pour le score QualIT (0 = pas de filtre)

    Returns:
        DataFrame avec les extractions et leurs scores
    """
    # Vérification du dataframe
    required_columns = {"author_id", "contribution"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Le dataframe doit contenir les colonnes suivantes : {required_columns}")
    
    # Extraction des idées et calcul des scores QualIT
    result = extract_ideas_from_df(df, system_prompt, user_template)
    data_extracted = compute_qualit_scores(df, result, embed_model, device)
    
    # Calcul de la métrique ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    def calc_rouge(row):
        try:
            scores = scorer.score(str(row['extraction']), str(row['contribution']))
            return pd.Series({
                'rouge_score_1gram': scores['rouge1'].fmeasure,
                'rouge_score_L': scores['rougeL'].fmeasure
            })
        except Exception:
            return pd.Series({'rouge_score_1gram': 0.0, 'rouge_score_L': 0.0})
    
    data_extracted[['rouge_score_1gram', 'rouge_score_L']] = data_extracted.apply(calc_rouge, axis=1)
    
    # Filtre des extractions échouées : présence de "[PARSE_FAIL]"
    if error_filtre == "Oui":
        data_extracted = data_extracted[~data_extracted['extraction'].str.contains('[PARSE_FAIL]', na=False)]
    
    # Filtre ROUGE (hallucinations)
    if rouge_filter > 0:
        data_extracted = data_extracted[data_extracted['rouge_score_1gram'] >= rouge_filter]
    
    # Filtre QualIT (qualité faible)
    if qualit_filter > 0:
        data_extracted = data_extracted[data_extracted['qualit_score'] >= qualit_filter]

    # Résultats finaux
    return data_extracted.reset_index(drop=True)