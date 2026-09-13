"""
CLYVO AI — Modelo preditivo de priorização de cuidados

O "Veterinary Clinical Dataset" (Kaggle) não traz uma coluna pronta de
diagnóstico/risco. Por isso este script primeiro DERIVA o rótulo de
prioridade (low/medium/high) a partir de sinais já presentes no dataset —
contagem de sintomas ativos, histórico médico relevante e idade avançada —
e depois treina um classificador para GENERALIZAR esse padrão para novos
pets (inclusive com combinações de sintomas que não apareceram no dataset).

Uso:
    1. Baixe o dataset real do Kaggle e salve em:
       data/veterinary_clinical_dataset.csv
       (ou rode generate_sample_data.py para testar com dados sintéticos)
    2. python train_model.py
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "veterinary_clinical_dataset.csv"  # CSV real do Kaggle
MODEL_PATH = BASE_DIR / "model" / "clyvo_ai_model.joblib"
ENCODER_PATH = BASE_DIR / "model" / "clyvo_ai_encoder.joblib"

# Colunas reais do Veterinary Clinical Dataset (Kaggle).
# AnimalName na verdade contém a espécie (Dog/cat), não um nome.
# Os sintomas têm ~900 valores de texto distintos (erros de digitação,
# sinônimos) — em vez de tratá-los como categorias cruas (o que o modelo
# não consegue generalizar bem), convertemos cada um em um score numérico
# de gravidade (symptom_severity) antes de treinar.
CATEGORICAL_COLS = ["AnimalName", "Breed", "MedicalHistory"]
NUMERIC_COLS = ["Age", "Weight_kg", "symptom_severity_total", "symptom_severity_max"]
TARGET_COL = "risk_priority"  # derivado, não vem pronto no dataset

SYMPTOM_COLS = ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4", "Symptom_5"]

# Palavras-chave associadas a maior gravidade clínica (peso 3) e gravidade
# moderada (peso 2). Qualquer sintoma que não bata com nenhuma das duas
# listas recebe peso 1 (leve/inespecífico). Esta é uma heurística — o
# dataset não traz um rótulo de gravidade pronto, então usamos conhecimento
# de domínio para aproximar o que um veterinário consideraria mais urgente.
CRITICAL_KEYWORDS = [
    "death", "seizure", "paraly", "breath", "collapse", "shock",
    "hemorrhage", "haemorrhage", "coma", "convuls", "distress",
]
MODERATE_KEYWORDS = [
    "pain", "fever", "vomit", "diarrh", "weak", "letharg", "swelling",
    "bleed", "anemia", "anaemia", "dehydrat", "lameness", "lame",
]

# Categorias de MedicalHistory observadas no dataset real, com peso de
# risco associado (0 = sem risco adicional, 3 = risco alto/crônico).
HISTORY_WEIGHTS = {
    "no previous conditions": 0,
    "vaccinated": 0,
    "not vaccinated": 1,
    "allergies": 1,
    "dental issues": 1,
    "parasite history": 1,
    "skin conditions history": 1,
    "recent surgery": 2,
    "previous kidney disease": 2,
    "previous heart condition": 2,
    "chronic illness": 3,
}


def symptom_severity(value: str) -> int:
    v = str(value).strip().lower()
    if any(k in v for k in CRITICAL_KEYWORDS):
        return 3
    if any(k in v for k in MODERATE_KEYWORDS):
        return 2
    return 1


def add_symptom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converte os 5 campos de texto de sintoma em features numéricas de
    gravidade, evitando tratar ~900 textos distintos como categorias cruas.
    """
    df = df.copy()
    severities = df[SYMPTOM_COLS].apply(lambda col: col.map(symptom_severity))
    df["symptom_severity_total"] = severities.sum(axis=1)
    df["symptom_severity_max"] = severities.max(axis=1)
    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva o rótulo de prioridade a partir de gravidade dos sintomas,
    histórico médico e idade.

    Usa divisão em tercis (quantis) sobre o score contínuo, para que a
    distribuição de low/medium/high se ajuste automaticamente aos dados
    reais, em vez de depender de um corte numérico fixo.
    """
    df = add_symptom_features(df)

    history_score = (
        df["MedicalHistory"].astype(str).str.strip().str.lower()
        .map(HISTORY_WEIGHTS).fillna(1)
    )
    senior = (df["Age"] >= 8).astype(int)

    score = df["symptom_severity_total"] + history_score + senior

    try:
        df[TARGET_COL] = pd.qcut(
            score, q=3, labels=["low", "medium", "high"], duplicates="drop"
        )
    except ValueError:
        df[TARGET_COL] = pd.cut(score, bins=3, labels=["low", "medium", "high"])
    df[TARGET_COL] = df[TARGET_COL].astype(str)
    return df


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    raw_required = {"AnimalName", "Breed", "MedicalHistory", "Age", "Weight_kg", *SYMPTOM_COLS}
    missing = raw_required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas esperadas ausentes no dataset: {missing}. "
            "Confira os nomes reais das colunas no CSV baixado do Kaggle."
        )
    return engineer_target(df)


def main():
    df = load_data(DATA_PATH)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = encoder.fit_transform(df[CATEGORICAL_COLS])
    X = pd.concat(
        [pd.DataFrame(X_cat, columns=CATEGORICAL_COLS), df[NUMERIC_COLS].reset_index(drop=True)],
        axis=1,
    )
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Distribuição do rótulo derivado (risk_priority) ===")
    print(df[TARGET_COL].value_counts())
    print("\n=== Resultado no conjunto de teste ===")
    print(classification_report(y_test, y_pred))

    import os
    os.makedirs(BASE_DIR / "model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")
    print(f"Encoder salvo em {ENCODER_PATH}")


if __name__ == "__main__":
    main()
