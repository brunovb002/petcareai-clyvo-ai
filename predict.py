"""
CLYVO AI — Predição de prioridade de cuidado para um pet

Carrega o modelo treinado e devolve o nível de prioridade (low/medium/high)
para os dados de um pet específico. Exposto como endpoint REST em app.py.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from train_model import (
    add_symptom_features,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "clyvo_ai_model.joblib"
ENCODER_PATH = BASE_DIR / "model" / "clyvo_ai_encoder.joblib"


@lru_cache(maxsize=1)
def carregar_modelo():
    """Lê o modelo e o encoder do disco uma única vez (e não a cada requisição)."""
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)


def predict_priority(pet: dict) -> dict:
    """
    pet: dicionário com as chaves AnimalName (especie), Breed, Age,
         Weight_kg, MedicalHistory, Symptom_1..Symptom_5
    """
    model, encoder = carregar_modelo()

    df = pd.DataFrame([pet])
    df = add_symptom_features(df)

    X_cat = encoder.transform(df[CATEGORICAL_COLS])
    X = pd.concat(
        [pd.DataFrame(X_cat, columns=CATEGORICAL_COLS), df[NUMERIC_COLS].reset_index(drop=True)],
        axis=1,
    )

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_

    return {
        "priority": prediction,
        "confidence": dict(zip(classes, [round(float(p), 3) for p in probabilities])),
    }


if __name__ == "__main__":
    # Exemplo: pet idoso, com doença crônica e sintomas graves
    example_pet = {
        "AnimalName": "Dog",
        "Breed": "Labrador",
        "Age": 11,
        "Weight_kg": 28.5,
        "MedicalHistory": "Chronic illness",
        "Symptom_1": "Difficulty in breathing",
        "Symptom_2": "Weakness",
        "Symptom_3": "Fever",
        "Symptom_4": "Lethargy",
        "Symptom_5": "Pain",
    }
    result = predict_priority(example_pet)
    print("Pet de exemplo:", example_pet)
    print("Resultado:", result)

    # Exemplo: pet jovem, saudável, sintomas leves
    healthy_pet = {
        "AnimalName": "cat",
        "Breed": "Siamese",
        "Age": 2,
        "Weight_kg": 4.2,
        "MedicalHistory": "No previous conditions",
        "Symptom_1": "Sneezing",
        "Symptom_2": "Coughing",
        "Symptom_3": "Sneezing",
        "Symptom_4": "Coughing",
        "Symptom_5": "Sneezing",
    }
    result2 = predict_priority(healthy_pet)
    print("\nPet de exemplo (saudável):", healthy_pet)
    print("Resultado:", result2)
