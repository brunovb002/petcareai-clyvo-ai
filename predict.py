"""
CLYVO AI — Predição de prioridade de cuidado para um pet

Carrega o modelo treinado e devolve o nível de prioridade (low/medium/high)
para os dados de um pet específico. Este é o núcleo do que futuramente vira
um endpoint (Sprint 4) consumido pela API/app do CLYVO VET.
"""
import joblib
import pandas as pd
from pathlib import Path
from train_model import (
    add_symptom_features,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "clyvo_ai_model.joblib"
ENCODER_PATH = BASE_DIR / "model" / "clyvo_ai_encoder.joblib"


def predict_priority(pet: dict) -> dict:
    """
    pet: dicionário com as chaves AnimalName (especie), Breed, Age,
         Weight_kg, MedicalHistory, Symptom_1..Symptom_5
    """
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

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
