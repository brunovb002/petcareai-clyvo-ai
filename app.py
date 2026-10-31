"""
CLYVO AI — API REST (Flask) que expõe o modelo de prioridade de cuidado.

Consumida pela API .NET do PetCareAI no endpoint POST /api/pets/{id}/avaliacao-risco:
a API junta os dados do pet (banco) com os sintomas relatados pelo tutor e chama POST /predict.
"""
from flask import Flask, jsonify, request

from predict import predict_priority

app = Flask(__name__)

CAMPOS_OBRIGATORIOS = ["AnimalName", "Breed", "Age", "Weight_kg", "MedicalHistory"]
SINTOMAS = [f"Symptom_{i}" for i in range(1, 6)]


@app.get("/health")
def health():
    return jsonify(status="Healthy")


@app.post("/predict")
def predict():
    pet = request.get_json(silent=True) or {}

    faltando = [campo for campo in CAMPOS_OBRIGATORIOS if campo not in pet]
    if faltando:
        return jsonify(erro=f"Campos obrigatórios ausentes: {', '.join(faltando)}"), 400
    if not any(pet.get(s) for s in SINTOMAS):
        return jsonify(erro="Informe pelo menos um sintoma (Symptom_1 a Symptom_5)."), 400

    # O modelo espera as 5 colunas de sintoma; as não informadas vão vazias
    for sintoma in SINTOMAS:
        pet.setdefault(sintoma, "")

    return jsonify(predict_priority(pet))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
