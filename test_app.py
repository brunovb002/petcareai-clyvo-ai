"""
Testes do endpoint do CLYVO AI. Rodar com: python -m unittest -v
"""
import unittest

from app import app

PET_GRAVE = {
    "AnimalName": "Dog",
    "Breed": "Labrador Retriever",
    "Age": 11,
    "Weight_kg": 28.5,
    "MedicalHistory": "Chronic illness",
    "Symptom_1": "Difficulty in breathing",
    "Symptom_2": "Weakness",
    "Symptom_3": "Fever",
    "Symptom_4": "Lethargy",
    "Symptom_5": "Pain",
}

PET_SAUDAVEL = {
    "AnimalName": "cat",
    "Breed": "Siamese",
    "Age": 2,
    "Weight_kg": 4.2,
    "MedicalHistory": "No previous conditions",
    "Symptom_1": "Sneezing",
}


class ClyvoAiApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_retorna_healthy(self):
        resposta = self.client.get("/health")

        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.get_json()["status"], "Healthy")

    def test_pet_idoso_com_sintomas_graves_tem_prioridade_alta(self):
        resposta = self.client.post("/predict", json=PET_GRAVE)

        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.get_json()["priority"], "high")

    def test_pet_jovem_saudavel_tem_prioridade_baixa(self):
        resposta = self.client.post("/predict", json=PET_SAUDAVEL)

        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.get_json()["priority"], "low")

    def test_confianca_traz_as_tres_classes_somando_1(self):
        confianca = self.client.post("/predict", json=PET_GRAVE).get_json()["confidence"]

        self.assertEqual(set(confianca), {"low", "medium", "high"})
        self.assertAlmostEqual(sum(confianca.values()), 1.0, places=2)

    def test_raca_desconhecida_nao_quebra_o_modelo(self):
        pet = {**PET_SAUDAVEL, "Breed": "Raça inexistente"}

        resposta = self.client.post("/predict", json=pet)

        self.assertEqual(resposta.status_code, 200)

    def test_sem_campos_obrigatorios_retorna_400(self):
        resposta = self.client.post("/predict", json={"AnimalName": "Dog"})

        self.assertEqual(resposta.status_code, 400)
        self.assertIn("Breed", resposta.get_json()["erro"])

    def test_sem_sintomas_retorna_400(self):
        pet = {k: v for k, v in PET_SAUDAVEL.items() if not k.startswith("Symptom")}

        resposta = self.client.post("/predict", json=pet)

        self.assertEqual(resposta.status_code, 400)


if __name__ == "__main__":
    unittest.main()
