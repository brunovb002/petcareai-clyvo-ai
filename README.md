# CLYVO AI — Assistente Inteligente de Saúde Pet

Componente de Inteligência Artificial do projeto **PetCareAI+**, solução do grupo para o desafio da **CLYVO VET** (FIAP Challenge 2026),
responsável por analisar dados do pet e gerar uma **priorização de cuidados**
(baixa / média / alta), apoiando o tutor a agir preventivamente antes de um
problema de saúde se agravar.

## Integrantes do Grupo

* **Bruno Vinicius Barbosa** – RM 566366
* **Guilherme de Andrade Martini** – RM 566087
* **Nathan Gonçalves Pereira Mendes** – RM 564666
* **Raphael Gomes Mancera** – RM 562279

## Problema de negócio

Tutores frequentemente perdem o timing de vacinas, consultas de rotina e
cuidados preventivos por falta de acompanhamento ativo. O CLYVO AI resolve
isso analisando o histórico e os sintomas do pet e sinalizando quando ele
precisa de atenção prioritária.

## Abordagem escolhida

**Modelo preditivo (classificação supervisionada)** usando Random Forest,
treinado sobre um dataset clínico veterinário real
([Veterinary Clinical Dataset](https://www.kaggle.com/datasets/lolhaterbro/veteriary-clinical-dataset),
Kaggle, 10.000 registros, licença MIT).

O dataset original não traz uma coluna pronta de diagnóstico/risco — apenas
`AnimalName` (na prática, a espécie), `Breed`, `Age`, `Weight_kg`,
`MedicalHistory` e `Symptom_1..Symptom_5` (texto livre, ~900 valores únicos
distintos, com bastante variação de digitação/sinônimos). Por isso o
rótulo de prioridade (`risk_priority`: low/medium/high) é **derivado**:

1. Cada sintoma recebe um **score de gravidade** (1 a 3) por correspondência
   com palavras-chave associadas a quadros críticos (`death`, `seizure`,
   `difficulty in breathing` etc.) ou moderados (`fever`, `pain`, `weakness`
   etc.), definidas com base em conhecimento de domínio.
2. Cada categoria de `MedicalHistory` recebe um peso de risco (0 a 3),
   por exemplo `chronic illness` pesa mais que `no previous conditions`.
3. Idade ≥ 8 anos soma +1 ao score (indicador de pet sênior).
4. O score final é dividido em tercis (`low`/`medium`/`high`), garantindo
   uma distribuição balanceada independente da distribuição bruta dos dados.

O modelo é então treinado com Random Forest para **generalizar** esse
padrão — inclusive para combinações de sintomas, raça e idade que não
apareceram exatamente iguais no dataset de treino.

## Dados utilizados

| Campo | Origem | Uso |
|---|---|---|
| AnimalName (espécie), Breed, Age, Weight_kg | Cadastro do pet (banco PetCareAI+) | Perfil demográfico |
| MedicalHistory | Prontuário do pet | Sinal de condição pré-existente |
| Symptom_1..Symptom_5 | Relato do tutor / consulta | Sinais clínicos atuais |
| risk_priority (target) | **Derivado** a partir das colunas acima (ver `engineer_target()` em `train_model.py`) | O que o modelo aprende a prever |

## Arquitetura de integração (planejada para a Sprint 4)

![Arquitetura CLYVO AI](docs/clyvo_ai_arquitetura.png)

```
App Mobile (tutor)
      |  consulta dados do pet
      v
API (.NET) ------> Banco de Dados (pets, vacinas, consultas)
      |
      |  chama endpoint de IA com os dados do pet
      v
Servico CLYVO AI (Python / Flask, expoe predict_priority)
      |
      v
Modelo treinado (Random Forest) -> prioridade + confianca
      |
      v
API devolve recomendacao ------> App exibe ao tutor
```

## Estrutura do projeto

```
clyvo_ai/
├── data/
│   └── veterinary_clinical_dataset.csv   # dataset real (Kaggle)
├── docs/
│   └── clyvo_ai_arquitetura.png          # diagrama arquitetural
├── model/
│   ├── clyvo_ai_model.joblib             # modelo treinado
│   └── clyvo_ai_encoder.joblib           # encoder das colunas categóricas
├── train_model.py                        # treina e salva o modelo
├── predict.py                            # carrega o modelo e faz predições
└── README.md
```

## Como usar

```bash
pip install pandas scikit-learn joblib

# 1. Baixe o dataset real do Kaggle em data/veterinary_clinical_dataset.csv

# 2. Treine o modelo:
python train_model.py

# 3. Rode uma predição de exemplo:
python predict.py
```

## Resultados parciais (Sprint 3)

Modelo treinado com o dataset real (10.000 registros): **90% de acurácia**
geral na classificação de prioridade (low/medium/high).

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| high | 0.95 | 0.71 | 0.81 |
| medium | 0.83 | 0.96 | 0.89 |
| low | 0.97 | 0.92 | 0.94 |

O recall menor na classe "high" é esperado — é a menor classe (1.946 de
10.000 registros) e a mais heterogênea, já que reúne casos que combinam
diferentes fatores de risco. Em produção (Sprint 4), esse é o ponto a
monitorar de perto, já que falsos negativos em "high" são o cenário mais
custoso (deixar de sinalizar um caso realmente urgente).

Testes qualitativos confirmam o comportamento esperado do modelo:
- Cão idoso, doença crônica, sintomas graves (dificuldade respiratória,
  fraqueza, febre) → **high priority (78.8% de confiança)**
- Gato jovem, saudável, sem histórico, sintomas leves → **low priority
  (99.7% de confiança)**

## Próximos passos (Sprint 4)

- Investigar o recall mais baixo da classe "high" (ex: ajustar pesos de
  classe no Random Forest, ou revisar a lista de palavras-chave críticas)
- Expor `predict_priority()` como endpoint REST (Flask)
- Integrar o endpoint à API .NET do PetCareAI+
- Exibir a recomendação na tela do app mobile
- Testar em cenário real de uso com dados do banco do PetCareAI+

## Tecnologias

Python, pandas, scikit-learn (RandomForestClassifier), joblib
