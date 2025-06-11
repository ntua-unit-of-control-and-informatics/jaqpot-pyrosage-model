# Jaqpot Pyrosage Model

Docker image for deploying Pyrosage AttentiveFP models on the Jaqpot platform. This service provides predictions for environmental and toxicity properties of chemical compounds using Graph Neural Networks.

## Available Models

### Classification Models
- **AMES**: Mutagenicity prediction
- **Endocrine_Disruption_NR-AR**: Androgen receptor disruption
- **Endocrine_Disruption_NR-AhR**: Aryl hydrocarbon receptor disruption  
- **Endocrine_Disruption_NR-ER**: Estrogen receptor disruption
- **Endocrine_Disruption_NR-aromatase**: Aromatase disruption
- **Irritation_Corrosion_Eye_Corrosion**: Eye corrosion prediction
- **Irritation_Corrosion_Eye_Irritation**: Eye irritation prediction

### Regression Models
- **FBA**: Bioaccumulation Factor
- **FBC**: Bioconcentration Factor
- **KH**: Henry's Law Constant
- **KOA**: Octanol-Air Partition Coefficient
- **KOC**: Soil/Water Partition Coefficient
- **KOW**: Octanol-Water Partition Coefficient (LogP)
- **LC50**: Aquatic toxicity
- **LD50_Zhu**: Acute oral toxicity
- **PLV**: Vapor pressure related
- **SW**: Water solubility
- **TBP**: Biodegradation related
- **TMP**: Melting point related
- **kAOH**: Aqueous hydroxyl rate
- **pKa_acidic**: Acidic pKa
- **pKa_basic**: Basic pKa
- **tbiodeg**: Biodegradation time
- **tfishbio**: Fish bioaccumulation time

## Usage

### Input Format

The service expects input with two required columns:
- `smiles`: SMILES string representation of the molecule
- `model_name`: Name of the model to use for prediction

Example input:
```json
{
  "dataset": {
    "input": [
      {
        "smiles": "CCO",
        "model_name": "AMES",
        "jaqpotRowId": 1
      },
      {
        "smiles": "c1ccccc1",
        "model_name": "KOW", 
        "jaqpotRowId": 2
      }
    ]
  }
}
```

### Output Format

The service returns predictions with metadata:

For classification models:
```json
{
  "predictions": [
    {
      "prediction": 0.23,
      "jaqpotMetadata": {
        "jaqpotRowId": 1,
        "model_name": "AMES",
        "model_type": "classification",
        "smiles": "CCO",
        "probability": 0.557,
        "predicted_class": 1
      }
    }
  ]
}
```

For regression models:
```json
{
  "predictions": [
    {
      "prediction": 2.15,
      "jaqpotMetadata": {
        "jaqpotRowId": 2,
        "model_name": "KOW",
        "model_type": "regression", 
        "smiles": "c1ccccc1"
      }
    }
  ]
}
```

## Building and Running

### Build Docker Image
```bash
docker build -t jaqpot-pyrosage-model .
```

### Run Locally
```bash
docker run -p 8000:8000 jaqpot-pyrosage-model
```

### Test Health Check
```bash
curl http://localhost:8000/health
```

### Test Prediction
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {
      "input": [
        {
          "smiles": "CCO",
          "model_name": "AMES",
          "jaqpotRowId": 1
        }
      ]
    },
    "model": {
      "dependent_features": [{"key": "prediction"}]
    }
  }'
```

## Model Details

All models are based on AttentiveFP (Attention-based Fingerprint) architecture:
- **Input**: Enhanced molecular graphs with 10-dimensional atom features and 6-dimensional bond features
- **Architecture**: Graph Neural Network with attention mechanism
- **Features**: 
  - Atom features: atomic number, degree, formal charge, hydrogens, radical electrons, aromaticity, ring membership, hybridization
  - Bond features: bond type, conjugation, ring membership

## Error Handling

If a prediction fails (e.g., invalid SMILES, unknown model), the service returns:
```json
{
  "predictions": [
    {
      "prediction": null,
      "error": "Error description",
      "jaqpotMetadata": {
        "jaqpotRowId": 1,
        "model_name": "AMES",
        "smiles": "invalid_smiles",
        "status": "failed"
      }
    }
  ]
}
```