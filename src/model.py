import os
import torch
from torch_geometric.nn import AttentiveFP
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit import RDLogger
import pandas as pd
from typing import Dict, List, Any
from jaqpot_api_client.models.prediction_request import PredictionRequest
from jaqpot_api_client.models.prediction_response import PredictionResponse

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


class PyrosageModelService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_info = {}
        self._load_all_models()

    def _load_all_models(self):
        """Load all available Pyrosage models"""
        models_dir = "models"
        
        # Load classification models
        classification_dir = os.path.join(models_dir, "classification")
        if os.path.exists(classification_dir):
            for file in os.listdir(classification_dir):
                if file.endswith("_attentivefp_best.pt"):
                    model_name = file.replace("_attentivefp_best.pt", "")
                    model_path = os.path.join(classification_dir, file)
                    try:
                        self.models[model_name] = self._load_model(model_path)
                        self.model_info[model_name] = {"type": "classification", "path": model_path}
                        print(f"Loaded classification model: {model_name}")
                    except Exception as e:
                        print(f"Failed to load model {model_name}: {e}")
        
        # Load regression models
        regression_dir = os.path.join(models_dir, "regression")
        if os.path.exists(regression_dir):
            for file in os.listdir(regression_dir):
                if file.endswith("_attentivefp_best.pt"):
                    model_name = file.replace("_attentivefp_best.pt", "")
                    model_path = os.path.join(regression_dir, file)
                    try:
                        self.models[model_name] = self._load_model(model_path)
                        self.model_info[model_name] = {"type": "regression", "path": model_path}
                        print(f"Loaded regression model: {model_name}")
                    except Exception as e:
                        print(f"Failed to load model {model_name}: {e}")
        
        print(f"Total models loaded: {len(self.models)}")

    def _load_model(self, model_path: str):
        """Load a single AttentiveFP model"""
        model_dict = torch.load(model_path, map_location=self.device)
        state_dict = model_dict['model_state_dict']
        hyperparams = model_dict['hyperparameters']

        # Create model with correct feature dimensions
        model = AttentiveFP(
            in_channels=10,  # Enhanced atom features (10 dimensions)
            hidden_channels=hyperparams["hidden_channels"],
            out_channels=1,
            edge_dim=6,  # Enhanced bond features (6 dimensions)
            num_layers=hyperparams["num_layers"],
            num_timesteps=hyperparams["num_timesteps"],
            dropout=hyperparams["dropout"],
        ).to(self.device)

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _smiles_to_data(self, smiles: str) -> Data:
        """Convert SMILES string to PyG Data object with enhanced features"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        # Enhanced atom features (10 dimensions)
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                # Hybridization as one-hot (3 dimensions)
                int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP),
                int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
                int(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
            ]
            atom_features.append(features)

        x = torch.tensor(atom_features, dtype=torch.float)

        # Enhanced bond features (6 dimensions)
        edges_list = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges_list.extend([[i, j], [j, i]])

            features = [
                # Bond type as one-hot (4 dimensions)
                int(bond.GetBondType() == Chem.rdchem.BondType.SINGLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.TRIPLE),
                int(bond.GetBondType() == Chem.rdchem.BondType.AROMATIC),
                # Additional features (2 dimensions)
                int(bond.GetIsConjugated()),
                int(bond.IsInRing())
            ]
            edge_features.extend([features, features])

        if not edges_list:  # Skip molecules with no bonds
            raise ValueError(f"Molecule has no bonds: {smiles}")

        edge_index = torch.tensor(edges_list, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make predictions using specified model"""
        # Convert input to DataFrame
        input_data = pd.DataFrame(request.dataset.input)
        
        # Validate required columns
        if 'smiles' not in input_data.columns:
            raise ValueError("Input must contain 'smiles' column")
        
        if 'model_name' not in input_data.columns:
            raise ValueError("Input must contain 'model_name' column")

        # Get dependent feature keys
        dependent_feature_keys = [feature.key for feature in request.model.dependent_features]

        prediction_results = []
        
        for i, row in input_data.iterrows():
            smiles = row['smiles']
            model_name = row['model_name']
            jaqpot_row_id = row.get('jaqpotRowId', i)
            
            try:
                # Check if model exists
                if model_name not in self.models:
                    available_models = list(self.models.keys())
                    raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
                
                # Get model and convert SMILES to graph
                model = self.models[model_name]
                model_type = self.model_info[model_name]['type']
                
                # Convert SMILES to graph data
                data = self._smiles_to_data(smiles)
                data = data.to(self.device)
                
                # Create batch for single molecule
                batch = Batch.from_data_list([data])
                
                # Make prediction
                with torch.no_grad():
                    prediction = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    pred_value = prediction.cpu().numpy().flatten()[0]

                # Prepare metadata (augmented below for classification)
                metadata = {
                    "jaqpotRowId": jaqpot_row_id,
                    "model_name": model_name,
                    "model_type": model_type,
                    "smiles": smiles,
                }

                # Build prediction output depending on model type
                if model_type == "classification":
                    # Convert logit to probability and hard class (0/1)
                    prob = torch.sigmoid(torch.tensor(pred_value)).item()
                    pred_class = int(prob >= 0.5)
                    if len(dependent_feature_keys) > 0:
                        prediction_dict = {dependent_feature_keys[0]: pred_class}
                    else:
                        prediction_dict = {"prediction": pred_class}
                    # Enrich metadata
                    metadata["probability"] = float(prob)
                    metadata["predicted_class"] = pred_class
                else:
                    # Regression returns continuous value
                    if len(dependent_feature_keys) > 0:
                        prediction_dict = {dependent_feature_keys[0]: float(pred_value)}
                    else:
                        prediction_dict = {"prediction": float(pred_value)}

                prediction_dict["jaqpotMetadata"] = metadata
                prediction_results.append(prediction_dict)
                
            except Exception as e:
                # Return error information for this row
                error_dict = {
                    "error": str(e),
                    "jaqpotMetadata": {
                        "jaqpotRowId": jaqpot_row_id,
                        "model_name": model_name,
                        "smiles": smiles,
                        "status": "failed"
                    }
                }
                if len(dependent_feature_keys) > 0:
                    error_dict[dependent_feature_keys[0]] = None
                else:
                    error_dict["prediction"] = None
                    
                prediction_results.append(error_dict)

        return PredictionResponse(predictions=prediction_results)

    def get_available_models(self) -> Dict[str, Any]:
        """Return information about available models"""
        return {
            "models": list(self.models.keys()),
            "model_info": self.model_info,
            "total_models": len(self.models)
        }
