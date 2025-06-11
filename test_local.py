#!/usr/bin/env python3
"""
Simple test script to validate the Pyrosage model service locally
"""

import requests
import json

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction():
    """Test a simple prediction"""
    test_data = {
        "dataset": {
            "input": [
                {
                    "smiles": "CCO",  # Ethanol
                    "model_name": "AMES",
                    "jaqpotRowId": 1
                },
                {
                    "smiles": "c1ccccc1",  # Benzene
                    "model_name": "KOW",
                    "jaqpotRowId": 2
                }
            ]
        },
        "model": {
            "dependent_features": [{"key": "prediction"}]
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/infer",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        print(f"Prediction status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Prediction results:")
            for i, pred in enumerate(result.get("predictions", [])):
                print(f"  Row {i+1}: {json.dumps(pred, indent=2)}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Prediction test failed: {e}")
        return False

def test_invalid_model():
    """Test prediction with invalid model name"""
    test_data = {
        "dataset": {
            "input": [
                {
                    "smiles": "CCO",
                    "model_name": "INVALID_MODEL",
                    "jaqpotRowId": 1
                }
            ]
        },
        "model": {
            "dependent_features": [{"key": "prediction"}]
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/infer",
            headers={"Content-Type": "application/json"},
            json=test_data
        )
        print(f"Invalid model test status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            pred = result.get("predictions", [{}])[0]
            if "error" in pred:
                print(f"Expected error caught: {pred['error']}")
                return True
        return False
    except Exception as e:
        print(f"Invalid model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Pyrosage Docker service...")
    print("Make sure the service is running on http://localhost:8000")
    print("=" * 50)
    
    # Run tests
    health_ok = test_health_check()
    print()
    
    pred_ok = test_prediction()
    print()
    
    error_ok = test_invalid_model()
    print()
    
    # Summary
    print("=" * 50)
    print("Test Summary:")
    print(f"Health check: {'✓' if health_ok else '✗'}")
    print(f"Prediction: {'✓' if pred_ok else '✗'}")
    print(f"Error handling: {'✓' if error_ok else '✗'}")
    
    all_passed = health_ok and pred_ok and error_ok
    print(f"Overall: {'✓ All tests passed' if all_passed else '✗ Some tests failed'}")