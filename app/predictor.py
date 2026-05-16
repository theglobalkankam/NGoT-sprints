import joblib 
import numpy as np 
from pathlib import Path 
from typing import Optional 
  
MODEL_PATH = Path('models/eta_model_latest.joblib') 
  
  
class ETAPredictor: 
    """ 
    Wraps the trained scikit-learn pipeline. 
    Handles loading, prediction, and confidence intervals. 
    """ 
  
    def __init__(self): 
        self._model = None 
        self._model_version = 'unknown' 
  
    def load(self, path: Path = MODEL_PATH) -> bool: 
        """Load the model from disk. Returns True if successful.""" 
        try: 
            self._model = joblib.load(path) 
            self._model_version = path.stem  # Use filename as version 
            print(f'Model loaded from {path}') 
            return True 
        except FileNotFoundError: 
            print(f'WARNING: Model file not found at {path}') 
            return False 
        except Exception as e: 
            print(f'ERROR loading model: {e}') 
            return False 
  
    @property 
    def is_loaded(self) -> bool: 
        return self._model is not None 
  
    @property
    def version(self) -> str: 
        return self._model_version 
  
    def predict(self, feature_vector: list[float]) -> tuple[float, float, float]: 
        """ 
        Run prediction on a feature vector. 
        Returns: (eta_minutes, confidence_low, confidence_high) 
        """ 
        if not self.is_loaded: 
            raise RuntimeError('Model is not loaded. Call load() first.') 
  
        X = np.array([feature_vector])  # Shape: (1, n_features) 
        eta = float(self._model.predict(X)[0]) 
  
        # Simple confidence interval: ±20% (in production, use proper uncertainty) 
        confidence_low  = max(0, eta * 0.80) 
        confidence_high = eta * 1.20 
  
        return round(eta, 1), round(confidence_low, 1), round(confidence_high, 1) 