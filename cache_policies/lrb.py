"""Learning Relaxed Belady (LRB) Cache Eviction Policy"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder
from .base import CachePolicy


class LRBPolicy(CachePolicy):
    """
    LRB eviction policy - uses ML models to predict future access patterns
    """
    
    def __init__(self, cache_size_mb: float = 100, 
                 model_prefix: str = "web_lrb_model",
                 model_dir: str = "ml/"):
        """
        Initialize LRB policy with trained models
        
        Args:
            cache_size_mb: Maximum cache size in megabytes
            model_prefix: Prefix for model files
            model_dir: Directory containing model files
        """
        super().__init__(cache_size_mb)
        self.models = {}
        self.model_info = {}
        self.label_encoders = {}
        self.model_dir = model_dir
        self.model_prefix = model_prefix
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained LRB models"""
        model_types = ['binary', 'distance', 'reuse']
        
        for model_type in model_types:
            model_path = os.path.join(self.model_dir, f"{self.model_prefix}_{model_type}.pkl")
            if os.path.exists(model_path):
                try:
                    self.models[model_type] = joblib.load(model_path)
                    print(f"Loaded {model_type} model from {model_path}")
                except Exception as e:
                    print(f"Failed to load {model_type} model: {e}")
        
        # Load model info if available
        info_path = os.path.join(self.model_dir, f"{self.model_prefix}_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
            except Exception as e:
                print(f"Failed to load model info: {e}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for LRB models"""
        df = df.copy()
        
        # Encode file types
        if 'type' in df.columns:
            le = LabelEncoder()
            # Handle unknown types
            df['type'] = df['type'].fillna('unknown')
            unique_types = df['type'].unique()
            le.fit(unique_types)
            df['file_type_encoded'] = le.transform(df['type'])
        else:
            df['file_type_encoded'] = 0
        
        # Add log-transformed size
        if 'size' in df.columns:
            df['log_file_size'] = np.log1p(df['size'])
        else:
            df['log_file_size'] = 0
        
        # Add derived features
        if 'age_seconds' in df.columns:
            df['age_hours'] = df['age_seconds'] / 3600
            df['age_days'] = df['age_seconds'] / 86400
        
        if 'access_count' in df.columns and 'age_hours' in df.columns:
            df['access_rate'] = df['access_count'] / (df['age_hours'] + 1)
            df['recency_score'] = df['access_count'] / np.log2(df['age_hours'] + 2)
        
        return df
    
    def _predict_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using LRB models"""
        df = self._prepare_features(df)
        
        # Define features expected by models
        base_features = ['access_count', 'age_since_last_access', 'file_type_encoded', 'log_file_size']
        
        # Initialize default eviction scores
        df['future_access_prob'] = 0.5
        df['next_access_distance'] = 100
        df['reuse_probability'] = 0.5
        
        # Binary prediction - probability of future access
        if 'binary' in self.models:
            try:
                features_available = [f for f in base_features if f in df.columns]
                if len(features_available) == len(base_features):
                    X = df[base_features]
                    proba = self.models['binary'].predict_proba(X)
                    df['future_access_prob'] = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
            except Exception as e:
                print(f"Binary prediction error: {e}")
        
        # Distance prediction - distance to next access
        if 'distance' in self.models:
            try:
                X = df[base_features]
                df['next_access_distance'] = np.maximum(0, self.models['distance'].predict(X))
            except Exception as e:
                print(f"Distance prediction error: {e}")
        
        # Reuse prediction
        if 'reuse' in self.models:
            try:
                X = df[base_features]
                df['reuse_probability'] = np.clip(self.models['reuse'].predict(X), 0, 1)
            except Exception as e:
                print(f"Reuse prediction error: {e}")
        
        # Calculate combined eviction score
        # Higher score = evict first
        # Components:
        # - Lower future access probability = higher eviction priority
        # - Longer distance to next access = higher eviction priority  
        # - Lower reuse probability = higher eviction priority
        
        weights = {'binary': 0.4, 'distance': 0.4, 'reuse': 0.2}
        
        eviction_score = (
            (1 - df['future_access_prob']) * weights['binary'] +
            (df['next_access_distance'] / df['next_access_distance'].max()) * weights['distance'] +
            (1 - df['reuse_probability']) * weights['reuse']
        )
        
        df['eviction_score'] = eviction_score
        
        return df
    
    def score_items(self, cache_index: Dict[str, Any]) -> pd.DataFrame:
        """
        Score items using LRB models
        """
        records = []
        now = datetime.utcnow()
        
        for file_id, metadata in cache_index.items():
            try:
                last_access = datetime.fromisoformat(metadata['last_access'])
                age_seconds = (now - last_access).total_seconds()
                
                records.append({
                    'file_id': file_id,
                    'size': metadata.get('size', 0),
                    'access_count': metadata.get('access_count', 0),
                    'age_since_last_access': age_seconds,
                    'age_seconds': age_seconds,
                    'type': metadata.get('type', 'unknown'),
                    'last_access': metadata['last_access']
                })
            except Exception as e:
                print(f"Error processing {file_id}: {e}")
                continue
        
        df = pd.DataFrame(records)
        if df.empty:
            return df
        
        # If models are loaded, use them for scoring
        if self.models:
            df = self._predict_scores(df)
        else:
            # Fallback to LRU-like behavior if models unavailable
            print("LRB models not available, falling back to LRU-like scoring")
            df['eviction_score'] = df['age_seconds']
        
        # Sort by eviction score (descending - highest score evicted first)
        return df.sort_values('eviction_score', ascending=False)
