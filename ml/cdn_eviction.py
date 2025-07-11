import os
import json
import joblib
import paramiko
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Dict, List, Tuple, Optional

CONFIG = {
    "cdn_host": "34.101.140.128", # 
    "cdn_user": "hnfxrt",
    "private_key_path": "/home/hnfxrt/gcp/gcp",
    "remote_cache_index": "/home/hnfxrt/cache_index.json",
    "remote_cache_dir": "/var/cache/nginx/media/",
    "local_model_paths": {
        "binary": "web_lrb_model_binary.pkl",      # Hit/miss prediction
        "distance": "web_lrb_model_distance.pkl",  # Distance to next access
        "reuse": "web_lrb_model_reuse.pkl"         # Reuse probability
    },
    "model_info_path": "web_lrb_model_info.json",
    "local_index_copy": "cache_index.json",
    "cache_limit_mb": 100,
    "supported_types": {"image", "video", "other"},
    "eviction_strategy": "lrb",  # ("lrb", "binary", "distance", "reuse")
    "ensemble_weights": {
        "binary": 0.4,    # Weight for future access probability
        "distance": 0.4,  # Weight for distance to next access
        "reuse": 0.2      # Weight for reuse probability
    }
}

# LRB Model
class LRBModelEnsemble:
    def __init__(self, model_paths: Dict[str, str], info_path: str):
        self.models = {}
        self.model_info = {}
        self.label_encoders = {}
        
        # Load model info if available
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
        
        # Load models
        for model_name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[model_name] = joblib.load(path)
                    print(f"Loaded LRB {model_name} model from {path}")
                except Exception as e:
                    print(f"Failed to load LRB {model_name} model: {e}")
            else:
                print(f"LRB model file not found: {path}")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Encode categorical features
        if "file_type" in df.columns:
            le = LabelEncoder()
            df["file_type_encoded"] = le.fit_transform(df["file_type"])
            self.label_encoders["file_type"] = le
        
        # Log transform file size (common in cache systems)
        if "file_size" in df.columns:
            df["log_file_size"] = df["file_size"].apply(lambda x: np.log1p(max(1, x)))
        
        # Time-based features for temporal patterns
        if "age_since_last_access" in df.columns:
            df["age_hours"] = df["age_since_last_access"] / 3600  # Convert to hours
            df["age_days"] = df["age_since_last_access"] / 86400  # Convert to days
        
        # Access pattern features
        if "access_count" in df.columns and "age_since_last_access" in df.columns:
            # Access rate per hour
            df["access_rate"] = df["access_count"] / (df["age_hours"] + 1)
            # Recency-weighted access score
            df["recency_score"] = df["access_count"] / np.log2(df["age_hours"] + 2)
        
        # Belady-inspired features: stack distance approximation
        if "access_count" in df.columns and "file_size" in df.columns:
            # Approximate stack distance using size and access patterns
            df["stack_distance_approx"] = df["file_size"] / (df["access_count"] + 1)
        
        return df
    
    def predict_belady_relaxed(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        df = self.prepare_features(df)
        
        # Define feature sets for each model
        base_features = ["access_count", "age_since_last_access", "file_type_encoded", "log_file_size"]
        extended_features = base_features + ["age_hours", "access_rate", "recency_score"]
        
        predictions = {}
        
        # Binary model - predicts if object will be accessed again (Belady's binary decision)
        if "binary" in self.models:
            try:
                features = [f for f in base_features if f in df.columns]
                if len(features) == len(base_features):
                    proba = self.models["binary"].predict_proba(df[features])
                    # Probability of future access
                    predictions["future_access_prob"] = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
                else:
                    print(f"Missing features for binary model: {set(base_features) - set(df.columns)}")
            except Exception as e:
                print(f"Error in binary prediction: {e}")
        
        # Distance model - predicts distance to next access (Belady's timing)
        if "distance" in self.models:
            try:
                features = [f for f in extended_features if f in df.columns]
                if features:
                    distance_pred = self.models["distance"].predict(df[features])
                    # Distance to next access (higher = further in future)
                    predictions["next_access_distance"] = np.maximum(0, distance_pred)
            except Exception as e:
                print(f"Error in distance prediction: {e}")
        
        # Reuse model - predicts reuse probability (Belady's utility)
        if "reuse" in self.models:
            try:
                features = [f for f in extended_features if f in df.columns]
                if features:
                    reuse_pred = self.models["reuse"].predict(df[features])
                    # Probability of reuse
                    predictions["reuse_probability"] = np.clip(reuse_pred, 0, 1)
            except Exception as e:
                print(f"Error in reuse prediction: {e}")
        
        # Add predictions to dataframe
        for pred_name, pred_values in predictions.items():
            df[pred_name] = pred_values
        
        return df, predictions
    
    def calculate_belady_score(self, df: pd.DataFrame, strategy: str = "lrb") -> pd.DataFrame:
        df, predictions = self.predict_belady_relaxed(df)
        
        if strategy == "lrb" and len(predictions) > 1:
            # LRB approach: combine predictions to approximate Belady's decision
            weights = CONFIG["ensemble_weights"]
            
            # Start with base eviction priority
            eviction_scores = []
            
            if "future_access_prob" in predictions:
                # Lower probability of future access = higher eviction priority
                access_score = (1 - df["future_access_prob"]) * weights.get("binary", 0.4)
                eviction_scores.append(access_score)
            
            if "next_access_distance" in predictions:
                # Longer distance to next access = higher eviction priority (core Belady principle)
                # Normalize distance scores
                max_distance = df["next_access_distance"].max()
                if max_distance > 0:
                    distance_score = (df["next_access_distance"] / max_distance) * weights.get("distance", 0.4)
                    eviction_scores.append(distance_score)
            
            if "reuse_probability" in predictions:
                # Lower reuse probability = higher eviction priority
                reuse_score = (1 - df["reuse_probability"]) * weights.get("reuse", 0.2)
                eviction_scores.append(reuse_score)
            
            if eviction_scores:
                df["belady_score"] = sum(eviction_scores)
            else:
                df["belady_score"] = 0.5  # Default neutral score
        
        elif strategy == "binary" and "future_access_prob" in predictions:
            df["belady_score"] = 1 - df["future_access_prob"]
        
        elif strategy == "distance" and "next_access_distance" in predictions:
            max_distance = df["next_access_distance"].max()
            df["belady_score"] = df["next_access_distance"] / max_distance if max_distance > 0 else 0.5
        
        elif strategy == "reuse" and "reuse_probability" in predictions:
            df["belady_score"] = 1 - df["reuse_probability"]
        
        else:
            # Fallback to LRU-like heuristic when LRB models unavailable
            print("Using fallback LRU-like heuristic (LRB models unavailable)")
            df["belady_score"] = (
                (df["age_since_last_access"] / df["age_since_last_access"].max()) * 0.6 +
                (1 / (df["access_count"] + 1)) * 0.4
            )
        
        # Sort by Belady score (higher score = evict first, following Belady's principle)
        return df.sort_values(by="belady_score", ascending=False)

# SSH Utilities
def establish_ssh_connection():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CONFIG["cdn_host"], username=CONFIG["cdn_user"], key_filename=CONFIG["private_key_path"])
    return ssh, ssh.open_sftp()

# Data Handling
def load_cache_index(sftp):
    sftp.get(CONFIG["remote_cache_index"], CONFIG["local_index_copy"])
    with open(CONFIG["local_index_copy"]) as f:
        return json.load(f)

def save_cache_index(sftp, cache_index, ssh):
    with open(CONFIG["local_index_copy"], "w") as f:
        json.dump(cache_index, f, indent=2)
    sftp.put(CONFIG["local_index_copy"], CONFIG["remote_cache_index"] + "_tmp")
    ssh.exec_command(f"mv {CONFIG['remote_cache_index']}_tmp {CONFIG['remote_cache_index']}")

def build_dataframe(cache_index):
    now = datetime.utcnow()
    records = []

    for file_id, info in cache_index.items():
        try:
            last_access = datetime.fromisoformat(info["last_access"])
            age = (now - last_access).total_seconds()
            size = info.get("size", 0)
            file_type = info.get("type", "other")
            if file_type not in CONFIG["supported_types"]:
                file_type = "other"
            
            # Handle missing access_count - use heuristic based on file_id structure
            access_count = info.get("access_count", 1)  # Default to 1 if missing
            
            # Alternative: estimate access count from file age and type
            if "access_count" not in info:
                # Heuristic: older files that still exist likely have more accesses
                days_old = age / 86400
                if file_type in ["image", "video"]:
                    # Media files: estimate based on age and size
                    size_mb = size / (1024 * 1024)
                    access_count = max(1, int(days_old * 0.5 + size_mb * 0.1))
                else:
                    # Other files: simpler heuristic
                    access_count = max(1, int(days_old * 0.2))
            
            records.append({
                "file_id": file_id,
                "access_count": access_count,
                "age_since_last_access": age,
                "file_type": file_type,
                "file_size": size,
                "last_access": info["last_access"],
                "creation_time": info.get("creation_time", info["last_access"]),
                "content_hash": file_id  # Store the hash for reference
            })
        except Exception as e:
            print(f"Error parsing {file_id}: {e}")
    
    return pd.DataFrame(records)

# Eviction Logic
def evict_files(df, cache_index, ssh, dry_run=False):
    current_size = df["file_size"].sum()
    limit_bytes = CONFIG["cache_limit_mb"] * 1024 * 1024
    evicted = []
    
    print(f"Current cache size: {current_size / (1024*1024):.2f} MB")
    print(f"Cache limit: {CONFIG['cache_limit_mb']} MB")
    
    if current_size <= limit_bytes:
        print("Cache is already under limit.")
        return evicted
    
    bytes_to_evict = current_size - limit_bytes
    bytes_evicted = 0
    
    print(f"Need to evict {bytes_to_evict / (1024*1024):.2f} MB")
    
    for _, row in df.iterrows():
        if bytes_evicted >= bytes_to_evict:
            break
        
        remote_path = os.path.join(CONFIG["remote_cache_dir"], row["file_id"])
        
        if dry_run:
            print(f"Would evict: {row['file_id']} (Belady Score: {row['belady_score']:.3f}, Size: {row['file_size']} bytes)")
            bytes_evicted += row["file_size"]
            evicted.append(row["file_id"])
        else:
            try:
                stdin, stdout, stderr = ssh.exec_command(f"rm -f '{remote_path}'")
                exit_status = stdout.channel.recv_exit_status()
                if exit_status == 0:
                    bytes_evicted += row["file_size"]
                    evicted.append(row["file_id"])
                    cache_index.pop(row["file_id"], None)
                    print(f"Evicted: {row['file_id']} (Belady Score: {row['belady_score']:.3f})")
                else:
                    print(f"Failed to delete {row['file_id']}: exit status {exit_status}")
            except Exception as e:
                print(f"Failed to delete {row['file_id']}: {e}")
    
    print(f"Evicted {len(evicted)} files, freed {bytes_evicted / (1024*1024):.2f} MB")
    return evicted

# Main
def main():
    try:
        # Initialize model ensemble
        model_ensemble = LRBModelEnsemble(
            CONFIG["local_model_paths"], 
            CONFIG["model_info_path"]
        )
        
        if not model_ensemble.models:
            print("No models loaded. Exiting.")
            return
        
        # Establish SSH connection
        ssh, sftp = establish_ssh_connection()
        
        # Load cache index
        cache_index = load_cache_index(sftp)
        
        # Build dataframe
        df = build_dataframe(cache_index)
        if df.empty:
            print("No valid entries found in cache index.")
            sftp.close()
            ssh.close()
            return
        
        print(f"Loaded {len(df)} cache entries")
        
        # Calculate LRB scores (Learning Relaxed Belady)
        df_scored = model_ensemble.calculate_belady_score(df, CONFIG["eviction_strategy"])
        
        # Show top candidates for eviction (highest Belady scores)
        print("\nTop 10 eviction candidates (Learning Relaxed Belady):")
        print(df_scored[["file_id", "belady_score", "file_size", "access_count", "age_since_last_access"]].head(10))
        
        # Perform eviction (set dry_run=True for testing)
        evicted = evict_files(df_scored, cache_index, ssh, dry_run=False)
        
        # Save updated cache index
        if evicted:
            save_cache_index(sftp, cache_index, ssh)
        
        # Close connections
        sftp.close()
        ssh.close()
        
        print(f"\nEviction complete. Removed {len(evicted)} files.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()