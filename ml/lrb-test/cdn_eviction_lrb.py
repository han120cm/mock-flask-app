import os
import json
import joblib
import paramiko
import pandas as pd
from datetime import datetime
import numpy as np
from typing import Dict
import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python3 cdn_eviction_lrb.py [sea|eu|us]")
    sys.exit(1)

REGION = sys.argv[1].lower()

HOSTS = {
    "sea": "1.1.1.1",
    "eu": "1.1.1.1",
    "us": "1.1.1.1"
}

if REGION not in HOSTS:
    print(f"Unknown region: {REGION}")
    sys.exit(1)

CONFIG = {
    "cdn_host": HOSTS[REGION],
    "cdn_user": "USER",
    "private_key_path": "/dir/to/priv/key",
    "remote_cache_index": "/location/to/cache_index.json",
    "remote_cache_dir": "/var/cache/nginx/media/",
    "local_model_paths": {
        "binary": "web_lrb_model_binary.pkl",
        "distance": "web_lrb_model_distance.pkl",
        "reuse": "web_lrb_model_reuse.pkl"
    },
    "model_info_path": "web_lrb_model_tuning_results.json",
    "local_index_copy": f"cache_index_{REGION}.json",
    "cache_limit_mb": 500,
    "look_ahead_window": 100,
    "eviction_strategy": "lrb",
    "ensemble_weights": {
        "binary": 0.4,
        "distance": 0.3,
        "reuse": 0.3
    }
}

class CacheJSONLRBScorer:
    """LRB scorer that works directly with cache JSON format."""

    def __init__(self, model_paths: Dict[str, str], info_path: str):
        self.models = {}
        self.feature_columns = []
        self.type_mapping = {}

        for model_name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[model_name] = joblib.load(path)
                    print(f"Loaded {model_name} model from {path}")
                except Exception as e:
                    print(f"Failed to load {model_name} model: {e}")
            else:
                print(f"Model file not found: {path}")

        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                    self.feature_columns = info.get("feature_columns", [])
                    self.type_mapping = info.get("type_mapping", {})
                print(f"Loaded model info: {len(self.feature_columns)} features")
            except Exception as e:
                print(f"Failed to load model info: {e}")

        if not self.type_mapping:
            self.type_mapping = {
                "video": 0, "audio": 1, "image": 2, "html": 3, "css": 4,
                "javascript": 5, "pdf": 6, "text": 7, "data": 8, "archive": 9,
                "other": 10, "unknown": 11
            }

    def prepare_cache_features(self, cache_data: Dict) -> pd.DataFrame:
        """Convert cache JSON data to model features."""
        if not cache_data:
            return pd.DataFrame()

        features_list = []
        for file_id, cache_info in cache_data.items():
            feature_row = {
                "size": float(cache_info.get("size", 0)),
                "type_encoded": int(self.type_mapping.get(cache_info.get("type", "unknown"), 11)),
                "access_count": int(cache_info.get("access_count", 1)),
                "age_since_last_access": float(cache_info.get("age_since_last_access", 0)),
                "age_hours": float(cache_info.get("age_hours", 0)),
                "log_file_size": float(cache_info.get("log_file_size", np.log1p(cache_info.get("size", 0)))),
                "access_rate": float(cache_info.get("access_rate", 1.0)),
                "recency_score": float(cache_info.get("recency_score", 1.0)),
                "stack_distance_approx": float(cache_info.get("stack_distance_approx", 1000))
            }
            features_list.append(feature_row)

        df = pd.DataFrame(features_list)
        if not df.empty and self.feature_columns:
            try:
                df = df[self.feature_columns]
                print(f"Prepared {len(df)} records with {len(df.columns)} features")
            except KeyError as e:
                print(f"Missing feature columns: {e}")
                return pd.DataFrame()
        return df

    def predict_lrb_scores(self, cache_data: Dict) -> Dict[str, float]:
        """Calculate LRB eviction scores for cache items."""
        if not self.models or not cache_data:
            print("Using fallback LRU-like scoring (no models available)")
            return self._fallback_scoring(cache_data)

        features_df = self.prepare_cache_features(cache_data)
        if features_df.empty:
            print("Failed to prepare features, using fallback")
            return self._fallback_scoring(cache_data)

        file_ids = list(cache_data.keys())
        scores = np.zeros(len(features_df))

        if "binary" in self.models:
            try:
                access_prob = self.models["binary"].predict_proba(features_df)[:, 1]
                scores += (1.0 - access_prob) * CONFIG["ensemble_weights"]["binary"]
            except Exception as e:
                print(f"Binary model error: {e}")

        if "distance" in self.models:
            try:
                distances = self.models["distance"].predict(features_df)
                max_distance = CONFIG["look_ahead_window"] * 3
                normalized_distances = np.clip(distances / max_distance, 0, 1)
                scores += normalized_distances * CONFIG["ensemble_weights"]["distance"]
            except Exception as e:
                print(f"Distance model error: {e}")

        if "reuse" in self.models:
            try:
                reuse_probs = self.models["reuse"].predict(features_df)
                scores += (1.0 - np.clip(reuse_probs, 0, 1)) * CONFIG["ensemble_weights"]["reuse"]
            except Exception as e:
                print(f"Reuse model error: {e}")

        if np.all(scores == 0):
            print("All models failed, using fallback scoring")
            return self._fallback_scoring(cache_data)

        return {file_ids[i]: scores[i] for i in range(len(file_ids))}

    def _fallback_scoring(self, cache_data: Dict) -> Dict[str, float]:
        """Simple LRU-like fallback scoring."""
        scores = {}
        for file_id, cache_info in cache_data.items():
            age_score = cache_info.get("age_since_last_access", 0) / 3600
            access_score = 1.0 / max(cache_info.get("access_count", 1), 1)
            scores[file_id] = age_score * 0.7 + access_score * 0.3
        return scores

    def debug_features(self, cache_data: Dict):
        """Debug feature preparation."""
        print("Debug: Feature Preparation")
        print(f"Cache items: {len(cache_data)}")
        if cache_data:
            first_key = list(cache_data.keys())[0]
            print(f"Sample cache item: {first_key}")
            print(f"Cache data: {cache_data[first_key]}")
        features_df = self.prepare_cache_features(cache_data)
        if not features_df.empty:
            print(f"Features shape: {features_df.shape}")
            print(f"Feature columns: {list(features_df.columns)}")

def establish_ssh_connection():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CONFIG["cdn_host"], username=CONFIG["cdn_user"], key_filename=CONFIG["private_key_path"])
    return ssh, ssh.open_sftp()

def load_cache_index(sftp):
    """Load cache index from remote server."""
    sftp.get(CONFIG["remote_cache_index"], CONFIG["local_index_copy"])
    with open(CONFIG["local_index_copy"]) as f:
        return json.load(f)

def save_cache_index(sftp, cache_index, ssh):
    """Save updated cache index to remote server."""
    with open(CONFIG["local_index_copy"], "w") as f:
        json.dump(cache_index, f, indent=2)
    sftp.put(CONFIG["local_index_copy"], CONFIG["remote_cache_index"] + "_tmp")
    ssh.exec_command(f"mv {CONFIG['remote_cache_index']}_tmp {CONFIG['remote_cache_index']}")

def validate_and_enrich_cache_data(cache_index: Dict) -> Dict:
    """Validate and enrich cache data to ensure all required features are present."""
    now = datetime.utcnow()
    enriched_cache = {}
    for file_id, info in cache_index.items():
        try:
            last_access = datetime.fromisoformat(info["last_access"])
            age_seconds = (now - last_access).total_seconds()
            size = info.get("size", 0)
            file_type = info.get("type", "other")
            valid_types = {"video", "audio", "image", "html", "css", "javascript",
                           "pdf", "text", "data", "archive", "other", "unknown"}
            if file_type not in valid_types:
                file_type = "other"
            access_count = info.get("access_count", 1)
            if access_count < 1:
                access_count = 1
            age_hours = age_seconds / 3600
            log_file_size = np.log1p(size)
            access_rate = access_count / max(age_hours, 0.1)
            recency_score = 1.0 / (1.0 + age_seconds / 3600)
            stack_distance_approx = size / (access_count + 1)
            enriched_cache[file_id] = {
                "last_access": info["last_access"],
                "size": size,
                "type": file_type,
                "access_count": access_count,
                "age_since_last_access": age_seconds,
                "age_hours": age_hours,
                "log_file_size": log_file_size,
                "access_rate": access_rate,
                "recency_score": recency_score,
                "stack_distance_approx": stack_distance_approx
            }
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue
    print(f"Processed {len(enriched_cache)} valid cache entries")
    return enriched_cache

def execute_sudo_command(ssh, command):
    """Execute a command with sudo."""
    try:
        full_command = f"sudo -n {command}"
        stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode().strip()
        stderr_text = stderr.read().decode().strip()
        return exit_code, stdout_text, stderr_text
    except Exception as e:
        return -1, "", str(e)

def perform_eviction(cache_data: Dict, lrb_scorer: CacheJSONLRBScorer, ssh, cache_index: Dict, dry_run=False):
    """Perform cache eviction using LRB scores."""
    current_size = sum(info["size"] for info in cache_data.values())
    limit_bytes = CONFIG["cache_limit_mb"] * 1024 * 1024
    print(f"Current cache size: {current_size / (1024*1024):.2f} MB")
    if current_size <= limit_bytes:
        print("Cache is already under limit")
        return []

    bytes_to_evict = current_size - limit_bytes
    print(f"Need to evict: {bytes_to_evict / (1024*1024):.2f} MB")
    lrb_scores = lrb_scorer.predict_lrb_scores(cache_data)
    sorted_candidates = sorted(lrb_scores.items(), key=lambda x: x[1], reverse=True)

    evicted = []
    bytes_evicted = 0
    for file_id, score in sorted_candidates:
        if bytes_evicted >= bytes_to_evict:
            break
        cache_info = cache_data[file_id]
        file_size = cache_info["size"]
        if dry_run:
            bytes_evicted += file_size
            evicted.append(file_id)
        else:
            try:
                remote_path = os.path.join(CONFIG["remote_cache_dir"], file_id)
                delete_cmd = f"rm -f '{remote_path}'"
                exit_code, stdout_text, stderr_text = execute_sudo_command(ssh, delete_cmd)
                if exit_code == 0:
                    bytes_evicted += file_size
                    evicted.append(file_id)
                    cache_index.pop(file_id, None)
                else:
                    print(f"Failed to delete {file_id}: exit code {exit_code}")
                    if stderr_text:
                        print(f"Error: {stderr_text}")
            except Exception as e:
                print(f"Error deleting {file_id}: {e}")
    print(f"Eviction complete: {len(evicted)} files, {bytes_evicted / (1024*1024):.2f} MB freed")
    return evicted

def main():
    try:
        print(f"Starting CDN Cache Eviction with LRB for {REGION}")
        result = subprocess.run(["/home/hnfxrt/ml-vm/lrb-new/./pull_index.sh", REGION], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to run pull_index.sh: {result.stderr}")
            return

        lrb_scorer = CacheJSONLRBScorer(CONFIG["local_model_paths"], CONFIG["model_info_path"])
        ssh, sftp = establish_ssh_connection()
        cache_index = load_cache_index(sftp)
        cache_data = validate_and_enrich_cache_data(cache_index)
        if not cache_data:
            print("No valid cache entries found")
            sftp.close()
            ssh.close()
            return

        if len(sys.argv) > 2 and sys.argv[2] == "--debug":
            lrb_scorer.debug_features(cache_data)

        evicted = perform_eviction(cache_data, lrb_scorer, ssh, cache_index, dry_run=False)
        if evicted:
            save_cache_index(sftp, cache_index, ssh)

        sftp.close()
        ssh.close()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
