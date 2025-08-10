import os
import json
import paramiko
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python3 cdn_eviction_lrb.py [sea|eu]")
    sys.exit(1)

REGION = sys.argv[1].lower()

HOSTS = {
    "sea": "34.128.85.243",
    "eu": "35.197.236.92",
    "us": "34.23.29.132"
}

if REGION not in HOSTS:
    print(f"Unknown region: {REGION}")
    sys.exit(1)

CONFIG = {
    "cdn_host": HOSTS[REGION],
    "cdn_user": "hnfxrt",
    "private_key_path": "/Users/feb/Documents/GitHub/mock-flask-app/id_rsa",
    "remote_cache_index": "/home/hnfxrt/cache_index.json",
    "remote_cache_dir": "/var/cache/nginx/media/",  # default location
    "local_index_copy": f"cache_index_lru_{REGION}.json",
    "cache_limit_mb": 500,
    "eviction_strategy": "lru",
    "metrics_log": f"lru_eviction_metrics_{REGION}.json"
}

class LRUEvictionScorer:
    """LRU (Least Recently Used) eviction scorer"""

    def __init__(self):
        self.cache_order = OrderedDict()

    def update_access_order(self, cache_data: Dict):
        """Update the LRU order based on cache data"""
        sorted_items = sorted(
            cache_data.items(),
            key=lambda x: datetime.fromisoformat(x[1]["last_access"])
        )
        self.cache_order.clear()
        for file_id, cache_info in sorted_items:
            self.cache_order[file_id] = cache_info["last_access"]

    def predict_lru_scores(self, cache_data: Dict) -> Dict[str, float]:
        """Calculate LRU eviction scores for cache items"""
        if not cache_data:
            return {}
        self.update_access_order(cache_data)

        scores = {}
        now = datetime.utcnow()
        for file_id, cache_info in cache_data.items():
            try:
                last_access = datetime.fromisoformat(cache_info["last_access"])
                age_seconds = (now - last_access).total_seconds()
                age_hours = age_seconds / 3600
                access_count = cache_info.get("access_count", 1)
                size_mb = cache_info.get("size", 0) / (1024 * 1024)

                frequency_penalty = 1.0 / max(access_count, 1)
                size_adjustment = max(0.1, min(1.0, size_mb / 10.0))
                final_score = age_hours * (1.0 + frequency_penalty * 0.1) * size_adjustment

                scores[file_id] = final_score
            except Exception:
                scores[file_id] = 999999.0
        return scores

    def get_eviction_candidates(self, cache_data: Dict, target_bytes: int) -> List[Tuple[str, float]]:
        """Get ordered list of eviction candidates to free target_bytes"""
        scores = self.predict_lru_scores(cache_data)
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        candidates = []
        bytes_to_free = 0
        for file_id, score in sorted_candidates:
            file_size = cache_data[file_id].get("size", 0)
            candidates.append((file_id, score))
            bytes_to_free += file_size
            if bytes_to_free >= target_bytes:
                break
        return candidates

def establish_ssh_connection():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CONFIG["cdn_host"], username=CONFIG["cdn_user"], key_filename=CONFIG["private_key_path"])
    return ssh, ssh.open_sftp()

def load_cache_index(sftp):
    """Load cache index from remote server"""
    sftp.get(CONFIG["remote_cache_index"], CONFIG["local_index_copy"])
    with open(CONFIG["local_index_copy"]) as f:
        return json.load(f)

def save_cache_index(sftp, cache_index, ssh):
    """Save updated cache index to remote server"""
    with open(CONFIG["local_index_copy"], "w") as f:
        json.dump(cache_index, f, indent=2)
    sftp.put(CONFIG["local_index_copy"], CONFIG["remote_cache_index"] + "_tmp")
    ssh.exec_command(f"mv {CONFIG['remote_cache_index']}_tmp {CONFIG['remote_cache_index']}")

def validate_cache_data(cache_index: Dict) -> Dict:
    """Validate cache data for LRU processing"""
    now = datetime.utcnow()
    validated_cache = {}
    for file_id, info in cache_index.items():
        try:
            if "last_access" not in info or "size" not in info:
                continue
            last_access = datetime.fromisoformat(info["last_access"])
            size = info.get("size", 0)
            if size <= 0:
                continue
            age_seconds = (now - last_access).total_seconds()
            validated_cache[file_id] = {
                "last_access": info["last_access"],
                "size": size,
                "type": info.get("type", "unknown"),
                "access_count": info.get("access_count", 1),
                "age_seconds": age_seconds
            }
        except Exception:
            continue
    return validated_cache

def execute_sudo_command(ssh, command):
    """Execute a command with sudo"""
    try:
        full_command = f"sudo -n {command}"
        stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode().strip()
        stderr_text = stderr.read().decode().strip()
        return exit_code, stdout_text, stderr_text
    except Exception as e:
        return -1, "", str(e)

def perform_lru_eviction(cache_data: Dict, lru_scorer: LRUEvictionScorer, ssh, cache_index: Dict, dry_run=False):
    """Perform cache eviction using LRU algorithm"""
    current_size = sum(info["size"] for info in cache_data.values())
    limit_bytes = CONFIG["cache_limit_mb"] * 1024 * 1024

    if current_size <= limit_bytes:
        return [], create_eviction_metrics([], 0, current_size, current_size)

    bytes_to_evict = current_size - limit_bytes
    eviction_candidates = lru_scorer.get_eviction_candidates(cache_data, bytes_to_evict)

    evicted = []
    bytes_evicted = 0
    for file_id, score in eviction_candidates:
        if bytes_evicted >= bytes_to_evict:
            break
        cache_info = cache_data[file_id]
        file_size = cache_info["size"]

        if dry_run:
            bytes_evicted += file_size
            evicted.append({
                "file_id": file_id,
                "size": file_size,
                "lru_score": score,
                "last_access": cache_info["last_access"],
                "access_count": cache_info["access_count"]
            })
        else:
            try:
                remote_path = os.path.join(CONFIG["remote_cache_dir"], file_id)
                delete_cmd = f"rm -f '{remote_path}'"
                exit_code, stdout_text, stderr_text = execute_sudo_command(ssh, delete_cmd)
                if exit_code == 0:
                    bytes_evicted += file_size
                    evicted.append({
                        "file_id": file_id,
                        "size": file_size,
                        "lru_score": score,
                        "last_access": cache_info["last_access"],
                        "access_count": cache_info["access_count"]
                    })
                    cache_index.pop(file_id, None)
            except Exception:
                pass

    metrics = create_eviction_metrics(evicted, bytes_evicted, current_size, current_size - bytes_evicted)
    return evicted, metrics

def create_eviction_metrics(evicted: List[Dict], bytes_evicted: int, size_before: int, size_after: int) -> Dict:
    """Create metrics for eviction operation"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "region": REGION,
        "strategy": "LRU",
        "files_evicted": len(evicted),
        "bytes_evicted": bytes_evicted,
        "mb_evicted": bytes_evicted / (1024 * 1024),
        "cache_size_before_mb": size_before / (1024 * 1024),
        "cache_size_after_mb": size_after / (1024 * 1024),
        "cache_limit_mb": CONFIG["cache_limit_mb"],
        "evicted_files": evicted[:10],
        "avg_file_size": bytes_evicted / len(evicted) if evicted else 0,
        "avg_lru_score": sum(f["lru_score"] for f in evicted) / len(evicted) if evicted else 0
    }

def save_metrics(metrics: Dict):
    """Save eviction metrics to file"""
    try:
        existing_metrics = []
        if os.path.exists(CONFIG["metrics_log"]):
            with open(CONFIG["metrics_log"], 'r') as f:
                existing_metrics = json.load(f)
        existing_metrics.append(metrics)
        with open(CONFIG["metrics_log"], 'w') as f:
            json.dump(existing_metrics, f, indent=2)
    except Exception:
        pass

def main():
    try:
        result = subprocess.run(["/home/hnfxrt/ml-vm/lru/./pull_index_lru.sh", REGION], capture_output=True, text=True)
        if result.returncode != 0:
            return

        lru_scorer = LRUEvictionScorer()
        ssh, sftp = establish_ssh_connection()
        cache_index = load_cache_index(sftp)
        cache_data = validate_cache_data(cache_index)
        if not cache_data:
            sftp.close()
            ssh.close()
            return

        dry_run = len(sys.argv) > 2 and sys.argv[2] == "--dry-run"
        evicted, metrics = perform_lru_eviction(cache_data, lru_scorer, ssh, cache_index, dry_run=dry_run)
        save_metrics(metrics)

        if evicted and not dry_run:
            save_cache_index(sftp, cache_index, ssh)

        sftp.close()
        ssh.close()
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
