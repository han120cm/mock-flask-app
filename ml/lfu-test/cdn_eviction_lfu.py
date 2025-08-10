import os
import json
import paramiko
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys
import subprocess

if len(sys.argv) < 2:
    print("Usage: python3 cdn_eviction_lfu.py [sea|eu]")
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
    "cdn_user": "hnfxrt",
    "private_key_path": "/home/hnfxrt/gcp/gcp",
    "remote_cache_index": "/home/hnfxrt/cache_index.json",
    "remote_cache_dir": "/var/cache/nginx/media/",
    "local_index_copy": f"cache_index_lfu_{REGION}.json",
    "cache_limit_mb": 500,
    "eviction_strategy": "lfu",
    "metrics_log": f"lfu_eviction_metrics_{REGION}.json",
    "frequency_decay_factor": 0.99,
    "time_window_hours": 24,
    "min_frequency_threshold": 1
}

class LFUEvictionScorer:
    def __init__(self):
        self.frequency_tracker = defaultdict(int)
        self.last_decay_time = datetime.utcnow()
        print("LFU Eviction Scorer initialized")
        print(f"Frequency decay factor: {CONFIG['frequency_decay_factor']}")
        print(f"Time window: {CONFIG['time_window_hours']} hours")

    def apply_frequency_decay(self):
        now = datetime.utcnow()
        time_since_decay = (now - self.last_decay_time).total_seconds() / 3600
        if time_since_decay >= 1.0:
            decay_factor = CONFIG["frequency_decay_factor"] ** time_since_decay
            decayed_count = 0
            for file_id in list(self.frequency_tracker.keys()):
                old_freq = self.frequency_tracker[file_id]
                new_freq = old_freq * decay_factor
                if new_freq < 0.1:
                    del self.frequency_tracker[file_id]
                    decayed_count += 1
                else:
                    self.frequency_tracker[file_id] = new_freq
            self.last_decay_time = now
            if decayed_count > 0:
                print(f"Applied frequency decay: {decayed_count} entries removed")

    def update_frequencies(self, cache_data: Dict):
        self.apply_frequency_decay()
        for file_id, cache_info in cache_data.items():
            access_count = cache_info.get("access_count", 1)
            last_access = datetime.fromisoformat(cache_info["last_access"])
            now = datetime.utcnow()
            hours_since_access = (now - last_access).total_seconds() / 3600
            time_weight = max(0.1, 1.0 - (hours_since_access / CONFIG["time_window_hours"]))
            weighted_frequency = access_count * time_weight
            self.frequency_tracker[file_id] = max(
                self.frequency_tracker.get(file_id, 0),
                weighted_frequency
            )
        print(f"Updated frequencies for {len(cache_data)} items")

    def predict_lfu_scores(self, cache_data: Dict) -> Dict[str, float]:
        if not cache_data:
            return {}
        self.update_frequencies(cache_data)
        scores = {}
        for file_id, cache_info in cache_data.items():
            try:
                frequency = self.frequency_tracker.get(file_id, CONFIG["min_frequency_threshold"])
                base_score = 1.0 / max(frequency, CONFIG["min_frequency_threshold"])
                last_access = datetime.fromisoformat(cache_info["last_access"])
                age_hours = (datetime.utcnow() - last_access).total_seconds() / 3600
                age_factor = 1.0 + (age_hours / (24 * 7))
                size_mb = cache_info.get("size", 0) / (1024 * 1024)
                size_factor = max(1.0, min(2.0, size_mb / 10.0))
                access_count = cache_info.get("access_count", 1)
                access_rate = access_count / max(age_hours, 1.0)
                pattern_factor = 1.0 + (1.0 / max(access_rate, 0.1))
                final_score = base_score * age_factor * size_factor * pattern_factor
                scores[file_id] = final_score
            except Exception as e:
                print(f"Error calculating LFU score for {file_id}: {e}")
                scores[file_id] = 999999.0
        print(f"Calculated LFU scores for {len(scores)} items")
        return scores

    def get_eviction_candidates(self, cache_data: Dict, target_bytes: int) -> List[Tuple[str, float, Dict]]:
        scores = self.predict_lfu_scores(cache_data)
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        bytes_to_free = 0
        for file_id, score in sorted_candidates:
            file_size = cache_data[file_id].get("size", 0)
            frequency = self.frequency_tracker.get(file_id, 0)
            metadata = {
                "frequency": frequency,
                "access_count": cache_data[file_id].get("access_count", 1),
                "size": file_size,
                "last_access": cache_data[file_id]["last_access"]
            }
            candidates.append((file_id, score, metadata))
            bytes_to_free += file_size
            if bytes_to_free >= target_bytes:
                break
        return candidates

    def debug_lfu_frequencies(self, cache_data: Dict, top_n: int = 10):
        print(f"Debug: LFU Frequencies (Top {top_n} least frequent items)")
        scores = self.predict_lfu_scores(cache_data)
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Total tracked frequencies: {len(self.frequency_tracker)}")
        for i, (file_id, score) in enumerate(sorted_items[:top_n]):
            cache_info = cache_data[file_id]
            frequency = self.frequency_tracker.get(file_id, 0)
            access_count = cache_info.get("access_count", 1)
            size_mb = cache_info.get("size", 0) / (1024 * 1024)
            last_access = cache_info["last_access"]
            print(f"  {i+1:2d}. {file_id[:50]}...")
            print(f"      LFU Score: {score:.4f}, Frequency: {frequency:.2f}")
            print(f"      Access Count: {access_count}, Size: {size_mb:.2f}MB")
            print(f"      Last Access: {last_access}")

    def get_frequency_stats(self) -> Dict:
        if not self.frequency_tracker:
            return {}
        frequencies = list(self.frequency_tracker.values())
        return {
            "total_tracked": len(frequencies),
            "avg_frequency": sum(frequencies) / len(frequencies),
            "min_frequency": min(frequencies),
            "max_frequency": max(frequencies),
            "low_freq_count": sum(1 for f in frequencies if f < 2.0),
            "high_freq_count": sum(1 for f in frequencies if f > 10.0)
        }

def establish_ssh_connection():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(CONFIG["cdn_host"], username=CONFIG["cdn_user"], key_filename=CONFIG["private_key_path"])
    return ssh, ssh.open_sftp()

def load_cache_index(sftp):
    sftp.get(CONFIG["remote_cache_index"], CONFIG["local_index_copy"])
    with open(CONFIG["local_index_copy"]) as f:
        return json.load(f)

def save_cache_index(sftp, cache_index, ssh):
    with open(CONFIG["local_index_copy"], "w") as f:
        json.dump(cache_index, f, indent=2)
    sftp.put(CONFIG["local_index_copy"], CONFIG["remote_cache_index"] + "_tmp")
    ssh.exec_command(f"mv {CONFIG['remote_cache_index']}_tmp {CONFIG['remote_cache_index']}")

def validate_cache_data(cache_index: Dict) -> Dict:
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
            access_count = info.get("access_count", None)
            if access_count is None or access_count < 1:
                age_days = (now - last_access).total_seconds() / 86400
                access_count = max(1, int(1 + age_days * 0.1))
            age_seconds = (now - last_access).total_seconds()
            validated_cache[file_id] = {
                "last_access": info["last_access"],
                "size": size,
                "type": info.get("type", "unknown"),
                "access_count": access_count,
                "age_seconds": age_seconds
            }
        except Exception as e:
            print(f"Error validating {file_id}: {e}")
            continue
    print(f"Validated {len(validated_cache)} cache entries for LFU processing")
    return validated_cache

def execute_sudo_command(ssh, command):
    try:
        full_command = f"sudo -n {command}"
        stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode().strip()
        stderr_text = stderr.read().decode().strip()
        return exit_code, stdout_text, stderr_text
    except Exception as e:
        return -1, "", str(e)

def perform_lfu_eviction(cache_data: Dict, lfu_scorer: LFUEvictionScorer, ssh, cache_index: Dict, dry_run=False):
    current_size = sum(info["size"] for info in cache_data.values())
    limit_bytes = CONFIG["cache_limit_mb"] * 1024 * 1024
    print(f"Current cache size: {current_size / (1024*1024):.2f} MB")
    print(f"Cache limit: {CONFIG['cache_limit_mb']} MB")
    if current_size <= limit_bytes:
        print("Cache is already under limit")
        return [], create_eviction_metrics([], 0, current_size, current_size, lfu_scorer)
    bytes_to_evict = current_size - limit_bytes
    print(f"Need to evict: {bytes_to_evict / (1024*1024):.2f} MB")
    freq_stats = lfu_scorer.get_frequency_stats()
    if freq_stats:
        print(f"Frequency stats: Avg: {freq_stats['avg_frequency']:.2f}, "
              f"Low freq (<2): {freq_stats['low_freq_count']}, "
              f"High freq (>10): {freq_stats['high_freq_count']}")
    print("Calculating LFU eviction priorities...")
    eviction_candidates = lfu_scorer.get_eviction_candidates(cache_data, bytes_to_evict)
    print(f"\nLFU Eviction Plan: {len(eviction_candidates)} candidates")
    print("Top 10 least frequently used items to evict:")
    for i, (file_id, score, metadata) in enumerate(eviction_candidates[:10]):
        size_mb = metadata["size"] / (1024 * 1024)
        frequency = metadata["frequency"]
        access_count = metadata["access_count"]
        last_access = metadata["last_access"]
        print(f"  {i+1:2d}. {file_id[:50]}...")
        print(f"      LFU Score: {score:.4f}, Frequency: {frequency:.2f}")
        print(f"      Access Count: {access_count}, Size: {size_mb:.2f}MB")
        print(f"      Last Access: {last_access}")
    evicted = []
    bytes_evicted = 0
    for file_id, score, metadata in eviction_candidates:
        if bytes_evicted >= bytes_to_evict:
            break
        file_size = metadata["size"]
        evicted_item = {
            "file_id": file_id,
            "size": file_size,
            "lfu_score": score,
            "frequency": metadata["frequency"],
            "access_count": metadata["access_count"],
            "last_access": metadata["last_access"]
        }
        if dry_run:
            print(f"Would evict (LFU): {file_id[:50]}... "
                  f"(Freq: {metadata['frequency']:.2f}, Size: {file_size} bytes)")
            bytes_evicted += file_size
            evicted.append(evicted_item)
        else:
            try:
                remote_path = os.path.join(CONFIG["remote_cache_dir"], file_id)
                delete_cmd = f"rm -f '{remote_path}'"
                exit_code, stdout_text, stderr_text = execute_sudo_command(ssh, delete_cmd)
                if exit_code == 0:
                    bytes_evicted += file_size
                    evicted.append(evicted_item)
                    cache_index.pop(file_id, None)
                    print(f"Evicted (LFU): {file_id[:50]}... "
                          f"(Freq: {metadata['frequency']:.2f}, Size: {file_size} bytes)")
                else:
                    print(f"Failed to delete {file_id}: exit code {exit_code}")
                    if stderr_text:
                        print(f"   Error: {stderr_text}")
            except Exception as e:
                print(f"Error deleting {file_id}: {e}")
    print(f"\nLFU Eviction complete: {len(evicted)} files, {bytes_evicted / (1024*1024):.2f} MB freed")
    metrics = create_eviction_metrics(evicted, bytes_evicted, current_size,
                                    current_size - bytes_evicted, lfu_scorer)
    return evicted, metrics

def create_eviction_metrics(evicted: List[Dict], bytes_evicted: int,
                          size_before: int, size_after: int, lfu_scorer: LFUEvictionScorer) -> Dict:
    freq_stats = lfu_scorer.get_frequency_stats()
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "region": REGION,
        "strategy": "LFU",
        "files_evicted": len(evicted),
        "bytes_evicted": bytes_evicted,
        "mb_evicted": bytes_evicted / (1024 * 1024),
        "cache_size_before_mb": size_before / (1024 * 1024),
        "cache_size_after_mb": size_after / (1024 * 1024),
        "cache_limit_mb": CONFIG["cache_limit_mb"],
        "evicted_files": evicted[:10],
        "avg_file_size": bytes_evicted / len(evicted) if evicted else 0,
        "avg_lfu_score": sum(f["lfu_score"] for f in evicted) / len(evicted) if evicted else 0,
        "avg_frequency": sum(f["frequency"] for f in evicted) / len(evicted) if evicted else 0,
        "avg_access_count": sum(f["access_count"] for f in evicted) / len(evicted) if evicted else 0,
        "frequency_stats": freq_stats,
        "config": {
            "decay_factor": CONFIG["frequency_decay_factor"],
            "time_window_hours": CONFIG["time_window_hours"],
            "min_frequency_threshold": CONFIG["min_frequency_threshold"]
        }
    }

def save_metrics(metrics: Dict):
    try:
        existing_metrics = []
        if os.path.exists(CONFIG["metrics_log"]):
            with open(CONFIG["metrics_log"], 'r') as f:
                existing_metrics = json.load(f)
        existing_metrics.append(metrics)
        with open(CONFIG["metrics_log"], 'w') as f:
            json.dump(existing_metrics, f, indent=2)
        print(f"Metrics saved to {CONFIG['metrics_log']}")
    except Exception as e:
        print(f"Failed to save metrics: {e}")

def main():
    try:
        print("Starting CDN Cache Eviction with LFU")
        print(f"Region: {REGION}")
        print(f"Host: {CONFIG['cdn_host']}")
        print("Strategy: Least Frequently Used (LFU)")
        print(f"Syncing cache index from remote with pull_index.sh {REGION} ...")
        result = subprocess.run(["/home/hnfxrt/ml-vm/lfu/./pull_index_lfu.sh", REGION], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to run pull_index_lfu.sh: {result.stderr}")
            return
        else:
            print(f"pull_index_lfu.sh output:\n{result.stdout}")
        lfu_scorer = LFUEvictionScorer()
        print("Connecting to CDN server...")
        ssh, sftp = establish_ssh_connection()
        print("Loading cache index...")
        cache_index = load_cache_index(sftp)
        print(f"Loaded {len(cache_index)} cache entries")
        print("Validating cache data for LFU processing...")
        cache_data = validate_cache_data(cache_index)
        if not cache_data:
            print("No valid cache entries found")
            sftp.close()
            ssh.close()
            return
        if len(sys.argv) > 2 and sys.argv[2] == "--debug":
            lfu_scorer.debug_lfu_frequencies(cache_data)
        dry_run = len(sys.argv) > 2 and sys.argv[2] == "--dry-run"
        if dry_run:
            print("Running in DRY-RUN mode - no files will be deleted")
        evicted, metrics = perform_lfu_eviction(cache_data, lfu_scorer, ssh, cache_index, dry_run=dry_run)
        save_metrics(metrics)
        if evicted and not dry_run:
            print("Saving updated cache index...")
            save_cache_index(sftp, cache_index, ssh)
        sftp.close()
        ssh.close()
        print(f"LFU Eviction complete: {len(evicted)} files processed")
        print(f"Metrics logged to: {CONFIG['metrics_log']}")
        if metrics:
            print(f"\nEviction Summary:")
            print(f"   Files evicted: {metrics['files_evicted']}")
            print(f"   Data freed: {metrics['mb_evicted']:.2f} MB")
            print(f"   Cache utilization: {metrics['cache_size_after_mb']:.1f}/{CONFIG['cache_limit_mb']} MB")
            print(f"   Avg frequency of evicted files: {metrics['avg_frequency']:.2f}")
    except Exception as e:
        print(f"Error in LFU eviction process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()