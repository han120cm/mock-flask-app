import os
import json
import hashlib
from datetime import datetime

# CONFIGURATION
CACHE_DIR = "/var/cache/nginx/media"
ACCESS_LOG = "/var/log/nginx/access.log"
OUTPUT_FILE = "cache_index.json"

# NGINX proxy_cache_key format: "$scheme://$proxy_host$request_uri"
PROXY_SCHEME = "https"
PROXY_HOST = "gcs_storage"
GCS_PREFIX = "/bucket-main-ta"

# Recognized extensions and types
TYPE_BY_EXT = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".mp4": "video", ".webm": "video", ".mkv": "video"
}
VALID_EXTENSIONS = tuple(TYPE_BY_EXT.keys())

# Map from access.log
print("🔍 Parsing access log...")

hash_type_map = {}

with open(ACCESS_LOG, "r") as f:
    for line in f:
        if "GET" not in line or "/static/" not in line:
            continue
        try:
            method, path = line.split('"')[1].split()[:2]
            ext = os.path.splitext(path)[1].lower()

            if ext not in VALID_EXTENSIONS:
                continue

            file_type = TYPE_BY_EXT.get(ext, "other")

            corrected_path = f"{GCS_PREFIX}{path}"
            cache_key_input = f"{PROXY_SCHEME}://{PROXY_HOST}{corrected_path}"
            cache_key_hash = hashlib.md5(cache_key_input.encode()).hexdigest()

            hash_type_map[cache_key_hash] = file_type
        except Exception:
            continue

print(f"✅ Extracted {len(hash_type_map)} CDN hashes from access.log.")

# Scan cache directory and match types
print("📂 Scanning cache directory...")

cache_index = {}

for root, _, files in os.walk(CACHE_DIR):
    for fname in files:
        try:
            full_path = os.path.join(root, fname)
            stat = os.stat(full_path)

            file_id = os.path.relpath(full_path, CACHE_DIR).replace("\\", "/")
            last_access = datetime.utcfromtimestamp(stat.st_atime).isoformat()
            size = stat.st_size

            hash_guess = os.path.splitext(fname)[0][-32:]
            file_type = hash_type_map.get(hash_guess, "other")

            cache_index[file_id] = {
                "last_access": last_access,
                "size": size,
                "type": file_type
            }
        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

# Save to cache_index.json
with open(OUTPUT_FILE, "w") as f:
    json.dump(cache_index, f, indent=2)

print(f"{OUTPUT_FILE} created with {len(cache_index)} entries.")