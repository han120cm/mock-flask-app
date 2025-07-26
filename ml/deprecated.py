import os
import json
import joblib
import paramiko
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import getpass

# -------------------------------
# Configuration
# -------------------------------
CDN_NODE_HOST = "34.101.140.128"
CDN_NODE_USER = "hnfxrt"
PRIVATE_KEY_PATH = "/home/hnfxrt/gcp/gcp"
SUDO_PASSWORD = None  # Will prompt if needed

REMOTE_CACHE_INDEX = "/home/hnfxrt/cache_index.json"
REMOTE_CACHE_DIR = "/var/cache/nginx/media/"
LOCAL_MODEL_PATH = "lrb_eviction_model.pkl"
LOCAL_INDEX_COPY = "cache_index.json"

CACHE_LIMIT_MB = 100
SUPPORTED_TYPES = {"image", "video"}

def execute_sudo_command(ssh, command, password=None):
    """Execute a command with sudo using different approaches"""
    try:
        # Method 1: Try with TTY and -n flag first
        full_command = f"sudo -n {command}"
        stdin, stdout, stderr = ssh.exec_command(full_command, get_pty=True)
        exit_code = stdout.channel.recv_exit_status()
        
        if exit_code == 0:
            stdout_text = stdout.read().decode().strip()
            stderr_text = stderr.read().decode().strip()
            return exit_code, stdout_text, stderr_text
        
        # Method 2: Try without TTY
        stdin, stdout, stderr = ssh.exec_command(full_command)
        exit_code = stdout.channel.recv_exit_status()
        
        stdout_text = stdout.read().decode().strip()
        stderr_text = stderr.read().decode().strip()
        
        return exit_code, stdout_text, stderr_text
    except Exception as e:
        return -1, "", str(e)

def find_cache_file_path(ssh, file_id):
    """Find the actual path of a cached file using find command"""
    try:
        # Method 1: Direct path construction (nginx cache structure)
        # file_id format: fc/90/1c097e982eea7d7a5c40529e129e90fc
        direct_path = os.path.join(REMOTE_CACHE_DIR, file_id)
        
        # Check if direct path exists
        check_cmd = f"sudo test -f '{direct_path}' && echo 'exists'"
        stdin, stdout, stderr = ssh.exec_command(check_cmd)
        result = stdout.read().decode().strip()
        
        if "exists" in result:
            return direct_path
        
        # Method 2: Find by the hash part (last part of file_id)
        hash_part = file_id.split('/')[-1]  # Get the last part
        find_cmd = f"sudo find /var/cache/nginx/media/ -name '*{hash_part}*' -type f"
        stdin, stdout, stderr = ssh.exec_command(find_cmd)
        exit_code = stdout.channel.recv_exit_status()
        
        if exit_code == 0:
            result = stdout.read().decode().strip()
            if result:
                paths = result.split('\n')
                # Return the first valid path
                for path in paths:
                    if path.strip():
                        return path.strip()
        
        # Method 3: List directory structure to debug
        print(f"🔍 Debug: Looking for {file_id}")
        list_cmd = f"sudo ls -la /var/cache/nginx/media/{file_id[:2]}/"
        stdin, stdout, stderr = ssh.exec_command(list_cmd)
        ls_result = stdout.read().decode().strip()
        print(f"Directory listing: {ls_result}")
        
        return None
    except Exception as e:
        print(f"Error finding file {file_id}: {e}")
        return None

# -------------------------------
# Connect via SSH
# -------------------------------
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(CDN_NODE_HOST, username=CDN_NODE_USER, key_filename=PRIVATE_KEY_PATH)
sftp = ssh.open_sftp()

print("📥 Downloading cache_index.json...")
sftp.get(REMOTE_CACHE_INDEX, LOCAL_INDEX_COPY)

print("📦 Loading model and cache index...")
model = joblib.load(LOCAL_MODEL_PATH)
with open(LOCAL_INDEX_COPY) as f:
    cache_index = json.load(f)

# -------------------------------
# Test sudo access first
# -------------------------------
print("🔐 Testing sudo access...")

# Debug: Check user and groups
stdin, stdout, stderr = ssh.exec_command("whoami && groups")
user_info = stdout.read().decode().strip()
print(f"User info: {user_info}")

# Debug: Test basic sudo
stdin, stdout, stderr = ssh.exec_command("sudo -n whoami", get_pty=True)
exit_code = stdout.channel.recv_exit_status()
stdout_text = stdout.read().decode().strip()
stderr_text = stderr.read().decode().strip()

print(f"Sudo test exit code: {exit_code}")
print(f"Sudo stdout: {stdout_text}")
print(f"Sudo stderr: {stderr_text}")

if exit_code != 0:
    print("❌ Sudo access failed. Trying alternative methods...")
    
    # Try without TTY
    stdin, stdout, stderr = ssh.exec_command("sudo -n whoami")
    exit_code2 = stdout.channel.recv_exit_status()
    stdout_text2 = stdout.read().decode().strip()
    stderr_text2 = stderr.read().decode().strip()
    
    print(f"Alternative sudo test exit code: {exit_code2}")
    print(f"Alternative sudo stdout: {stdout_text2}")
    print(f"Alternative sudo stderr: {stderr_text2}")
    
    if exit_code2 != 0:
        print("❌ All sudo methods failed. Check visudo configuration.")
        sftp.close()
        ssh.close()
        exit()

print("✅ Sudo access confirmed")

# -------------------------------
# Debug: Check nginx cache structure
# -------------------------------
print("🔍 Analyzing nginx cache structure...")
stdin, stdout, stderr = ssh.exec_command("sudo find /var/cache/nginx/media/ -type f | head -10")
sample_files = stdout.read().decode().strip()
print(f"Sample cache files:\n{sample_files}")

stdin, stdout, stderr = ssh.exec_command("sudo ls -la /var/cache/nginx/media/ | head -10")
cache_dirs = stdout.read().decode().strip()
print(f"Cache directory structure:\n{cache_dirs}")

# -------------------------------
# Build DataFrame for prediction
# -------------------------------
now = datetime.utcnow()
records = []

for file_id, info in cache_index.items():
    try:
        last_access = datetime.fromisoformat(info["last_access"])
        age = (now - last_access).total_seconds()
        size = info.get("size", 0)
        file_type = info.get("type", "other")
        if file_type not in SUPPORTED_TYPES:
            file_type = "other"

        records.append({
            "file_id": file_id,
            "access_count": file_id.count("_"),
            "age_since_last_access": age,
            "file_type": file_type,
            "file_size": size,
            "log_file_size": np.log1p(max(1, size))
        })
    except Exception as e:
        print(f"⚠️ Error parsing {file_id}: {e}")

df = pd.DataFrame(records)
if df.empty:
    print("❌ No valid entries found in cache index.")
    sftp.close()
    ssh.close()
    exit()

# -------------------------------
# Predict hit probabilities
# -------------------------------
print("🔮 Predicting hit probabilities...")
le = LabelEncoder()
df["file_type_encoded"] = le.fit_transform(df["file_type"])

X = df[["access_count", "age_since_last_access", "file_type_encoded", "log_file_size"]]
df["hit_prob"] = model.predict_proba(X)[:, 1]
df = df.sort_values(by="hit_prob")

# -------------------------------
# Eviction process
# -------------------------------
print("🧹 Starting eviction...")
current_size = df["file_size"].sum()
limit_bytes = CACHE_LIMIT_MB * 1024 * 1024
evicted = []

for _, row in df.iterrows():
    if current_size <= limit_bytes:
        break

    # Find the actual file path
    file_id = row["file_id"]
    
    # Try multiple path formats
    possible_paths = [
        os.path.join(REMOTE_CACHE_DIR, file_id),  # Direct path
        os.path.join(REMOTE_CACHE_DIR, file_id.replace('/', '')),  # Flattened
        f"/var/cache/nginx/media/{file_id}",  # Full path
    ]
    
    actual_path = None
    for path in possible_paths:
        check_cmd = f"sudo test -f '{path}' && echo 'exists'"
        stdin, stdout, stderr = ssh.exec_command(check_cmd)
        result = stdout.read().decode().strip()
        if "exists" in result:
            actual_path = path
            break
    
    if not actual_path:
        # Last resort: find by hash
        hash_part = file_id.split('/')[-1]
        find_cmd = f"sudo find /var/cache/nginx/media/ -name '*{hash_part}*' -type f -print -quit"
        stdin, stdout, stderr = ssh.exec_command(find_cmd)
        result = stdout.read().decode().strip()
        if result:
            actual_path = result
    
    if not actual_path:
        print(f"❌ File not found: {file_id}")
        continue

    try:
        # Delete with sudo
        delete_cmd = f"rm -f '{actual_path}'"
        exit_code, stdout_text, stderr_text = execute_sudo_command(ssh, delete_cmd, SUDO_PASSWORD)
        
        if exit_code == 0:
            print(f"✅ Deleted: {actual_path}")
            current_size -= row["file_size"]
            evicted.append(file_id)
            cache_index.pop(file_id, None)
        else:
            print(f"🚫 Failed to delete {actual_path}: {stderr_text}")

    except Exception as e:
        print(f"🚫 SSH error for {actual_path}: {e}")

# -------------------------------
# Update cache index
# -------------------------------
print("💾 Updating cache index...")
with open(LOCAL_INDEX_COPY, "w") as f:
    json.dump(cache_index, f, indent=2)

sftp.put(LOCAL_INDEX_COPY, "/home/hnfxrt/cache_index_tmp.json")
ssh.exec_command("mv /home/hnfxrt/cache_index_tmp.json /home/hnfxrt/cache_index.json")

sftp.close()
ssh.close()

print(f"\n🧹 Evicted {len(evicted)} files. Cache size now under {CACHE_LIMIT_MB} MB.")