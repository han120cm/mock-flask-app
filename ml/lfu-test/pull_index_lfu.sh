#!/bin/bash

SCRIPT="generate_cache_index.py"
REMOTE_SCRIPT_PATH="/home/user/$SCRIPT"
REMOTE_OUTPUT="cache_index.json"
LOCAL_SAVE_DIR="local/save/dir"
KEY="/dir/to/priv/key"
USER="USER"

# change to server actual ip"
declare -A HOSTS
HOSTS[sea]="1.1.1.1" 
HOSTS[eu]="1.1.1.1"
HOSTS[us]="1.1.1.1"

TARGET=$1

if [[ -z "${HOSTS[$TARGET]}" ]]; then
  echo "Usage: $0 [sea|eu]"
  exit 1
fi

HOST=${HOSTS[$TARGET]}

echo " Copying $SCRIPT to $TARGET ($HOST)..."
scp -i $KEY "$SCRIPT" "$USER@$HOST:$REMOTE_SCRIPT_PATH"

echo " Running script on $TARGET ($HOST)..."
ssh -i $KEY "$USER@$HOST" "sudo python3 $REMOTE_SCRIPT_PATH"

echo "Downloading $REMOTE_OUTPUT from $TARGET ($HOST)..."
scp -i $KEY "$USER@$HOST:/home/hnfxrt/$REMOTE_OUTPUT" "$LOCAL_SAVE_DIR/cache_index_lfu_${TARGET}.json"

echo " Saved to $LOCAL_SAVE_DIR/cache_index_${TARGET}.json"
