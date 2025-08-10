#!/bin/bash

SCRIPT="generate_cache_index.py"
REMOTE_SCRIPT_PATH="/home/hnfxrt/$SCRIPT"
REMOTE_OUTPUT="cache_index.json"
LOCAL_SAVE_DIR="."
KEY="/Users/feb/Documents/GitHub/mock-flask-app/id_rsa"
USER="hnfxrt"

TARGET=$1
HOST=""

if [[ "$TARGET" == "sea" ]]; then
  HOST="34.128.85.243"
elif [[ "$TARGET" == "eu" ]]; then
  HOST="35.197.236.92"
elif [[ "$TARGET" == "us" ]]; then
  HOST="34.23.29.132"
else
  echo "Usage: $0 [sea|eu|us]"
  exit 1
fi

echo " Copying $SCRIPT to $TARGET ($HOST)..."
scp -o IdentitiesOnly=yes -i $KEY "ml/$SCRIPT" "$USER@$HOST:$REMOTE_SCRIPT_PATH"

echo " Running script on $TARGET ($HOST)..."
ssh -o IdentitiesOnly=yes -i $KEY "$USER@$HOST" "sudo python3 $REMOTE_SCRIPT_PATH"

echo "Downloading $REMOTE_OUTPUT from $TARGET ($HOST)..."
scp -o IdentitiesOnly=yes -i $KEY "$USER@$HOST:/home/hnfxrt/$REMOTE_OUTPUT" "$LOCAL_SAVE_DIR/cache_index_lfu_${TARGET}.json"

echo " Saved to $LOCAL_SAVE_DIR/cache_index_lfu_${TARGET}.json"