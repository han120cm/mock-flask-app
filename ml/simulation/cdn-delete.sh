#!/bin/bash

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

echo "Deleting cache at /var/cache/nginx/media/ on $TARGET ($HOST)..."

ssh -o IdentitiesOnly=yes -i "$KEY" "$USER@$HOST" "sudo rm -rf /var/cache/nginx/media/*" && \
echo "Cache cleared on $TARGET ($HOST)"