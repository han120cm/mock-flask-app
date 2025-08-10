#!/bin/bash

KEY="dir/to/priv/key"
USER="USER"

# change to server actual ip"
declare -A HOSTS
HOSTS[sea]="1.1.1.1" 
HOSTS[eu]="1.1.1.1"
HOSTS[us]="1.1.1.1"

TARGET=$1

if [[ -z "${HOSTS[$TARGET]}" ]]; then
  echo "Usage: $0 [sea|eu|us]"
  exit 1
fi

HOST=${HOSTS[$TARGET]}

echo "Deleting cache at /var/cache/nginx/media/ on $TARGET ($HOST)..."

ssh -i "$KEY" "$USER@$HOST" "sudo rm -rf /var/cache/nginx/media/*" && \
echo "Cache cleared on $TARGET ($HOST)"
