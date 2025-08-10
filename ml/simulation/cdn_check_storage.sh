#!/bin/bash

KEY="/Users/feb/Documents/GitHub/mock-flask-app/id_rsa"
USER="hnfxrt"

# change to server actual ip"
declare -A HOSTS
HOSTS[sea]="34.128.85.243" 
HOSTS[eu]="35.197.236.92"
HOSTS[us]="34.23.29.132"

for TARGET in "${!HOSTS[@]}"; do
  HOST=${HOSTS[$TARGET]}

  (
    OUTPUT=$(ssh -i "$KEY" "$USER@$HOST" "sudo du -sh /var/cache/nginx/media/" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
      echo -e "[$TARGET] $HOST => $OUTPUT"
    else
      echo -e "[$TARGET] $HOST => FAILED to connect or run command"
    fi
  ) &
done

wait
