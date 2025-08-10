#!/bin/bash

KEY="dir/to/priv/key"
USER="USER"

# change to server actual ip"
declare -A HOSTS
HOSTS[sea]="1.1.1.1" 
HOSTS[eu]="1.1.1.1"
HOSTS[us]="1.1.1.1"

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
