#!/bin/zsh
# push local er/ code to the Lightning studio; set LIGHTNING_SSH=user@ssh.lightning.ai
scp -o BatchMode=yes -q -r "$(dirname "$0")/er" "${LIGHTNING_SSH:?set LIGHTNING_SSH}":/teamspace/studios/this_studio/ 2>&1 | grep -v Permanently
