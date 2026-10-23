#!/bin/bash
# Keep the JupyterHub server alive while one job runs: POST activity to the Hub every 4 min until PID $1 exits.
LOG=logs/keep_alive_job.log
echo "$(date -u +%FT%TZ) keep-alive for pid $1" >> "$LOG"
while kill -0 "$1" 2>/dev/null; do
  curl -s -o /dev/null -X POST -H "Authorization: token $JUPYTERHUB_API_TOKEN" -H "Content-Type: application/json" \
       "$JUPYTERHUB_ACTIVITY_URL" -d '{"last_activity": "'"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"'"}'
  sleep 240
done
echo "$(date -u +%FT%TZ) pid $1 finished, keep-alive stopped" >> "$LOG"
