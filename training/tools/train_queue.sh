#!/usr/bin/env bash
# Sequential GPU training queue (recipe v5, 2026-09-30). Reads the NEXT job from runs/queue.txt each time the previous
# one ends, so the queue can be edited while it runs. One job per line, fields separated by '|':
#   body | task id | run name | init: <experiment>/<load-run dir> | max iterations | watch_gate args (gates / its / flags)
# A line starting with '#' is skipped; a finished job is prefixed with '#done '. Stops when no job is left or when
# runs/queue.stop exists. Each run's watcher starts next to its training and gates the checkpoints on CPU MuJoCo.
cd "$(dirname "$0")/.." || exit 1
Q=runs/queue.txt
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a runs/queue.log; }
while true; do
  [ -f runs/queue.stop ] && { log "stop file found — queue ends"; break; }
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | head -1)
  [ -z "$line" ] && { log "queue empty"; break; }
  IFS='|' read -r body task name init its watch <<<"$line"
  body=$(echo $body); task=$(echo $task); name=$(echo $name); init=$(echo $init); its=$(echo $its)
  exp=${init%%/*}; loadrun=${init#*/}
  log "START $name: $task (body $body, init $init, $its its)"
  POOLYMPIC_BODY=$body uv run train "$task" --log-root runs --agent.resume True --agent.load-run "$loadrun" \
      --agent.load-checkpoint model_0.pt --agent.run-name "$name" --agent.max-iterations "$its" > "runs/$name.log" 2>&1 &
  tpid=$!
  # the run directory appears once training starts; then attach the watcher
  for _ in $(seq 1 120); do
    rundir=$(ls -d runs/$exp/*_"$name" 2>/dev/null | tail -1); [ -n "$rundir" ] && break; sleep 5
  done
  if [ -n "$rundir" ] && [ -n "$(echo $watch)" ]; then
    POOLYMPIC_BODY=$body nohup uv run python tools/watch_gate.py "$rundir" "$name" $watch > "runs/${name}_watch.log" 2>&1 &
  fi
  wait $tpid; rc=$?
  log "END $name rc=$rc ($(grep -a 'Learning iteration' runs/$name.log | tail -1 | tr -s ' '))"
  # mark this job done (first matching undone line)
  python - "$Q" "$line" <<'EOF'
import sys
p, line = sys.argv[1], sys.argv[2]
rows = open(p, encoding="utf-8").read().splitlines()
for i, r in enumerate(rows):
    if r == line:
        rows[i] = "#done " + r
        break
open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
EOF
done
