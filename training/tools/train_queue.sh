#!/usr/bin/env bash
# Sequential GPU training queue (recipe v5, 2026-09-30). Reads the NEXT job from runs/queue.txt each time the previous
# one ends, so the queue can be edited while it runs. One job per line, fields separated by '|':
#   body | task id | run name | init | max iterations | watch_gate args (gates / its / flags) | prep (optional)
#   init: <experiment>/<load-run dir>  resume from that dir's model_0.pt (tools/warm_start.py / plain_init.py output)
#         scratch:<experiment>         train from scratch (no resume)
#   prep: a shell command run first with POOLYMPIC_BODY set (e.g. warm_start.py building this job's init dir from the
#         previous run's checkpoint; `runs/<exp>/*_<name>` globs work). A failing prep skips the job ('#skip ').
# A line starting with '#' is skipped; a finished job is prefixed with '#done '. Stops when no job is left or when
# runs/queue.stop exists. Each run's watcher starts next to its training and gates the checkpoints on CPU MuJoCo.
# Detached start (survives the Claude session), from training/:
#   Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CurrentDirectory = '<repo>\training'; CommandLine =
#     'cmd /c "C:\Program Files\Git\bin\bash.exe" tools/train_queue.sh > runs\queue_launcher.log 2>&1' }
cd "$(dirname "$0")/.." || exit 1
Q=runs/queue.txt
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a runs/queue.log; }
mark() {   # mark <prefix> <line>: prefix the first matching undone line
  python - "$Q" "$1" "$2" <<'EOF'
import sys
p, prefix, line = sys.argv[1], sys.argv[2], sys.argv[3]
rows = open(p, encoding="utf-8").read().splitlines()
for i, r in enumerate(rows):
    if r == line:
        rows[i] = prefix + " " + r
        break
open(p, "w", encoding="utf-8").write("\n".join(rows) + "\n")
EOF
}
while true; do
  [ -f runs/queue.stop ] && { log "stop file found — queue ends"; break; }
  line=$(grep -v '^\s*#' "$Q" | grep -v '^\s*$' | head -1)
  [ -z "$line" ] && { log "queue empty"; break; }
  IFS='|' read -r body task name init its watch prep <<<"$line"
  body=$(echo $body); task=$(echo $task); name=$(echo $name); init=$(echo $init); its=$(echo $its)
  prep=$(echo "$prep" | sed 's/^ *//; s/ *$//')
  if [ -n "$prep" ]; then
    log "PREP $name: $prep"
    if ! POOLYMPIC_BODY=$body bash -c "$prep" >> "runs/${name}_prep.log" 2>&1; then
      log "SKIP $name: prep failed (runs/${name}_prep.log)"; mark "#skip" "$line"; continue
    fi
  fi
  if [ "${init%%:*}" = "scratch" ]; then
    exp=${init#scratch:}; resume=()
  else
    exp=${init%%/*}; loadrun=${init#*/}
    resume=(--agent.resume True --agent.load-run "$loadrun" --agent.load-checkpoint model_0.pt)
  fi
  log "START $name: $task (body $body, init $init, $its its)"
  POOLYMPIC_BODY=$body uv run train "$task" --log-root runs "${resume[@]}" \
      --agent.run-name "$name" --agent.max-iterations "$its" > "runs/$name.log" 2>&1 &
  tpid=$!
  # the run directory appears once training starts; then attach the watcher
  rundir=""
  for _ in $(seq 1 120); do
    rundir=$(ls -d runs/$exp/*_"$name" 2>/dev/null | tail -1); [ -n "$rundir" ] && break; sleep 5
  done
  if [ -n "$rundir" ] && [ -n "$(echo $watch)" ]; then
    POOLYMPIC_BODY=$body nohup uv run python tools/watch_gate.py "$rundir" "$name" $watch > "runs/${name}_watch.log" 2>&1 &
  fi
  wait $tpid; rc=$?
  log "END $name rc=$rc ($(grep -a 'Learning iteration' runs/$name.log | tail -1 | tr -s ' '))"
  mark "#done" "$line"
done
