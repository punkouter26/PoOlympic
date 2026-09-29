#!/bin/bash
# Run C# in the open Unity Editor (Pipeline eval); prints the JSON result. usage: tools/unity_eval.sh 'return 1+1;' [timeout_ms]
unity command eval --no-banner --caller plugin --skill unity-cli --timeout ${2:-120000} --code "$1" --format json 2>&1 | python -c "
import sys,json
raw=sys.stdin.read()
try:
  j=json.loads(raw)
  r=j.get('data',{}).get('result')
  print(json.dumps(r,indent=1)[:6000] if r is not None else raw[:3000])
except Exception: print(raw[:3000])"
