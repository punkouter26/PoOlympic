#!/bin/bash
# usage: tools/unity_shot.sh <out png, project-relative>  (Play mode)
cd "C:/Users/punko/Downloads/PoOlympic"
./tools/unity_eval.sh "return PoOlympic.Editor.CaptureTools.Screenshot(\"$1\");" | grep -o '"result": ".*"'
