#!/bin/bash
# usage: tools/unity_capture_scene.sh (Play-mode Game-view screenshot of a scene, portrait 1080x1920) <scene path> <out png (project-relative)> <seconds in play> [extra C# before capture]
cd "C:/Users/punko/Downloads/PoOlympic"
./tools/unity_eval.sh "UnityEditor.SceneManagement.EditorSceneManager.OpenScene(\"$1\"); return PoOlympic.Editor.CaptureTools.SetPortrait();" >/dev/null
unity command editor_play --no-banner --caller plugin --skill unity-cli >/dev/null 2>&1
sleep $3
[ -n "$4" ] && ./tools/unity_eval.sh "$4" | grep result
./tools/unity_eval.sh "return PoOlympic.Editor.CaptureTools.Screenshot(\"$2\");" | grep result
sleep 2
unity command editor_stop --no-banner --caller plugin --skill unity-cli >/dev/null 2>&1
sleep 3
