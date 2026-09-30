<#
Pre-flight for a training run on this laptop (RTX 5070 Ti Laptop 12 GB, Core Ultra 9 275HX). Checks only — it never
closes apps or deletes runs (both are the user's call). AGENTS.md: close the editors for 30+ min runs, start TensorBoard,
review obsolete runs.

Usage (from training/):  pwsh tools/preflight.ps1            # report + start TensorBoard if it is not running
#>
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Write-Host "== GPU"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total,clocks.sm --format=csv,noheader
$gpuT = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
if ($gpuT -ge 70) { Write-Host "  WARN idle GPU at $gpuT C: laptop will thermal-throttle during training (elevate the laptop, check the vendor fan/performance mode)" }

Write-Host "== Apps competing for GPU / CPU (close for runs > 30 min; the MuJoCo viewer alone costs +55-80 % per iteration)"
$heavy = Get-Process | Where-Object { $_.ProcessName -match '^(Unity|blender|UnrealEditor|.*Win64-Shipping|EpicGamesLauncher)$' }
if ($heavy) { $heavy | Group-Object ProcessName | ForEach-Object { "  {0} x{1}  {2:N1} GB RAM" -f $_.Name, $_.Count, (($_.Group | Measure-Object WorkingSet64 -Sum).Sum / 1GB) } }
else { Write-Host "  none" }
$py = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'train|play|eval_cpu|bench_env' }
if ($py) { Write-Host "  running python jobs:"; $py | ForEach-Object { "    pid {0}: {1}" -f $_.ProcessId, ($_.CommandLine -replace '.*python(.exe)?"?\s*', '') } }

Write-Host "== Power"
powercfg /getactivescheme
$bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($bat -and $bat.BatteryStatus -ne 2) { Write-Host "  WARN on battery: plug in (the GPU power limit drops on battery)" }
$hags = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers' -Name HwSchMode -ErrorAction SilentlyContinue).HwSchMode
Write-Host ("  hardware-accelerated GPU scheduling: {0}" -f $(if ($hags -eq 2) { 'on' } else { 'off' }))

Write-Host "== TensorBoard"
$tb = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'tensorboard' -and $_.CommandLine -match 'logdir' }
if ($tb) { Write-Host "  running (http://localhost:6006)" }
else {
    Start-Process -WindowStyle Hidden -FilePath "uv" -ArgumentList "run", "tensorboard", "--logdir", "runs", "--port", "6006" -WorkingDirectory $root
    Write-Host "  started: http://localhost:6006"
}

Write-Host "== Runs (review obsolete ones before a long run; keep the warm-start sources of parity/brains/*.json)"
Get-ChildItem (Join-Path $root 'runs') -Directory | ForEach-Object {
    $exp = $_
    Get-ChildItem $exp.FullName -Directory | ForEach-Object {
        $mb = (Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
        "  {0,-18} {1,-40} {2,7:N0} MB" -f $exp.Name, $_.Name, $mb
    }
}
$used = Get-ChildItem (Join-Path $root '..\parity\brains') -Filter *.onnx.json | ForEach-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).checkpoint } | Where-Object { $_ }
Write-Host "  checkpoints referenced by deployed brains:"; $used | ForEach-Object { "    $_" }
