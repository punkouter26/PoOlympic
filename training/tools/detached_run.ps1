<#
One training run + its checkpoint watcher, meant to be started OUTSIDE the Claude session's process tree so it survives
the session ending (rs_v1, rs_v4 and rs_v5 all died with their sessions). Launch it through WMI:

  Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CurrentDirectory = '<repo>\training';
      CommandLine = 'pwsh -NoProfile -WindowStyle Hidden -File tools\detached_run.ps1 -Body mattbio -Task <id> ...' }

Logs: runs/<Name>.log (training), runs/<Name>_watch.log (watcher). Stop: kill the python processes (preflight.ps1 lists them).
#>
param(
    [Parameter(Mandatory)] [string] $Body,
    [Parameter(Mandatory)] [string] $Task,
    [Parameter(Mandatory)] [string] $Name,          # run name (dir suffix)
    [Parameter(Mandatory)] [string] $Experiment,    # e.g. mattbio_stance
    [Parameter(Mandatory)] [string] $LoadRun,       # dir under runs/<Experiment>
    [Parameter(Mandatory)] [string] $Checkpoint,    # e.g. model_1050.pt
    [Parameter(Mandatory)] [int] $Iterations,
    [string] $TrainArgs = '',                       # extra train flags, e.g. '--agent.algorithm.entropy-coef 0.0025'
    [string] $WatchPrefix = '',                     # parity/watch_<prefix>.jsonl
    [string] $WatchArgs = ''                        # e.g. '--gates S,2 --its 1150,1350,1549'
)
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:POOLYMPIC_BODY = $Body
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe" }

$trainArgv = @('run', 'train', $Task, '--log-root', 'runs', '--agent.resume', 'True', '--agent.load-run', $LoadRun,
    '--agent.load-checkpoint', $Checkpoint, '--agent.run-name', $Name, '--agent.max-iterations', "$Iterations") +
    ($TrainArgs -split '\s+' | Where-Object { $_ })
$train = Start-Process $uv -ArgumentList $trainArgv -NoNewWindow -PassThru `
    -RedirectStandardOutput "runs\$Name.log" -RedirectStandardError "runs\$Name.err.log"

if ($WatchArgs) {
    $runDir = $null
    for ($i = 0; $i -lt 120 -and -not $runDir; $i++) {
        Start-Sleep 5
        $runDir = Get-ChildItem "runs\$Experiment" -Directory -Filter "*_$Name" -ErrorAction SilentlyContinue |
            Sort-Object Name | Select-Object -Last 1
    }
    if ($runDir) {
        $prefix = if ($WatchPrefix) { $WatchPrefix } else { $Name }
        $watchArgv = @('run', 'python', 'tools/watch_gate.py', $runDir.FullName, $prefix) + ($WatchArgs -split '\s+' | Where-Object { $_ })
        Start-Process $uv -ArgumentList $watchArgv -NoNewWindow `
            -RedirectStandardOutput "runs\${Name}_watch.log" -RedirectStandardError "runs\${Name}_watch.err.log" | Out-Null
    }
}
$train.WaitForExit()
