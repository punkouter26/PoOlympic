<#
Sequential GPU training queue (PowerShell port of train_queue.sh, 2026-10-01). The bash version, started detached
through WMI, still received a Ctrl-C when a Claude turn ended ("^C" in its launcher log, 2026-09-30 23:43) and took
g1_v1 down with it; detached_run.ps1 under Windows PowerShell never did. Start it the same way:

  Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CurrentDirectory = '<repo>\training'; CommandLine =
    'cmd /c C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File tools\train_queue.ps1 > runs\queue_launcher.log 2>&1' }

Reads the NEXT job from runs/queue.txt each time the previous one ends, so the queue can be edited while it runs. One
job per line, fields separated by '|':
  body | task id | run name | init | max iterations | watch_gate args | prep (optional)
  init: <experiment>/<load-run dir>  resume from that dir's model_0.pt (tools/warm_start.py / plain_init.py output)
        scratch:<experiment>         train from scratch
  prep: `uv` arguments run first with POOLYMPIC_BODY set, e.g. `run python tools/warm_start.py {run:grandma_rung1/g1_v1}/model_1500.pt ...`
        ({run:<experiment>/<name>} = the newest runs/<experiment>/*_<name> directory). A failing prep skips the job.
'#' lines are skipped; a finished job is prefixed '#done ', a skipped one '#skip '. Stops when no job is left or when
runs/queue.stop exists. Log: runs/queue.log.
#>
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$Q = 'runs\queue.txt'
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe" }
function Log($m) { $l = "[{0}] {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m; Add-Content runs\queue.log $l; Write-Output $l }
function Mark($prefix, $line) {
    $rows = Get-Content $Q -Encoding UTF8
    for ($i = 0; $i -lt $rows.Count; $i++) { if ($rows[$i] -eq $line) { $rows[$i] = "$prefix $line"; break } }
    [IO.File]::WriteAllLines((Resolve-Path $Q), $rows, (New-Object Text.UTF8Encoding $false))
}
function RunDir($exp, $name) {
    $d = Get-ChildItem "runs\$exp" -Directory -Filter "*_$name" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if ($d) { $d.FullName } else { $null }
}
while ($true) {
    if (Test-Path runs\queue.stop) { Log 'stop file found - queue ends'; break }
    $line = Get-Content $Q -Encoding UTF8 | Where-Object { $_ -notmatch '^\s*#' -and $_.Trim() } | Select-Object -First 1
    if (-not $line) { Log 'queue empty'; break }
    $f = $line.Split('|') | ForEach-Object { $_.Trim() }
    $body, $task, $name, $init, $its, $watch = $f[0..5]
    $prep = if ($f.Count -gt 6) { $f[6] } else { '' }
    $env:POOLYMPIC_BODY = $body
    if ($prep) {
        $bad = $false
        $prep = [regex]::Replace($prep, '\{run:([^/}]+)/([^}]+)\}', { param($m) $d = RunDir $m.Groups[1].Value $m.Groups[2].Value; if (-not $d) { $script:bad = $true }; "$d" })
        Log "PREP ${name}: uv $prep"
        if (-not $bad) {
            $p = Start-Process $uv -ArgumentList ($prep -split '\s+' | Where-Object { $_ }) -NoNewWindow -PassThru -Wait `
                -RedirectStandardOutput "runs\${name}_prep.log" -RedirectStandardError "runs\${name}_prep.err.log"
            $bad = $p.ExitCode -ne 0
        }
        if ($bad) { Log "SKIP ${name}: prep failed (runs\${name}_prep.err.log)"; Mark '#skip' $line; continue }
    }
    if ($init -like 'scratch:*') { $exp = $init.Substring(8); $resume = @() }
    else { $exp, $loadrun = $init.Split('/', 2); $resume = @('--agent.resume', 'True', '--agent.load-run', $loadrun, '--agent.load-checkpoint', 'model_0.pt') }
    Log "START ${name}: $task (body $body, init $init, $its its)"
    $argv = @('run', 'train', $task, '--log-root', 'runs') + $resume + @('--agent.run-name', $name, '--agent.max-iterations', "$its")
    $train = Start-Process $uv -ArgumentList $argv -NoNewWindow -PassThru -RedirectStandardOutput "runs\$name.log" -RedirectStandardError "runs\$name.err.log"
    $rundir = $null
    for ($i = 0; $i -lt 120 -and -not $rundir -and -not $train.HasExited; $i++) { Start-Sleep 5; $rundir = RunDir $exp $name }
    if ($rundir -and $watch) {
        $wargv = @('run', 'python', 'tools/watch_gate.py', $rundir, $name) + ($watch -split '\s+' | Where-Object { $_ })
        Start-Process $uv -ArgumentList $wargv -NoNewWindow -RedirectStandardOutput "runs\${name}_watch.log" -RedirectStandardError "runs\${name}_watch.err.log" | Out-Null
    }
    $train.WaitForExit()
    $last = (Select-String -Path "runs\$name.log" -Pattern 'Learning iteration' | Select-Object -Last 1).Line -replace '[^\x20-\x7e]', '' -replace '\[\d+m', '' -replace '\s+', ' '
    Log "END $name rc=$($train.ExitCode) ($last)"
    Mark '#done' $line
}
