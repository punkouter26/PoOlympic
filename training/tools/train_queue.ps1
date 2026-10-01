<#
Sequential GPU training queue (PowerShell port of train_queue.sh, 2026-10-01). Start it detached through WMI:

  Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CurrentDirectory = '<repo>\training'; CommandLine =
    'cmd /c C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File tools\train_queue.ps1 > runs\queue_launcher.log 2>&1' }

Reads the NEXT job from runs/queue.txt each time the previous one ends, so the queue can be edited while it runs. One
job per line, fields separated by '|':
  body | task id | run name | init | max iterations | watch_gate args | prep (optional)
  init: <experiment>/<load-run dir>  resume from that dir's model_0.pt (tools/warm_start.py / plain_init.py output)
        scratch:<experiment>         train from scratch
  prep: `uv` arguments run first with POOLYMPIC_BODY set, e.g. `run python tools/warm_start.py {run:grandma_rung1/g1_v1}/model_1500.pt ...`
        ({run:<experiment>/<name>} = the newest runs/<experiment>/*_<name> directory). A failing prep skips the job.
'#' lines are skipped; a finished job is prefixed '#done ', a skipped one '#skip ', a failed one '#fail '. Stops when no
job is left or when runs/queue.stop exists. Log: runs/queue.log.

Ctrl-C hardening (2026-10-01): a console Ctrl-C of unknown origin killed the queue and its training twice (23:43 under
Git bash, 09:09 under this script: "^C" in runs/queue_launcher.log, ~5 min into a new run, no traceback). Now
  - this process ignores Ctrl-C (TreatControlCAsInput),
  - training and watcher run in their OWN hidden consoles (-WindowStyle Hidden, not -NoNewWindow),
  - a run that ends before its last iteration is RESUMED from its newest checkpoint (up to 3 times; the resumed part is a
    second runs/<experiment>/*_<name> directory, iteration numbers continue), instead of being marked done.
#>
$ErrorActionPreference = 'Continue'
try { [Console]::TreatControlCAsInput = $true } catch {}
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
function LastIteration($log) {
    $m = Select-String -Path $log -Pattern 'Learning iteration (\d+)/(\d+)' -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($m) { [int]$m.Matches[0].Groups[1].Value, [int]$m.Matches[0].Groups[2].Value } else { -1, -1 }
}
function NewestCheckpoint($dir) {
    $best = -1
    Get-ChildItem $dir -Filter 'model_*.pt' -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.BaseName -match '^model_(\d+)$') { $n = [int]$Matches[1]; if ($n -gt $best) { $best = $n } } }
    $best
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
            $p = Start-Process $uv -ArgumentList ($prep -split '\s+' | Where-Object { $_ }) -WindowStyle Hidden -PassThru -Wait `
                -RedirectStandardOutput "runs\${name}_prep.log" -RedirectStandardError "runs\${name}_prep.err.log"
            $bad = $p.ExitCode -ne 0
        }
        if ($bad) { Log "SKIP ${name}: prep failed (runs\${name}_prep.err.log)"; Mark '#skip' $line; continue }
    }
    if ($init -like 'scratch:*') { $exp = $init.Substring(8); $resume = @() }
    else { $exp, $loadrun = $init.Split('/', 2); $resume = @('--agent.resume', 'True', '--agent.load-run', $loadrun, '--agent.load-checkpoint', 'model_0.pt') }
    $todo = [int]$its; $from = 0; $ok = $false
    for ($attempt = 0; $attempt -le 3; $attempt++) {
        $suffix = if ($attempt) { ".r$attempt" } else { '' }
        Log ("START ${name}: $task (body $body, init $init, $todo its" + $(if ($attempt) { ", resume $attempt from it $from" } else { '' }) + ')')
        $before = RunDir $exp $name
        $argv = @('run', 'train', $task, '--log-root', 'runs') + $resume + @('--agent.run-name', $name, '--agent.max-iterations', "$todo")
        $train = Start-Process $uv -ArgumentList $argv -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput "runs\$name$suffix.log" -RedirectStandardError "runs\$name$suffix.err.log"
        $rundir = $null
        for ($i = 0; $i -lt 120 -and -not $rundir -and -not $train.HasExited; $i++) {
            Start-Sleep 5; $d = RunDir $exp $name; if ($d -and $d -ne $before) { $rundir = $d } }
        if ($rundir -and $watch) {
            # gates at or below the resume point were already run in the first part
            $w = [regex]::Replace($watch, '--its\s+(\S+)', { param($m) '--its ' + (($m.Groups[1].Value.Split(',') | Where-Object { [int]$_ -gt $from }) -join ',') })
            if ($w -notmatch '--its\s*($|--)') {
                $wargv = @('run', 'python', 'tools/watch_gate.py', $rundir, $name) + ($w -split '\s+' | Where-Object { $_ })
                $watcher = Start-Process $uv -ArgumentList $wargv -WindowStyle Hidden -PassThru `
                    -RedirectStandardOutput "runs\${name}${suffix}_watch.log" -RedirectStandardError "runs\${name}${suffix}_watch.err.log"
            }
        }
        $train.WaitForExit()
        $last, $max = LastIteration "runs\$name$suffix.log"
        Log "END $name rc=$($train.ExitCode) (Learning iteration $last/$max)"
        if ($last -ge 0 -and $last -ge $max - 1) { $ok = $true; break }
        # ended early (killed): resume from the newest checkpoint of the part that just ran
        $ck = if ($rundir) { NewestCheckpoint $rundir } else { -1 }
        if ($ck -lt 0 -or $max -lt 0) { Log "FAIL ${name}: ended at it $last with no checkpoint to resume from"; break }
        $from = $ck; $todo = $max - $ck
        $resume = @('--agent.resume', 'True', '--agent.load-run', (Split-Path $rundir -Leaf), '--agent.load-checkpoint', "model_$ck.pt")
        Log "RESUME ${name}: ended at it $last of $max, newest checkpoint model_$ck.pt"
        # the first part's watcher would wait forever for checkpoints that now land in the resumed part's directory
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'watch_gate.py' -and $_.CommandLine -like "*$(Split-Path $rundir -Leaf)*" -and $_.Name -notmatch 'powershell|cmd' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Start-Sleep 20
    }
    Mark $(if ($ok) { '#done' } else { '#fail' }) $line
}
