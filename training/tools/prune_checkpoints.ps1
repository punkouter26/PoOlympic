# Prune training/runs: keep, per run, the newest checkpoint, the gate iterations and warm starts named in
# runs/queue.txt, and every checkpoint a brain sidecar in parity/brains points at; delete the other model_*.pt and
# the git/PoOlympic.diff snapshots over 5 MB (one run stored a 163 MB diff of binary files).
#   pwsh training/tools/prune_checkpoints.ps1            # dry run: prints the plan
#   pwsh training/tools/prune_checkpoints.ps1 -Delete    # deletes (training/runs is git-ignored: not recoverable)
param([switch]$Delete)

$root = Split-Path (Split-Path $PSScriptRoot)
$runs = Join-Path $root 'training/runs'
$queue = Get-Content (Join-Path $runs 'queue.txt') -ErrorAction SilentlyContinue
$sidecars = Get-ChildItem (Join-Path $root 'parity/brains') -Filter *.json
$doomed = @()

foreach ($run in Get-ChildItem $runs -Directory -Recurse -Depth 1) {
    $pts = Get-ChildItem $run.FullName -Filter model_*.pt |
        ForEach-Object { [pscustomobject]@{ File = $_; It = [int]($_.BaseName -replace 'model_', '') } } | Sort-Object It
    if (-not $pts) { continue }
    $name = $run.Name -replace '^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_', ''
    $keep = [System.Collections.Generic.HashSet[int]]::new()
    [void]$keep.Add($pts[-1].It)
    foreach ($line in $queue) {
        $cols = $line.TrimStart('#').Split('|') | ForEach-Object { $_.Trim() }
        if ($cols.Count -ge 6 -and $cols[2] -eq $name -and $cols[5] -match '--its\s+([\d,]+)') {
            $Matches[1].Split(',') | ForEach-Object { [void]$keep.Add([int]$_) }
        }
        foreach ($m in [regex]::Matches($line, "\{run:[^}]*/$([regex]::Escape($name))\}/model_(\d+)\.pt")) { [void]$keep.Add([int]$m.Groups[1].Value) }
    }
    $sidecars | Select-String -Pattern ([regex]::Escape($run.Name) + '[\\/]+model_(\d+)\.pt') -List |
        ForEach-Object { [void]$keep.Add([int]$_.Matches[0].Groups[1].Value) }
    $del = @($pts | Where-Object { -not $keep.Contains($_.It) } | ForEach-Object { $_.File })
    $diff = Get-Item (Join-Path $run.FullName 'git/PoOlympic.diff') -ErrorAction SilentlyContinue
    if ($diff -and $diff.Length -gt 5MB) { $del += $diff }
    if ($del.Count -eq 0) { continue }
    '{0,-28} keep {1,-24} delete {2,3} files {3,5:N0} MB' -f "$($run.Parent.Name)/$name", (($keep | Sort-Object) -join ','), $del.Count, (($del | Measure-Object Length -Sum).Sum / 1MB)
    $doomed += $del
}

'total: {0} files, {1:N0} MB{2}' -f $doomed.Count, (($doomed | Measure-Object Length -Sum).Sum / 1MB), $(if ($Delete) { ' - deleting' } else { ' (dry run; add -Delete)' })
if ($Delete) { $doomed | Remove-Item -Confirm:$false }
