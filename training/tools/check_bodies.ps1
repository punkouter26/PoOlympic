# Import poolympic.tasks once per body (a process = one body) and check the registry: every SUPERSEDED id of the body's
# family is a registered task. Run from training/ after every edit to poolympic/tasks/__init__.py.
$fail = 0
foreach ($b in "matt", "mattbio", "zombie", "grandma", "kim") {
    $env:POOLYMPIC_BODY = $b
    $out = uv run python -c "import poolympic.tasks as T; from mjlab.tasks.registry import list_tasks; ours = [t for t in list_tasks() if t.startswith('PoOlympic-')]; fam = 'PoOlympic-' + {'matt': 'Matt-', 'mattbio': 'MattBio-', 'zombie': 'Zombie-', 'grandma': 'Grandma-', 'kim': 'Kim-'}['$b']; bad = [t for t in T.SUPERSEDED if t.startswith(fam) and t not in ours]; print('BODY $b', len(ours), 'tasks,', sum(t in T.SUPERSEDED for t in ours), 'superseded, unknown superseded ids', bad); raise SystemExit(1 if bad else 0)" 2>&1
    if ($LASTEXITCODE -ne 0) { $fail = 1; $out | Select-Object -Last 12 } else { $out | Select-String "^BODY" }
}
$env:POOLYMPIC_BODY = $null
exit $fail
