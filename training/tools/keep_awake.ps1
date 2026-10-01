<#
Keep the laptop out of Modern Standby while the training queue runs (2026-10-01: with a 5 min idle sleep on AC the
system entered standby 5 min into getup_rev_v2 and froze it for 53 min, Kernel-Power events 506 / 507). Holds an
ES_SYSTEM_REQUIRED request (no power-plan change, the display may still turn off) and exits when no train_queue.ps1 or
`train` process is left. Start it detached like the queue:

  Invoke-CimMethod Win32_Process -MethodName Create -Arguments @{ CurrentDirectory = '<repo>\training'; CommandLine =
    'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File tools\keep_awake.ps1' }
#>
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM_REQUIRED = [uint32]1
function Busy {
    [bool](Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and ($_.CommandLine -match 'train_queue\.ps1' -or $_.CommandLine -match '\brun train PoOlympic') })
}
Start-Sleep 5
while (Busy) {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    Start-Sleep 30
}
[void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
