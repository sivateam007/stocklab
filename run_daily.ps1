# Daily routine: run after market close (15:30 IST), weekdays.
# Register once with:
#   schtasks /create /tn "Stocklab daily" /tr "powershell -NoProfile -File \"D:\stock market\run_daily.ps1\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 18:30

$ErrorActionPreference = "Stop"
Set-Location "D:\stock market"

$logDir = "reports\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$stamp = Get-Date -Format "yyyyMMdd"
$log = Join-Path $logDir "daily_$stamp.log"

function Log($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss')  $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

Log "=== stocklab daily start ==="
try {
    python -m stocklab ingest  2>&1 | Tee-Object -FilePath $log -Append
    python -m stocklab screen  2>&1 | Tee-Object -FilePath $log -Append
    Log "=== done ==="
}
catch {
    Log "FAILED: $_"
    exit 1
}
