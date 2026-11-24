# Improved main model (degradation augmentation + 14 epochs). Waits until run_all.ps1 has finished
# (crop-only ablation evaluated and no other training/eval/leakage process running), then trains and
# evaluates r18_arcface_aug. Resumable: restart it and training continues from the last finished epoch.
$py = ".\.venv\Scripts\python.exe"
$cfg = "r18_arcface_aug"
function Test-Busy {
    $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -match 'scripts[/\\](train|evaluate|04_leakage_check)\.py' -and $_.CommandLine -notmatch $cfg }
    return [bool]$p
}
while (-not (Test-Path "outputs/r18_arcface_croponly/results.json") -or (Test-Busy)) {
    Start-Sleep -Seconds 300
}
Write-Output "== $(Get-Date -Format 'yyyy-MM-dd HH:mm') starting $cfg"
if (-not (Test-Path "outputs/$cfg/results.json")) {
    & $py -W ignore scripts/train.py --config "configs/$cfg.yaml" --workers 4
    if ($LASTEXITCODE -ne 0) { Write-Output "training $cfg failed"; exit 1 }
    Write-Output "== evaluating $cfg"
    & $py -W ignore scripts/evaluate.py --run "outputs/$cfg"
}
Write-Output "== $(Get-Date -Format 'yyyy-MM-dd HH:mm') done"
