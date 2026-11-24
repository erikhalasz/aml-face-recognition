# Train and evaluate the three ResNet-18 runs back to back. Each run is resumable:
# if this script is interrupted, just start it again and it continues from the last finished epoch.
$py = ".\.venv\Scripts\python.exe"
foreach ($cfg in "r18_arcface", "r18_softmax", "r18_arcface_croponly") {
    if (Test-Path "outputs/$cfg/results.json") { Write-Output "== $cfg already evaluated, skipping"; continue }
    Write-Output "== training $cfg"
    & $py -W ignore scripts/train.py --config "configs/$cfg.yaml" --workers 4
    if ($LASTEXITCODE -ne 0) { Write-Output "training $cfg failed"; exit 1 }
    Write-Output "== evaluating $cfg"
    & $py -W ignore scripts/evaluate.py --run "outputs/$cfg"
}
& $py -W ignore scripts/04_leakage_check.py --run outputs/r18_arcface
