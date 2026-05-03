# ULGM ablation: fixed gamma2 vs linear gamma2 warmup (GPU if available; omit --no-cuda).
# Run from SDT directory (where train.py is), e.g.:
#   cd D:\...\SDT\SDT
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_gamma2_ablation.ps1
# Do not pass --no-cuda so PyTorch uses GPU when available.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
New-Item -ItemType Directory -Force -Path "logs_gamma2_warmup" | Out-Null

$common = @(
  "train.py", "--Dataset", "IEMOCAP",
  "--iemocap-pkl-path", "data/iemocap_multimodal_features_rppg_ses01_v3.pkl",
  "--iemocap-session-prefixes", "Ses01",
  "--use-rppg", "--iemocap-rppg-npz-path", "data/iemocap_rppg_features_ses01_v3.npz",
  "--gamma-1", "1", "--gamma-2", "1", "--gamma-3", "1",
  "--epochs", "45", "--batch-size", "8", "--split-seed", "42",
  "--ulgm-lambda-min", "0.0005", "--ulgm-lambda-max", "0.005",
  "--ulgm-e-delay", "3", "--ulgm-e-ramp", "8",
  "--ulgm-alpha-t", "0.4", "--ulgm-alpha-a", "0.2", "--ulgm-alpha-v", "0.2", "--ulgm-alpha-r", "0.2",
  "--ulgm-normalize-alpha",
  "--no-pseudo-ulgm"
)

Write-Host "=== A: fixed gamma2 (no warmup, gamma-2-warmup-epochs=0) ==="
python @common --gamma-2-warmup-epochs 0 2>&1 | Tee-Object -FilePath "logs_gamma2_warmup/runA_fixed_gamma2.log"

Write-Host "=== B: progressive gamma2 (0 -> 1 over 15 epochs) ==="
python @common --gamma-2-start 0 --gamma-2-warmup-epochs 15 2>&1 | Tee-Object -FilePath "logs_gamma2_warmup/runB_gamma2_warmup15.log"

Write-Host "Done. Compare F-Score lines in logs_gamma2_warmup\*.log"
