# Usage: .\scp_to_pod.ps1 -Port 12345 -PodHost 1.2.3.4
# Example: .\scp_to_pod.ps1 -Port 13585 -PodHost 91.199.227.82
param(
    [Parameter(Mandatory=$true)][int]$Port,
    [Parameter(Mandatory=$true)][string]$PodHost
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== SCP to pod root@${PodHost}:${Port} ===" -ForegroundColor Cyan

scp -P $Port -i "$HOME\.ssh\id_ed25519" -o StrictHostKeyChecking=no `
    "$ScriptDir\train_avery.py" `
    "$ScriptDir\setup_competition.sh" `
    "$ScriptDir\run_competition.sh" `
    "$ScriptDir\upload_results.sh" `
    "root@${PodHost}:/workspace/"

if ($LASTEXITCODE -eq 0) {
    Write-Host "" 
    Write-Host "=== Files uploaded. Next steps: ===" -ForegroundColor Green
    Write-Host "  ssh -p $Port -i ~/.ssh/id_ed25519 root@$PodHost"
    Write-Host "  chmod +x /workspace/*.sh"
    Write-Host "  bash /workspace/setup_competition.sh"
    Write-Host "  # Wait for dataset download (~5-10 min)"
    Write-Host "  bash /workspace/run_competition.sh"
} else {
    Write-Host "SCP failed. Check pod is running and SSH is ready." -ForegroundColor Red
}
