# Script to start AlphaZero training
# Can be executed and closed - continues in the background

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptPath

Write-Host "🚀 Starting AlphaZero training..." -ForegroundColor Green
Write-Host "📁 Directory: $ScriptPath" -ForegroundColor Cyan
Write-Host "⏰ Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Tip: You can close this window. Training continues!" -ForegroundColor Yellow
Write-Host "📊 To view progress: Get-Content training_log.txt -Tail 20 -Wait" -ForegroundColor Yellow
Write-Host ""

# Execute training and save logs
python train.py 2>&1 | Tee-Object -FilePath "training_log.txt"

Write-Host ""
Write-Host "✅ Training completed!" -ForegroundColor Green
Write-Host "⏰ End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
