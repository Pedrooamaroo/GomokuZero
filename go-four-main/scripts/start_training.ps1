# Script para iniciar treino AlphaZero
# Pode ser executado e fechado - continua em background

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptPath

Write-Host "🚀 Iniciando treino AlphaZero..." -ForegroundColor Green
Write-Host "📁 Pasta: $ScriptPath" -ForegroundColor Cyan
Write-Host "⏰ Início: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Dica: Podes fechar esta janela. O treino continua!" -ForegroundColor Yellow
Write-Host "📊 Para ver progresso: Get-Content treino_log.txt -Tail 20 -Wait" -ForegroundColor Yellow
Write-Host ""

# Executa treino e guarda logs
python train.py 2>&1 | Tee-Object -FilePath "treino_log.txt"

Write-Host ""
Write-Host "✅ Treino concluído!" -ForegroundColor Green
Write-Host "⏰ Fim: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
