$ErrorActionPreference = 'Stop'

$config = Get-Content '.\user_data\config_evo.json' -Raw | ConvertFrom-Json
if ($config.dry_run -ne $true) {
    throw 'SEGURANCA: recusado. dry_run precisa ser true.'
}
if ($config.trading_mode -ne 'spot') {
    throw 'SEGURANCA: recusado. trading_mode precisa ser spot.'
}

New-Item -ItemType Directory -Force -Path '.\user_data\logs' | Out-Null
Write-Host 'Iniciando EVO 1.0 em DRY-RUN. Nenhuma ordem real sera enviada.' -ForegroundColor Cyan
docker compose up -d
Write-Host ''
Write-Host 'EVO iniciada.' -ForegroundColor Green
Write-Host 'Para acompanhar: .\watch-logs.ps1'
Write-Host 'Para gerar diagnostico: .\export-diagnostics.ps1'
Write-Host 'Para parar: .\stop.ps1'
