$ErrorActionPreference = 'Stop'
Write-Host 'Parando EVO Trading AI 1.0...' -ForegroundColor Yellow
docker compose down
Write-Host 'EVO parada.' -ForegroundColor Green
