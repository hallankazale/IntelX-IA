$ErrorActionPreference = 'Stop'
Write-Host 'Logs ao vivo da EVO 1.0 - Ctrl+C apenas sai da visualizacao; nao para o bot.' -ForegroundColor Cyan
docker compose logs -f --tail 200 freqtrade
