$ErrorActionPreference = 'Stop'

Write-Host '=== EVO Trading AI 1.0 - Setup ===' -ForegroundColor Cyan

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host 'Docker nao encontrado.' -ForegroundColor Red
    Write-Host 'O Freqtrade recomenda Docker no Windows. Instale o Docker Desktop e execute este script novamente.' -ForegroundColor Yellow
    exit 1
}

try {
    docker info *> $null
} catch {
    Write-Host 'Docker esta instalado, mas o Docker Desktop/engine nao esta rodando.' -ForegroundColor Red
    Write-Host 'Abra o Docker Desktop, espere ficar pronto e execute .\setup.ps1 novamente.' -ForegroundColor Yellow
    exit 1
}

$dirs = @(
    '.\user_data\logs',
    '.\user_data\models',
    '.\user_data\data',
    '.\user_data\backtest_results',
    '.\diagnostics'
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

$config = Get-Content '.\user_data\config_evo.json' -Raw | ConvertFrom-Json
if ($config.dry_run -ne $true) {
    throw 'SEGURANCA: config_evo.json precisa permanecer com dry_run=true nesta versao.'
}
if ($config.trading_mode -ne 'spot') {
    throw 'SEGURANCA: EVO 1.0 aceita somente spot.'
}

Write-Host 'Baixando/atualizando imagem Freqtrade stable_freqai...' -ForegroundColor Cyan
docker compose pull

Write-Host 'Validando estrategia no container...' -ForegroundColor Cyan
docker compose run --rm freqtrade list-strategies --strategy-path /freqtrade/user_data/strategies | Tee-Object -Variable strategies | Out-Host
if (($strategies | Out-String) -notmatch 'EvoFreqAiStrategy') {
    throw 'A estrategia EvoFreqAiStrategy nao foi encontrada pelo Freqtrade.'
}

Write-Host ''
Write-Host 'Setup concluido.' -ForegroundColor Green
Write-Host 'Proximo comando: .\start-dry-run.ps1' -ForegroundColor Green
