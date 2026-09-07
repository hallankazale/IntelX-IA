$ErrorActionPreference = 'Stop'

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = 'py'
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = 'python'
} else {
    throw 'Python nao encontrado. A EVO precisa do Python apenas para empacotar o diagnostico.'
}

Write-Host 'Gerando diagnostico sanitizado da EVO...' -ForegroundColor Cyan
& $python '.\tools\export_diagnostics.py'
Write-Host 'Pronto. Envie o ZIP mais recente da pasta diagnostics aqui no ChatGPT.' -ForegroundColor Green
