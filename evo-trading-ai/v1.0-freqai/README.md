# EVO Trading AI 1.0 — Freqtrade/FreqAI

Base de produção segura para a EVO usando Freqtrade/FreqAI como infraestrutura de mercado e execução.

## Estado atual

- Dry-run obrigatório (`dry_run: true`).
- Spot long-only, sem alavancagem.
- Binance como exchange inicial.
- BTC/USDT e ETH/USDT.
- FreqAI com LightGBMRegressor e retreinamento periódico.
- Logs persistentes.
- Exportação de pacote de diagnóstico ZIP para análise externa.
- Nenhuma chave real é necessária para o dry-run.

## Objetivo

A EVO só deve avançar para capital real se demonstrar retorno líquido positivo, drawdown controlado e consistência fora da amostra e em dry-run ao vivo. Não existe garantia de lucro.

## Instalação recomendada no Windows

Freqtrade recomenda Docker no Windows. Instale Docker Desktop e então execute:

```powershell
cd $HOME\Desktop\EVO-Trading-AI-1.0\evo-trading-ai\v1.0-freqai
Set-ExecutionPolicy -Scope Process Bypass -Force
.\setup.ps1
.\start-dry-run.ps1
```

Nesta primeira build a API/FreqUI fica desabilitada por segurança. O acompanhamento é feito por `watch-logs.ps1` e pelo pacote de diagnóstico.

## Diagnóstico

A qualquer momento, gere um ZIP para enviar ao ChatGPT:

```powershell
.\export-diagnostics.ps1
```

O arquivo será criado em `diagnostics/` e inclui manifesto, configuração sanitizada, logs, trades exportáveis quando disponíveis, inventário de modelos FreqAI e resultados de backtest existentes. O coletor remove campos conhecidos de chave, secret, senha e token.

## Estrutura

- `user_data/strategies/EvoFreqAiStrategy.py`: cérebro/estratégia EVO.
- `user_data/config_evo.json`: configuração dry-run.
- `docker-compose.yml`: engine Freqtrade/FreqAI estável.
- `setup.ps1`: prepara diretórios, baixa a imagem e valida a estratégia.
- `start-dry-run.ps1`: inicia dry-run com logs.
- `watch-logs.ps1`: acompanha logs ao vivo.
- `stop.ps1`: para o bot.
- `export-diagnostics.ps1`: gera diagnóstico ZIP.
- `tools/export_diagnostics.py`: coletor e sanitizador.

## Regras de segurança

1. `dry_run` deve permanecer `true` nesta versão.
2. Sem futures, margem, short ou alavancagem.
3. Não colocar API keys no GitHub.
4. Não habilitar saque em nenhuma futura chave de exchange.
5. Não avançar para real por causa de um único backtest positivo.
