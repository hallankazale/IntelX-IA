# EVO Trading AI PC v0.7.1

Laboratório **paper-only**. Preços históricos BTCUSDT são reais; ordens, saldo e lucro são simulados.

## Correções da v0.7.1
- mantém o cérebro `HistGradientBoostingRegressor` com 19 features e horizontes 3/6/12 candles;
- substitui thresholds fixos por calibração adaptativa baseada em quantis da janela de tuning;
- aplica piso de entrada baseado no custo estimado de ida e volta (fee + slippage) mais margem de segurança;
- se nenhum threshold produzir amostra mínima, declara `sinal insuficiente` em vez de usar fallback arbitrário;
- folds só contam como elegíveis se calibração for válida e houver pelo menos 5 trades na validação;
- embargo temporal de aproximadamente 14 dias entre treino e validação de cada fold;
- gap interno de 3 dias entre treino do modelo e tuning quando houver espaço;
- dashboard mostra P90/P95/P97.5/P99, threshold escolhido, sinais, trades e exposição;
- holdout final continua bloqueado.

## Rodar no Windows
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run.ps1
```

Abre em `http://127.0.0.1:8768`.

Sem API key, sem corretora conectada, sem dinheiro real e sem alavancagem.
