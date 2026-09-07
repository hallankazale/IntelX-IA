# EVO Trading AI PC v0.7

Laboratório de pesquisa **paper-only**. Preços históricos BTCUSDT são reais; ordens, saldo e lucro são simulados.

## O que mudou
- cérebro supervisionado contínuo com `HistGradientBoostingRegressor`;
- 19 features contínuas;
- alvos de 3, 6 e 12 candles;
- decisão por consenso equivalente a 30 minutos;
- thresholds de entrada/saída escolhidos somente dentro da janela de treino;
- 3 folds walk-forward;
- gap temporal antes de cada validação e antes do holdout;
- benchmark 25% com custos comparáveis;
- métricas: retorno, alpha, drawdown, PF, win rate, Sharpe/Sortino diários, CVaR, exposição e acurácia direcional;
- holdout final permanece trancado nesta versão de desenvolvimento.

## Executar no Windows
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run.ps1
```

Abre em `http://127.0.0.1:8767`.

## Importante
Não existe botão "treinar 12 horas" na v0.7. Repetir o mesmo histórico por horas não adiciona informação a um modelo supervisionado. O próximo passo só é liberado se o walk-forward mostrar consistência.

Sem API key, sem corretora conectada, sem dinheiro real e sem alavancagem.
