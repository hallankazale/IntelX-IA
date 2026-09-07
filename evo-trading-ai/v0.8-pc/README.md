# EVO Trading AI PC v0.8

Laboratório **paper-only**. Usa candles históricos reais BTCUSDT; ordens, saldo e P&L são simulados.

## Mudança principal
A v0.7.1 mostrou que regressão de retorno não gerava sinais grandes o suficiente para superar custos de forma robusta.  
A v0.8 troca o alvo por **eventos de mercado** e pergunta algo mais útil:

> qual a probabilidade de o preço atingir uma barreira de alta antes de uma barreira de queda?

### Cérebro
- `HistGradientBoostingClassifier`
- 31 features contínuas
- horizontes 1h / 2h / 4h
- labels triple-barrier conservadores
- consenso `p_up`, `p_down` e `edge = p_up - p_down`
- calibração de política só dentro da janela de tuning

### Protocolo
- ~365 dias de BTCUSDT 5m
- 3 folds walk-forward
- embargo externo de ~14 dias
- embargo interno de ~3 dias
- execução no **open do candle seguinte**, evitando preencher a ordem no mesmo close usado para decidir
- holdout final continua trancado

### Risco
- spot long-only
- 25% de exposição por posição
- sem alavancagem
- stop 0,65%
- take 1,50%
- máximo 4h por posição
- fee 0,10% por lado
- slippage 0,02% por lado

## Windows
```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run.ps1
```

Abre em `http://127.0.0.1:8769`.

## Importante
Backtest não garante desempenho futuro. Nenhuma corretora está conectada e nenhuma ordem real é enviada.
