# EVO Trading AI PC v0.6

Laboratório de aprendizado para BTC/USDT com dados históricos públicos reais e execução 100% paper/simulada.

## Mudanças principais
- 180 dias de candles 5m, com gap de 14 dias para não reutilizar o holdout da v0.5.
- Contexto 5m + 15m + 1h sem usar candle futuro/parcial.
- Estado inclui tendência, RSI, ATR, volume, retornos e contexto da posição.
- Q-Learning esparso com exploração forçada de ações ainda não tentadas por estado.
- Recompensa prioriza retorno líquido, alpha moderado, drawdown e penalidade leve de turnover.
- 3 folds walk-forward + modelo final treinado apenas no bloco de desenvolvimento.
- Holdout final permanece fechado em testes curtos; só é aberto ao fim de sessão longa (6h+).
- Benchmark de 25% agora inclui fee e slippage comparáveis.
- Métricas: retorno, alpha, drawdown, PF, win rate, Sharpe diário, Sortino diário e CVaR 95% diário.

## Rodar
```powershell
.\run.ps1
```
Dashboard: http://127.0.0.1:8766

Não há ordens reais, API key de corretora ou promessa de lucro.
