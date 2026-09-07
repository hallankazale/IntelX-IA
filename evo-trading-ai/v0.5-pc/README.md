# EVO Trading AI PC v0.5

Laboratório local de aprendizado para BTC/USDT.

## O que faz
- baixa ~60 dias de candles públicos reais de 5 minutos da Binance;
- usa 70% treino / 15% validação / 15% teste não visto;
- Q-Learning com action masking;
- ações válidas: flat -> HOLD/BUY; posicionado -> HOLD/SELL;
- benchmark ajustado para a mesma exposição máxima de 25%;
- custos simulados: fee 0,10% e slippage 0,02%;
- Risk Engine: posição máx. 25%, stop 3%, take 6%, máx. 4h;
- salva checkpoint local;
- dashboard web local em http://127.0.0.1:8765.

## Executar no Windows
Abra PowerShell nesta pasta:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run.ps1
```

Use primeiro **Teste 10 min**. Só rode 12h se BUY e SELL estiverem acumulando amostras.

## Segurança
Não usa chave de API, não envia ordens à corretora e não movimenta dinheiro real.
