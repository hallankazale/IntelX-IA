package com.evotradingai.core;

public final class TradingMetrics {
    public final double returnPct;
    public final double buyHoldPct;
    public final double alphaPct;
    public final double maxDrawdownPct;
    public final double profitFactor;
    public final double winRatePct;
    public final double sharpe;
    public final int trades;
    public final double finalEquity;

    public TradingMetrics(double returnPct, double buyHoldPct, double alphaPct,
                          double maxDrawdownPct, double profitFactor, double winRatePct,
                          double sharpe, int trades, double finalEquity) {
        this.returnPct = returnPct;
        this.buyHoldPct = buyHoldPct;
        this.alphaPct = alphaPct;
        this.maxDrawdownPct = maxDrawdownPct;
        this.profitFactor = profitFactor;
        this.winRatePct = winRatePct;
        this.sharpe = sharpe;
        this.trades = trades;
        this.finalEquity = finalEquity;
    }
}
