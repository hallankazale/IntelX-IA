package com.evotradingai.core;

public final class Portfolio {
    public static final double FEE_RATE=0.0010, SLIPPAGE_RATE=0.0002, POSITION_FRACTION=0.25, STOP_LOSS=0.03, TAKE_PROFIT=0.06;
    public static final int MAX_HOLD_BARS=48;
    private double cash,asset,entryPrice,tradeCostBasis; private int barsHeld,trades,wins; private double grossProfit,grossLoss;
    public Portfolio(double startingCash){ cash=startingCash; }
    public boolean inPosition(){ return asset>1e-12; }
    public double equity(double markPrice){ return cash+asset*markPrice; }
    public void applyAction(int action,double price){ if(action==TradingModel.ACTION_BUY&&!inPosition()) buy(price); else if(action==TradingModel.ACTION_SELL&&inPosition()) sell(price); }
    private void buy(double price){
        double allocation=cash*POSITION_FRACTION; if(allocation<1e-9)return;
        double execution=price*(1+SLIPPAGE_RATE), fee=allocation*FEE_RATE, net=allocation-fee;
        asset=net/execution; cash-=allocation; entryPrice=execution; tradeCostBasis=allocation; barsHeld=0;
    }
    private void sell(double price){
        double execution=price*(1-SLIPPAGE_RATE), gross=asset*execution, fee=gross*FEE_RATE, net=gross-fee, pnl=net-tradeCostBasis;
        cash+=net; asset=0; entryPrice=0; tradeCostBasis=0; barsHeld=0; trades++;
        if(pnl>=0){ wins++; grossProfit+=pnl; } else grossLoss+=-pnl;
    }
    public void applyRiskRules(Candle next){
        if(!inPosition())return; barsHeld++; double stop=entryPrice*(1-STOP_LOSS), take=entryPrice*(1+TAKE_PROFIT);
        if(next.low<=stop) sell(stop); else if(next.high>=take) sell(take); else if(barsHeld>=MAX_HOLD_BARS) sell(next.close);
    }
    public void closeAt(double price){ if(inPosition())sell(price); }
    public int trades(){return trades;} public int wins(){return wins;} public double grossProfit(){return grossProfit;} public double grossLoss(){return grossLoss;}
}
