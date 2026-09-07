package com.evotradingai.core;

import java.util.ArrayList;
import java.util.List;

public final class PaperTrader {
    public TradingMetrics evaluate(PreparedMarket market,double[][] q,int start,int endExclusive){
        TradingModel model=new TradingModel(q,0); Portfolio p=new Portfolio(1000);
        int first=Math.max(market.warmup,start), last=Math.min(endExclusive-1,market.size()-2);
        if(last<=first+5)return new TradingMetrics(0,0,0,0,0,0,0,0,1000);
        double peak=1000,maxDd=0,prev=1000; List<Double> returns=new ArrayList<>();
        for(int i=first;i<=last;i++){
            Candle current=market.candles.get(i), next=market.candles.get(i+1);
            int state=market.stateAt(i,p.inPosition()); p.applyAction(model.greedyAction(state),current.close); p.applyRiskRules(next);
            double equity=p.equity(next.close); peak=Math.max(peak,equity); maxDd=Math.max(maxDd,(peak-equity)/peak); returns.add(equity/prev-1); prev=equity;
        }
        double finalPrice=market.candles.get(last+1).close; p.closeAt(finalPrice); double finalEquity=p.equity(finalPrice);
        double ret=(finalEquity/1000-1)*100, bh=(finalPrice/market.candles.get(first).close-1)*100, alpha=ret-bh;
        double pf=p.grossLoss()<1e-9?(p.grossProfit()>0?99:0):p.grossProfit()/p.grossLoss();
        double wr=p.trades()==0?0:p.wins()*100.0/p.trades();
        return new TradingMetrics(ret,bh,alpha,maxDd*100,pf,wr,sharpe(returns),p.trades(),finalEquity);
    }
    private double sharpe(List<Double> v){ if(v.size()<10)return 0; double m=0; for(double x:v)m+=x; m/=v.size(); double var=0; for(double x:v){double d=x-m;var+=d*d;} var/=Math.max(1,v.size()-1); double sd=Math.sqrt(var); return sd<1e-12?0:(m/sd)*Math.sqrt(12.0*24*365); }
}
