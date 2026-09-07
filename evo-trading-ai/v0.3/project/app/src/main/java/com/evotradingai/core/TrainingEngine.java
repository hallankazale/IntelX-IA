package com.evotradingai.core;

import java.util.Random;

public final class TrainingEngine {
    public static final int EPISODE_BARS=512;
    public double trainOneEpisode(PreparedMarket market,TradingModel model,int trainStart,int trainEndExclusive,Random random){
        int minStart=Math.max(market.warmup,trainStart), latestStart=trainEndExclusive-EPISODE_BARS-2;
        if(latestStart<=minStart) throw new IllegalArgumentException("Faixa de treino insuficiente.");
        int start=minStart+random.nextInt(latestStart-minStart+1), end=Math.min(trainEndExclusive-1,start+EPISODE_BARS);
        Portfolio p=new Portfolio(1000); double peak=1000,totalReward=0;
        for(int i=start;i<end;i++){
            Candle current=market.candles.get(i), next=market.candles.get(i+1);
            double before=Math.max(1e-9,p.equity(current.close)); int state=market.stateAt(i,p.inPosition()); int action=model.chooseAction(state,random,true);
            p.applyAction(action,current.close); p.applyRiskRules(next);
            double after=Math.max(1e-9,p.equity(next.close)); peak=Math.max(peak,after); double drawdown=Math.max(0,(peak-after)/peak);
            double reward=100*Math.log(after/before)-drawdown*0.10; int nextState=market.stateAt(i+1,p.inPosition());
            model.update(state,action,reward,nextState); totalReward+=reward;
        }
        p.closeAt(market.candles.get(end).close); model.finishEpisode(); return totalReward;
    }
}
