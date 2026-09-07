package com.evotradingai.core;

import java.util.Arrays;
import java.util.Random;

public final class TradingModel {
    public static final int ACTION_HOLD = 0, ACTION_BUY = 1, ACTION_SELL = 2, ACTION_COUNT = 3;
    private final double[][] q;
    private long episodes;

    public TradingModel() { q = new double[FeatureEngine.STATE_COUNT][ACTION_COUNT]; }
    public TradingModel(double[][] initialQ, long episodes) {
        this();
        if (initialQ != null) for (int i=0;i<Math.min(initialQ.length,q.length);i++) if (initialQ[i]!=null) System.arraycopy(initialQ[i],0,q[i],0,Math.min(initialQ[i].length,ACTION_COUNT));
        this.episodes = Math.max(0, episodes);
    }
    public long episodes(){ return episodes; }
    public double epsilon(){ return Math.max(0.05, 0.35 * Math.exp(-episodes / 1800.0)); }
    public int chooseAction(int state, Random r, boolean explore){ return explore && r.nextDouble()<epsilon() ? r.nextInt(ACTION_COUNT) : greedyAction(state); }
    public int greedyAction(int state){ int best=0; for(int a=1;a<ACTION_COUNT;a++) if(q[state][a]>q[state][best]) best=a; return best; }
    public void update(int state,int action,double reward,int nextState){
        double alpha=Math.max(0.025,0.11/Math.sqrt(1.0+episodes/1200.0)), gamma=0.97;
        double nextBest=q[nextState][greedyAction(nextState)];
        q[state][action]+=alpha*(reward+gamma*nextBest-q[state][action]);
    }
    public void finishEpisode(){ episodes++; }
    public double[][] snapshot(){ double[][] c=new double[q.length][ACTION_COUNT]; for(int i=0;i<q.length;i++) c[i]=Arrays.copyOf(q[i],ACTION_COUNT); return c; }
}
