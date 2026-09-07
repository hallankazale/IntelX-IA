package com.evotradingai.core;

import java.util.List;

public final class PreparedMarket {
    public final List<Candle> candles;
    public final int[] baseStates;
    public final int warmup;

    public PreparedMarket(List<Candle> candles, int[] baseStates, int warmup) {
        this.candles = candles;
        this.baseStates = baseStates;
        this.warmup = warmup;
    }

    public int stateAt(int index, boolean inPosition) {
        return baseStates[index] * 2 + (inPosition ? 1 : 0);
    }

    public int size() {
        return candles.size();
    }
}
