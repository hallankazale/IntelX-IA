package com.evotradingai.core;

public final class Candle {
    public final long openTime;
    public final double open;
    public final double high;
    public final double low;
    public final double close;
    public final double volume;

    public Candle(long openTime, double open, double high, double low, double close, double volume) {
        this.openTime = openTime;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.volume = volume;
    }
}
