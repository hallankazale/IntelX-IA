package com.evotradingai.core;

import java.util.List;

public final class FeatureEngine {
    public static final int BASE_STATE_COUNT = 36;
    public static final int STATE_COUNT = 72;
    public static final int WARMUP = 30;

    public PreparedMarket prepare(List<Candle> candles) {
        if (candles == null || candles.size() < 200) throw new IllegalArgumentException("Poucos candles para preparar o mercado.");
        int n = candles.size();
        double[] close = new double[n], high = new double[n], low = new double[n], volume = new double[n];
        for (int i = 0; i < n; i++) {
            Candle c = candles.get(i); close[i] = c.close; high[i] = c.high; low[i] = c.low; volume[i] = c.volume;
        }
        double[] ema9 = ema(close, 9), ema21 = ema(close, 21), rsi14 = rsi(close, 14), atr14 = atr(high, low, close, 14), volumeSma20 = sma(volume, 20);
        int[] states = new int[n];
        for (int i = 0; i < n; i++) {
            double price = Math.max(close[i], 1e-9);
            double trendDiff = (ema9[i] - ema21[i]) / price;
            int trend = trendDiff > 0.001 ? 2 : (trendDiff < -0.001 ? 0 : 1);
            int momentum = rsi14[i] > 58.0 ? 2 : (rsi14[i] < 42.0 ? 0 : 1);
            int volatility = atr14[i] / price > 0.006 ? 1 : 0;
            int highVolume = volume[i] > Math.max(volumeSma20[i], 1e-9) * 1.20 ? 1 : 0;
            states[i] = (((trend * 3 + momentum) * 2 + volatility) * 2 + highVolume);
        }
        return new PreparedMarket(candles, states, WARMUP);
    }

    private double[] ema(double[] v, int p) {
        double[] out = new double[v.length]; double a = 2.0 / (p + 1.0); out[0] = v[0];
        for (int i = 1; i < v.length; i++) out[i] = a * v[i] + (1.0 - a) * out[i - 1];
        return out;
    }

    private double[] sma(double[] v, int p) {
        double[] out = new double[v.length]; double sum = 0;
        for (int i = 0; i < v.length; i++) { sum += v[i]; if (i >= p) sum -= v[i - p]; out[i] = sum / Math.max(1, Math.min(p, i + 1)); }
        return out;
    }

    private double[] rsi(double[] close, int p) {
        double[] out = new double[close.length]; out[0] = 50; double g = 0, l = 0;
        for (int i = 1; i < close.length; i++) {
            double ch = close[i] - close[i - 1], gain = Math.max(0, ch), loss = Math.max(0, -ch);
            if (i <= p) { g += gain; l += loss; if (i == p) { g /= p; l /= p; } }
            else { g = ((g * (p - 1)) + gain) / p; l = ((l * (p - 1)) + loss) / p; }
            if (i < p) out[i] = 50; else if (l < 1e-12) out[i] = 100; else out[i] = 100 - 100 / (1 + g / l);
        }
        return out;
    }

    private double[] atr(double[] high, double[] low, double[] close, int p) {
        double[] tr = new double[close.length]; tr[0] = high[0] - low[0];
        for (int i = 1; i < close.length; i++) tr[i] = Math.max(high[i]-low[i], Math.max(Math.abs(high[i]-close[i-1]), Math.abs(low[i]-close[i-1])));
        double[] out = new double[close.length]; double avg = tr[0]; out[0] = avg;
        for (int i = 1; i < tr.length; i++) { avg = i < p ? ((avg * i) + tr[i]) / (i + 1) : ((avg * (p - 1)) + tr[i]) / p; out[i] = avg; }
        return out;
    }
}
