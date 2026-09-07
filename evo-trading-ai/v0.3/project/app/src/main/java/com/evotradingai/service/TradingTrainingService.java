package com.evotradingai.service;

import android.app.*;
import android.content.*;
import android.content.pm.ServiceInfo;
import android.os.*;

import com.evotradingai.MainActivity;
import com.evotradingai.R;
import com.evotradingai.core.*;
import com.evotradingai.data.*;

import org.json.JSONObject;

import java.util.*;
import java.util.concurrent.*;

public final class TradingTrainingService extends Service {
    public static final String ACTION_START = "com.evotradingai.START";
    public static final String ACTION_STOP = "com.evotradingai.STOP";
    public static final String EXTRA_HOURS = "hours";

    private static final String CHANNEL = "evo_trade_training";
    private static final int NID = 7731;
    private static final long BATCH = 5000;
    private static final long DIAG = 3600000;
    private static final long GAP = 25000;

    private TrainingStateRepository states;
    private MarketDataRepository marketRepo;
    private PreparedMarket market;
    private ScheduledExecutorService exec;
    private PowerManager.WakeLock wake;
    private boolean tickerStarted = false;

    private final FeatureEngine features = new FeatureEngine();
    private final TrainingEngine trainer = new TrainingEngine();
    private final PaperTrader paper = new PaperTrader();
    private final Random random = new Random();

    @Override public void onCreate() {
        super.onCreate();
        states = new TrainingStateRepository(this);
        marketRepo = new MarketDataRepository(this);
        createChannel();
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? null : intent.getAction();
        if (ACTION_STOP.equals(action)) {
            stopTraining();
            return START_NOT_STICKY;
        }

        TrainingState s = states.load();
        if (ACTION_START.equals(action) && !s.running) {
            long now = System.currentTimeMillis();
            int h = intent == null ? 12 : Math.max(1, Math.min(24, intent.getIntExtra(EXTRA_HOURS, 12)));
            s.running = true;
            s.phase = "PREPARANDO DADOS REAIS · v0.4";
            s.error = "";
            s.sessionStartAt = now;
            s.sessionEndAt = now + TimeUnit.HOURS.toMillis(h);
            s.lastBatchAt = 0;
            s.nextDiagnosticAt = now + DIAG;
            s.activeRuntimeMs = 0;
            s.missedRuntimeMs = 0;
            s.interruptions = 0;
            s.reports.clear();
            states.save(s);
        }

        if (!s.running) {
            stopSelf();
            return START_NOT_STICKY;
        }

        startFg("Preparando dados reais BTC/USDT para EVO v0.4...");
        acquireWake();
        ensureExecutor();
        exec.execute(this::prepare);
        return START_STICKY;
    }

    private synchronized void prepare() {
        try {
            if (market == null) {
                List<Candle> candles = loadMarket();
                market = features.prepare(candles);
                TrainingState s = states.load();
                s.candleCount = candles.size();
                s.dataStartAt = candles.get(0).openTime;
                s.dataEndAt = candles.get(candles.size() - 1).openTime;
                s.trainEnd = (int) (candles.size() * 0.70);
                s.validationEnd = (int) (candles.size() * 0.85);
                s.phase = "TREINANDO v0.4";
                states.save(s);
            }
            if (!tickerStarted) {
                tickerStarted = true;
                exec.scheduleAtFixedRate(this::tick, 0, BATCH, TimeUnit.MILLISECONDS);
            }
        } catch (Throwable e) {
            fail("Dados reais: " + msg(e));
        }
    }

    private List<Candle> loadMarket() throws Exception {
        boolean fresh = marketRepo.exists() && System.currentTimeMillis() - marketRepo.modifiedAt() < TimeUnit.HOURS.toMillis(12);
        if (fresh) return marketRepo.load();
        notifyText("Baixando ~60 dias de candles reais...");
        List<Candle> c = new MarketDataClient().downloadRecentHistory();
        marketRepo.save(c);
        return c;
    }

    private void tick() {
        try {
            TrainingState s = states.load();
            if (!s.running) {
                stopSafely();
                return;
            }
            long now = System.currentTimeMillis();
            if (now >= s.sessionEndAt) {
                diagnose(s, now, true);
                s.running = false;
                s.phase = "CONCLUÍDO v0.4";
                states.save(s);
                notifyText("EVO v0.4 concluída — teste final não visto pronto.");
                stopSafely();
                return;
            }

            if (s.lastBatchAt > 0) {
                long gap = now - s.lastBatchAt;
                if (gap > GAP) {
                    s.interruptions++;
                    s.missedRuntimeMs += Math.max(0, gap - BATCH);
                }
            }

            TradingModel model = new TradingModel(s.q, s.episodes, s.actionCounts);
            trainer.trainOneEpisode(market, model, market.warmup, s.trainEnd, random);
            s.q = model.snapshot();
            s.actionCounts = model.actionCountsSnapshot();
            s.episodes = model.episodes();
            s.activeRuntimeMs += BATCH;
            s.lastBatchAt = now;
            s.phase = "TREINANDO v0.4";

            if (now >= s.nextDiagnosticAt) {
                diagnose(s, now, false);
                s.nextDiagnosticAt = now + DIAG;
            }
            states.save(s);

            if (s.episodes % 12 == 0) {
                notifyText(String.format(Locale.getDefault(),
                        "%,d episódios • %d%% continuidade • BUY %,d / SELL %,d",
                        s.episodes, s.continuityPercent(),
                        s.actionCounts[TradingModel.ACTION_BUY],
                        s.actionCounts[TradingModel.ACTION_SELL]));
            }
        } catch (Throwable e) {
            TrainingState s = states.load();
            s.interruptions++;
            s.error = msg(e);
            states.save(s);
        }
    }

    private void diagnose(TrainingState s, long time, boolean fin) throws Exception {
        TradingMetrics v = paper.evaluate(market, s.q, s.trainEnd, s.validationEnd);
        double riskBenchmark = Portfolio.POSITION_FRACTION * v.buyHoldPct;
        double riskAlpha = v.returnPct - riskBenchmark;
        double score = riskAlpha - 0.70 * v.maxDrawdownPct + Math.min(3, v.profitFactor) * 2 + 0.10 * v.sharpe;

        if (v.trades >= 10 && score > s.bestValidationScore) {
            s.bestValidationScore = score;
            s.bestQ = copy(s.q);
        }

        JSONObject r = json(v);
        r.put("timestamp", time);
        r.put("final", fin);
        r.put("episodes", s.episodes);
        r.put("continuity", s.continuityPercent());
        r.put("interruptions", s.interruptions);
        r.put("holdSamples", s.actionCounts[TradingModel.ACTION_HOLD]);
        r.put("buySamples", s.actionCounts[TradingModel.ACTION_BUY]);
        r.put("sellSamples", s.actionCounts[TradingModel.ACTION_SELL]);

        int positiveRiskAlphaReports = 0;
        for (int i = 0; i < Math.min(3, s.reports.size()); i++) {
            JSONObject old = s.reports.get(i);
            if (old.optDouble("riskAlphaPct", -999) > 0 && old.optInt("trades", 0) >= 10) positiveRiskAlphaReports++;
        }

        boolean actionCoverage = s.actionCounts[TradingModel.ACTION_BUY] >= 100
                && s.actionCounts[TradingModel.ACTION_SELL] >= 50;

        boolean candidate = s.episodes >= 500
                && v.trades >= 20
                && v.returnPct > 0
                && riskAlpha > 0
                && v.profitFactor >= 1.10
                && v.maxDrawdownPct <= 10
                && s.continuityPercent() >= 90
                && actionCoverage
                && (s.reports.size() < 2 || positiveRiskAlphaReports >= 2);

        r.put("candidate", candidate);

        if (fin) {
            double[][] q = s.bestValidationScore > -Double.MAX_VALUE / 2 ? s.bestQ : s.q;
            TradingMetrics t = paper.evaluate(market, q, s.validationEnd, market.size());
            JSONObject tj = json(t);
            r.put("test", tj);
            double testRiskAlpha = tj.optDouble("riskAlphaPct", -999);
            r.put("laboratoryReady", candidate
                    && t.trades >= 10
                    && t.returnPct > 0
                    && testRiskAlpha > 0
                    && t.profitFactor >= 1.05
                    && t.maxDrawdownPct <= 12);
        } else {
            r.put("laboratoryReady", false);
        }

        s.reports.add(0, r);
        while (s.reports.size() > 24) s.reports.remove(s.reports.size() - 1);
    }

    private JSONObject json(TradingMetrics m) throws Exception {
        JSONObject o = new JSONObject();
        double riskBenchmark = Portfolio.POSITION_FRACTION * m.buyHoldPct;
        o.put("returnPct", m.returnPct);
        o.put("buyHoldPct", m.buyHoldPct);
        o.put("riskBenchmarkPct", riskBenchmark);
        o.put("alphaPct", m.alphaPct);
        o.put("riskAlphaPct", m.returnPct - riskBenchmark);
        o.put("maxDrawdownPct", m.maxDrawdownPct);
        o.put("profitFactor", m.profitFactor);
        o.put("winRatePct", m.winRatePct);
        o.put("sharpe", m.sharpe);
        o.put("trades", m.trades);
        return o;
    }

    private double[][] copy(double[][] q) {
        double[][] c = new double[q.length][];
        for (int i = 0; i < q.length; i++) c[i] = Arrays.copyOf(q[i], q[i].length);
        return c;
    }

    private void ensureExecutor() {
        if (exec == null || exec.isShutdown()) exec = Executors.newScheduledThreadPool(2);
    }

    private void acquireWake() {
        if (wake != null && wake.isHeld()) return;
        PowerManager p = (PowerManager) getSystemService(POWER_SERVICE);
        wake = p.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "EvoTradingAI:Training");
        wake.acquire(TimeUnit.HOURS.toMillis(13));
    }

    private void startFg(String text) {
        Notification n = notification(text);
        if (Build.VERSION.SDK_INT >= 34) startForeground(NID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        else startForeground(NID, n);
    }

    private Notification notification(String text) {
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pi = PendingIntent.getActivity(this, 1, open, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Intent stop = new Intent(this, TradingTrainingService.class).setAction(ACTION_STOP);
        PendingIntent spi = PendingIntent.getService(this, 2, stop, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this, CHANNEL)
                .setSmallIcon(R.drawable.ic_evo)
                .setContentTitle("EVO Trading AI v0.4 — treino real")
                .setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text))
                .setContentIntent(pi)
                .setOngoing(true)
                .addAction(new Notification.Action.Builder(null, "PARAR", spi).build())
                .build();
    }

    private void notifyText(String t) {
        ((NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE)).notify(NID, notification(t));
    }

    private void createChannel() {
        NotificationChannel c = new NotificationChannel(CHANNEL, "Treino EVO Trading AI", NotificationManager.IMPORTANCE_LOW);
        ((NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE)).createNotificationChannel(c);
    }

    private void stopTraining() {
        TrainingState s = states.load();
        s.running = false;
        s.phase = "PARADO";
        states.save(s);
        stopSafely();
    }

    private void fail(String m) {
        TrainingState s = states.load();
        s.running = false;
        s.phase = "ERRO";
        s.error = m;
        states.save(s);
        notifyText(m);
        stopSafely();
    }

    private String msg(Throwable e) {
        return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
    }

    private void stopSafely() {
        if (exec != null) {
            exec.shutdownNow();
            exec = null;
        }
        tickerStarted = false;
        if (wake != null && wake.isHeld()) wake.release();
        wake = null;
        stopForeground(false);
        stopSelf();
    }

    @Override public void onDestroy() {
        if (exec != null) exec.shutdownNow();
        if (wake != null && wake.isHeld()) wake.release();
        super.onDestroy();
    }

    @Override public IBinder onBind(Intent i) { return null; }
}
