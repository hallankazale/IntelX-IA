package com.evotradingai;

import android.Manifest;
import android.app.*;
import android.content.*;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.*;
import android.provider.Settings;
import android.view.*;
import android.widget.*;

import com.evotradingai.data.*;
import com.evotradingai.service.TradingTrainingService;

import org.json.JSONObject;

import java.text.*;
import java.util.*;

public final class MainActivity extends Activity {
    private final Handler handler = new Handler(Looper.getMainLooper());
    private TrainingStateRepository repo;
    private TextView status, phase, data, metrics, diag, error;
    private Button start, stop, reset;
    private final DecimalFormat d = new DecimalFormat("0.00");
    private final SimpleDateFormat dt = new SimpleDateFormat("dd/MM HH:mm", Locale.getDefault());
    private final Runnable refresh = new Runnable() {
        public void run() { render(); handler.postDelayed(this, 2500); }
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        repo = new TrainingStateRepository(this);
        setContentView(ui());
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 77);
        }
    }

    @Override protected void onResume() { super.onResume(); handler.removeCallbacks(refresh); handler.post(refresh); }
    @Override protected void onPause() { handler.removeCallbacks(refresh); super.onPause(); }

    private ScrollView ui() {
        ScrollView sc = new ScrollView(this);
        sc.setBackgroundColor(getColor(R.color.bg));
        LinearLayout root = v();
        root.setPadding(dp(16), dp(22), dp(16), dp(30));
        sc.addView(root);

        root.addView(t("REAL-MARKET LEARNING LAB · v0.4", 11, R.color.muted, Typeface.BOLD));
        root.addView(t("EVO Trading AI", 31, R.color.text, Typeface.BOLD));
        status = t("PARADO", 12, R.color.muted, Typeface.BOLD);
        root.addView(status, mt(8));

        LinearLayout hero = card();
        hero.addView(t("BTC / USDT · candles 5 min", 27, R.color.text, Typeface.BOLD));
        hero.addView(t("v0.4 corrige a política de 'sempre esperar': ações inválidas são bloqueadas e a recompensa compara a EVO ao mercado com a mesma exposição de 25%.", 12, R.color.positive, 0), mt(8));
        phase = t("Fase: parado", 13, R.color.muted, Typeface.BOLD);
        hero.addView(phase, mt(10));
        root.addView(hero, mt(18));

        LinearLayout m = card();
        metrics = t("Episódios: 0\nTempo ativo: 0,0 h\nContinuidade: 100%", 14, R.color.text, Typeface.BOLD);
        m.addView(metrics);
        root.addView(m, mt(10));

        LinearLayout dc = card();
        dc.addView(t("DADOS REAIS · ~60 dias · split 70/15/15", 15, R.color.text, Typeface.BOLD));
        data = t("Serão baixados ao iniciar.", 13, R.color.muted, 0);
        dc.addView(data, mt(10));
        root.addView(dc, mt(10));

        LinearLayout c = card();
        c.addView(t("MODO NOITE REAL · v0.4", 15, R.color.text, Typeface.BOLD));
        start = btn("INICIAR TREINO DE 12 HORAS", true);
        start.setOnClickListener(vv -> start());
        c.addView(start, mt(10));
        stop = btn("PARAR TREINO", false);
        stop.setOnClickListener(vv -> stop());
        c.addView(stop, mt(8));
        Button bat = btn("CONFIGURAÇÃO DE BATERIA", false);
        bat.setOnClickListener(vv -> battery());
        c.addView(bat, mt(8));
        root.addView(c, mt(10));

        LinearLayout dg = card();
        dg.addView(t("DIAGNÓSTICO HORÁRIO", 15, R.color.text, Typeface.BOLD));
        diag = t("Primeiro diagnóstico após 1 hora real.", 13, R.color.muted, 0);
        diag.setLineSpacing(0, 1.18f);
        dg.addView(diag, mt(10));
        root.addView(dg, mt(10));

        LinearLayout risk = card();
        risk.addView(t("RISK ENGINE", 15, R.color.text, Typeface.BOLD));
        risk.addView(t("Sem alavancagem\nExposição máx. 25%\nStop 3% · Take 6%\nMáx. 4h por posição\nFee 0,10% · slippage 0,02%\nAction masking: sem SELL vazio / BUY duplicado\nTeste final usa dados nunca vistos", 13, R.color.text, 0), mt(8));
        root.addView(risk, mt(10));

        error = t("", 12, R.color.negative, 0);
        root.addView(error, mt(8));

        reset = btn("REINICIAR MODELO", false);
        reset.setTextColor(getColor(R.color.negative));
        reset.setOnClickListener(vv -> {
            TrainingState s = repo.load();
            if (s.running) Toast.makeText(this, "Pare o treino primeiro.", Toast.LENGTH_SHORT).show();
            else { repo.reset(); render(); }
        });
        root.addView(reset, mt(10));
        root.addView(t("Importante: preços são reais, mas nenhuma ordem é enviada à corretora. Resultado de backtest/paper trading não garante lucro futuro.", 11, R.color.muted, 0), mt(12));
        return sc;
    }

    private void start() {
        Intent i = new Intent(this, TradingTrainingService.class)
                .setAction(TradingTrainingService.ACTION_START)
                .putExtra(TradingTrainingService.EXTRA_HOURS, 12);
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i); else startService(i);
        Toast.makeText(this, "Treino v0.4 de 12h iniciado.", Toast.LENGTH_LONG).show();
    }

    private void stop() {
        Intent i = new Intent(this, TradingTrainingService.class).setAction(TradingTrainingService.ACTION_STOP);
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i); else startService(i);
    }

    private void battery() {
        try { startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)); }
        catch (Exception e) {
            startActivity(new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:" + getPackageName())));
        }
    }

    private void render() {
        TrainingState s = repo.load();
        status.setText(s.running ? "● RODANDO EM SEGUNDO PLANO" : "● PARADO");
        status.setTextColor(getColor(s.running ? R.color.positive : R.color.muted));
        phase.setText("Fase: " + s.phase);
        metrics.setText(String.format(Locale.getDefault(),
                "Episódios: %,d\nTempo ativo: %.1f h\nContinuidade: %d%%\nInterrupções: %d\nAmostras HOLD %,d · BUY %,d · SELL %,d",
                s.episodes, s.activeRuntimeMs / 3600000.0, s.continuityPercent(), s.interruptions,
                s.actionCounts[0], s.actionCounts[1], s.actionCounts[2]));

        if (s.candleCount > 0) {
            data.setText(String.format(Locale.getDefault(),
                    "%,d candles reais\n%s → %s\nTreino %,d · Validação %,d · Teste %,d",
                    s.candleCount, dt.format(new Date(s.dataStartAt)), dt.format(new Date(s.dataEndAt)),
                    s.trainEnd, s.validationEnd - s.trainEnd, s.candleCount - s.validationEnd));
        } else data.setText("Os dados serão baixados automaticamente ao iniciar.");

        error.setText(s.error == null || s.error.isBlank() ? "" : "Erro: " + s.error);
        diagnostic(s);
        start.setEnabled(!s.running);
        stop.setEnabled(s.running);
        reset.setEnabled(!s.running);
    }

    private void diagnostic(TrainingState s) {
        if (s.reports.isEmpty()) {
            diag.setText("Primeiro diagnóstico após 1 hora real.\nContinuidade: " + s.continuityPercent() + "%");
            return;
        }
        JSONObject r = s.reports.get(0);
        StringBuilder b = new StringBuilder();
        b.append(r.optBoolean("final") ? "RELATÓRIO FINAL" : "RELATÓRIO HORÁRIO")
                .append(" — ").append(dt.format(new Date(r.optLong("timestamp"))))
                .append("\n\nRetorno paper: ").append(d.format(r.optDouble("returnPct"))).append("%")
                .append("\nBuy & hold 100%: ").append(d.format(r.optDouble("buyHoldPct"))).append("%")
                .append("\nBenchmark risco 25%: ").append(d.format(r.optDouble("riskBenchmarkPct"))).append("%")
                .append("\nAlpha ajustado 25%: ").append(d.format(r.optDouble("riskAlphaPct"))).append("%")
                .append("\nDrawdown: ").append(d.format(r.optDouble("maxDrawdownPct"))).append("%")
                .append("\nProfit factor: ").append(d.format(r.optDouble("profitFactor")))
                .append("\nWin rate: ").append(d.format(r.optDouble("winRatePct"))).append("%")
                .append("\nSharpe: ").append(d.format(r.optDouble("sharpe")))
                .append("\nOperações: ").append(r.optInt("trades"))
                .append("\nAmostras: HOLD ").append(r.optLong("holdSamples"))
                .append(" · BUY ").append(r.optLong("buySamples"))
                .append(" · SELL ").append(r.optLong("sellSamples"));

        if (r.optBoolean("final")) {
            JSONObject x = r.optJSONObject("test");
            if (x != null) {
                b.append("\n\nTESTE NÃO VISTO")
                        .append("\nRetorno: ").append(d.format(x.optDouble("returnPct"))).append("%")
                        .append("\nBenchmark 25%: ").append(d.format(x.optDouble("riskBenchmarkPct"))).append("%")
                        .append("\nAlpha ajustado: ").append(d.format(x.optDouble("riskAlphaPct"))).append("%")
                        .append("\nDrawdown: ").append(d.format(x.optDouble("maxDrawdownPct"))).append("%")
                        .append("\nProfit factor: ").append(d.format(x.optDouble("profitFactor")))
                        .append("\nOperações: ").append(x.optInt("trades"));
            }
            b.append("\n\nSTATUS: ").append(r.optBoolean("laboratoryReady") ? "APTA NO LABORATÓRIO" : "AINDA NÃO APTA");
        } else {
            b.append("\n\nStatus: ").append(r.optBoolean("candidate") ? "candidata a aprovação" : "em treinamento");
        }
        diag.setText(b.toString());
    }

    private LinearLayout v() { LinearLayout l = new LinearLayout(this); l.setOrientation(LinearLayout.VERTICAL); return l; }
    private LinearLayout card() { LinearLayout l = v(); l.setPadding(dp(16), dp(16), dp(16), dp(16)); l.setBackgroundColor(getColor(R.color.card)); return l; }
    private TextView t(String s, int sp, int color, int style) { TextView x = new TextView(this); x.setText(s); x.setTextSize(sp); x.setTextColor(getColor(color)); x.setTypeface(Typeface.DEFAULT, style); return x; }
    private Button btn(String s, boolean primary) { Button b = new Button(this); b.setText(s); b.setTextColor(getColor(R.color.text)); b.setTextSize(13); b.setAllCaps(false); b.setTypeface(Typeface.DEFAULT, Typeface.BOLD); b.setBackgroundColor(getColor(primary ? R.color.primary : R.color.card2)); return b; }
    private LinearLayout.LayoutParams mt(int m) { LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT); p.topMargin = dp(m); return p; }
    private int dp(int v) { return (int) (v * getResources().getDisplayMetrics().density + .5f); }
}
