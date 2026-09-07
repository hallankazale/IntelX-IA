package com.evotradingai.core;

import java.util.Random;

public final class TrainingEngine {
    public static final int EPISODE_BARS = 512;

    public double trainOneEpisode(PreparedMarket market, TradingModel model, int trainStart, int trainEndExclusive, Random random) {
        int minStart = Math.max(market.warmup, trainStart);
        int latestStart = trainEndExclusive - EPISODE_BARS - 2;
        if (latestStart <= minStart) throw new IllegalArgumentException("Faixa de treino insuficiente.");

        int start = minStart + random.nextInt(latestStart - minStart + 1);
        int end = Math.min(trainEndExclusive - 1, start + EPISODE_BARS);
        Portfolio p = new Portfolio(1000);
        double peak = 1000;
        double totalReward = 0;

        for (int i = start; i < end; i++) {
            Candle current = market.candles.get(i);
            Candle next = market.candles.get(i + 1);
            boolean inPositionBefore = p.inPosition();
            double before = Math.max(1e-9, p.equity(current.close));
            int state = market.stateAt(i, inPositionBefore);
            int action = model.chooseAction(state, inPositionBefore, random, true);

            p.applyAction(action, current.close);
            p.applyRiskRules(next);

            double after = Math.max(1e-9, p.equity(next.close));
            peak = Math.max(peak, after);
            double drawdown = Math.max(0, (peak - after) / peak);

            double agentLogReturn = Math.log(after / before);
            double marketLogReturn = Math.log(Math.max(1e-9, next.close) / Math.max(1e-9, current.close));
            double riskMatchedBenchmark = Portfolio.POSITION_FRACTION * marketLogReturn;

            // v0.4: reward relativo ao mercado com a MESMA exposição máxima de 25%.
            // Assim, ficar fora quando BTC sobe tem custo de oportunidade; ficar fora em queda pode ser bom.
            double reward = 100.0 * (agentLogReturn - riskMatchedBenchmark) - drawdown * 0.12;

            // Penalidade mínima apenas para quebrar inatividade crônica em movimentos positivos claros.
            if (!inPositionBefore && action == TradingModel.ACTION_HOLD && marketLogReturn > 0.0015) {
                reward -= 0.02;
            }

            int nextState = market.stateAt(i + 1, p.inPosition());
            model.update(state, action, reward, nextState);
            totalReward += reward;
        }

        p.closeAt(market.candles.get(end).close);
        model.finishEpisode();
        return totalReward;
    }
}
