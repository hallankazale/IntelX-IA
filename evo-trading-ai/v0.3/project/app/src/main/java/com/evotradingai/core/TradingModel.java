package com.evotradingai.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public final class TradingModel {
    public static final int ACTION_HOLD = 0, ACTION_BUY = 1, ACTION_SELL = 2, ACTION_COUNT = 3;
    private final double[][] q;
    private final long[] actionCounts;
    private long episodes;

    public TradingModel() {
        q = new double[FeatureEngine.STATE_COUNT][ACTION_COUNT];
        actionCounts = new long[ACTION_COUNT];
    }

    public TradingModel(double[][] initialQ, long episodes) {
        this(initialQ, episodes, null);
    }

    public TradingModel(double[][] initialQ, long episodes, long[] initialCounts) {
        this();
        if (initialQ != null) {
            for (int i = 0; i < Math.min(initialQ.length, q.length); i++) {
                if (initialQ[i] != null) {
                    System.arraycopy(initialQ[i], 0, q[i], 0, Math.min(initialQ[i].length, ACTION_COUNT));
                }
            }
        }
        if (initialCounts != null) {
            System.arraycopy(initialCounts, 0, actionCounts, 0, Math.min(initialCounts.length, ACTION_COUNT));
        }
        this.episodes = Math.max(0, episodes);
    }

    public long episodes() { return episodes; }

    public double epsilon() {
        return Math.max(0.08, 0.45 * Math.exp(-episodes / 2500.0));
    }

    public int chooseAction(int state, boolean inPosition, Random r, boolean explore) {
        int[] valid = validActions(inPosition);
        int action;
        if (explore && (episodes < 50 || r.nextDouble() < epsilon())) {
            action = valid[r.nextInt(valid.length)];
        } else {
            action = greedyAction(state, inPosition, r);
        }
        actionCounts[action]++;
        return action;
    }

    public int greedyAction(int state, boolean inPosition) {
        return greedyAction(state, inPosition, null);
    }

    private int greedyAction(int state, boolean inPosition, Random randomForTie) {
        int[] valid = validActions(inPosition);
        double bestValue = -Double.MAX_VALUE;
        List<Integer> ties = new ArrayList<>(2);
        for (int action : valid) {
            double value = q[state][action];
            if (value > bestValue + 1e-12) {
                bestValue = value;
                ties.clear();
                ties.add(action);
            } else if (Math.abs(value - bestValue) <= 1e-12) {
                ties.add(action);
            }
        }
        if (ties.size() == 1) return ties.get(0);
        if (randomForTie != null) return ties.get(randomForTie.nextInt(ties.size()));
        int hash = state * 31 + (inPosition ? 17 : 7);
        return ties.get(Math.floorMod(hash, ties.size()));
    }

    public void update(int state, int action, double reward, int nextState) {
        double alpha = Math.max(0.025, 0.11 / Math.sqrt(1.0 + episodes / 1200.0));
        double gamma = 0.97;
        boolean nextInPosition = (nextState & 1) == 1;
        double nextBest = q[nextState][greedyAction(nextState, nextInPosition)];
        q[state][action] += alpha * (reward + gamma * nextBest - q[state][action]);
    }

    public void finishEpisode() { episodes++; }

    public double[][] snapshot() {
        double[][] copy = new double[q.length][ACTION_COUNT];
        for (int i = 0; i < q.length; i++) copy[i] = Arrays.copyOf(q[i], ACTION_COUNT);
        return copy;
    }

    public long[] actionCountsSnapshot() {
        return Arrays.copyOf(actionCounts, ACTION_COUNT);
    }

    private int[] validActions(boolean inPosition) {
        return inPosition
                ? new int[]{ACTION_HOLD, ACTION_SELL}
                : new int[]{ACTION_HOLD, ACTION_BUY};
    }
}
