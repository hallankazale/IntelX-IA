package com.evotradingai.data;

import android.content.Context;
import android.content.SharedPreferences;
import org.json.JSONObject;

public final class TrainingStateRepository {
    private final SharedPreferences prefs;

    public TrainingStateRepository(Context c) {
        prefs = c.getSharedPreferences("evo_trading_state", Context.MODE_PRIVATE);
    }

    public synchronized TrainingState load() {
        String raw = prefs.getString("state_json", null);
        if (raw == null) return new TrainingState();
        try {
            TrainingState state = TrainingState.fromJson(new JSONObject(raw));
            if (state.modelVersion != TrainingState.CURRENT_MODEL_VERSION) {
                TrainingState fresh = new TrainingState();
                save(fresh);
                return fresh;
            }
            return state;
        } catch (Exception e) {
            return new TrainingState();
        }
    }

    public synchronized void save(TrainingState s) {
        try {
            prefs.edit().putString("state_json", s.toJson().toString()).commit();
        } catch (Exception ignored) { }
    }

    public synchronized void reset() {
        prefs.edit().clear().commit();
    }
}
