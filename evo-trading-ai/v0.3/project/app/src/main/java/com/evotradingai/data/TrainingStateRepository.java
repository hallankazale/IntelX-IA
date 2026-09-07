package com.evotradingai.data;

import android.content.Context;
import android.content.SharedPreferences;
import org.json.JSONObject;

public final class TrainingStateRepository {
    private final SharedPreferences prefs;
    public TrainingStateRepository(Context c){prefs=c.getSharedPreferences("evo_trading_state",Context.MODE_PRIVATE);}
    public synchronized TrainingState load(){String raw=prefs.getString("state_json",null);if(raw==null)return new TrainingState();try{return TrainingState.fromJson(new JSONObject(raw));}catch(Exception e){return new TrainingState();}}
    public synchronized void save(TrainingState s){try{prefs.edit().putString("state_json",s.toJson().toString()).commit();}catch(Exception ignored){}}
    public synchronized void reset(){prefs.edit().clear().commit();}
}
