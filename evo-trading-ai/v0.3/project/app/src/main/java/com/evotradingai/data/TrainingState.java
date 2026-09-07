package com.evotradingai.data;

import com.evotradingai.core.FeatureEngine;
import org.json.*;
import java.util.*;

public final class TrainingState {
    public boolean running; public String phase="PARADO",error=""; public long sessionStartAt,sessionEndAt,lastBatchAt,nextDiagnosticAt,activeRuntimeMs,missedRuntimeMs; public int interruptions,candleCount,trainEnd,validationEnd; public long dataStartAt,dataEndAt,episodes; public double bestValidationScore=-Double.MAX_VALUE;
    public double[][] q=new double[FeatureEngine.STATE_COUNT][3],bestQ=new double[FeatureEngine.STATE_COUNT][3]; public final List<JSONObject> reports=new ArrayList<>();
    public JSONObject toJson()throws Exception{
        JSONObject o=new JSONObject(); o.put("running",running);o.put("phase",phase);o.put("error",error);o.put("sessionStartAt",sessionStartAt);o.put("sessionEndAt",sessionEndAt);o.put("lastBatchAt",lastBatchAt);o.put("nextDiagnosticAt",nextDiagnosticAt);o.put("activeRuntimeMs",activeRuntimeMs);o.put("missedRuntimeMs",missedRuntimeMs);o.put("interruptions",interruptions);o.put("candleCount",candleCount);o.put("dataStartAt",dataStartAt);o.put("dataEndAt",dataEndAt);o.put("trainEnd",trainEnd);o.put("validationEnd",validationEnd);o.put("episodes",episodes);o.put("bestValidationScore",bestValidationScore);o.put("q",matrix(q));o.put("bestQ",matrix(bestQ)); JSONArray a=new JSONArray();for(JSONObject r:reports)a.put(r);o.put("reports",a);return o;
    }
    public static TrainingState fromJson(JSONObject o)throws Exception{
        TrainingState s=new TrainingState();s.running=o.optBoolean("running");s.phase=o.optString("phase","PARADO");s.error=o.optString("error","");s.sessionStartAt=o.optLong("sessionStartAt");s.sessionEndAt=o.optLong("sessionEndAt");s.lastBatchAt=o.optLong("lastBatchAt");s.nextDiagnosticAt=o.optLong("nextDiagnosticAt");s.activeRuntimeMs=o.optLong("activeRuntimeMs");s.missedRuntimeMs=o.optLong("missedRuntimeMs");s.interruptions=o.optInt("interruptions");s.candleCount=o.optInt("candleCount");s.dataStartAt=o.optLong("dataStartAt");s.dataEndAt=o.optLong("dataEndAt");s.trainEnd=o.optInt("trainEnd");s.validationEnd=o.optInt("validationEnd");s.episodes=o.optLong("episodes");s.bestValidationScore=o.optDouble("bestValidationScore",-Double.MAX_VALUE);JSONArray q=o.optJSONArray("q");if(q!=null)s.q=matrix(q);JSONArray b=o.optJSONArray("bestQ");if(b!=null)s.bestQ=matrix(b);JSONArray rs=o.optJSONArray("reports");if(rs!=null)for(int i=0;i<rs.length();i++)s.reports.add(rs.getJSONObject(i));return s;
    }
    private static JSONArray matrix(double[][] m)throws Exception{JSONArray a=new JSONArray();for(double[] row:m){JSONArray r=new JSONArray();for(double v:row)r.put(v);a.put(r);}return a;}
    private static double[][] matrix(JSONArray a)throws Exception{double[][] m=new double[FeatureEngine.STATE_COUNT][3];for(int i=0;i<Math.min(a.length(),m.length);i++){JSONArray r=a.getJSONArray(i);for(int j=0;j<Math.min(3,r.length());j++)m[i][j]=r.getDouble(j);}return m;}
    public int continuityPercent(){long t=activeRuntimeMs+missedRuntimeMs;return t<=0?100:(int)Math.max(0,Math.min(100,Math.round(activeRuntimeMs*100.0/t)));}
}
