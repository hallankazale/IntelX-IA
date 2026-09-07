package com.evotradingai.data;

import com.evotradingai.core.Candle;
import org.json.JSONArray;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public final class MarketDataClient {
    private static final long INTERVAL_MS=300000L,HISTORY_MS=60L*24*60*60*1000;
    private static final String[] BASES={"https://data-api.binance.vision","https://api.binance.com","https://api1.binance.com"};
    public List<Candle> downloadRecentHistory() throws Exception {
        long end=System.currentTimeMillis(),cursor=end-HISTORY_MS; List<Candle> all=new ArrayList<>();
        while(cursor<end){
            JSONArray batch=fetchBatch(cursor); if(batch.length()==0)break; long last=cursor;
            for(int i=0;i<batch.length();i++){
                JSONArray r=batch.getJSONArray(i); long t=r.getLong(0); if(!all.isEmpty()&&t<=all.get(all.size()-1).openTime)continue;
                all.add(new Candle(t,r.getDouble(1),r.getDouble(2),r.getDouble(3),r.getDouble(4),r.getDouble(5))); last=t;
            }
            if(last<=cursor)break; cursor=last+INTERVAL_MS; if(batch.length()<1000)break; Thread.sleep(80);
        }
        if(all.size()<5000)throw new IllegalStateException("A fonte retornou poucos candles: "+all.size());
        return all;
    }
    private JSONArray fetchBatch(long start) throws Exception {
        Exception last=null; for(String base:BASES){ try{return request(base+"/api/v3/klines?symbol=BTCUSDT&interval=5m&startTime="+start+"&limit=1000");}catch(Exception e){last=e;} }
        throw last==null?new IllegalStateException("Falha de rede."):last;
    }
    private JSONArray request(String endpoint) throws Exception {
        HttpURLConnection c=(HttpURLConnection)new URL(endpoint).openConnection(); c.setRequestMethod("GET"); c.setConnectTimeout(12000); c.setReadTimeout(15000); c.setRequestProperty("Accept","application/json"); c.setRequestProperty("User-Agent","EVO-Trading-AI/0.3");
        int status=c.getResponseCode(); if(status!=200)throw new IllegalStateException("HTTP "+status); StringBuilder b=new StringBuilder();
        try(BufferedReader r=new BufferedReader(new InputStreamReader(c.getInputStream(), StandardCharsets.UTF_8))){String line;while((line=r.readLine())!=null)b.append(line);}finally{c.disconnect();}
        return new JSONArray(b.toString());
    }
}
