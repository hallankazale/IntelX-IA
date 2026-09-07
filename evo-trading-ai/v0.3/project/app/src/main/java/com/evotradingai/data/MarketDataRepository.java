package com.evotradingai.data;

import android.content.Context;
import com.evotradingai.core.Candle;
import java.io.*;
import java.util.*;

public final class MarketDataRepository {
    private final File file;
    public MarketDataRepository(Context c){file=new File(c.getFilesDir(),"btcusdt_5m_market.bin");}
    public boolean exists(){return file.exists()&&file.length()>1000;}
    public long modifiedAt(){return file.exists()?file.lastModified():0;}
    public void save(List<Candle> candles)throws Exception{
        File temp=new File(file.getParentFile(),file.getName()+".tmp");
        try(DataOutputStream out=new DataOutputStream(new BufferedOutputStream(new FileOutputStream(temp)))){
            out.writeInt(candles.size()); for(Candle c:candles){out.writeLong(c.openTime);out.writeDouble(c.open);out.writeDouble(c.high);out.writeDouble(c.low);out.writeDouble(c.close);out.writeDouble(c.volume);}
        }
        if(file.exists()&&!file.delete())throw new IllegalStateException("Falha ao substituir dados."); if(!temp.renameTo(file))throw new IllegalStateException("Falha ao salvar dados.");
    }
    public List<Candle> load()throws Exception{
        List<Candle> list=new ArrayList<>(); try(DataInputStream in=new DataInputStream(new BufferedInputStream(new FileInputStream(file)))){
            int count=in.readInt(); if(count<1||count>100000)throw new IllegalStateException("Arquivo inválido.");
            for(int i=0;i<count;i++)list.add(new Candle(in.readLong(),in.readDouble(),in.readDouble(),in.readDouble(),in.readDouble(),in.readDouble()));
        } return list;
    }
}
