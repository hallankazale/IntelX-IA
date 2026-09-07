from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class EvoFreqAiStrategy(IStrategy):
    """EVO Trading AI 1.0.

    Long-only spot strategy for FreqAI. The target is a future return net of a
    conservative fixed round-trip cost estimate. This version is dry-run only.
    """

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 240
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hard risk limits for the first live-market laboratory.
    stoploss = -0.015
    minimal_roi = {"0": 0.020}

    # 0.10% fee + 0.02% slippage per side => ~0.24% round trip.
    round_trip_cost = 0.0024
    min_predicted_net_return = 0.0030
    max_hold_hours = 4

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 3},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 2,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 144,
                "trade_limit": 10,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.05,
            },
        ]

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-atr-pct-period"] = ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        dataframe["%-ema-distance-period"] = dataframe["close"] / dataframe["%-ema-period"] - 1.0
        dataframe["%-relative-volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2.0)
        dataframe["%-bb-width-period"] = (bb["upper"] - bb["lower"]) / bb["mid"]
        dataframe["%-bb-position-period"] = (
            (dataframe["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
        )
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-log-return"] = np.log(dataframe["close"] / dataframe["close"].shift(1))
        dataframe["%-range-pct"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe["%-raw-volume"] = dataframe["volume"]
        dataframe["%-raw-price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        hour = dataframe["date"].dt.hour
        dow = dataframe["date"].dt.dayofweek
        dataframe["%-hour-sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        dataframe["%-hour-cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        dataframe["%-dow-sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        dataframe["%-dow-cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        horizon = int(self.freqai_info["feature_parameters"]["label_period_candles"])
        future_return = dataframe["close"].shift(-horizon) / dataframe["close"] - 1.0
        dataframe["&-evo_net_return"] = future_return - self.round_trip_cost
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter = (
            (dataframe["do_predict"] == 1)
            & (dataframe["&-evo_net_return"] >= self.min_predicted_net_return)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[enter, ["enter_long", "enter_tag"]] = (1, "evo_freqai_edge")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long = (
            (dataframe["do_predict"] == 1)
            & (dataframe["&-evo_net_return"] <= 0.0)
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[exit_long, ["exit_long", "exit_tag"]] = (1, "evo_edge_gone")
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        if current_time - trade.open_date_utc >= timedelta(hours=self.max_hold_hours):
            return "evo_max_hold_4h"
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        if side != "long":
            return False
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False
        last = dataframe.iloc[-1]
        # Do not chase a price more than 0.25% above the analyzed close.
        if rate > float(last["close"]) * 1.0025:
            return False
        return bool(last.get("do_predict", 0) == 1)
