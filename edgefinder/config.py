"""Central configuration for the backtest.

Everything that defines the experiment lives here so runs are reproducible and
easy to tweak. Costs are set to realistic post-Budget-2026 levels for Indian
*delivery* equity (we deliberately avoid intraday/F&O where costs and whales
destroy retail edges).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostModel:
    """Round-trip cost assumptions for NSE delivery equity (long-only).

    All values are fractions (e.g. 0.001 == 0.1% == 10 bps). These are charged
    on traded notional, NOT on profit — exactly how STT works in India.

    Defaults are intentionally conservative so the backtest does not flatter the
    strategy. Tune to your actual broker.
    """

    stt_sell: float = 0.001          # Securities Transaction Tax, delivery sell side (0.1%)
    exchange_txn: float = 0.0000297  # NSE transaction charge (~0.00297%) per side
    sebi_charges: float = 0.000001   # SEBI turnover fee (~Rs 10 / crore) per side
    stamp_duty_buy: float = 0.00015  # stamp duty, buy side only (0.015%)
    gst: float = 0.18                # GST on (brokerage + exchange + sebi) charges
    brokerage_per_side: float = 0.0  # discount brokers: Rs 0 on delivery
    slippage_bps: float = 5.0        # modelled slippage per side, in basis points

    def _per_side_taxed(self) -> float:
        return self.exchange_txn + self.sebi_charges + self.brokerage_per_side

    def buy_cost(self) -> float:
        """All-in buy-side cost as a fraction of notional bought."""
        slip = self.slippage_bps / 10_000.0
        taxed = self._per_side_taxed()
        return self.stamp_duty_buy + taxed + self.gst * taxed + slip

    def sell_cost(self) -> float:
        """All-in sell-side cost as a fraction of notional sold (STT lives here)."""
        slip = self.slippage_bps / 10_000.0
        taxed = self._per_side_taxed()
        return self.stt_sell + taxed + self.gst * taxed + slip

    def round_trip_cost(self) -> float:
        """Approximate all-in round-trip cost as a fraction of notional."""
        return self.buy_cost() + self.sell_cost()


@dataclass
class BacktestConfig:
    # --- Universe & data ---
    universe: str = "nifty200"        # see edgefinder/universe.py
    start: str = "2015-01-01"
    end: str | None = None            # None == today
    cache_dir: str = "data_cache"
    benchmark: str = "^NSEI"          # NIFTY 50 index

    # --- Strategy ---
    rebalance: str = "ME"             # pandas offset: 'ME' month-end, 'W-FRI' weekly
    top_n: int = 25                   # number of stocks held
    min_history_days: int = 260       # require ~1y history before a stock is eligible
    weighting: str = "inv_vol"        # 'equal' or 'inv_vol' (lower-vol names get more)
    turnover_buffer: float = 0.5      # keep a holding while ranked within
                                      # (1+buffer)*top_n; reduces churn/costs
    strategy: str = "ml"              # 'ml' (learned ranking) or 'lowvol'
                                      # (rank by lowest risk; the documented
                                      # low-volatility anomaly, no training)

    # --- Regime / trend filter (point-in-time) ---
    # When the benchmark closes below its long moving average, scale exposure
    # down toward cash. A classic, well-documented way to cut drawdowns and
    # often improve risk-adjusted returns. Uses only data up to each rebalance.
    regime_filter: bool = False
    regime_ma_days: int = 200         # benchmark MA lookback (trading days)
    risk_off_exposure: float = 0.0    # 0.0 = full cash when risk-off; 0.5 = half

    # --- Model / walk-forward ---
    # We retrain the ranking model at each rebalance using only PAST data.
    train_min_periods: int = 24       # min rebalance periods before model trades
    label_horizon: int = 1            # predict next-period forward return (in periods)
    use_lightgbm: bool = True         # fall back to sklearn if unavailable

    # --- Costs & risk ---
    costs: CostModel = field(default_factory=CostModel)
    annualization: int = 12           # set to 52 for weekly rebalance

    # --- Output ---
    out_dir: str = "results"
    random_state: int = 42

    def resolved_end(self) -> str:
        import datetime as _dt

        return self.end or _dt.date.today().isoformat()
