import pandas as pd
from loguru import logger

def simple_backtest(df, signal_col='predicted_signal', price_col='close', output_path=None, model_name=None, run_id=None):
    df = df.copy()
    df['shifted_signal'] = df[signal_col].shift(1)
    df['return'] = df[price_col].pct_change()
    df['strategy_return'] = df['shifted_signal'] * df['return']
    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()
    logger.info(f"Backtest cumulative return: {df['cum_strategy_return'].iloc[-1]:.4f}")
    if output_path and model_name and run_id:
        df.to_csv(f"{output_path}/backtest_{model_name}_{run_id}.csv", index=False)
    return df
