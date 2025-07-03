import numpy as np
import pandas as pd

def get_empty_metrics():
    """Return empty metrics structure when no trades occur"""
    return {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'total_pnl': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'profit_factor': 0,
        'gross_profit': 0,
        'gross_loss': 0
    }

def calculate_all_metrics(trades):
    assert all('symbol' in t for t in trades), "Missing symbol in trade records!"
    trades = [t for t in trades if t.get('status') == 'closed']
    if not trades:
        return {'error': 'No closed trades'}
    
    # Convert % to decimals
    returns = [t['pnl']/100 for t in trades]  
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    
    # Handle edge cases
    if not wins or not losses:
        return {
            'win_rate': 0,
            'sharpe_ratio': -5 if not wins else 0,
            'max_drawdown': abs(min(returns)) if returns else 0
        }
    
    return {
        'win_rate': len(wins)/len(trades),
        'sharpe_ratio': np.mean(returns)/np.std(returns) * np.sqrt(365),
        'max_drawdown': abs(min(returns)),
        'profit_factor': sum(wins)/abs(sum(losses))
    }

def calculate_cagr(trades):
    """Calculate Compound Annual Growth Rate from trades"""
    if not trades:
        return 0
    
    # Convert trades to DataFrame for date calculations
    trades_df = pd.DataFrame(trades)
    if len(trades_df) < 2:
        return 0
    
    start_date = pd.to_datetime(trades_df['date'].min())
    end_date = pd.to_datetime(trades_df['date'].max())
    days = (end_date - start_date).days
    
    if days == 0:
        return 0
    
    start_equity = 10000  # Starting with $10k for simulation
    total_pnl = sum(trade['pnl'] for trade in trades)
    end_equity = start_equity + (total_pnl * start_equity / 100)  # Convert % to dollar amount
    
    return ((end_equity / start_equity) ** (365/days) - 1) * 100

def calculate_sortino(trades):
    """Calculate Sortino Ratio from trades"""
    if not trades:
        return 0
    
    returns = [trade['pnl'] / 100 for trade in trades]  # Convert from % to decimal
    downside_returns = [r for r in returns if r < 0]
    
    if not downside_returns:
        return 0
    
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0
    
    return np.mean(returns) / downside_std

def calculate_ulcer(trades):
    """Calculate Ulcer Index from trades"""
    if not trades:
        return 0
    
    # Calculate running equity curve
    equity = 10000
    equity_values = [equity]
    
    for trade in trades:
        equity += (trade['pnl'] / 100) * equity_values[0]  # Apply percentage return
        equity_values.append(equity)
    
    equity_series = pd.Series(equity_values)
    max_equity = equity_series.cummax()
    drawdown = ((max_equity - equity_series) / max_equity) * 100
    
    return np.sqrt((drawdown ** 2).mean())

def calculate_expectancy(trades):
    """Calculate trading expectancy from trades"""
    if not trades:
        return 0
    
    wins = [trade for trade in trades if trade['pnl'] > 0]
    losses = [trade for trade in trades if trade['pnl'] <= 0]
    
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([trade['pnl'] for trade in wins]) if wins else 0
    avg_loss = np.mean([trade['pnl'] for trade in losses]) if losses else 0
    
    return (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

def empty_metrics():
    """Return empty metrics when no trades"""
    return {k: 0 for k in [
        'total_trades', 'win_rate', 'profit_factor', 'max_drawdown',
        'gross_profit', 'gross_loss', 'cagr', 'sortino_ratio', 'ulcer_index',
        'expectancy', 'avg_win', 'avg_loss'
    ]}


def _calculate_sharpe(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0
    excess_returns = np.array(returns) - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)

def _calculate_max_drawdown(returns):
    """Calculate max drawdown from a list of returns (in %)"""
    if not returns:
        return 0.0
    
    cumulative = 100 * (1 + np.array(returns)).cumprod()  # Start with $100
    peak = cumulative.max()
    trough = cumulative[cumulative.argmax():].min()
    return (trough - peak) / peak * 100  # Return as percentage
