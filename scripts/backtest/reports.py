import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from datetime import datetime

def generate_html_report(result):
    """Generate complete HTML report with metrics and charts"""
    os.makedirs('backtest_reports', exist_ok=True)

    symbol = result['symbol']
    metrics = result['metrics']
    trades = result['trades']

    # Create interactive Plotly equity curve
    fig = go.Figure()
    
    if trades:
        # Calculate cumulative PnL for equity curve
        cumulative_pnl = 0
        equity_data = []
        dates = []
        
        for trade in trades:
            if trade.get('status') == 'closed' and 'exit_time' in trade:
                cumulative_pnl += trade.get('pnl', 0)
                equity_data.append(cumulative_pnl)
                dates.append(trade['exit_time'])
        
        if equity_data:
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity_data,
                name='Equity Curve',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative PnL:</b> %{y:.2f}%<extra></extra>'
            ))
    
    fig.update_layout(
        title=f'{symbol} Interactive Equity Curve',
        xaxis_title='Date',
        yaxis_title='Cumulative PnL (%)',
        template='plotly_white',
        hovermode='x unified'
    )
    
    # Save interactive chart as separate HTML file
    interactive_chart_file = f"backtest_reports/{symbol}_interactive_equity.html"
    fig.write_html(interactive_chart_file)

    # Create static plots for backup
    plot_files = []
    plot_files.append(create_equity_curve_plot(trades, symbol))
    plot_files.append(create_monthly_heatmap(trades, symbol))

    # Generate main HTML report
    html = f"""
    <html>
    <head>
        <title>Backtest Report - {symbol}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
            .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
            .good {{ background-color: #e6f7e6; }}
            .bad {{ background-color: #ffe6e6; }}
            .neutral {{ background-color: #e6f3ff; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .chart-container {{ margin: 20px 0; }}
            .interactive-link {{ 
                display: inline-block; 
                margin: 10px 0; 
                padding: 10px 20px; 
                background-color: #2E86AB; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
            }}
            .interactive-link:hover {{ background-color: #1a5a7a; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report - {symbol}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Key Metrics</h2>
        <div class="metrics">
            {generate_metric_cards(metrics)}
        </div>

        <h2>Equity Curve</h2>
        <div class="chart-container">
            <a href="{symbol}_interactive_equity.html" class="interactive-link" target="_blank">
                📊 View Interactive Equity Curve
            </a>
            <br>
            <img src="{plot_files[0]}" alt="Static Equity Curve">
        </div>

        <h2>Monthly Performance</h2>
        <img src="{plot_files[1]}" alt="Monthly Heatmap">

        <h2>Trade History</h2>
        {generate_trade_table(trades)}
    </body>
    </html>
    """

    report_file = f"backtest_reports/{symbol}_report.html"
    with open(report_file, 'w') as f:
        f.write(html)

    return report_file

def generate_metric_cards(metrics):
    """Generate HTML for metric cards with color coding"""
    cards = []

    # Define thresholds for good/bad metrics
    thresholds = {
        'win_rate': (60, 40),
        'profit_factor': (1.5, 1.0),
        'max_drawdown': (15, 25),
        'cagr': (20, 5),
        'sortino_ratio': (2.0, 1.0)
    }

    for name, value in metrics.items():
        if name in thresholds:
            good, bad = thresholds[name]
            if value >= good:
                cls = "good"
            elif value <= bad:
                cls = "bad"
            else:
                cls = "neutral"
        else:
            cls = "neutral"

        cards.append(f"""
        <div class="metric-card {cls}">
            <h3>{name.replace('_', ' ').title()}</h3>
            <p>{round(value, 2)}</p>
        </div>
        """)

    return '\n'.join(cards)

def generate_trade_table(trades):
    """Generate HTML table of trades"""
    if not trades:
        return "<p>No trades executed</p>"

    rows = []
    for trade in trades:
        pnl_class = "good" if trade.get('pnl', 0) > 0 else "bad"
        rows.append(f"""
        <tr>
            <td>{trade['date']}</td>
            <td>{trade['type']}</td>
            <td>{round(trade['entry_price'], 2)}</td>
            <td>{round(trade['exit_price'], 2) if trade['exit_price'] else '-'}</td>
            <td class="{pnl_class}">{round(trade.get('pnl', 0), 2)}%</td>
        </tr>
        """)

    return f"""
    <table>
        <tr>
            <th>Date</th>
            <th>Type</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>PnL %</th>
        </tr>
        {''.join(rows)}
    </table>
    """

def create_equity_curve_plot(trades, symbol):
    """Fixed version that handles trade data correctly"""
    import pandas as pd
    import plotly.graph_objects as go
    
    # Convert trades to DataFrame
    df = pd.DataFrame([t for t in trades if t['status'] == 'closed'])
    if len(df) == 0:
        return None
    
    # Calculate cumulative PnL
    df['cum_pnl'] = df['pnl'].cumsum()
    df['date'] = df['exit_time']  # Use exit_time as date
    
    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cum_pnl'],
        mode='lines',
        name='Equity Curve'
    ))
    return fig

def create_monthly_heatmap(trades, symbol):
    """Simplified monthly heatmap without date column dependency"""
    try:
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        if not closed_trades:
            fig = go.Figure()
            fig.add_annotation(text="No trades to display", showarrow=False)
            return fig

        # Create simple monthly returns bar chart instead of heatmap
        monthly_data = {}
        for trade in closed_trades:
            month_key = trade['exit_time'].strftime('%Y-%m')
            monthly_data[month_key] = monthly_data.get(month_key, 0) + trade.get('pnl', 0)

        fig = go.Figure(go.Bar(
            x=list(monthly_data.keys()),
            y=list(monthly_data.values()),
            marker_color=['green' if val > 0 else 'red' for val in monthly_data.values()]
        ))
        
        fig.update_layout(
            title=f"{symbol} Monthly Performance",
            xaxis_title="Month",
            yaxis_title="PnL (%)"
        )
        return fig
        
    except Exception as e:
        print(f"⚠️ Simplified monthly chart error: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(text="Chart error", showarrow=False)
        return fig