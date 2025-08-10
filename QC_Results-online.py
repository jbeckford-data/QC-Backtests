import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import json
import glob
import os
import re
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import requests

@st.cache_data(ttl=3600, show_spinner="Fetching latest backtest data...")  # Cache for 1 hour
# Fetch files from github repo
def fetch_github_json_files():
    token = st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN"))
        if not token:
            st.error("❌ GitHub token not found in secrets or environment variables")
            return {}
    headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3.raw"  # Get raw content directly
        }
    base_url = "https://api.github.com/repos/jbeckford-data/QC-Backtests/contents/Data"
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    files = response.json()

    data_files = {}
    for file in files:
        if file['name'].endswith('.json'):
            raw_url = file['download_url']
            r = requests.get(raw_url)
            r.raise_for_status()
            data_files[file['name']] = r.json()
    return data_files

#Used for Label naming
def fmt_hhmm(s):
    s = s.strip()
    if len(s) == 4:
        hh,mm = s[:2], s[2:]
    elif len(s) == 3:
        hh,mm = s[0], s[1:]
    else:
        return s
    return f"{hh.zfill(2)}:{mm}"

#Use some logic to update name of files to something meaningful, readable.
def suggest(name):
    base = name.replace('.json','')
    y = re.search(r'-(\d{4})-(\d{4})$', base)
    years = f"{y.group(1)}–{y.group(2)}" if y else ''
    if y:
        base = base[:y.start()]
    prefix = ''
    safe_prefix = ''
    if base.startswith('2_entries-'):
        prefix = '2 entries, '
        safe_prefix = '2_entries_'
        base = base[len('2_entries-'):]
    tr = re.search(r'(\d{3,4})-(\d{3,4})', base)
    if tr:
        t1 = fmt_hhmm(tr.group(1)); t2 = fmt_hhmm(tr.group(2))
        display_time = f"{t1}–{t2}"
        safe_time = f"{t1.replace(':','-')}-{t2.replace(':','-')}"
        base = base[:tr.start()] + base[tr.end():]
    else:
        s = re.search(r'(\d{1,2})_(\d{2})', base)
        if s:
            display_time = f"{int(s.group(1))}:{s.group(2)}"
            safe_time = f"{s.group(1).zfill(2)}-{s.group(2)}"
            base = base[:s.start()] + base[s.end():]
        else:
            display_time = ''
            safe_time = ''
    p = re.search(r'(\d{1,2}(?:_\d{1,2})+)', base)
    if p:
        params = p.group(1)
        display_params = 'params ' + '/'.join(params.split('_'))
        safe_params = 'params_' + '-'.join(params.split('_'))
    else:
        display_params = ''
        safe_params = ''
    display_parts = [x for x in [prefix + display_time if display_time else prefix.rstrip(', '),
                                 display_params, years] if x]
    display = ' — '.join(display_parts)
    safe_parts = [x for x in [safe_prefix.rstrip('_'), safe_time, safe_params, (y.group(1)+'-'+y.group(2) if y else '')] if x]
    safe = '_'.join(safe_parts)
    return display, safe

# Download All Files
def find_all_json(folder_path):
    return [(os.path.basename(p), p) for p in glob.glob(os.path.join(folder_path, "**", "*.json"), recursive=True)]

# Download SPX historical
# This is used to calc exercised values.
def download_spx(start, end=datetime.now()):
        spx_df = yf.download('^GSPC', auto_adjust=True, start=start)
        spx_df = spx_df.droplevel(level=1, axis=1)[['Close']]
        spx_df['Close'] = round(spx_df['Close'],2)    
        return spx_df

# Update Times from Zulu to New York
def convert_time(df, columns):
    ny_tz = pytz.timezone("America/New_York")
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].dt.tz_convert(ny_tz)
    return df

# Splits symbol.value into symbol and characteristics
def process_symbol(row):
    base, option = row.split()
    date = datetime.strptime(option[:6], "%y%m%d")
    cp = option[6]
    strike_str = option[7:]
    strike_price = float(strike_str[:-3] + '.' + strike_str[-3:-1])
    return pd.Series({'Symbol': base, 'Date': date, 'OpType': 'Call' if cp == 'C' else 'Put', 'Strike': strike_price})

def correct_exercised(df):
    # Case where trades did not sell and exercise at end of day.
    # A few corrections: time is corrected from next day, 1am, to 4pm NY of trade date.
    # And value is calculated correctly.
    df1 = df.merge(spx_df, left_on='Date', right_on='Date')
    df1['value'] = np.where(
        (df1['value'] != 0) & (df1['OpType'] == 'Call'),
        (df1['Close'] - df1['Strike']) * df1['quantity'],
        np.where(
            (df1['value'] != 0) & (df1['OpType'] == 'Put'),
            (df1['Strike'] - df1['Close']) * df1['quantity'],
            0
        )
    )
    df1.set_index(df.index, inplace=True)
    df1['Date'] = pd.to_datetime(df1['Date'])
    ny_tz = pytz.timezone("America/New_York")
    market_close_time = df1['Date'].apply(lambda x: ny_tz.localize(
        x.replace(hour=16, minute=0, second=0, microsecond=0)
    ))
    df1['time'] = market_close_time
    df1['createdTime'] = market_close_time
    df1['lastFillTime'] = market_close_time
    df1['Date'] = df1['Date'].dt.date    
    return df1

def build_equity_chart(data):
    equity = {
        'timestamps': [x[0] for x in data['charts']['Strategy Equity']['series']['Equity']['values']],
        'open': [x[1] for x in data['charts']['Strategy Equity']['series']['Equity']['values']],
        'high': [x[2] for x in data['charts']['Strategy Equity']['series']['Equity']['values']],
        'low': [x[3] for x in data['charts']['Strategy Equity']['series']['Equity']['values']],
        'close': [x[4] for x in data['charts']['Strategy Equity']['series']['Equity']['values']]
    }
    drawdown = {
        'timestamps': [x[0] for x in data['charts']['Drawdown']['series']['Equity Drawdown']['values']],
        'values': [x[1] for x in data['charts']['Drawdown']['series']['Equity Drawdown']['values']]
    }
    equity_dates = [datetime.fromtimestamp(ts) for ts in equity['timestamps']]
    drawdown_dates = [datetime.fromtimestamp(ts) for ts in drawdown['timestamps']]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('Equity Curve', 'Drawdown')
    )
    
    # equity curve
    fig.add_trace(go.Candlestick(
        x=equity_dates,
        open=equity['open'],
        high=equity['high'],
        low=equity['low'],
        close=equity['close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False
    ), row=1, col=1)
    
    #drawdown
    fig.add_trace(go.Scatter(
        x=drawdown_dates,
        y=drawdown['values'],
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.3)',
        line=dict(color='red')
    ), row=2, col=1)

    fig.update_layout(
        title_text='Strategy Performance with Drawdown',
        height=700,
        hovermode='x unified',
        xaxis=dict(rangeslider=dict(visible=False))
    )
    fig.update_yaxes(title_text="Equity Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dash'), row=2, col=1)
    return fig

def format_orders_df(df):
    # Format 'tradeOpen' as hh:mm:ss
    df = df.copy()
    if 'tradeOpen' in df.columns:
        df['tradeOpen'] = df['tradeOpen'].apply(
        lambda td: "{:02d}:{:02d}:{:02d}".format(
            int(td.total_seconds() // 3600),
            int((td.total_seconds() % 3600) // 60),
            int(td.total_seconds() % 60)
        )
    )

    return df

# Streamlit Formatting - Update to readable if desired.
# Formatting is done regardless.  View is toggled in the app.
# Both Summary & Orders tabs.    
def format_table(df, mode="summary"):
    df = df.copy()

    if mode == "summary":

        if 'Percent Profitable' in df.columns:
            df['Percent Profitable'] = df['Percent Profitable'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")

        currency_cols = ['Avg Positive Day', 'Avg Negative Day', 'Avg Day', 'Avg Credit Received', 'Cumulative']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")

        if 'Sharpe Ratio' in df.columns:
            df['Sharpe Ratio'] = df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")

        if 'Avg TIF' in df.columns:
            # If Avg TIF is timedelta or float seconds, convert to HH:MM:SS
            def format_tif(val):
                if pd.isna(val):
                    return ""
                if isinstance(val, pd.Timedelta):
                    print('Val:' + str(val))
                    total_seconds = int(val.total_seconds())
                else:
                    print("Val3:" + str(val))
                    total_seconds = int(val)  # assuming seconds as int or float
                h = total_seconds // 3600
                m = (total_seconds % 3600) // 60
                s = total_seconds % 60
                return f"{h:02d}:{m:02d}:{s:02d}"
            df['Avg TIF'] = df['Avg TIF'].apply(format_tif)

        # Ensure integer columns are int string
        int_cols = ['Positive Days', 'Negative Days', 'Total Days']
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

        # Reorder or keep columns as is

    elif mode == "orders":
        # Format Date as MM/DD/YYYY string
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y')

        # Format tradeOpen timedelta as HH:MM:SS string
        if 'tradeOpen' in df.columns:
            def format_timedelta(td):
                if pd.isna(td):
                    return ""
                total_seconds = int(td.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            df['tradeOpen'] = df['tradeOpen'].apply(format_timedelta)

        # Format currency columns: Price, Value
        currency_cols = ['Price', 'Value']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")

        # Format Quantity as integer string
        if 'Quantity' in df.columns:
            df['Quantity'] = df['Quantity'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

        # Optionally reorder columns
        expected_cols = ['Date', 'Symbol', 'Quantity', 'Price', 'Value', 'OpType', 'Strike', 'tradeOpen']
        df = df[[col for col in expected_cols if col in df.columns]]

    else:
        raise ValueError("mode must be 'summary' or 'orders'")

    return df




# --- Processing function ---
def process_files_from_dict(json_files_dict, spx_df):
    results = {}
    orders_store = {}
    raw_data_store = {}
    for filename, data in json_files_dict.items():
        raw_data_store[filename] = data
        if 'statistics' in data and 'Sharpe Ratio' in data['statistics']:
            sharpe_ratio = float(data['statistics']['Sharpe Ratio'])
        orders_df = pd.json_normalize(data["orders"].values())
        orders_df['time'] = pd.to_datetime(orders_df['time'])
        orders_df = convert_time(orders_df, ['time', 'createdTime', 'lastFillTime', 'canceledTime'])
        orders_df[['Symbol','Date','OpType','Strike']] = orders_df['symbol.value'].apply(process_symbol)

        exercised = orders_df[orders_df['type'] == 6]
        exercised = correct_exercised(exercised)
        orders_df.loc[exercised.index] = exercised
        date_to_created = orders_df.loc[orders_df['type'] == 8].groupby('Date')['createdTime'].first()
        orders_df.loc[orders_df['type'] == 6, 'createdTime'] = orders_df.loc[orders_df['type'] == 6, 'Date'].map(date_to_created)
        orders_df['tradeOpen'] = orders_df['lastFillTime'] - orders_df['createdTime']
        orders_df = orders_df.query("status==3")
        orders_df['value'] *= -1
        orders_df['Quantity'] = orders_df['quantity']
        orders_df['Price'] = orders_df['price']
        orders_df['Value'] = orders_df['value']
        
        daily_summary = orders_df.groupby('Date').agg(
            value_sum=('value', 'sum'),
            tradeOpen=('tradeOpen', lambda s: s[s != pd.Timedelta(0)].mean())
        ).reset_index()
        daily_summary = daily_summary[daily_summary['Date'] < '2025-01-01']
        daily_summary['cumulative_value'] = daily_summary['value_sum'].cumsum()
        avg_all = daily_summary['value_sum'].mean()
        print(f'Average all: {type(avg_all)}')
        positive_days = daily_summary[daily_summary['value_sum'] > 0]
        negative_days = daily_summary[daily_summary['value_sum'] < 0]
        total_positive = len(positive_days)
        total_negative = len(negative_days)
        total_days = len(daily_summary)
        percent_profitable = total_positive / (total_positive + total_negative) if (total_positive + total_negative) > 0 else None
        average_positive = positive_days['value_sum'].mean() if total_positive > 0 else 0
        average_negative = negative_days['value_sum'].mean() if total_negative > 0 else 0
        avg_tif = daily_summary['tradeOpen'].mean()

        def credit_group(group):
            total_sum = group['value'].sum()
            positive_vals = group.loc[group['value'] > 0, 'value']
            avg_positive = positive_vals.mean() if not positive_vals.empty else 0
            return pd.Series({ 'Sum Value': total_sum, 'Average Positive Value': avg_positive})
        result = orders_df.groupby('groupOrderManager.id').apply(credit_group).reset_index()
        avg_credit = result[result['Sum Value'] > 0]['Sum Value'].mean()

        summary = pd.DataFrame([{
            'Positive Days': total_positive,
            'Negative Days': total_negative,
            'Total Days': total_days,
            'Percent Profitable': percent_profitable,
            'Avg Positive Day': average_positive * 100.0,
            'Avg Negative Day': average_negative * 100.0,
            'Avg Day': avg_all * 100.0,
            'Sharpe Ratio': round(sharpe_ratio, 3) if sharpe_ratio is not None else np.nan,
            'Avg TIF': pd.to_timedelta(avg_tif.total_seconds(), unit='s'),
            'Avg Credit Received': avg_credit * 100.0,
            'Cumulative': daily_summary['cumulative_value'].iloc[-1] * 100.0
        }])

        columns = ['Date','Symbol','Quantity','Price','Value','OpType','Strike','tradeOpen']
        orders_df = orders_df[columns]
        disp, _ = suggest(filename)
        results[disp] = summary
        orders_store[disp] = orders_df
        raw_data_store[disp] = data

    final_df = pd.concat(results.values(), keys=results.keys())    
    final_df = final_df.reset_index(level=1, drop=True).rename_axis('Label').reset_index()
    return final_df, orders_store, raw_data_store

# --- Streamlit UI ---
st.set_page_config(page_title="QC Backtest Details", layout="wide")
# Help button & text (at top)
col1, col2 = st.columns([9,1])
with col1:
    st.title("QC Backtest Details")
with col2:
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    if st.button("Help"):
        st.session_state.show_help = not st.session_state.show_help

if st.session_state.show_help:
    help_text = """
    ### QuantConnect Backtest Summary Help
    
    This app displays detailed results from numerous QuantConnect backtests, including performance metrics, order details, and equity curves.
    
    **Summary Tab:**  
    Shows aggregate statistics such as profitability, average gains/losses, sharpe ratio, cumulative returns, and average time in trade.
    
    **Orders Tab:**  
    Lists all executed orders with details on symbol, quantity, price, value, option type, strike, and trade duration.
    
    **Graph Tab:**  
    Visualizes the strategy’s equity curve and drawdown over time.
    
    Use the dropdown to select different single backtests. Toggle between formatted and numeric views for better sorting or readability.
    
    The data is sourced from QuantConnect backtest JSON exports.
    
    Source files can be found [here](https://github.com/jbeckford-data/QC-Backtests/tree/main/Data).
    """
    st.markdown(help_text)

if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

# Show Run button only if not already clicked
if not st.session_state.run_clicked:
    if st.button("Run"):
        with st.spinner("Loading ... this can take a few minutes"):
            st.session_state.run_clicked = True
            spx_df = download_spx(start='2021-10-01', end='2025-08-01')
            json_files = fetch_github_json_files()
            st.session_state.summary_df, st.session_state.orders_dict, st.session_state.raw_data_dict = process_files_from_dict(json_files, spx_df)
            st.session_state.selected_label = list(st.session_state.orders_dict.keys())[0]
            
# Show dropdown only after run
if st.session_state.run_clicked:
    st.selectbox(
        "Select Label",
        list(st.session_state.orders_dict.keys()),
        key='selected_label'
    )
    
    format_toggle = st.checkbox("Numeric View", value=False)

    st.caption("""
    Note: Due to Streamlit's formatting, values displayed with symbols 
    like \$ or \% are treated as strings for sorting purposes. 
    This means, for example, "\$140" is sorted less than "\$3." 
    To enable proper numeric sorting, set the view to numeric (unformatted). 
    Uncheck mainly for easier reading.""")
    


tab1, tab2, tab3 = st.tabs(["Summary", "Orders", "Graph"])

with tab1:
    if st.session_state.run_clicked:
        if format_toggle:
            st.dataframe(st.session_state.summary_df, use_container_width=True)
        else:
            formatted_summary = format_table(st.session_state.summary_df, mode='summary')
            st.dataframe(formatted_summary, use_container_width=True)
    else:
        st.write('Run to see summary...')
        st.dataframe(pd.DataFrame())

with tab2:
    if st.session_state.run_clicked:
        orders_df = st.session_state.orders_dict[st.session_state.selected_label]
        if format_toggle:
            st.dataframe(orders_df, use_container_width=True)            
        else:
            formatted_orders = format_table(orders_df, mode='orders')
            st.dataframe(formatted_orders, use_container_width=True)
    else:
        st.write('Run to see orders...')
        st.dataframe(pd.DataFrame())

with tab3:
    if st.session_state.run_clicked:
        fig = build_equity_chart(st.session_state.raw_data_dict[st.session_state.selected_label])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No graph to display yet.")


