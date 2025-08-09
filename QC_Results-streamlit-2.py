# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 09:09:28 2025

@author: johnn
"""

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
from st_aggrid import AgGrid, GridOptionsBuilder


# --- Hardcoded folder path ---
folder_path = "C:\\users\\johnn\\downloads\\QC Exports"

# --- Original Functions (unchanged) ---
def fmt_hhmm(s):
    s = s.strip()
    if len(s) == 4:
        hh,mm = s[:2], s[2:]
    elif len(s) == 3:
        hh,mm = s[0], s[1:]
    else:
        return s
    return f"{hh.zfill(2)}:{mm}"

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

def find_all_json(folder_path):
    return [(os.path.basename(p), p) for p in glob.glob(os.path.join(folder_path, "**", "*.json"), recursive=True)]

def download_spx(start, end=datetime.now()):
        spx_df = yf.download('^GSPC', auto_adjust=True, start=start)
        spx_df = spx_df.droplevel(level=1, axis=1)[['Close']]
        spx_df['Close'] = round(spx_df['Close'],2)
        
        # #Temp fix while yfinance doesn't work in Spyder
        # spx_df = pd.read_csv('C:\\Users\\johnn\\Backtesting\\gspc.csv')
        # spx_df['Date'] = pd.to_datetime(spx_df['Date'])
    
        return spx_df

def convert_time(df, columns):
    ny_tz = pytz.timezone("America/New_York")
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = df[col].dt.tz_convert(ny_tz)
    return df

def process_symbol(row):
    base, option = row.split()
    date = datetime.strptime(option[:6], "%y%m%d")
    cp = option[6]
    strike_str = option[7:]
    strike_price = float(strike_str[:-3] + '.' + strike_str[-3:-1])
    return pd.Series({'Symbol': base, 'Date': date, 'OpType': 'Call' if cp == 'C' else 'Put', 'Strike': strike_price})

def correct_exercised(df):
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

# --- Chart function ---
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

def display_summary_with_aggrid(df):
    df = df.reset_index(drop=True)  # drop index to avoid confusion

    
    # Setting 1
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_grid_options(domLayout='normal')
    # Optional: enable sorting, filtering etc.
    gb.configure_default_column(editable=False, filter=True, sortable=True, autoSize=True)
    gridOptions = gb.build()
    
    AgGrid(
        df,
        gridOptions=gridOptions,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=True,
        height=400,
        theme='light'
    )
    
    ## Setting 2
    # gb = GridOptionsBuilder.from_dataframe(df)
    # # Key settings:
    # gb.configure_default_column(
    #     autoSize=True,          # Columns fit content
    #     suppressSizeToFit=True,  # Disable full-width stretching
    #     resizable=True,          # Allow manual resizing
    # )
    
    # # Ensure grid doesn't default to full width
    # gridOptions = gb.build()
    # gridOptions["domLayout"] = "normal"  # Avoid 'autoHeight' or 'print'
    
    # AgGrid(
    #     df,
    #     gridOptions=gridOptions,
    #     height=400,
    #     theme='light',
    #     fit_columns_on_grid_load=False,  # Critical: Disable default fit
    # )

# --- Processing function ---
def process_files(folder, spx_df):
    results = {}
    orders_store = {}
    raw_data_store = {}
    files = find_all_json(folder)
    for filename, src in files:
        with open(src) as f:
            data = json.load(f)
        raw_data_store[filename] = data
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

        # orders_df.drop(['id','contingentId','brokerId','priceCurrency','tag','securityType','direction','isMarketable',
        #                 'priceAdjustmentMode','symbol.id','symbol.permtick','symbol.underlying.value','symbol.underlying.id',
        #                 'symbol.underlying.permtick','groupOrderManager.quantity','groupOrderManager.direction','canceledTime'], 
        #                axis=1, inplace=True)


        daily_summary = orders_df.groupby('Date').agg(
            value_sum=('value', 'sum'),
            tradeOpen=('tradeOpen', lambda s: s[s != pd.Timedelta(0)].mean())
        ).reset_index()
        daily_summary = daily_summary[daily_summary['Date'] < '2025-01-01']
        daily_summary['cumulative_value'] = daily_summary['value_sum'].cumsum()
        avg_all = daily_summary['value_sum'].mean()
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
            'Percent Profitable': "{:.2%}".format(percent_profitable),
            'Average Positive Day': "${:,.2f}".format(average_positive * 100.0),
            'Average Negative Day': "${:,.2f}".format(average_negative * 100.0),
            'Average Day': "${:,.4f}".format(avg_all * 100.0),
            'Average TIF': pd.to_datetime(avg_tif.total_seconds(), unit='s').strftime('%H:%M:%S'),
            'Average Credit Received': "${:,.2f}".format(avg_credit * 100.0),
            'Cumulative': "${:,.2f}".format(daily_summary['cumulative_value'].iloc[-1] * 100.0)
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
st.set_page_config(page_title="Backtest Details", layout="wide")
st.title("Backtest Details")

tab1, tab2, tab3 = st.tabs(["Summary", "Orders", "Graph"])

if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if st.button("Run"):
    with st.spinner("Running..."):
        st.session_state.run_clicked = True
        spx_df = download_spx(start='2021-10-01', end='2025-08-01')
        st.session_state.summary_df, st.session_state.orders_dict, st.session_state.raw_data_dict = process_files(folder_path, spx_df)
        st.session_state.selected_label = list(st.session_state.orders_dict.keys())[0]  # default first label


if st.session_state.run_clicked:
    # Shared dropdown for Orders + Graph
    st.session_state.selected_label = st.selectbox(
        "Select File Label", list(st.session_state.orders_dict.keys()),
        index=list(st.session_state.orders_dict.keys()).index(st.session_state.selected_label)
    )

with tab1:
    if st.session_state.run_clicked:
        # st.dataframe(st.session_state.summary_df)
        # st.write(st.session_state.summary_df.style.hide(axis="index"))
        display_summary_with_aggrid(st.session_state.summary_df)


    else:
        st.dataframe(pd.DataFrame())

with tab2:
    if st.session_state.run_clicked:
        # st.dataframe(st.session_state.orders_dict[st.session_state.selected_label])
        display_summary_with_aggrid(st.session_state.orders_dict[st.session_state.selected_label])

    else:
        st.dataframe(pd.DataFrame())

with tab3:
    if st.session_state.run_clicked:
        fig = build_equity_chart(st.session_state.raw_data_dict[st.session_state.selected_label])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No graph to display yet.")
