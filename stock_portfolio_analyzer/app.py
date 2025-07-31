import os
import pandas as pd
import yfinance as yf
from scipy.optimize import newton
import json
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

def load_and_clean_data(files):
    """
    Loads and merges multiple CSV files into a single DataFrame.
    """
    all_data = []
    expected_columns = [
        "trade_type", "header", "data_discriminator", "asset_category", "currency", 
        "symbol", "datetime", "quantity", "t_price", "c_price", "proceeds", 
        "comm_fee", "basis", "realized_pl", "mtm_pl", "code"
    ]
    
    for file in files:
        df = pd.read_csv(
            file,
            header=None,
            skiprows=1,
            encoding="utf-8",
            on_bad_lines='skip'
        )
        df = df.iloc[:, :len(expected_columns)]
        all_data.append(df)
        
    master_df = pd.concat(all_data, ignore_index=True)
    master_df.dropna(how='all', inplace=True)
    master_df.columns = expected_columns
    
    master_df['datetime'] = pd.to_datetime(master_df['datetime'], errors='coerce').dt.tz_localize('UTC')
    numeric_cols = ['quantity', 't_price', 'c_price', 'proceeds', 'comm_fee', 'basis', 'realized_pl', 'mtm_pl']
    for col in numeric_cols:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col].astype(str).str.replace(',', ''), errors='coerce')

    # Filter out rows that are not 'Order' trades
    master_df = master_df[master_df['data_discriminator'] == 'Order'].copy()
    master_df.dropna(subset=['datetime', 'symbol', 'quantity', 't_price'], inplace=True)
    
    return master_df

def get_split_history(symbol):
    stock = yf.Ticker(symbol)
    return stock.splits

def adjust_for_splits(trades_df, split_data):
    """
    Adjusts historical trade data for stock splits.
    """
    trades_df['adj_quantity'] = trades_df['quantity']
    trades_df['adj_t_price'] = trades_df['t_price']

    for symbol, splits in split_data.items():
        if not splits.empty:
            splits = splits.sort_index()
            for split_date, split_ratio in splits.items():
                
                # FIX: Check if the timestamp is naive before localizing, otherwise convert.
                if split_date.tzinfo is None:
                    split_date_aware = split_date.tz_localize('UTC')
                else:
                    split_date_aware = split_date.tz_convert('UTC')

                # Find all trades for the symbol that occurred before the split
                trades_to_adjust = (trades_df['symbol'] == symbol) & (trades_df['datetime'] < split_date_aware)
                
                # Adjust quantity and price
                trades_df.loc[trades_to_adjust, 'adj_quantity'] *= split_ratio
                trades_df.loc[trades_to_adjust, 'adj_t_price'] /= split_ratio
                
    # Recalculate proceeds based on adjusted values
    trades_df['adjusted_cashflow'] = trades_df['adj_quantity'] * trades_df['adj_t_price']
    
    return trades_df

def get_currency_history(symbols, start_date, end_date):
    return yf.download(symbols, start=start_date, end=end_date)['Close']

def get_price_history(symbols, start_date, end_date):
    price_data = yf.download(symbols, start=start_date, end=end_date)['Close']
    # Forward-fill to ensure non-trading days have price data
    return price_data.ffill()

def calculate_transaction_prices_in_all_currencies(trades_df, currency_data):
    """
    Calculates the transaction price in USD, INR, and SGD.
    """
    # Forward-fill missing currency data
    currency_data.ffill(inplace=True)

    # Merge currency data with trades data
    trades_df['trade_date'] = trades_df['datetime'].dt.date
    currency_data.index = currency_data.index.date
    
    merged_df = pd.merge(trades_df, currency_data, left_on='trade_date', right_index=True, how='left')
    merged_df.ffill(inplace=True) # Handle any remaining NaNs

    merged_df['cashflow_usd'] = merged_df['adjusted_cashflow']
    merged_df['cashflow_inr'] = merged_df['adjusted_cashflow'] * merged_df['USD_to_INR']
    merged_df['cashflow_sgd'] = merged_df['adjusted_cashflow'] * merged_df['USD_to_SGD']
    
    return merged_df

def calculate_daily_portfolio_value(trades_df, price_history, currency_history):
    # Ensure all dataframes are consistently timezone-aware (UTC)
    trades_df['datetime'] = pd.to_datetime(trades_df['datetime']).dt.tz_convert('UTC')
    
    if price_history.index.tz is None:
        price_history.index = price_history.index.tz_localize('UTC')
    else:
        price_history.index = price_history.index.tz_convert('UTC')

    if currency_history.index.tz is None:
        currency_history.index = currency_history.index.tz_localize('UTC')
    else:
        currency_history.index = currency_history.index.tz_convert('UTC')
        
    start_date = trades_df['datetime'].min()
    end_date = price_history.index.max()
    all_dates = pd.date_range(start=start_date, end=end_date, tz='UTC')
    
    daily_holdings = pd.DataFrame(index=all_dates)
    
    valid_symbols = [s for s in trades_df['symbol'].unique() if s in price_history.columns]

    for symbol in valid_symbols:
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        daily_trades = symbol_trades.groupby(symbol_trades['datetime'].dt.date)['adj_quantity'].sum()
        daily_trades.index = pd.to_datetime(daily_trades.index).tz_localize('UTC')
        
        daily_quantity = daily_trades.cumsum().reindex(all_dates, method='ffill').fillna(0)
        daily_holdings[symbol] = daily_quantity

    # ** THE FIX IS HERE **
    # Reindex price_history to match all_dates and forward-fill missing values
    aligned_prices = price_history.reindex(all_dates, method='ffill')
            
    # Calculate daily value in USD
    daily_value_usd = daily_holdings.mul(aligned_prices[valid_symbols]).sum(axis=1)
    
    # Create the final portfolio value DataFrame
    portfolio_value_df = pd.DataFrame(index=daily_value_usd.index)
    portfolio_value_df['Portfolio_Value_USD'] = daily_value_usd
    
    # ** THE FIX IS HERE **
    # Reindex currency_history to match all_dates and forward-fill missing values
    aligned_currencies = currency_history.reindex(all_dates, method='ffill')

    # Join with aligned currency data for conversion
    portfolio_value_df = portfolio_value_df.join(aligned_currencies)
    
    # Calculate values in other currencies
    portfolio_value_df['Portfolio_Value_INR'] = portfolio_value_df['Portfolio_Value_USD'] * portfolio_value_df['USD_to_INR']
    portfolio_value_df['Portfolio_Value_SGD'] = portfolio_value_df['Portfolio_Value_USD'] * portfolio_value_df['USD_to_SGD']
    
    first_trade_date = trades_df['datetime'].min().floor('D')
    return portfolio_value_df[portfolio_value_df.index >= first_trade_date]


def xirr(values, dates):
    """
    Custom XIRR function using scipy's Newton-Raphson method.
    """
    if len(values) != len(dates):
        raise ValueError("values and dates must have the same length")

    # Sort by date
    unique_dates = sorted(list(set(zip(dates, values))))
    
    # Aggregate cashflows on the same day
    flows_by_date = {}
    for d, v in unique_dates:
        flows_by_date[d] = flows_by_date.get(d, 0) + v
        
    chron_order = sorted(flows_by_date.items())
    
    dates, values = zip(*chron_order)
    
    t0 = dates[0]
    
    def xnpv(rate):
        return sum([cf / (1 + rate)**((t - t0).days / 365.0) for t, cf in zip(dates, values)])
    
    try:
        return newton(xnpv, 0.1) # Start with a guess of 10%
    except (RuntimeError, OverflowError):
        # If Newton's method fails, it might be because of no solution or multiple solutions
        return None


def calculate_xirr_for_holdings(trades_df, price_history):
    xirr_results = {}
    today = pd.to_datetime('today').tz_localize('UTC')
    
    for symbol in trades_df['symbol'].unique():
        if symbol and isinstance(symbol, str):
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            
            dates = list(symbol_trades['datetime'])
            values = list(-symbol_trades['adjusted_cashflow'])
            
            current_quantity = symbol_trades['adj_quantity'].sum()
            if current_quantity > 0:
                # Handle both Series and DataFrame for price_history
                if isinstance(price_history, pd.DataFrame):
                    last_price = price_history[symbol].iloc[-1]
                else: # It's a Series
                    last_price = price_history.iloc[-1]
                
                current_value = current_quantity * last_price
                dates.append(today)
                values.append(current_value)
            
            rate = xirr(values, dates)
            xirr_results[symbol] = rate * 100 if rate is not None else "N/A"
            
    return xirr_results

def get_latest_news(symbols):
    """
    Fetches real, latest news for holdings using NewsAPI.
    """
    load_dotenv() # Load environment variables from .env file
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        return {s: [{"title": "NewsAPI key is not configured.", "source": {"name": "System"}}] for s in symbols}

    newsapi = NewsApiClient(api_key=api_key)
    news_results = {}
    
    valid_symbols = [s for s in symbols if s and isinstance(s, str) and not s.startswith('$')]

    for symbol in valid_symbols:
        try:
            # Fetch news from NewsAPI
            all_articles = newsapi.get_everything(q=symbol, language='en', sort_by='relevancy', page_size=5)
            
            if all_articles['articles']:
                news_results[symbol] = all_articles['articles']
            else:
                news_results[symbol] = [{"title": f"No recent news found for {symbol}", "source": {"name": "System"}}]

        except Exception as e:
            print(f"Could not fetch news for {symbol}: {e}")
            news_results[symbol] = [{"title": f"Error fetching news for {symbol}", "source": {"name": "System"}}]
            
    return news_results

