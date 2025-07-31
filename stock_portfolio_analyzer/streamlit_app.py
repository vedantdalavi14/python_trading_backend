import streamlit as st
import pandas as pd
import numpy as np # Import numpy
import plotly.express as px
from app import (
    load_and_clean_data,
    get_split_history,
    adjust_for_splits,
    get_currency_history,
    get_price_history,
    calculate_transaction_prices_in_all_currencies,
    calculate_daily_portfolio_value,
    calculate_xirr_for_holdings,
    get_latest_news,
)

# Set the page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Stock Portfolio Analyzer")

st.title("📈 Stock Portfolio Performance Analyzer")
st.markdown("An interactive dashboard to track your portfolio's performance, value, and news.")

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """
    Loads all financial data and performs calculations.
    This function is cached to avoid re-running on every interaction.
    """
    files_to_process = ['Stock_trading_2023.csv', 'Stock_trading_2024.csv', 'Stock_trading_2025.csv']
    all_trades = load_and_clean_data(files_to_process)

    if all_trades.empty:
        return None

    unique_symbols = [s for s in all_trades['symbol'].unique() if s and isinstance(s, str)]

    split_data = {s: get_split_history(s) for s in unique_symbols}
    all_trades_adj = adjust_for_splits(all_trades.copy(), split_data)

    start_date = all_trades_adj['datetime'].min().date()
    end_date = pd.to_datetime('today').date()

    currency_pairs = ['USDINR=X', 'SGD=X']
    currency_data = get_currency_history(currency_pairs, start_date, end_date)
    currency_data.rename(columns={'USDINR=X': 'USD_to_INR', 'SGD=X': 'USD_to_SGD'}, inplace=True)
    stock_price_data = get_price_history(unique_symbols, start_date, end_date)

    # Filter out symbols that failed to download and create a clean trades DataFrame
    valid_symbols = [s for s in unique_symbols if s in stock_price_data.columns and not stock_price_data[s].isnull().all()]
    valid_trades_adj = all_trades_adj[all_trades_adj['symbol'].isin(valid_symbols)].copy()

    # Now, calculate transaction prices for the valid trades
    all_trades_final = calculate_transaction_prices_in_all_currencies(valid_trades_adj, currency_data.copy())

    # And proceed with calculations using only valid data
    daily_portfolio_value = calculate_daily_portfolio_value(all_trades_final, stock_price_data[valid_symbols], currency_data.copy())
    xirr_results = calculate_xirr_for_holdings(all_trades_final, stock_price_data[valid_symbols]) 
    news_data = get_latest_news(valid_symbols)

    return {
        "daily_portfolio_value": daily_portfolio_value,
        "xirr": xirr_results,
        "news": news_data,
        "holdings": all_trades_final,
        "transactions_df": all_trades_final[['datetime', 'symbol', 'adj_quantity', 'adj_t_price', 'currency', 'cashflow_usd']]
    }

# --- Main App Logic ---
try:
    data = load_data()

    if data is None:
        st.error("Failed to load or process data. Please ensure your CSV files are in the correct directory and are not empty.")
    else:
        st.header("📊 Portfolio Overview")

        # ** THE FIX IS HERE **
        # Convert XIRR results to a DataFrame and handle formatting for display
        xirr_df = pd.DataFrame(data['xirr'].items(), columns=['Symbol', 'XIRR'])
        # Ensure XIRR is a numeric column, coercing errors to NaN (Not a Number)
        xirr_df['XIRR'] = pd.to_numeric(xirr_df['XIRR'].astype(str).str.replace('%', ''), errors='coerce')

        st.dataframe(
            xirr_df,
            use_container_width=True,
            column_config={
                "Symbol": "Stock Ticker",
                "XIRR": st.column_config.NumberColumn(
                    "XIRR (%)",
                    help="Extended Internal Rate of Return. Indicates the annualized return.",
                    format="%.2f %%" # Format as percentage with 2 decimal places
                )
            },
            hide_index=True
        )

        st.header("🗓️ Daily Portfolio Value")
        currency_option = st.selectbox("Select Currency", ['USD', 'INR', 'SGD'])
        
        value_col = f'Portfolio_Value_{currency_option}'
        fig = px.line(
            data['daily_portfolio_value'],
            x=data['daily_portfolio_value'].index,
            y=value_col,
            title=f"Portfolio Value in {currency_option}",
            labels={'x': 'Date', 'y': f'Value ({currency_option})'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.header("🗞️ Holdings Details & News")
        
        # Use a filtered list of symbols for the selectbox
        holding_symbols = sorted([s for s in data['holdings']['symbol'].unique() if not s.startswith('$')])
        selected_stock = st.selectbox("Select a stock to view details", holding_symbols)

        # --- Display formatted transaction data ---
        st.subheader(f"Transaction Summary for {selected_stock}")
        stock_trades = data['transactions_df'][data['transactions_df']['symbol'] == selected_stock]
        st.dataframe(
            stock_trades,
            use_container_width=True,
            column_config={
                "datetime": "Date & Time",
                "adj_quantity": "Quantity",
                "adj_t_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "currency": "Currency",
                "cashflow_usd": st.column_config.NumberColumn("Cash Flow (USD)", format="$%.2f")
            },
            hide_index=True
        )
        
        st.subheader(f"Latest News for {selected_stock}")
        stock_news = data['news'].get(selected_stock, [])
        if stock_news:
            for news_item in stock_news:
                title = news_item.get('title', 'No Title')
                source = news_item.get('source', {}).get('name', 'Unknown Source')
                url = news_item.get('url')
                
                if url:
                    st.markdown(f"- **{title}** ({source}) - [Link]({url})")
                else:
                    st.markdown(f"- **{title}** ({source}) - *Link not available*")
        else:
            st.write("No news available for this stock.")

except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.info("There might be an issue with the data in the CSV files or with fetching data from the API. Please check the console for more details.")