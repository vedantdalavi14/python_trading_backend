# Stock Portfolio Analyzer

This is an interactive web application that provides a comprehensive analysis of a stock portfolio. The application reads trade data from CSV files, calculates daily portfolio values in multiple currencies, computes the annualized return (XIRR) for each holding, and fetches the latest financial news from top-tier sources.

The dashboard is built with Streamlit and uses Plotly for interactive charts, providing a clean and user-friendly interface to track your portfolio's performance.

## Key Features

- **Automated Portfolio Analysis**: Reads and processes raw trade data from multiple CSV files.
- **Stock Split Adjustments**: Automatically handles stock splits by adjusting historical trade quantities and prices to ensure accurate calculations.
- **Multi-Currency Support**: Computes and displays portfolio values in USD, INR, and SGD.
- **Daily Performance Tracking**: Calculates the total value of the portfolio for each day.
- **Annualized Return (XIRR)**: Computes the Extended Internal Rate of Return for each individual holding.
- **Curated Financial News**: Fetches the latest, most relevant news for each holding from top financial sources like Bloomberg, Reuters, and CNBC using the NewsAPI.
- **Interactive UI**: A simple and clean web interface built with Streamlit allows for easy visualization and interaction with the data.

---

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/downloads)

### 2. Clone the Repository

```bash
git clone <https://github.com/vedantdalavi14/python_trading_backend>
cd <stock_portfolio_analyzer>
```

### 3. Create a Virtual Environment (Recommended)

It's highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r stock_portfolio_analyzer/requirements.txt
```

### 5. Set Up the Environment File

The application uses the NewsAPI to fetch news articles. You will need a free API key to use this feature.

1.  Go to [**newsapi.org**](https://newsapi.org) and get your free developer API key.
2.  In the `stock_portfolio_analyzer` directory, create a new file named `.env`.
3.  Open the `.env` file and add the following line, pasting your own API key inside the quotes:

    ```
    NEWS_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
    ```

---

## How to Run the Application

Once you have completed the setup, you can run the Streamlit application with the following command from the project's root directory:

```bash
streamlit run stock_portfolio_analyzer/streamlit_app.py
```

This will open the application in a new tab in your web browser.

---

## Project Structure

```
backend_internship/
├── stock_portfolio_analyzer/
│   ├── app.py                      # Core data processing and financial calculations.
│   ├── streamlit_app.py            # The main Streamlit UI application.
│   ├── requirements.txt            # List of all Python dependencies.
│   ├── .env                        # Stores the NewsAPI key (created by you).
│   ├── .gitignore                  # Ensures the .env file is not committed to Git.
│   ├── Stock_trading_2023.csv      # Your portfolio's trade data.
│   ├── Stock_trading_2024.csv      # Your portfolio's trade data.
│   └── Stock_trading_2025.csv      # Your portfolio's trade data.
└── README.md                       # This file.
```
