#!pip install yfinance
#!pip install yahoofinancials

# import pandas as pd
# import yfinance as yf
# from yahoofinancials import YahooFinancials

# infy_df = yf.download('INFY.NS', 
#                       start='2000-01-01', 
#                       end='2023-07-15',
#                       progress=False,
# )


from yahoo_fin.stock_info import get_data

infy_df = get_data("INFY.NS", start_date="01/01/2000", end_date="15/07/2023", index_as_date=True, interval="1d")
print(infy_df.head())
print(len(infy_df))