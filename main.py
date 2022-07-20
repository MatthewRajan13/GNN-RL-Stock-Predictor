import math
import torch
import requests
import bs4 as bs
import pandas as pd
import yfinance as yf

# Set the start and end date
START_DATE = '2019-07-01'
END_DATE = '2021-07-12'


def main():

    # Get list of all S&P 500 tickers
    html = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)


def check_correlation(ticker1, ticker2):
    # Get the data
    ticker1_data = yf.download(ticker1, START_DATE, END_DATE)
    ticker2_data = yf.download(ticker2, START_DATE, END_DATE)

    # Calculate Deviation
    ticker1_mean = ticker1_data['Adj Close'].mean()
    ticker2_mean = ticker2_data['Adj Close'].mean()

    ticker1_tensor = torch.tensor(ticker1_data['Adj Close'], dtype=float)
    ticker1_deviation = torch.sub(ticker1_tensor, ticker1_mean)

    ticker2_tensor = torch.tensor(ticker2_data['Adj Close'], dtype=float)
    ticker2_deviation = torch.sub(ticker2_tensor, ticker2_mean)

    # Calculate covariance
    cov = float(torch.dot(ticker1_deviation, ticker2_deviation))

    # Calculate standard deviation
    ticker1_deviation_sq = float(torch.dot(ticker1_deviation, ticker1_deviation))
    ticker2_deviation_sq = float(torch.dot(ticker2_deviation, ticker2_deviation))

    sd = math.sqrt(ticker1_deviation_sq * ticker2_deviation_sq)

    # Calculate correlation
    correlation = cov / sd

    if abs(correlation) > .5:  # Arbitrary right now, ignores edge weights, need to figure out
        correlation = 1

    return correlation


if __name__ == "__main__":
    main()
