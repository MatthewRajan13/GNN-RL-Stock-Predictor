import math
import torch
import requests
import bs4 as bs
import yfinance as yf

# Set the start and end date
START_DATE = '2022-05-01'
END_DATE = '2022-06-01'


def main():
    tickers = get_sp_list()
    filled_tickers = fill_sp(tickers)
    adj_matrix = get_adj_matrix(filled_tickers)
    print(adj_matrix)
    # y = yf.download('DVN', START_DATE, END_DATE)
    # print(y)


def get_sp_list():
    # Scrape Wikipedia for S&P 500 list
    html = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(html.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        if '.' in ticker:
            ticker = ticker.replace('.', '-')
        tickers.append(ticker)

    return tickers


def fill_sp(tickers):
    filled_tickers = []
    for ticker in tickers:
        data = yf.download(ticker, START_DATE, END_DATE)
        # TODO: Figure out how to try again downloading data if it fails the first time, yfinance super finicky
        filled_tickers.append(data)

    return filled_tickers


def get_adj_matrix(tickers):
    adj_matrix = torch.zeros(500, 500)
    for ticker1 in range(500):
        for ticker2 in range(500):
            if ticker1 == ticker2:
                pass
            else:
                print("Stock #: {} vs {}".format(ticker1, ticker2))
                correlation = check_correlation(tickers[ticker1], tickers[ticker2])
                if correlation:
                    adj_matrix[ticker1][ticker2] = 1

    return adj_matrix


def check_correlation(ticker1, ticker2):
    # Clean Dataframe
    ticker1 = ticker1.dropna()
    ticker2 = ticker2.dropna()

    # Calculate Deviation
    ticker1_mean = ticker1['Adj Close'].mean()
    ticker2_mean = ticker2['Adj Close'].mean()

    ticker1_tensor = torch.tensor(ticker1['Adj Close'], dtype=float)
    ticker1_deviation = torch.sub(ticker1_tensor, ticker1_mean)

    ticker2_tensor = torch.tensor(ticker2['Adj Close'], dtype=float)
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
        correlation = True
    else:
        correlation = False

    return correlation


if __name__ == "__main__":
    main()