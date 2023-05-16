import ta
import torch
import warnings
import requests
import bs4 as bs
import numpy as np
import pandas as pd
import yfinance as yf
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import plotly.graph_objects as go
import torch.nn.functional as F
from gnn_autoencoder import GNNAutoEncoder

warnings.filterwarnings('ignore')

# Set the start and end date
START_DATE = '2022-05-01'


def main():
    tickers = ['BAL', 'BNO', 'CANE', 'CORN', 'COW', 'CPER', 'IAU', 'JO', 'SLV',
               'SOYB', 'UGA', 'UNG', 'USO', 'WEAT']
    filled_tickers, price= fill_sp(tickers)

    # Graph attributes
    edge_index, edge_weights, correlation_matrix, node_features, node_labels = get_graph_attributes(filled_tickers, price, tickers)

    data = Data(x=node_features, y=node_labels, edge_index=edge_index, edge_attr=edge_weights, num_nodes=len(tickers))

    plot_graph(data, tickers)

    # Define the dimensions for the autoencoder
    input_dim = data.x.size(2)
    hidden_dim = 64
    encoding_dim = 16

    # Create the autoencoder model
    model = GNNAutoEncoder(input_dim, hidden_dim, encoding_dim)

    # move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)

    # Set the model to training mode
    model.train()

    # Initialize an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    num_epochs = 100

    encoded_data = train_gnn(model, data, optimizer, num_epochs)


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
    data = pd.DataFrame()
    price = pd.DataFrame()
    for ticker in tickers:
        ticker_data = yf.download(ticker, start=START_DATE).fillna(0)
        price[ticker] = ticker_data['Adj Close']
        ticker_data['SMA'] = ta.trend.sma_indicator(ticker_data['Adj Close'], window=20).fillna(0)
        ticker_data['RSI'] = ta.momentum.rsi(ticker_data['Adj Close']).fillna(0)
        ticker_data['MACD'] = ta.trend.macd(ticker_data['Adj Close']).fillna(0)

        scaler = MinMaxScaler()
        ticker_data = pd.DataFrame(scaler.fit_transform(ticker_data),
                                 columns=ticker_data.columns, index=ticker_data.index)
        data[ticker] = pd.Series(ticker_data[['Adj Close', 'SMA', 'RSI', 'MACD']].values.tolist(), index=ticker_data.index)

    data.index.name = 'Date'

    return data, price


def get_adj_matrix(tickers):
    returns_df = tickers.pct_change()
    adj_matrix = returns_df.corr()

    return adj_matrix


def get_graph_attributes(filled_tickers, price, tickers):
    adj_matrix = get_adj_matrix(price)
    adj_matrix = adj_matrix.fillna(0)
    adj_array = adj_matrix.to_numpy()
    adj_tensor = torch.from_numpy(adj_array).float()
    edge_index = torch.nonzero(adj_tensor, as_tuple=False).t().contiguous()
    edge_weights = adj_tensor[edge_index[0], edge_index[1]]

    mask = torch.abs(edge_weights) >= .3
    edge_index = edge_index[:, mask]
    edge_weights = edge_weights[mask]

    node_features = torch.tensor(np.array(filled_tickers.values.tolist()), dtype=torch.float)
    node_labels = torch.tensor(range(len(tickers)))

    return edge_index, edge_weights, adj_matrix, node_features, node_labels


def plot_graph(data, tickers):
    # Convert to NetworkX graph
    graph = to_networkx(data)
    pos = nx.spring_layout(graph)

    fig = go.Figure()

    for i, node in enumerate(graph.nodes()):
        fig.add_trace(go.Scatter(x=[pos[node][0]],
                                 y=[pos[node][1]],
                                 mode='markers+text',
                                 marker=dict(size=10),
                                 text=tickers[i],
                                 textposition="middle right"))
    edge_trace = go.Scatter(x=[], y=[], mode='lines',
                            line=dict(width=0.5, color='grey'), hoverinfo='none')

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    fig.add_trace(edge_trace)
    fig.update_layout(
        title="Ticker Correlation Graph",
        title_font_size=24,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig.show()


def train_gnn(model, data, optimizer, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.mse_loss(output, data.x)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Set the model to evaluation mode
    model.eval()

    # Encode the input data
    with torch.no_grad():
        encoded_data = model(data.x, data.edge_index)

    return encoded_data


if __name__ == "__main__":
    main()
