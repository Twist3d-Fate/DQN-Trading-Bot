# DQN-Trading-Bot
A Deep Q Learning Network Applied to Trading Using Lumibot and Alpaca

## Setup
Create an account at [Alpaca](https://alpaca.markets/) and generate an API key and secret key to connect to the agent. 

## Functionality
- This model has rudimentary multi-symbol capabilities, allowing the user to enter a list of symbols for the model to train on and trade.
- The model can be trained through reinforcement learning on each symbol in a Gymnasium trading environment using historical stock data and be saved to the user's personal computer.
- The model has backtesting capabilities through Lumibot and live trading functionality through Alpaca Markets.

## Backtesting Results 
The bot had strong volatility matched returns against the market in backtesting from 2023 to 2024 and 2024 to 2025, averaging to around a 20% annual return. In this backtesting, the bot traded on the following symbol list: "AAPL", "GOOGL", "MSFT", TSLA".

<p align="center">
  <a href="#"><img src="https://github.com/Twist3d-Fate/DQN-Trading-Bot/blob/main/23-24%20Returns.png"></a>
  <a href="#"><img src="https://github.com/Twist3d-Fate/DQN-Trading-Bot/blob/main/24-25%20Returns.png"></a>
</p>

## TODO Improvements 
- The model currently trades one symbol at a time in backtesting, is trained separately on each symbol, and is saved separately on a per-symbol basis. The next step for this functionality is to collectively train the mdoel on all symbols and enable it to make trades on any symbol during backtesting and live trading.
- Try other strategies like a DDQN, ARIMA or an LSTM-based architecture, FinBERT for sentiment analysis, or an ensemble of multiple models.
- Enable live trading with other brokerages besides Alpaca.
- Use more data for decision making, like technical and fundamental analysis of the data conducted by an AI model. 
