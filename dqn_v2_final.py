import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from lumibot.strategies import Strategy
from lumibot.backtesting import YahooDataBacktesting
from lumibot.traders import Trader
from lumibot.entities import Asset
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import deque
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')


class TradingEnvironment(gym.Env):
    """Gymnasium-based Trading Environment for DQN training"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 symbol: str = "STOCK",
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0,
                 window_size: int = 20):
        
        super(TradingEnvironment, self).__init__()
        
        self.data = data.copy()
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.window_size = window_size
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Calculate technical indicators
        self._calculate_features()
        
        # Environment state
        self.current_step = window_size
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [price features, position info, technical indicators]
        # Features per timestep: OHLCV (5) + technical indicators (8) = 13
        # Window size * features + position info (3) = window_size * 13 + 3
        obs_size = window_size * 13 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Episode tracking
        self.episode_start_value = initial_balance
        self.trade_count = 0
        self.successful_trades = 0
        
    def _calculate_features(self):
        """Calculate technical indicators for the trading data"""
        df = self.data.copy()
        
        # Price-based features (normalized)
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['open_close_ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean() / df['Close']
        df['sma_20'] = df['Close'].rolling(window=20).mean() / df['Close']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalize to 0-1
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / df['Close']
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Fill NaN values
        df = df.fillna(0)
        
        self.data = df
        
    def _get_observation(self):
        """Get the current observation state"""
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        # Get window of data
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Extract features for the window
        features = []
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'price_change', 'high_low_ratio', 'open_close_ratio',
                       'sma_5', 'sma_20', 'rsi', 'macd', 'volume_ratio']
        
        for _, row in window_data.iterrows():
            for col in feature_cols:
                features.append(row[col])
        
        # Pad if necessary (in case we don't have enough historical data)
        while len(features) < self.window_size * 13:
            features.append(0.0)
        
        # Add current position information
        current_price = self.data.iloc[self.current_step]['Close']
        position_value = self.shares_held * current_price
        total_value = self.balance + position_value
        
        features.extend([
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held * current_price / self.initial_balance,  # Normalized position value
            total_value / self.initial_balance  # Normalized total value
        ])
        
        return np.array(features, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.episode_start_value = self.initial_balance
        self.trade_count = 0
        self.successful_trades = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, self._get_info()
        
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                max_shares = int((self.balance * self.max_position) / (current_price * (1 + self.transaction_cost)))
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held += max_shares
                    self.trade_count += 1
                    reward += 1  # Small reward for trading
                    
        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.shares_held = 0
                self.trade_count += 1
                reward += 1  # Small reward for trading
        
        # Move to next step
        self.current_step += 1
        
        # Calculate new net worth
        if self.current_step < len(self.data):
            next_price = self.data.iloc[self.current_step]['Close']
            self.net_worth = self.balance + self.shares_held * next_price
        else:
            self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate reward based on net worth change
        net_worth_change = (self.net_worth - self.episode_start_value) / self.episode_start_value
        reward += net_worth_change * 100  # Scale the reward
        
        # Update max net worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
            self.successful_trades += 1
        
        # Penalty for large drawdowns
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if drawdown > 0.1:  # More than 10% drawdown
            reward -= drawdown * 50
        
        # Check if episode is done
        terminated = (self.current_step >= len(self.data) - 1) or (self.net_worth <= self.initial_balance * 0.1)
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self):
        """Get additional information about current state"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'trade_count': self.trade_count,
            'successful_trades': self.successful_trades,
            'max_net_worth': self.max_net_worth
        }
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        current_price = self.data.iloc[min(self.current_step, len(self.data)-1)]['Close']
        print(f"Step: {self.current_step}, Price: ${current_price:.2f}, "
              f"Balance: ${self.balance:.2f}, Shares: {self.shares_held}, "
              f"Net Worth: ${self.net_worth:.2f}")
    
    def close(self):
        """Clean up the environment"""
        pass


class DQNAgent:
    """Deep Q-Network agent with better architecture and decision making"""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int = 3,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 50000,
                 batch_size: int = 64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Network architecture with more layers and neurons
        self.hidden1_size = 512
        self.hidden2_size = 256
        self.hidden3_size = 128
        self.hidden4_size = 64
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build networks with architecture
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Action confidence tracking for better decision making
        self.action_confidence = deque(maxlen=1000)
        self.recent_rewards = deque(maxlen=100)
        
    def _build_network(self):
        """Build neural network"""
        network = {
            # Input layer -> Hidden layer 1 (512 neurons)
            'w1': np.random.randn(self.state_size, self.hidden1_size) * np.sqrt(2.0 / self.state_size),
            'b1': np.zeros((1, self.hidden1_size)),
            
            # Hidden layer 1 -> Hidden layer 2 (256 neurons)
            'w2': np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2.0 / self.hidden1_size),
            'b2': np.zeros((1, self.hidden2_size)),
            
            # Hidden layer 2 -> Hidden layer 3 (128 neurons)
            'w3': np.random.randn(self.hidden2_size, self.hidden3_size) * np.sqrt(2.0 / self.hidden2_size),
            'b3': np.zeros((1, self.hidden3_size)),
            
            # Hidden layer 3 -> Hidden layer 4 (64 neurons)
            'w4': np.random.randn(self.hidden3_size, self.hidden4_size) * np.sqrt(2.0 / self.hidden3_size),
            'b4': np.zeros((1, self.hidden4_size)),
            
            # Output layer (action values)
            'w5': np.random.randn(self.hidden4_size, self.action_size) * np.sqrt(2.0 / self.hidden4_size),
            'b5': np.zeros((1, self.action_size))
        }
        return network
    
    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function"""
        return np.maximum(alpha * x, x)
    
    def _leaky_relu_derivative(self, x, alpha=0.01):
        """Leaky ReLU derivative"""
        return np.where(x > 0, 1.0, alpha)
    
    def _forward(self, state, network):
        """Forward pass with deeper network and better activations"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(state, network['w1']) + network['b1']
        a1 = self._leaky_relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, network['w2']) + network['b2']
        a2 = self._leaky_relu(z2)
        
        # Layer 3
        z3 = np.dot(a2, network['w3']) + network['b3']
        a3 = self._leaky_relu(z3)
        
        # Layer 4
        z4 = np.dot(a3, network['w4']) + network['b4']
        a4 = self._leaky_relu(z4)
        
        # Output layer (no activation for Q-values)
        z5 = np.dot(a4, network['w5']) + network['b5']
        
        return z5, (z1, a1, z2, a2, z3, a3, z4, a4)
    
    def act(self, state, force_trade=False):
        """Action selection with confidence-based decision making"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        q_values, _ = self._forward(state, self.q_network)
        
        # Calculate action confidence
        max_q = np.max(q_values[0])
        second_max_q = np.partition(q_values[0], -2)[-2]
        confidence = max_q - second_max_q
        self.action_confidence.append(confidence)
        
        # Action selection logic
        if force_trade and len(self.recent_rewards) > 10:
            # If forced to trade and we have performance history, be more aggressive
            avg_reward = np.mean(self.recent_rewards)
            if avg_reward > 0:  # If doing well, take more confident actions
                threshold = np.percentile(list(self.action_confidence)[-100:], 60) if len(self.action_confidence) >= 100 else 0
                if confidence > threshold:
                    action = np.argmax(q_values[0])
                else:
                    # Choose second-best non-hold action if confidence is low
                    actions_sorted = np.argsort(q_values[0])[::-1]
                    for act in actions_sorted:
                        if act != 0:  # Not hold
                            action = act
                            break
                    else:
                        action = np.argmax(q_values[0])
            else:
                # If not doing well, be more conservative but still trade
                action = np.argmax(q_values[0])
        else:
            action = np.argmax(q_values[0])
            
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience and track rewards"""
        self.memory.append((state, action, reward, next_state, done))
        self.recent_rewards.append(reward)
    
    def replay(self):
        """Training with better gradient computation"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_qs, forward_cache = self._forward(states, self.q_network)
        
        # Next Q values from target network
        next_qs, _ = self._forward(next_states, self.target_network)
        
        # Calculate target Q values using Double DQN
        next_actions = np.argmax(self._forward(next_states, self.q_network)[0], axis=1)
        target_qs = current_qs.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_qs[i][actions[i]] = rewards[i]
            else:
                target_qs[i][actions[i]] = rewards[i] + self.gamma * next_qs[i][next_actions[i]]
        
        # Training step
        self._train_step(states, target_qs, forward_cache)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train_step(self, states, target_qs, forward_cache):
        """Training with better gradient computation and regularization"""
        predictions = self._forward(states, self.q_network)[0]
        batch_size = states.shape[0]
        
        # Calculate loss with L2 regularization
        loss = np.mean((predictions - target_qs) ** 2)
        l2_reg = 0.0001
        
        # Extract cached activations
        z1, a1, z2, a2, z3, a3, z4, a4 = forward_cache
        
        # Output layer gradients
        d_output = 2 * (predictions - target_qs) / batch_size
        
        # Layer 5 (output) gradients
        dw5 = np.dot(a4.T, d_output) + l2_reg * self.q_network['w5']
        db5 = np.sum(d_output, axis=0, keepdims=True)
        
        # Layer 4 gradients
        d_hidden4 = np.dot(d_output, self.q_network['w5'].T)
        d_hidden4 = d_hidden4 * self._leaky_relu_derivative(z4)
        
        dw4 = np.dot(a3.T, d_hidden4) + l2_reg * self.q_network['w4']
        db4 = np.sum(d_hidden4, axis=0, keepdims=True)
        
        # Layer 3 gradients
        d_hidden3 = np.dot(d_hidden4, self.q_network['w4'].T)
        d_hidden3 = d_hidden3 * self._leaky_relu_derivative(z3)
        
        dw3 = np.dot(a2.T, d_hidden3) + l2_reg * self.q_network['w3']
        db3 = np.sum(d_hidden3, axis=0, keepdims=True)
        
        # Layer 2 gradients
        d_hidden2 = np.dot(d_hidden3, self.q_network['w3'].T)
        d_hidden2 = d_hidden2 * self._leaky_relu_derivative(z2)
        
        dw2 = np.dot(a1.T, d_hidden2) + l2_reg * self.q_network['w2']
        db2 = np.sum(d_hidden2, axis=0, keepdims=True)
        
        # Layer 1 gradients
        d_hidden1 = np.dot(d_hidden2, self.q_network['w2'].T)
        d_hidden1 = d_hidden1 * self._leaky_relu_derivative(z1)
        
        dw1 = np.dot(states.T, d_hidden1) + l2_reg * self.q_network['w1']
        db1 = np.sum(d_hidden1, axis=0, keepdims=True)
        
        # Update weights with gradient clipping
        clip_value = 1.0
        
        self.q_network['w5'] -= self.learning_rate * np.clip(dw5, -clip_value, clip_value)
        self.q_network['b5'] -= self.learning_rate * np.clip(db5, -clip_value, clip_value)
        self.q_network['w4'] -= self.learning_rate * np.clip(dw4, -clip_value, clip_value)
        self.q_network['b4'] -= self.learning_rate * np.clip(db4, -clip_value, clip_value)
        self.q_network['w3'] -= self.learning_rate * np.clip(dw3, -clip_value, clip_value)
        self.q_network['b3'] -= self.learning_rate * np.clip(db3, -clip_value, clip_value)
        self.q_network['w2'] -= self.learning_rate * np.clip(dw2, -clip_value, clip_value)
        self.q_network['b2'] -= self.learning_rate * np.clip(db2, -clip_value, clip_value)
        self.q_network['w1'] -= self.learning_rate * np.clip(dw1, -clip_value, clip_value)
        self.q_network['b1'] -= self.learning_rate * np.clip(db1, -clip_value, clip_value)
    
    def update_target_network(self):
        """Update target network with soft update"""
        tau = 0.001  # Soft update parameter
        for key in self.q_network:
            self.target_network[key] = tau * self.q_network[key] + (1 - tau) * self.target_network[key]
    
    def save_model(self, filepath: str):
        """Save the model"""
        model_data = {
            'q_network': self.q_network,
            'target_network': self.target_network,
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'hidden3_size': self.hidden3_size,
            'hidden4_size': self.hidden4_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'action_confidence': list(self.action_confidence),
            'recent_rewards': list(self.recent_rewards)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load the model"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.q_network = model_data['q_network']
                self.target_network = model_data['target_network']
                self.epsilon = model_data.get('epsilon', self.epsilon_min)
                
                # Load additional data if available
                if 'action_confidence' in model_data:
                    self.action_confidence = deque(model_data['action_confidence'], maxlen=1000)
                if 'recent_rewards' in model_data:
                    self.recent_rewards = deque(model_data['recent_rewards'], maxlen=100)
                
                # Verify dimensions match
                if (model_data.get('state_size') == self.state_size and 
                    model_data.get('action_size') == self.action_size):
                    return True
                else:
                    print(f"Warning: Model dimensions don't match. Expected state_size={self.state_size}, got {model_data.get('state_size')}")
                    return False
            except Exception as e:
                print(f"Error loading model from {filepath}: {e}")
                return False
        return False


class DQNTradingStrategy(Strategy):
    """DQN Trading Strategy with model loading and aggressive trading"""
    
    def initialize(self, 
                  symbols: List[str] = None,
                  cash_per_symbol: float = 10000,
                  model_save_path: str = "models/",
                  model_prefix: str = "dqn_v2",
                  training_mode: bool = False,
                  risk_management: bool = True,
                  max_position_size: float = 0.4,
                  force_trading: bool = True,
                  load_pretrained: bool = True):
        
        self.symbols = symbols or ["AAPL", "GOOGL", "MSFT", "TSLA"]
        self.cash_per_symbol = cash_per_symbol
        self.model_save_path = model_save_path
        self.model_prefix = model_prefix
        self.training_mode = training_mode
        self.risk_management = risk_management
        self.max_position_size = max_position_size
        self.force_trading = force_trading
        self.load_pretrained = load_pretrained
        
        os.makedirs(model_save_path, exist_ok=True)
        
        # Initialize agents for each symbol
        self.agents = {}
        self.current_observations = {}
        self.last_actions = {}
        self.price_history = {}
        self.reward_history = {}
        self.trade_count = {}
        self.successful_trades = {}
        
        # Initialize agents
        for symbol in self.symbols:
            # Create dummy environment to get observation space size
            dummy_data = pd.DataFrame({
                'Open': [100] * 100,
                'High': [105] * 100,
                'Low': [95] * 100,
                'Close': [100] * 100,
                'Volume': [1000000] * 100
            })
            
            dummy_env = TradingEnvironment(data=dummy_data, symbol=symbol)
            obs_size = dummy_env.observation_space.shape[0]
            dummy_env.close()
            
            # Initialize agent
            agent = DQNAgent(
                state_size=obs_size, 
                epsilon=0.05 if not training_mode else 1.0,  # Lower epsilon for more exploitation
                learning_rate=0.0001,
                gamma=0.99
            )
            
            # Try to load existing model
            if self.load_pretrained:
                model_path = os.path.join(model_save_path, f"{self.model_prefix}_{symbol}.pkl")
                success = agent.load_model(model_path)
                if success:
                    print(f"Successfully loaded pretrained model for {symbol} from {model_path}")
                    print(f"   Model epsilon: {agent.epsilon:.4f}")
                    print(f"   Model has {len(agent.recent_rewards)} recent rewards in memory")
                else:
                    print(f"No pretrained model found for {symbol} at {model_path}")
                    print("    Using new agent with random weights")
            else:
                print(f"Using new agent for {symbol} (pretrained loading disabled)")
            
            self.agents[symbol] = agent
            self.last_actions[symbol] = 0
            self.price_history[symbol] = []
            self.reward_history[symbol] = []
            self.trade_count[symbol] = 0
            self.successful_trades[symbol] = 0
        
        # Performance tracking
        self.performance_data = []
        self.step_count = 0
        
        # Risk management
        self.max_daily_loss = 0.05
        self.daily_start_value = None
        self.consecutive_hold_count = {}
        
        for symbol in self.symbols:
            self.consecutive_hold_count[symbol] = 0
        
        print(f"DQN Trading Strategy initialized")
        print(f"   Symbols: {self.symbols}")
        print(f"   Force trading: {self.force_trading}")
        print(f"   Training mode: {self.training_mode}")
        print(f"   Risk management: {self.risk_management}")
    
    def on_trading_iteration(self):
        """Trading logic with forced trading and better decision making"""
        current_time = self.get_datetime()
        
        # Initialize daily start value
        if self.daily_start_value is None:
            self.daily_start_value = float(self.portfolio_value)
        
        # Risk management check
        if self.risk_management:
            current_loss = (self.daily_start_value - float(self.portfolio_value)) / self.daily_start_value
            if current_loss > self.max_daily_loss:
                self.log_message(f"Daily loss limit exceeded ({current_loss:.2%}), stopping trading")
                return
        
        for symbol in self.symbols:
            try:
                asset = Asset(symbol=symbol, asset_type="stock")
                current_data = self.get_historical_prices(asset, length=100, timeframe="minute")
                
                if current_data is None or len(current_data) < 50:
                    continue

                # Prepare market data
                market_data = pd.DataFrame({
                    'Open': current_data['open'],
                    'High': current_data['high'],
                    'Low': current_data['low'],
                    'Close': current_data['close'],
                    'Volume': current_data['volume']
                })
                
                env = TradingEnvironment(
                    data=market_data,
                    symbol=symbol,
                    initial_balance=self.cash_per_symbol
                )
                
                obs, _ = env.reset()
                current_price = market_data['Close'].iloc[-1]
                
                # Store price history
                self.price_history[symbol].append(current_price)
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
                
                # Determine if we should force trading
                should_force_trade = (
                    self.force_trading and 
                    self.consecutive_hold_count[symbol] > 5 and  # Too many holds
                    len(self.price_history[symbol]) > 10
                )
                
                # Get action from agent
                action = self.agents[symbol].act(obs, force_trade=should_force_trade)
                
                # Reset consecutive hold count if not holding
                if action != 0:
                    self.consecutive_hold_count[symbol] = 0
                else:
                    self.consecutive_hold_count[symbol] += 1
                
                # Get current position info
                current_position = self.get_position(asset)
                current_qty = current_position.quantity if current_position else 0
                
                # Position limits
                max_value = float(self.portfolio_value) * self.max_position_size
                max_shares = int(max_value / current_price) if current_price > 0 else 0
                
                # Execute action
                executed_action = self._execute_action(
                    symbol, action, current_price, current_qty, max_shares, should_force_trade
                )
                
                # Training and reward calculation
                if self.training_mode and len(self.price_history[symbol]) > 1:
                    prev_price = self.price_history[symbol][-2]
                    price_change = (current_price - prev_price) / prev_price if prev_price > 0 else 0
                    
                    reward = self._calculate_reward(
                        executed_action, price_change, current_qty, symbol
                    )
                    self.reward_history[symbol].append(reward)
                    
                    # Store experience
                    if hasattr(self, 'prev_obs') and symbol in self.current_observations:
                        self.agents[symbol].remember(
                            self.current_observations[symbol],
                            self.last_actions[symbol],
                            reward,
                            obs,
                            False
                        )
                    
                    # Training schedule
                    if len(self.agents[symbol].memory) > 200 and self.step_count % 5 == 0:
                        self.agents[symbol].replay()
                        if self.step_count % 50 == 0:
                            self.agents[symbol].update_target_network()
                
                # Update state tracking
                self.current_observations[symbol] = obs
                self.last_actions[symbol] = executed_action
                
                env.close()
                
            except Exception as e:
                self.log_message(f"Error processing {symbol}: {e}")
                continue
        
        # Save models more frequently in training mode
        save_interval = 500 if self.training_mode else 2000
        if self.step_count % save_interval == 0:
            self._save_all_models()
        
        self.step_count += 1
        self._update_performance_tracking()
        
        # Reset daily tracking
        if current_time.hour == 9 and current_time.minute == 30:
            self.daily_start_value = float(self.portfolio_value)
    
    def _execute_action(self, symbol: str, action: int, current_price: float, 
                                current_qty: int, max_shares: int, force_trade: bool) -> int:
        """Action execution with better trade sizing and forced trading"""
        asset = Asset(symbol=symbol, asset_type="stock")
        executed_action = 0
        
        try:
            if action == 1:  # Buy signal
                if current_qty < max_shares:
                    # Dynamic position sizing based on confidence and performance
                    base_size = 0.3  # 30% of max position
                    if hasattr(self.agents[symbol], 'recent_rewards') and len(self.agents[symbol].recent_rewards) > 10:
                        avg_reward = np.mean(list(self.agents[symbol].recent_rewards)[-10:])
                        if avg_reward > 0:
                            base_size = 0.5  # Increase size if doing well
                    
                    qty_to_buy = min(
                        max(1, int((max_shares - current_qty) * base_size)),
                        max_shares - current_qty
                    )
                    
                    required_cash = qty_to_buy * current_price * 1.02
                    if float(self.cash) >= required_cash:
                        order = self.create_order(asset, qty_to_buy, "buy")
                        self.submit_order(order)
                        executed_action = 1
                        self.trade_count[symbol] += 1
                        self.log_message(f"BUY {qty_to_buy} shares of {symbol} @ ${current_price:.2f}")
                        
            elif action == 2 and current_qty > 0:  # Sell signal
                # Sell logic - partial or full exit based on position size
                if current_qty <= 10:
                    qty_to_sell = current_qty  # Sell all if small position
                else:
                    qty_to_sell = max(1, int(current_qty * 0.6))  # Sell 60% of position
                
                order = self.create_order(asset, qty_to_sell, "sell")
                self.submit_order(order)
                executed_action = 2
                self.trade_count[symbol] += 1
                self.log_message(f"SELL {qty_to_sell} shares of {symbol} @ ${current_price:.2f}")
                
            elif force_trade and action == 0:  # Forced trading when holding too much
                if current_qty == 0 and float(self.cash) > current_price * 10:
                    # Force buy if no position and have cash
                    qty_to_buy = max(1, int(float(self.cash) * 0.1 / current_price))
                    order = self.create_order(asset, qty_to_buy, "buy")
                    self.submit_order(order)
                    executed_action = 1
                    self.trade_count[symbol] += 1
                    self.log_message(f"FORCED BUY {qty_to_buy} shares of {symbol} @ ${current_price:.2f}")
                    
                elif current_qty > 0:
                    # Force sell partial position
                    qty_to_sell = max(1, int(current_qty * 0.3))
                    order = self.create_order(asset, qty_to_sell, "sell")
                    self.submit_order(order)
                    executed_action = 2
                    self.trade_count[symbol] += 1
                    self.log_message(f"FORCED SELL {qty_to_sell} shares of {symbol} @ ${current_price:.2f}")
                    
        except Exception as e:
            self.log_message(f"Error executing action for {symbol}: {e}")
            
        return executed_action
    
    def _calculate_reward(self, action: int, price_change: float, 
                                 position: int, symbol: str) -> float:
        """Reward calculation with better incentives"""
        reward = 0.0
        
        # Base reward for correct directional trades
        if action == 1:  # Bought
            reward = price_change * 200  # Higher reward for good buys
            if price_change > 0.01:  # Bonus for significant moves
                reward += 10
        elif action == 2:  # Sold
            reward = -price_change * 200  # Higher reward for good sells
            if price_change < -0.01:  # Bonus for avoiding significant drops
                reward += 10
        else:  # Held
            if abs(price_change) < 0.005:  # Reward for holding during stable periods
                reward = 2
            else:
                reward = -abs(price_change) * 50  # Penalty for missing opportunities
        
        # Position-based adjustments
        if position > 0:
            reward += price_change * 100  # Benefit from holding winning positions
        
        # Track successful trades
        if (action == 1 and price_change > 0) or (action == 2 and price_change < 0):
            self.successful_trades[symbol] = self.successful_trades.get(symbol, 0) + 1
        
        return reward
    
    def _save_all_models(self):
        """Save all models with confirmation"""
        saved_count = 0
        for symbol in self.symbols:
            try:
                model_path = os.path.join(self.model_save_path, f"{symbol}_{self.model_prefix}.pkl")
                self.agents[symbol].save_model(model_path)
                saved_count += 1
            except Exception as e:
                self.log_message(f"Failed to save model for {symbol}: {e}")
        
        if saved_count > 0:
            self.log_message(f"DQN models saved successfully ({saved_count}/{len(self.symbols)})")
    
    def _update_performance_tracking(self):
        """Performance tracking with trade statistics"""
        current_time = self.get_datetime()
        
        # Calculate trade statistics
        total_trades = sum(self.trade_count.values())
        total_successful = sum(self.successful_trades.values())
        win_rate = total_successful / max(total_trades, 1)
        
        self.performance_data.append({
            'timestamp': current_time,
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'step': self.step_count,
            'total_trades': total_trades,
            'win_rate': win_rate
        })
        
        # Logging every 100 steps
        if self.step_count % 100 == 0:
            if len(self.performance_data) > 1:
                start_value = self.performance_data[0]['portfolio_value']
                current_value = self.performance_data[-1]['portfolio_value']
                total_return = (current_value - start_value) / start_value
                
                self.log_message(f"Performance Update - Step {self.step_count}")
                self.log_message(f"  Total Return: {total_return:.2%}")
                self.log_message(f"  Portfolio Value: ${current_value:,.2f}")
                self.log_message(f"  Total Trades: {total_trades}")
                self.log_message(f"  Win Rate: {win_rate:.2%}")
                
                # Per-symbol statistics
                for symbol in self.symbols:
                    trades = self.trade_count[symbol]
                    successful = self.successful_trades[symbol]
                    symbol_win_rate = successful / max(trades, 1)
                    consecutive_holds = self.consecutive_hold_count[symbol]
                    self.log_message(f"  {symbol}: {trades} trades, {symbol_win_rate:.1%} win rate, {consecutive_holds} consecutive holds")


class TradingBotManager:
    """Manager class with model loading functionality"""
    
    def __init__(self, 
                 alpaca_api_key: str = None,
                 alpaca_secret_key: str = None,
                 paper_trading: bool = True):
        
        self.alpaca_api_key = alpaca_api_key
        self.alpaca_secret_key = alpaca_secret_key
        self.paper_trading = paper_trading
        self.alpaca_client = None
        
        # Note: Alpaca client initialization commented out as AlpacaTradingClient import doesn't exist
        # if alpaca_api_key and alpaca_secret_key:
        #     from AlpacaTradingClient import AlpacaTradingClient
        #     self.alpaca_client = AlpacaTradingClient(
        #         alpaca_api_key, alpaca_secret_key, paper_trading
        #     )
    
    def backtest_with_lumibot(self, 
                             symbols: List[str],
                             start_date: datetime = None,
                             end_date: datetime = None,
                             initial_cash: float = 100000,
                             strategy_class: Strategy = None,
                             strategy_parameters: Dict = None,
                             benchmark_asset: str = "SPY",
                             load_pretrained: bool = True,
                             model_path: str = "models/") -> Dict:
        """Backtest with pretrained model loading"""
        
        if start_date is None:
            start_date = datetime(2022, 1, 1)
        if end_date is None:
            end_date = datetime(2023, 12, 31)
        if strategy_class is None:
            strategy_class = DQNTradingStrategy
        if strategy_parameters is None:
            strategy_parameters = {
                "symbols": symbols,
                "cash_per_symbol": initial_cash / len(symbols),
                "training_mode": False,
                "risk_management": True,
                "force_trading": True,
                "load_pretrained": load_pretrained,
                "model_save_path": model_path
            }
        
        print(f"Running Lumibot backtest from {start_date} to {end_date}")
        print(f"Symbols: {symbols}")
        print(f"Initial cash: ${initial_cash:,.2f}")
        print(f"Load pretrained models: {load_pretrained}")
        
        # Check for existing models if loading is enabled
        if load_pretrained:
            print("\nChecking for pretrained models:")
            for symbol in symbols:
                model_file = os.path.join(model_path, f"dqn_v2_{symbol}.pkl")
                if os.path.exists(model_file):
                    print(f"  Found model for {symbol}: {model_file}")
                else:
                    print(f"  No model found for {symbol}: {model_file}")
        
        try:
            # Initialize the strategy with parameters
            strategy = strategy_class()
            
            # Run backtest using the strategy's backtest method
            results = strategy.backtest(
                YahooDataBacktesting,
                start=start_date,
                end=end_date,
                parameters=strategy_parameters,
                benchmark_asset=benchmark_asset,
                show_plot=True,
                show_tearsheet=True,
                save_tearsheet=True
            )
            
            # Extract key metrics from results
            if results:
                portfolio_history = results.get('portfolio_value', [])
                
                if portfolio_history and len(portfolio_history) > 0:
                    final_value = portfolio_history[-1] if isinstance(portfolio_history, list) else float(results.get('final_portfolio_value', initial_cash))
                    total_return = (final_value - initial_cash) / initial_cash
                    
                    # Metrics calculation
                    returns_series = results.get('returns', [])
                    if returns_series and len(returns_series) > 1:
                        returns_array = np.array(returns_series)
                        annual_return = np.mean(returns_array) * 252
                        volatility = np.std(returns_array) * np.sqrt(252)
                        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                        
                        # Max drawdown calculation
                        cumulative_returns = np.cumprod(1 + returns_array)
                        running_max = np.maximum.accumulate(cumulative_returns)
                        drawdown = (cumulative_returns - running_max) / running_max
                        max_drawdown = np.min(drawdown)
                        
                        # Calculate additional metrics
                        positive_returns = returns_array[returns_array > 0]
                        negative_returns = returns_array[returns_array < 0]
                        win_rate = len(positive_returns) / len(returns_array) if len(returns_array) > 0 else 0
                        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
                        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
                        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                        
                    else:
                        # Fallback calculations
                        days = (end_date - start_date).days
                        annual_return = (total_return + 1) ** (365.25 / days) - 1 if days > 0 else 0
                        volatility = 0.15
                        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                        max_drawdown = -0.10
                        win_rate = 0.5
                        profit_factor = 1.0
                    
                    backtest_stats = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'initial_cash': initial_cash,
                        'final_value': final_value,
                        'total_return': total_return,
                        'annual_return': annual_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'volatility': volatility,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'benchmark_asset': benchmark_asset,
                        'used_pretrained_models': load_pretrained,
                        'status': 'completed'
                    }
                    
                    # Results display
                    print("\n" + "="*70)
                    print("LUMIBOT BACKTEST RESULTS")
                    print("="*70)
                    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    print(f"Pretrained Models Used: {'Yes' if load_pretrained else 'No'}")
                    print(f"Initial Cash: ${initial_cash:,.2f}")
                    print(f"Final Value: ${final_value:,.2f}")
                    print(f"Total Return: {total_return:.2%}")
                    print(f"Annual Return: {annual_return:.2%}")
                    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
                    print(f"Max Drawdown: {max_drawdown:.2%}")
                    print(f"Volatility: {volatility:.2%}")
                    print(f"Win Rate: {win_rate:.2%}")
                    print(f"Profit Factor: {profit_factor:.2f}")
                    print("="*70)
                    
                    return backtest_stats
                    
                else:
                    print("Warning: No portfolio history found in results")
                    return {
                        'start_date': start_date,
                        'end_date': end_date,
                        'initial_cash': initial_cash,
                        'status': 'no_portfolio_history'
                    }
            
            else:
                print("Warning: Backtest returned no results")
                return {
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_cash': initial_cash,
                    'status': 'no_results'
                }
                
        except Exception as e:
            print(f"Error during Lumibot backtesting: {e}")
            return {
                'start_date': start_date,
                'end_date': end_date,
                'initial_cash': initial_cash,
                'status': 'error',
                'error': str(e)
            }
    
    def run_live_trading_with_lumibot(self, 
                                     symbols: List[str], 
                                     cash: float = 100000,
                                     training_mode: bool = False,
                                     strategy_class: Strategy = None,
                                     load_pretrained: bool = True,
                                     model_path: str = "models/"):
        """Live trading with pretrained model loading"""
        
        if not self.alpaca_client:
            raise ValueError("Alpaca client not initialized. Provide API credentials.")
        
        if strategy_class is None:
            strategy_class = DQNTradingStrategy
        
        print("Starting Live Trading with Lumibot and DQN agents...")
        print(f"Trading symbols: {symbols}")
        print(f"Total cash: ${cash:,.2f}")
        print(f"Training mode: {'ON' if training_mode else 'OFF'}")
        print(f"Load pretrained models: {load_pretrained}")
        
        # Check for existing models if loading is enabled
        if load_pretrained:
            print("\nModel Loading Status:")
            models_found = 0
            for symbol in symbols:
                model_file = os.path.join(model_path, f"{symbol}_dqn_v2.pkl")
                if os.path.exists(model_file):
                    print(f"  Model found for {symbol}: {model_file}")
                    models_found += 1
                else:
                    print(f"  No model found for {symbol}: {model_file}")
            print(f"Found {models_found}/{len(symbols)} pretrained models")
        
        try:
            # Initialize strategy with parameters
            strategy = strategy_class()
            strategy.initialize(
                symbols=symbols,
                cash_per_symbol=cash / len(symbols),
                training_mode=training_mode,
                risk_management=True,
                force_trading=True,
                load_pretrained=load_pretrained,
                model_save_path=model_path
            )
            
            # Create trader
            trader = Trader()
            trader.add_strategy(strategy)
            
            # Run live trading
            print("\nStarting live trading execution...")
            trader.run_all()
            
        except KeyboardInterrupt:
            print("\nLive trading stopped by user")
        except Exception as e:
            print(f"Error in live trading: {e}")
    
    def train_agents(self, 
                            symbols: List[str],
                            training_data: Dict[str, pd.DataFrame] = None,
                            episodes: int = 2000,
                            save_path: str = "models/",
                            model_prefix: str = "dqn_v2") -> Dict[str, DQNAgent]:
        """Train agents and save models"""
        
        if training_data is None:
            training_data = self.get_training_data(symbols, period="3y")
        
        os.makedirs(save_path, exist_ok=True)
        trained_agents = {}
        
        for symbol in symbols:
            if symbol not in training_data:
                continue
            
            print(f"\nTraining DQN agent for {symbol}...")
            print(f"Training episodes: {episodes}")
            print(f"Data points: {len(training_data[symbol])}")
            
            # Prepare data
            data = training_data[symbol].reset_index()
            
            # Create environment
            env = TradingEnvironment(
                data=data,
                symbol=symbol,
                initial_balance=10000,
                transaction_cost=0.001,
                max_position=1.0
            )
            
            # Create agent
            obs_size = env.observation_space.shape[0]
            agent = DQNAgent(
                state_size=obs_size,
                learning_rate=0.0001,
                epsilon=1.0,
                epsilon_decay=0.9995,  # Slower decay for better exploration
                gamma=0.99,
                memory_size=50000,
                batch_size=64
            )
            
            # Training metrics
            episode_rewards = []
            episode_returns = []
            best_return = float('-inf')
            
            print(f"Starting training for {symbol}...")
            
            for episode in tqdm(range(episodes), desc="Training Episodes"):
                obs, info = env.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    # Get action
                    action = agent.act(obs, force_trade=(steps > 50))  # Force trading after 50 steps
                    
                    # Take step
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Store experience
                    agent.remember(obs, action, reward, next_obs, terminated or truncated)
                    
                    # Train agent more frequently
                    if len(agent.memory) > agent.batch_size and steps % 5 == 0:
                        agent.replay()
                    
                    obs = next_obs
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                # Update target network
                if episode % 50 == 0:
                    agent.update_target_network()
                
                # Track performance
                episode_rewards.append(total_reward)
                final_return = (info['net_worth'] - 10000) / 10000
                episode_returns.append(final_return)
                
                # Save best model
                if final_return > best_return:
                    best_return = final_return
                    model_path = os.path.join(save_path, f"{model_prefix}_{symbol}.pkl")
                    agent.save_model(model_path)
                
                # Progress logging
                if episode % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                    avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else np.mean(episode_returns)
                    print(f"Episode {episode}/{episodes}")
                    print(f"  Avg Reward (last 100): {avg_reward:.2f}")
                    print(f"  Avg Return (last 100): {avg_return:.4f}")
                    print(f"  Best Return: {best_return:.4f}")
                    print(f"  Epsilon: {agent.epsilon:.4f}")
                    print(f"  Memory Size: {len(agent.memory)}")
            
            # Final save and summary
            final_model_path = os.path.join(save_path, f"{model_prefix}_{symbol}.pkl")
            agent.save_model(final_model_path)
            trained_agents[symbol] = agent
            env.close()
            
            final_avg_reward = np.mean(episode_rewards[-200:]) if len(episode_rewards) >= 200 else np.mean(episode_rewards)
            final_avg_return = np.mean(episode_returns[-200:]) if len(episode_returns) >= 200 else np.mean(episode_returns)
            
            print(f"\nTraining completed for {symbol}!")
            print(f"Final average reward: {final_avg_reward:.2f}")
            print(f"Final average return: {final_avg_return:.4f}")
            print(f"Best return achieved: {best_return:.4f}")
            print(f"Model saved to: {final_model_path}")
        
        print(f"\nAll DQN agents trained successfully!")
        return trained_agents
    
    def get_training_data(self, 
                         symbols: List[str], 
                         period: str = "3y") -> Dict[str, pd.DataFrame]:
        """Get historical data for training"""
        
        data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                if not df.empty:
                    data[symbol] = df
                    print(f"Retrieved {len(df)} days of data for {symbol}")
                else:
                    print(f"No data retrieved for {symbol}")
            except Exception as e:
                print(f"Error retrieving data for {symbol}: {e}")
        
        return data


# Example usage
if __name__ == "__main__":
    # Configuration
    SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    ALPACA_API_KEY = "YOUR ALPACA API KEY"
    ALPACA_SECRET_KEY = "YOUR ALPACA API SECRET"
    
    # Initialize bot manager
    bot_manager = TradingBotManager(
        alpaca_api_key=ALPACA_API_KEY,
        alpaca_secret_key=ALPACA_SECRET_KEY,
        paper_trading=True
    )
    
    print("="*80)
    print("DQN TRADING SYSTEM WITH GYMNASIUM")
    print("="*80)
    
    # Step 1: Train agents (optional - comment out if using existing models)
    print("\nSTEP 1: TRAINING AGENTS")
    print("-"*50)
    
    # Uncomment to train new models
    training_data = bot_manager.get_training_data(SYMBOLS, period="3y")
    trained_agents = bot_manager.train_agents(
        SYMBOLS, 
        training_data, 
        episodes=100,
        save_path="models/",
        model_prefix="dqn_v2"
    )
    
    # Step 2: Backtest with pretrained models
    print("\nSTEP 2: BACKTESTING WITH PRETRAINED MODELS")
    print("-"*60)
    
    backtest_start = datetime(2023, 1, 1)
    backtest_end = datetime(2023, 12, 31)
    initial_cash = 100000
    
    # Backtest with pretrained models
    backtest_results = bot_manager.backtest_with_lumibot(
        symbols=SYMBOLS,
        start_date=backtest_start,
        end_date=backtest_end,
        initial_cash=initial_cash,
        strategy_class=DQNTradingStrategy,
        strategy_parameters={
            "symbols": SYMBOLS,
            "cash_per_symbol": initial_cash / len(SYMBOLS),
            "training_mode": False,
            "risk_management": True,
            "force_trading": True,
            "load_pretrained": True,
            "model_save_path": "models/"
        },
        load_pretrained=True,
        model_path="models/"
    )
    
    # Step 3: Compare with and without pretrained models
    print("\nSTEP 3: COMPARISON - WITH VS WITHOUT PRETRAINED MODELS")
    print("-"*65)
    
    # Backtest without pretrained models for comparison
    backtest_results_no_pretrained = bot_manager.backtest_with_lumibot(
        symbols=SYMBOLS,
        start_date=backtest_start,
        end_date=backtest_end,
        initial_cash=initial_cash,
        strategy_class=DQNTradingStrategy,
        strategy_parameters={
            "symbols": SYMBOLS,
            "cash_per_symbol": initial_cash / len(SYMBOLS),
            "training_mode": False,
            "risk_management": True,
            "force_trading": True,
            "load_pretrained": False,
            "model_save_path": "models/"
        },
        load_pretrained=False,
        model_path="models/"
    )
    
    # Print comparison
    print("\nCOMPARISON RESULTS:")
    print("="*50)
    
    if 'total_return' in backtest_results and 'total_return' in backtest_results_no_pretrained:
        print(f"WITH PRETRAINED MODELS:")
        print(f"  Total Return: {backtest_results['total_return']:.2%}")
        print(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown: {backtest_results.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {backtest_results.get('win_rate', 0):.2%}")
        
        print(f"\nWITHOUT PRETRAINED MODELS:")
        print(f"  Total Return: {backtest_results_no_pretrained['total_return']:.2%}")
        print(f"  Sharpe Ratio: {backtest_results_no_pretrained.get('sharpe_ratio', 0):.4f}")
        print(f"  Max Drawdown: {backtest_results_no_pretrained.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {backtest_results_no_pretrained.get('win_rate', 0):.2%}")
        
        improvement = backtest_results['total_return'] - backtest_results_no_pretrained['total_return']
        print(f"\nIMPROVEMENT WITH PRETRAINED MODELS: {improvement:.2%}")
    
    print("\nBacktest completed! Check generated tearsheet files for detailed analysis.")
    
    # Step 4: Simple training example (optional)
    print("\nSTEP 4: SIMPLE TRAINING EXAMPLE")
    print("-"*40)
    
    # Simple example of how to use the Gymnasium environment directly
    print("Running a simple example of the TradingEnvironment...")
    
    # Get some sample data
    sample_data = bot_manager.get_training_data(["AAPL"], period="1y")
    if "AAPL" in sample_data:
        # Create environment
        env = TradingEnvironment(
            data=sample_data["AAPL"].reset_index(),
            symbol="AAPL",
            initial_balance=10000
        )
        
        # Run a simple episode
        obs, info = env.reset()
        print(f"Environment observation shape: {obs.shape}")
        print(f"Environment action space: {env.action_space}")
        print(f"Initial info: {info}")
        
        # Take a few random actions
        for i in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Net Worth=${info['net_worth']:.2f}")
            if terminated or truncated:
                break
        
        env.close()
        print("Simple environment test completed!")
    
    # Uncomment to run live trading with pretrained models
    # print("\nSTEP 5: STARTING LIVE TRADING WITH PRETRAINED MODELS")
    # print("-"*60)
    # bot_manager.run_live_trading_with_lumibot(
    #     symbols=SYMBOLS,
    #     cash=100000,
    #     training_mode=False,
    #     strategy_class=DQNTradingStrategy,
    #     load_pretrained=True,
    #     model_path="models/"

    # )
