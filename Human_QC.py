import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import numpy as np
import random
import pickle
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

# Set the D-Wave API token
os.environ["DWAVE_API_TOKEN"] = ""

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def log_activity(message):
    logging.info(message)
    print(message)  # Also print to console for real-time feedback

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    log_activity(f"Data read from {file_path}, shape: {df.shape}")
    log_activity(f"Columns in the dataset: {df.columns.tolist()}")
    return df

def preprocess_data(data):
    # Convert date columns to datetime or drop them
    for column in data.columns:
        if 'date' in column.lower():
            data[column] = pd.to_datetime(data[column], errors='coerce')
            data[column] = data[column].map(datetime.toordinal)  # Convert dates to ordinal numbers

    # Drop any remaining non-numeric columns
    data = data.select_dtypes(include=[int, float])

    # Drop rows with any NaN values
    data = data.dropna()

    return data

def detect_support_resistance(data):
    data['support'] = data['Close/Last'].rolling(window=20).min()
    data['resistance'] = data['Close/Last'].rolling(window=20).max()
    return data

def train_model(data):
    data = preprocess_data(data)
    data = detect_support_resistance(data)
    target_column = 'Close/Last'  # Assuming 'Close/Last' is the target column for the model
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    log_activity("Model trained successfully")
    return model, X_test, y_test, X.columns

class SimulatedEnvironment:
    def __init__(self, historical_data, feature_columns):
        self.historical_data = historical_data
        self.current_step = 0
        self.feature_columns = feature_columns
        self.trades_per_day = 0
        self.daily_trades_limit = 1
        self.initial_balance = 100000  # Example initial balance
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.hold_duration = 0
        self.number_of_shares = 1

    def reset(self):
        self.current_step = 0
        self.trades_per_day = 0
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0
        self.hold_duration = 0
        log_activity("Simulation reset")
        return self.historical_data.iloc[self.current_step]

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.historical_data):
            done = True
            next_state = None
        else:
            done = False
            next_state = self.historical_data.iloc[self.current_step]

        reward = self.calculate_reward(action)
        log_activity(f"Step: {self.current_step}, Action: {action}, Reward: {reward}, Balance: {self.balance}.")
        return next_state, reward, done

    def calculate_reward(self, action):
        if self.current_step >= len(self.historical_data):
            return 0  # No reward if we're beyond the end of the data

        current_price = self.historical_data.iloc[self.current_step]['Close/Last']
        transaction_fee = 0  # Default transaction fee

        # Calculate transaction fee based on Charles Schwab's structure
        trade_amount = current_price * 100  # Assuming 100 shares per trade
        if trade_amount < 100:
            transaction_fee = 0
        else:
            transaction_fee = min(8.5 / 100 * trade_amount, 74.95)

        if self.position is None and action == 1:  # Buy
            self.position = 'long'
            self.entry_price = current_price
            self.hold_duration = 0
            self.balance -= transaction_fee
            return -transaction_fee  # Immediate cost for entering a position

        if self.position == 'long':
            self.hold_duration += 1
            if action == 2:  # Sell
                profit = (current_price - self.entry_price) * 100 - transaction_fee  # Net profit after transaction fee
                self.balance += profit
                reward = profit
                self.position = None
                self.entry_price = 0
                self.hold_duration = 0
                return reward

        # Reward for holding a position
        if self.position == 'long':
            return 0.01 * self.hold_duration  # Small reward for holding

        return 0  # No reward for other actions

def evaluate_model_performance(model, X_test, y_test):
    score = model.score(X_test, y_test)
    log_activity(f"Model R^2 score: {score}")
    return score

def analyze_sentiment(state):
    if 'Close/Last' in state.index and 'Open' in state.index:
        if state['Close/Last'] > state['Open']:
            return 1  # Positive sentiment
        elif state['Close/Last'] < state['Open']:
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment
    else:
        return 0  # Default to neutral if data is missing

def solve_qubo_with_dwave(bqm):
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm, num_reads=10)
    return response

def q_learning_train(env, model, episodes=1000, alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.01, decay_rate=0.995):
    q_table = {}

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            features = state[env.feature_columns].values.reshape(1, -1)
            sentiment_score = analyze_sentiment(state)
            state_key = (tuple(features[0]), sentiment_score)

            if state_key not in q_table:
                q_table[state_key] = [0, 0, 0]

            if random.uniform(0, 1) < epsilon:
                action = random.choice([0, 1, 2])
            else:
                action = np.argmax(q_table[state_key])

            next_state, reward, done = env.step(action)
            next_features = next_state[env.feature_columns].values.reshape(1, -1) if next_state is not None else None
            next_sentiment_score = analyze_sentiment(next_state) if next_state is not None else 0
            next_state_key = (tuple(next_features[0]), next_sentiment_score) if next_features is not None else None

            if next_state_key not in q_table:
                q_table[next_state_key] = [0, 0, 0]

            if done:
                q_table[state_key][action] = reward
            else:
                linear = q_table[state_key]
                quadratic = {(i, j): 0 for i in range(len(linear)) for j in range(i+1, len(linear))}
                bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
                response = solve_qubo_with_dwave(bqm)
                dwave_result = next(response.data())
                q_table[state_key][action] = (1 - alpha) * q_table[state_key][action] + alpha * (
                        reward + gamma * dwave_result.energy)

            state = next_state

        epsilon = max(min_epsilon, epsilon * decay_rate)

    return q_table

# Example usage
def main():
    # Load data
    data = read_csv_data('data/Historic_data_SPY.csv')

    # Detect support and resistance levels
    data = detect_support_resistance(data)

    # Train the model
    model, X_test, y_test, feature_columns = train_model(data)

    # Initialize simulated environment
    sim_env = SimulatedEnvironment(data, feature_columns)

    # Train Q-learning agent
    q_table = q_learning_train(sim_env, model, episodes=1)

    # Save the Q-table for use in the trading script
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

    # Save the model
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Evaluate model performance
    score = evaluate_model_performance(model, X_test, y_test)
    log_activity(f"Simulation completed. Model R^2 score: {score}")

if __name__ == "__main__":
    main()
