# Quantum_Finance
Quantum Finance Trading Bot
This repository contains two Python scripts: Human_QC.py and Testing_Simulation.py. These scripts implement a trading bot using Q-learning for reinforcement learning and the Alpaca API for trading simulations. The Human_QC.py script incorporates quantum computing via the D-Wave quantum annealer for solving QUBO problems. The Testing_Simulation.py script simulates trading using historical and real-time data fetched from Alpaca.

Files and Functionality
Human_QC.py
This script trains a Q-learning model to make trading decisions and integrates quantum computing for optimization.

Key Functions:
log_activity(message): Logs and prints messages for real-time feedback.
read_csv_data(file_path): Reads CSV data from a specified file path.
preprocess_data(data): Preprocesses the data by converting date columns to ordinal numbers and removing non-numeric columns.
detect_support_resistance(data): Adds support and resistance levels to the data.
train_model(data): Trains a RandomForestRegressor model on the preprocessed data.
SimulatedEnvironment: Class to simulate a trading environment with methods for resetting, stepping through the simulation, and calculating rewards.
analyze_sentiment(state): Analyzes sentiment based on the state.
solve_qubo_with_dwave(bqm): Solves a QUBO problem using D-Wave's quantum annealer.
q_learning_train(env, model, episodes, alpha, gamma, epsilon, min_epsilon, decay_rate): Trains the Q-learning model using the simulated environment and quantum computing for optimization.
main(): Main function to run the training and save the Q-table and model.

Testing_Simulation.py
This script uses the trained Q-learning model and Alpaca API to simulate trading for a given period.

Key Functions:
log_activity(message): Logs and prints messages for real-time feedback.
debug_activity(message): Logs debug messages for detailed trace.
get_alpaca_data(symbol, start_date, end_date): Fetches historical data from Alpaca API.
preprocess_data(data): Preprocesses the data by removing non-numeric columns and NaN values.
detect_support_resistance(data): Adds support and resistance levels to the data.
SimulatedEnvironment: Class to simulate a trading environment with methods for resetting, stepping through the simulation, and calculating rewards.
analyze_sentiment(state): Analyzes sentiment based on the state.
make_decision(state, q_table, feature_columns): Makes trading decisions using the Q-table.
main(): Main function to load the trained model and Q-table, fetch data from Alpaca, and simulate trading for a specified period.
Requirements
Python 3.7 or later
pandas
numpy
scikit-learn
alpaca-trade-api
dimod
dwave-system
Setup
Clone the repository.
Install the required packages using pip:
bash
Copy code
pip install pandas numpy scikit-learn alpaca-trade-api dimod dwave-system
Set up your Alpaca API credentials in Testing_Simulation.py.
Set up your D-Wave API credentials in Human_QC.py.
Usage
Training the Model with Quantum Computing
Run the Human_QC.py script to train the Q-learning model and save the Q-table and trained model:
bash
Copy code
python Human_QC.py
Simulating Trading with Alpaca Data
Run the Testing_Simulation.py script to simulate trading using the trained model and real-time data from Alpaca:
bash
Copy code
python Testing_Simulation.py
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Alpaca API
D-Wave Systems
This README provides an overview of the functionality and setup of the trading bot program. For detailed explanations of individual functions and classes, refer to the comments and docstrings within the code files.
