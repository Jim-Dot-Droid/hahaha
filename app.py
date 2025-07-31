import streamlit as st
import pandas as pd
import numpy as np
import os

# File paths
HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
FLAT_FILE = "sol_balance.txt"
FIXED_FILE = "fixed_balance.txt"

# Constants
INITIAL_BALANCE = 0.1
FLAT_BET = 0.01
FIXED_BET = 0.02

# Load/Save Functions
@st.cache_data
def load_csv(file):
    df = pd.read_csv(file)
    return df['multiplier'].tolist()

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        return df['multiplier'].tolist()
    return []

def save_history(data):
    pd.DataFrame({'multiplier': data}).to_csv(HISTORY_FILE, index=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(columns=['prediction', 'actual', 'correct'])

def save_result(prediction, actual):
    correct = ((prediction == "Above") and actual > 2.0) or ((prediction == "Under") and actual <= 2.0)
    result_df = load_results()
    result_df.loc[len(result_df)] = [prediction, actual, correct]
    result_df.to_csv(RESULTS_FILE, index=False)
    update_flat_balance(prediction, actual)
    if prediction == "Above":
        update_fixed_balance(actual)

# Balance Handlers
def get_flat_balance():
    if os.path.exists(FLAT_FILE):
        with open(FLAT_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def get_fixed_balance():
    if os.path.exists(FIXED_FILE):
        with open(FIXED_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def update_flat_balance(prediction, actual):
    balance = get_flat_balance()
    if prediction == "Above":
        balance += FLAT_BET if actual > 2.0 else -FLAT_BET
        with open(FLAT_FILE, "w") as f:
            f.write(str(balance))

def update_fixed_balance(actual):
    balance = get_fixed_balance()
    balance += FIXED_BET if actual > 2.0 else -FIXED_BET
    with open(FIXED_FILE, "w") as f:
        f.write(str(balance))

# Logic
def normalize_input(value):
    return value / 100 if value > 10 else value

def compute_improved_confidence(data, threshold=2.0, trend_window=10):
    if not data:
        return 0.5, 0.5
    data = np.array(data)
    n = len(data)
    weights = np.linspace(0.5, 1.0, n)
    base_score = np.average((data > threshold).astype(int), weights=weights)

    recent = data[-trend_window:] if n >= trend_window else data
    trend_score = np.mean(recent > threshold) if len(recent) > 0 else 0.5

    streak = 1
    for i in range(n-2, -1, -1):
        if (data[i] > threshold and data[i+1] > threshold) or (data[i] <= threshold and data[i+1] <= threshold):
            streak += 1
        else:
            break
    streak_impact = min(streak * 0.02, 0.2)
    streak_score = streak_impact if data[-1] <= threshold else -streak_impact

    combined = (0.4 * base_score) + (0.45 * trend_score) + (0.15 * (0.5 + streak_score))
    return max(0, min(combined, 1)), 1 - max(0, min(combined, 1))

def reset_balance():
    for f in [FLAT_FILE, FIXED_FILE]:
        if os.path.exists(f):
            os.remove(f)

# Streamlit App
def main():
    st.title("Crash Game Predictor with SOL Flat & Fixed Betting")

    if "history" not in st.session_state:
        st.session_state.history = load_history()

    uploaded_file = st.file_uploader("Upload multipliers CSV", type=["csv"])
    if uploaded_file:
        st.session_state.history = load_csv(uploaded_file)
        save_history(st.session_state.history)
        st.success(f"Loaded {len(st.session_state.history)} multipliers from file.")

    st.subheader("Manual Input")
    new_val = st.text_input("Enter a new multiplier or percentage (e.g., 1.87 or 187)")
    if st.button("Add"):
        try:
            val = float(new_val)
            val = normalize_input(val)
            if "last_prediction" in st.session_state:
                save_result(st.session_state.last_prediction, val)
                del st.session_state.last_prediction
            st.session_state.history.append(val)
            save_history(st.session_state.history)
            st.success(f"Added {val}x")
        except:
            st.error("Invalid number.")

    if st.button("Reset All Data"):
        st.session_state.history = []
        save_history([])
        reset_balance()
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        st.success("All data and results cleared.")

    if st.session_state.history:
        data = st.session_state.history
        st.write(f"Entries so far: **{len(data)}**")
        st.progress(min(len(data) / 20, 1.0))

        above_conf, under_conf = compute_improved_confidence(data)
        st.subheader("Prediction")
        if above_conf > under_conf:
            st.session_state.last_prediction = "Above"
            st.write(f"Prediction: **Above 200%** ({above_conf:.1%} confidence)")
        else:
            st.session_state.last_prediction = "Under"
            st.write(f"Prediction: **Under 200%** ({under_conf:.1%} confidence)")
    else:
        st.write("Add data to get prediction.")

    st.subheader("Accuracy Tracker")
    results_df = load_results()
    if not results_df.empty:
        total = len(results_df)
        correct = results_df['correct'].sum()
        acc = correct / total if total else 0
        st.metric("Total Predictions", total)
        st.metric("Correct Predictions", int(correct))
        st.metric("Accuracy Rate", f"{acc:.1%}")
        st.dataframe(results_df[::-1].reset_index(drop=True))
    else:
        st.write("No predictions have been verified yet.")

    st.subheader("💰 SOL Balance Tracker")
    st.metric("Flat Bet Balance (0.01 SOL per 'Above')", f"{get_flat_balance():.4f} SOL")
    st.metric("Fixed Bet Balance (0.02 SOL only when predicted 'Above')", f"{get_fixed_balance():.4f} SOL")
    st.caption("You start with 0.1 SOL. Flat and fixed bets are only made when the model predicts 'Above' 2.0x.")

if __name__ == "__main__":
    main()

