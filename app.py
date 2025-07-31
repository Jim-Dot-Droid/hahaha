
import streamlit as st
import pandas as pd
import numpy as np
import os

HISTORY_FILE = "history.csv"
RESULTS_FILE = "results.csv"
BALANCE_FILE = "sol_balance.txt"
MARTINGALE_FILE = "martingale_balance.txt"
INITIAL_BALANCE = 0.1
BET_AMOUNT = 0.01

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
    update_balance(prediction, actual)
    update_martingale(prediction, actual)

def normalize_input(value):
    if value > 10:
        return value / 100
    return value

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
    combined = max(0, min(combined, 1))
    return combined, 1 - combined

def get_balance():
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def get_martingale():
    if os.path.exists(MARTINGALE_FILE):
        with open(MARTINGALE_FILE, "r") as f:
            return float(f.read())
    return INITIAL_BALANCE

def update_balance(prediction, actual):
    balance = get_balance()
    if prediction == "Above":
        if actual > 2.0:
            balance += BET_AMOUNT
        else:
            balance -= BET_AMOUNT
        with open(BALANCE_FILE, "w") as f:
            f.write(str(balance))

def update_martingale(prediction, actual):
    balance = get_martingale()
    history = load_results()
    last_bet = BET_AMOUNT
    streak = 0

    # Find last martingale amount
    for i in reversed(range(len(history))):
        row = history.iloc[i]
        if row["prediction"] == "Above":
            if not row["correct"]:
                streak += 1
            else:
                break

    martingale_bet = BET_AMOUNT * (2 ** streak)

    if prediction == "Above":
        if actual > 2.0:
            balance += martingale_bet
        else:
            balance -= martingale_bet

        with open(MARTINGALE_FILE, "w") as f:
            f.write(str(balance))

def reset_balance():
    for f in [BALANCE_FILE, MARTINGALE_FILE]:
        if os.path.exists(f):
            os.remove(f)

def main():
    st.title("Crash Game Predictor with Sol + Martingale Tracker")

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

    st.subheader("💰 Sol Tracker")
    st.metric("Flat Betting Balance", f"{get_balance():.4f} SOL")
    st.metric("Martingale Balance", f"{get_martingale():.4f} SOL")
    st.caption("You start with 0.1 SOL. Each 'Above' prediction places a 0.01 SOL flat bet or Martingale doubling after losses.")

if __name__ == "__main__":
    main()
