import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import os
from PIL import Image

# ------------------------------
# 1️⃣ Page config & theme
# ------------------------------
st.set_page_config(
    page_title="Ben 10 Battle Simulator",
    page_icon="🟢",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #000000;
    color: white;
}
h1, h2, h3 {
    color: #39ff14;
}
.stButton>button {
    background-color: #39ff14;
    color: black;
    border-radius: 8px;
    font-weight: bold;
}
.card {
    border: 2px solid #39ff14;
    padding: 15px;
    border-radius: 10px;
    background-color: #111111;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# 2️⃣ Load data & model
# ------------------------------
@st.cache_data
def load_data():
    aliens = pd.read_csv("data/processed/aliens_with_archetypes.csv")
    stats = pd.read_csv("data/ben10_aliens.csv")
    return aliens, stats

@st.cache_resource
def load_model():
    return joblib.load("model/battle_model.pkl")

aliens, alien_stats = load_data()
battle_model = load_model()

# ------------------------------
# 3️⃣ Helper functions
# ------------------------------
def build_battle_features(a, b):
    A = alien_stats[alien_stats["alien_name"] == a].iloc[0]
    B = alien_stats[alien_stats["alien_name"] == b].iloc[0]

    features = {
        "strength_diff": A["strength_level"] - B["strength_level"],
        "speed_diff": A["speed_level"] - B["speed_level"],
        "intelligence_diff": A["intelligence"] - B["intelligence"],
        "strength_gap": abs(A["strength_level"] - B["strength_level"]),
        "speed_gap": abs(A["speed_level"] - B["speed_level"]),
        "intelligence_gap": abs(A["intelligence"] - B["intelligence"]),
        "strength_ratio": A["strength_level"] / (B["strength_level"] + 1),
        "speed_ratio": A["speed_level"] / (B["speed_level"] + 1),
        "intelligence_ratio": A["intelligence"] / (B["intelligence"] + 1),
        "power_diff": (
            (A["strength_level"] + A["speed_level"]) -
            (B["strength_level"] + B["speed_level"])
        ),
        "total_diff": (
            (A["strength_level"] + A["speed_level"] + A["intelligence"]) -
            (B["strength_level"] + B["speed_level"] + B["intelligence"])
        ),
        "max_stat_diff": max(
            A["strength_level"], A["speed_level"], A["intelligence"]
        ) - max(
            B["strength_level"], B["speed_level"], B["intelligence"]
        ),
        "min_stat_diff": min(
            A["strength_level"], A["speed_level"], A["intelligence"]
        ) - min(
            B["strength_level"], B["speed_level"], B["intelligence"]
        ),
        "variance_diff": np.var(
            [A["strength_level"], A["speed_level"], A["intelligence"]]
        ) - np.var(
            [B["strength_level"], B["speed_level"], B["intelligence"]]
        ),
        "speed_advantage": int(A["speed_level"] > B["speed_level"]),
        "speed_dominance": (
            A["speed_level"] *
            (A["speed_level"] / (B["speed_level"] + 1))
        )
    }

    return pd.DataFrame([features])

def simulate_battle(a, b):
    X = build_battle_features(a, b)
    prob = battle_model.predict_proba(X)[0][1]
    winner = a if random.random() < prob else b
    return winner, prob

def load_description(alien_name):
    md_path = f"data/dataset/{alien_name}/info.md"
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()
    return "No description available."

def display_alien_card_full(alien_name):
    alien = aliens[aliens["alien_name"] == alien_name].iloc[0]
    
    # Image
    img_folder = f"data/dataset/{alien_name}"
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if img_files:
        img_path = os.path.join(img_folder, img_files[0])
        img = Image.open(img_path)
        st.image(img, width=200)
    
    # Description
    description = load_description(alien_name)
    
    st.markdown(f"""
    <div class="card">
        <h3>{alien_name}</h3>
        <p><b>Archetype:</b> {alien.archetype}</p>
        <p><b>Strength:</b> {alien.strength_level} &nbsp; 
           <b>Speed:</b> {alien.speed_level} &nbsp; 
           <b>Intelligence:</b> {alien.intelligence}</p>
        <p><b>Description:</b> {description[:200]}...</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# 4️⃣ App header
# ------------------------------
st.title("🟢 Ben 10 Battle Simulator")
st.subheader("ML-driven alien battles & tournament simulation")

# ------------------------------
# 5️⃣ 1v1 Battle Cards
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    alien_a = st.selectbox("Alien A", aliens["alien_name"])
    display_alien_card_full(alien_a)

with col2:
    alien_b = st.selectbox("Alien B", aliens["alien_name"], index=1)
    display_alien_card_full(alien_b)

if st.button("Simulate Battle"):
    winner, prob = simulate_battle(alien_a, alien_b)
    st.markdown(f"""
    <div class="card">
        <h2>🏆 Winner: {winner}</h2>
        <p>Win Probability for {alien_a}: {prob:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# 6️⃣ Tournament Simulator
# ------------------------------
st.divider()
st.header("🏆 Tournament Simulation")

num_runs = st.slider("Number of Tournaments", 50, 500, 200)

if st.button("Run Tournament"):
    winners = []

    for _ in range(num_runs):
        pool = aliens["alien_name"].tolist()
        random.shuffle(pool)

        while len(pool) > 1:
            next_round = []
            for i in range(0, len(pool), 2):
                if i + 1 >= len(pool):
                    next_round.append(pool[i])
                else:
                    w, _ = simulate_battle(pool[i], pool[i+1])
                    next_round.append(w)
            pool = next_round

        winners.append(pool[0])

    result = pd.Series(winners).value_counts()

    st.subheader("Most Dominant Aliens")
    st.bar_chart(result.head(10))
