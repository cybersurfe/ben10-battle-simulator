import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict, List

# ------------------------------
#  Page config & theme
# ------------------------------
st.set_page_config(
    page_title="Ben 10 Battle Simulator",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS
st.markdown("""
<style>
/* Main theme */
body {
    background-color: #000000;
    color: white;
}

h1, h2, h3 {
    color: #39ff14;
    text-shadow: 0 0 10px #39ff14;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(135deg, #39ff14 0%, #2ecc71 100%);
    color: black;
    border-radius: 12px;
    font-weight: bold;
    border: none;
    padding: 0.5rem 2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(57, 255, 20, 0.3);
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(57, 255, 20, 0.5);
}

/* Card styling */
.card {
    border: 2px solid #39ff14;
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #111111 0%, #1a1a1a 100%);
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(57, 255, 20, 0.2);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(57, 255, 20, 0.4);
}

/* Winner announcement */
.winner-card {
    border: 3px solid #39ff14;
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    text-align: center;
    animation: pulse 2s infinite;
    box-shadow: 0 0 30px rgba(57, 255, 20, 0.6);
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 30px rgba(57, 255, 20, 0.6); }
    50% { box-shadow: 0 0 50px rgba(57, 255, 20, 0.9); }
}

/* Stats display */
.stat-box {
    background-color: #1a1a1a;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #39ff14;
    margin: 5px 0;
}

/* Sidebar */
.css-1d391kg {
    background-color: #0a0a0a;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background-color: #39ff14;
}

/* Metrics */
.css-1xarl3l {
    color: #39ff14 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
#  Load data & model with error handling
# ------------------------------
@st.cache_data
def load_data():
    """Load alien data with proper error handling"""
    try:
        # Try multiple possible paths
        possible_paths = [
            ("data/processed/aliens_with_archetypes.csv", "data/ben10_aliens.csv"),
            ("../data/processed/aliens_with_archetypes.csv", "../data/ben10_aliens.csv"),
        ]
        
        for aliens_path, stats_path in possible_paths:
            if os.path.exists(aliens_path) and os.path.exists(stats_path):
                aliens = pd.read_csv(aliens_path)
                stats = pd.read_csv(stats_path)
                return aliens, stats
        
        # If files don't exist, create sample data
        st.warning(" Data files not found. Using sample data for demonstration.")
        return create_sample_data()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

@st.cache_resource
def load_model():
    """Load battle model with error handling"""
    try:
        possible_paths = [
            "model/battle_model.pkl",
            "../model/battle_model.pkl",
            "battle_model.pkl"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return joblib.load(path)
        
        st.error(" Model file not found. Please ensure battle_model.pkl is in the correct location.")
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data if files are missing"""
    sample_aliens = pd.DataFrame({
        'alien_name': ['Four Arms', 'XLR8', 'Grey Matter', 'Heatblast', 'Diamondhead'],
        'archetype': ['Tank', 'Speedster', 'Genius', 'Ranged', 'Tank'],
        'strength_level': [9, 4, 2, 6, 8],
        'speed_level': [3, 10, 3, 5, 4],
        'intelligence': [4, 5, 10, 6, 5]
    })
    return sample_aliens, sample_aliens.copy()

# Initialize session state
if 'battle_history' not in st.session_state:
    st.session_state.battle_history = []
if 'tournament_results' not in st.session_state:
    st.session_state.tournament_results = None

# Load data
aliens, alien_stats = load_data()
battle_model = load_model()

# ------------------------------
# Enhanced Helper functions
# ------------------------------
def build_battle_features(a: str, b: str) -> pd.DataFrame:
    """Build feature vector for battle prediction"""
    try:
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
    except Exception as e:
        st.error(f"Error building features: {e}")
        return pd.DataFrame()

def simulate_battle(a: str, b: str) -> Tuple[str, float]:
    """Simulate a battle between two aliens"""
    if battle_model is None:
        # Fallback to simple stat comparison
        A = alien_stats[alien_stats["alien_name"] == a].iloc[0]
        B = alien_stats[alien_stats["alien_name"] == b].iloc[0]
        
        score_a = A["strength_level"] + A["speed_level"] + A["intelligence"]
        score_b = B["strength_level"] + B["speed_level"] + B["intelligence"]
        
        prob = score_a / (score_a + score_b)
        winner = a if random.random() < prob else b
        return winner, prob
    
    X = build_battle_features(a, b)
    if X.empty:
        return a, 0.5
        
    prob = battle_model.predict_proba(X)[0][1]
    winner = a if random.random() < prob else b
    return winner, prob

def load_description(alien_name: str) -> str:
    """Load alien description from markdown file"""
    possible_paths = [
        f"data/dataset/{alien_name}/info.md",
        f"../data/dataset/{alien_name}/info.md"
    ]
    
    for md_path in possible_paths:
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
    
    return "A powerful alien from the Omnitrix with unique abilities."

def get_alien_image_path(alien_name: str) -> str:
    """Get path to alien image"""
    possible_folders = [
        f"data/dataset/{alien_name}",
        f"../data/dataset/{alien_name}"
    ]
    
    for img_folder in possible_folders:
        if os.path.exists(img_folder):
            img_files = [f for f in os.listdir(img_folder) 
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if img_files:
                return os.path.join(img_folder, img_files[0])
    
    return None

def display_alien_stats_radar(alien_name: str):
    """Display alien stats as a radar chart"""
    alien = alien_stats[alien_stats["alien_name"] == alien_name].iloc[0]
    
    categories = ['Strength', 'Speed', 'Intelligence']
    values = [
        alien["strength_level"],
        alien["speed_level"],
        alien["intelligence"]
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(57, 255, 20, 0.3)',
        line=dict(color='#39ff14', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                gridcolor='rgba(57, 255, 20, 0.3)'
            ),
            angularaxis=dict(
                gridcolor='rgba(57, 255, 20, 0.3)'
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#39ff14'),
        height=250,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def display_alien_card_compact(alien_name: str, col):
    """Display compact alien card"""
    with col:
        alien = aliens[aliens["alien_name"] == alien_name].iloc[0]
        
        # Image
        img_path = get_alien_image_path(alien_name)
        if img_path:
            try:
                img = Image.open(img_path)
                st.image(img, width=200)
            except Exception:
                st.write("✓")
        else:
            st.markdown(f"<h1 style='text-align: center;'></h1>", unsafe_allow_html=True)
        
        # Stats
        st.markdown(f"""
        <div class="card">
            <h3 style='text-align: center;'>{alien_name}</h3>
            <p style='text-align: center;'><b>Type:</b> {alien.archetype}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Radar chart
        fig = display_alien_stats_radar(alien_name)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stat bars
        st.markdown(f"""
        <div class="stat-box">
            <b>Strength:</b> {alien.strength_level}/10
        </div>
        <div class="stat-box">
            <b> Speed:</b> {alien.speed_level}/10
        </div>
        <div class="stat-box">
            <b> Intelligence:</b> {alien.intelligence}/10
        </div>
        """, unsafe_allow_html=True)

def create_comparison_chart(alien_a: str, alien_b: str):
    """Create comparison bar chart"""
    A = alien_stats[alien_stats["alien_name"] == alien_a].iloc[0]
    B = alien_stats[alien_stats["alien_name"] == alien_b].iloc[0]
    
    stats = ['Strength', 'Speed', 'Intelligence']
    values_a = [A["strength_level"], A["speed_level"], A["intelligence"]]
    values_b = [B["strength_level"], B["speed_level"], B["intelligence"]]
    
    fig = go.Figure(data=[
        go.Bar(name=alien_a, x=stats, y=values_a, marker_color='#39ff14'),
        go.Bar(name=alien_b, x=stats, y=values_b, marker_color='#ff6b6b')
    ])
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,17,17,0.8)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Head-to-Head Stats", font=dict(color='#39ff14'))
    )
    
    return fig

# ------------------------------
# Sidebar Configuration
# ------------------------------
with st.sidebar:
    st.markdown("##  Settings")
    
    st.markdown("###  Battle Mode")
    battle_mode = st.radio(
        "Choose mode:",
        ["1v1 Battle", "Tournament", "Battle History", "Stats Explorer"],
        label_visibility="collapsed"
    )
    
    if battle_mode == "Tournament":
        st.markdown("###  Tournament Settings")
        num_runs = st.slider("Number of Simulations", 50, 1000, 200)
        include_stats = st.checkbox("Show detailed statistics", value=True)
    
    st.divider()
    
    st.markdown("###  Quick Stats")
    st.metric("Total Aliens", len(aliens))
    st.metric("Battle Simulations", len(st.session_state.battle_history))
    
    if st.session_state.battle_history:
        most_common = pd.Series([b['winner'] for b in st.session_state.battle_history]).mode()[0]
        st.metric("Most Victorious", most_common)
    
    st.divider()
    
    if st.button(" Clear History"):
        st.session_state.battle_history = []
        st.session_state.tournament_results = None
        st.rerun()

# ------------------------------
# Main Content
# ------------------------------
st.title("Ben 10 Battle Simulator")
st.markdown("""
<p style='font-size: 1.2em; color: #39ff14;'>
ML-Powered Alien Combat Analysis & Tournament System
</p>
""", unsafe_allow_html=True)

# ------------------------------
# 6️ Mode: 1v1 Battle
# ------------------------------
if battle_mode == "1v1 Battle":
    st.header(" One-on-One Battle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alien_a = st.selectbox(" Select Fighter A", aliens["alien_name"].sort_values(), key="alien_a")
    
    with col2:
        alien_b = st.selectbox(" Select Fighter B", 
                              aliens["alien_name"].sort_values(), 
                              index=min(1, len(aliens)-1), 
                              key="alien_b")
    
    # Display aliens
    col1, col2 = st.columns(2)
    display_alien_card_compact(alien_a, col1)
    display_alien_card_compact(alien_b, col2)
    
    # Comparison chart
    st.plotly_chart(create_comparison_chart(alien_a, alien_b), use_container_width=True)
    
    # Battle button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("SIMULATE BATTLE", use_container_width=True):
            with st.spinner("Battle in progress..."):
                winner, prob = simulate_battle(alien_a, alien_b)
                
                # Store in history
                st.session_state.battle_history.append({
                    'alien_a': alien_a,
                    'alien_b': alien_b,
                    'winner': winner,
                    'probability': prob
                })
                
                # Display result
                st.markdown("<br>", unsafe_allow_html=True)
                
                if winner == alien_a:
                    win_prob = prob
                    loser_prob = 1 - prob
                else:
                    win_prob = 1 - prob
                    loser_prob = prob
                
                st.markdown(f"""
                <div class="winner-card">
                    <h1> WINNER: {winner}</h1>
                    <p style='font-size: 1.5em; margin: 20px 0;'>Victory Probability: {win_prob:.1%}</p>
                    <p style='font-size: 1.2em; color: #888;'>vs {alien_a if winner == alien_b else alien_b} ({loser_prob:.1%})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show probability breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric( alien_a + " Win Chance", f"{prob:.1%}")
                with col2:
                    st.metric("Match Certainty", f"{abs(2*prob - 1):.1%}")
                with col3:
                    st.metric(alien_b + " Win Chance", f"{1-prob:.1%}")

# ------------------------------
#  Mode: Tournament
# ------------------------------
elif battle_mode == "Tournament":
    st.header(" Tournament Simulation")
    
    st.info(f"Simulating {num_runs} tournaments with {len(aliens)} aliens")
    
    if st.button("START TOURNAMENT", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        winners = []
        alien_names = aliens["alien_name"].tolist()
        
        for i in range(num_runs):
            # Update progress
            progress_bar.progress((i + 1) / num_runs)
            status_text.text(f"Running tournament {i+1}/{num_runs}...")
            
            # Run single tournament
            pool = alien_names.copy()
            random.shuffle(pool)
            
            while len(pool) > 1:
                next_round = []
                for j in range(0, len(pool), 2):
                    if j + 1 >= len(pool):
                        next_round.append(pool[j])
                    else:
                        w, _ = simulate_battle(pool[j], pool[j+1])
                        next_round.append(w)
                pool = next_round
            
            winners.append(pool[0])
        
        progress_bar.empty()
        status_text.empty()
        
        # Store results
        result = pd.Series(winners).value_counts()
        st.session_state.tournament_results = result
        
        # Display results
        st.success(f" Completed {num_runs} tournaments!")
        
        # Top performers
        st.subheader(" Top 10 Champions")
        
        top_10 = result.head(10)
        
        # Create bar chart
        fig = px.bar(
            x=top_10.values,
            y=top_10.index,
            orientation='h',
            labels={'x': 'Championships Won', 'y': 'Alien'},
            color=top_10.values,
            color_continuous_scale=['#1a1a1a', '#39ff14']
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(17,17,17,0.8)',
            font=dict(color='white'),
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        if include_stats:
            st.subheader(" Tournament Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Unique Winners", len(result))
            with col2:
                st.metric("Dominant Champion", result.index[0])
            with col3:
                st.metric("Win Rate", f"{result.iloc[0]/num_runs:.1%}")
            with col4:
                concentration = result.iloc[0] / result.sum()
                st.metric("Dominance Factor", f"{concentration:.1%}")
            
            # Archetype analysis
            st.subheader(" Performance by Archetype")
            
            winner_df = result.reset_index()
            winner_df.columns = ["alien_name", "wins"]
            winner_df = winner_df.merge(aliens[["alien_name", "archetype"]], on="alien_name")
            
            archetype_wins = winner_df.groupby("archetype")["wins"].sum().sort_values(ascending=False)
            
            fig = px.pie(
                values=archetype_wins.values,
                names=archetype_wins.index,
                title="Championship Distribution by Type",
                color_discrete_sequence=px.colors.sequential.Greens
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
#  Mode: Battle History
# ------------------------------
elif battle_mode == "Battle History":
    st.header(" Battle History")
    
    if not st.session_state.battle_history:
        st.info("No battles recorded yet. Start some 1v1 battles to build history!")
    else:
        # Create dataframe
        history_df = pd.DataFrame(st.session_state.battle_history)
        
        st.metric("Total Battles", len(history_df))
        
        # Recent battles
        st.subheader("Recent Battles")
        
        for idx, battle in enumerate(reversed(st.session_state.battle_history[-10:])):
            with st.expander(f"Battle {len(st.session_state.battle_history) - idx}: {battle['alien_a']} vs {battle['alien_b']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Fighter A:** {battle['alien_a']}")
                with col2:
                    st.write(f"**Fighter B:** {battle['alien_b']}")
                with col3:
                    st.write(f"**Winner:**  {battle['winner']}")
                
                st.progress(battle['probability'])
                st.caption(f"Win probability: {battle['probability']:.1%}")
        
        # Statistics
        st.subheader("Win Rate Analysis")
        
        all_aliens_in_history = list(history_df['alien_a']) + list(history_df['alien_b'])
        participation = pd.Series(all_aliens_in_history).value_counts()
        
        wins = history_df['winner'].value_counts()
        
        win_rate_df = pd.DataFrame({
            'Battles': participation,
            'Wins': wins
        }).fillna(0)
        
        win_rate_df['Win Rate'] = (win_rate_df['Wins'] / win_rate_df['Battles'] * 100).round(1)
        win_rate_df = win_rate_df.sort_values('Win Rate', ascending=False)
        
        st.dataframe(win_rate_df.head(10), use_container_width=True)

# ------------------------------
# Mode: Stats Explorer
# ------------------------------
elif battle_mode == "Stats Explorer":
    st.header("Alien Statistics Explorer")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        archetype_filter = st.multiselect(
            "Filter by Archetype",
            options=aliens["archetype"].unique(),
            default=aliens["archetype"].unique()
        )
    
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Name", "Strength", "Speed", "Intelligence", "Total Power"]
        )
    
    # Filter data
    filtered_aliens = aliens[aliens["archetype"].isin(archetype_filter)].copy()
    filtered_aliens["total_power"] = (
        filtered_aliens["strength_level"] + 
        filtered_aliens["speed_level"] + 
        filtered_aliens["intelligence"]
    )
    
    # Sort
    sort_map = {
        "Name": "alien_name",
        "Strength": "strength_level",
        "Speed": "speed_level",
        "Intelligence": "intelligence",
        "Total Power": "total_power"
    }
    
    filtered_aliens = filtered_aliens.sort_values(sort_map[sort_by], ascending=False)
    
    # Display
    st.subheader(f"Showing {len(filtered_aliens)} aliens")
    
    # 3D scatter plot
    fig = px.scatter_3d(
        filtered_aliens,
        x='strength_level',
        y='speed_level',
        z='intelligence',
        color='archetype',
        hover_name='alien_name',
        title='Alien Stats Distribution (3D)',
        labels={
            'strength_level': 'Strength',
            'speed_level': 'Speed',
            'intelligence': 'Intelligence'
        }
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top aliens by category
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Strongest")
        top_strength = filtered_aliens.nlargest(5, 'strength_level')[['alien_name', 'strength_level']]
        for idx, row in top_strength.iterrows():
            st.write(f"**{row['alien_name']}**: {row['strength_level']}/10")
    
    with col2:
        st.subheader("Fastest")
        top_speed = filtered_aliens.nlargest(5, 'speed_level')[['alien_name', 'speed_level']]
        for idx, row in top_speed.iterrows():
            st.write(f"**{row['alien_name']}**: {row['speed_level']}/10")
    
    with col3:
        st.subheader("Smartest")
        top_intel = filtered_aliens.nlargest(5, 'intelligence')[['alien_name', 'intelligence']]
        for idx, row in top_intel.iterrows():
            st.write(f"**{row['alien_name']}**: {row['intelligence']}/10")

# ------------------------------
# Footer
# ------------------------------
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Ben 10 Battle Simulator v2.0 | Powered by Machine Learning</p>
    <p style='font-size: 0.9em;'>Omnitrix technology simulated | All alien abilities considered</p>
</div>
""", unsafe_allow_html=True)
