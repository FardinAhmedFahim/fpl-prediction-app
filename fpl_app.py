# fpl_app.py

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import combinations

# --- Data Acquisition ---
def fetch_latest_fpl_data(url="https://fantasy.premierleague.com/api/bootstrap-static/"):
    """
    Fetches the latest FPL data from the given API URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {url}: {e}")
        return None

fpl_data = fetch_latest_fpl_data()

if fpl_data is None:
    st.stop() # Stop the app if data fetching fails

# --- Data Preprocessing and Feature Engineering ---
def preprocess_data(data):
    """
    Cleans the data and creates features for prediction.
    """
    df = pd.DataFrame(data['elements'])

    df['form'] = pd.to_numeric(df['form'], errors='coerce')
    df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce') / 10.0 # now_cost is in tenths of a million

    df['points_per_million'] = np.where(df['now_cost'] > 0, df['total_points'] / df['now_cost'], 0)

    processed_df = df[['id', 'first_name', 'second_name', 'element_type', 'team', 'selected_by_percent',
                             'now_cost', 'value_season', 'total_points', 'form', 'points_per_million']].copy()

    processed_df.fillna(0, inplace=True)

    return processed_df

processed_fpl_data = preprocess_data(fpl_data)

# --- Model Selection and Training ---
# Using the processed_fpl_data for training as done in previous steps
features = ['selected_by_percent', 'now_cost', 'value_season', 'total_points', 'form', 'points_per_million']
target = 'total_points'

X = processed_fpl_data[features]
y = processed_fpl_data[target]

# Splitting data for evaluation purposes (not strictly for real-time prediction on new data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set for accuracy tracking
y_pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)


# --- Prediction for current and next gameweek ---
# Predict points for all players in the latest fetched data
prediction_features = ['selected_by_percent', 'now_cost', 'value_season', 'total_points', 'form', 'points_per_million']
X_predict = processed_fpl_data[prediction_features]
processed_fpl_data['predicted_total_points'] = model.predict(X_predict)


# --- Team Selection (best 11 and 15) ---
MAX_BUDGET = 100.0
MAX_PLAYERS_PER_TEAM = 3
GOALKEEPER = 1
DEFENDER = 2
MIDFIELDER = 3
FORWARD = 4

def select_best_15(players_df, max_budget, max_players_per_team):
    """
    Selects the best 15 players based on predicted points, budget, and team limits.
    """
    sorted_players = players_df.sort_values(by='predicted_total_points', ascending=False).copy()
    best_15_players = []
    current_budget = max_budget
    team_player_counts = {}
    position_counts_15 = {GOALKEEPER: 0, DEFENDER: 0, MIDFIELDER: 0, FORWARD: 0}

    for index, player in sorted_players.iterrows():
        player_cost = player['now_cost']
        player_team = player['team']
        player_position = player['element_type']

        if current_budget - player_cost < 0:
            continue
        if team_player_counts.get(player_team, 0) >= max_players_per_team:
            continue

        best_15_players.append(player)
        current_budget -= player_cost
        team_player_counts[player_team] = team_player_counts.get(player_team, 0) + 1
        position_counts_15[player_position] += 1

        if len(best_15_players) == 15:
            break

    return pd.DataFrame(best_15_players), current_budget, position_counts_15

best_15_df, remaining_budget, position_counts_15 = select_best_15(processed_fpl_data, MAX_BUDGET, MAX_PLAYERS_PER_TEAM)


def select_best_11(best_15_df):
    """
    Selects the best starting 11 from the best 15 players based on predicted points and valid formations.
    """
    valid_formations = [
        (1, 3, 2, 1), (1, 3, 3, 1), (1, 3, 4, 1), (1, 3, 5, 1),
        (1, 4, 2, 1), (1, 4, 3, 1), (1, 4, 4, 1), (1, 4, 5, 1),
        (1, 5, 2, 1), (1, 5, 3, 1), (1, 5, 4, 1),
        (1, 3, 2, 2), (1, 4, 2, 2), (1, 4, 3, 2), (1, 5, 2, 2),
        (1, 3, 3, 2), (1, 4, 4, 2), (1, 3, 4, 2), (1, 5, 3, 2), (1, 4, 5, 2), (1, 3, 5, 2), (1, 5, 4, 2),
        (1, 3, 1, 3), (1, 4, 1, 3), (1, 5, 1, 3), # Note: FPL requires min 2 midfielders, these might be invalid depending on exact rules
        (1, 3, 2, 3), (1, 4, 2, 3), (1, 5, 2, 3), (1, 3, 3, 3), (1, 4, 3, 3), (1, 5, 3, 3)
    ]
    valid_formations = [f for f in valid_formations if sum(f) == 11]

    goalkeepers_15 = best_15_df[best_15_df['element_type'] == GOALKEEPER].sort_values(by='predicted_total_points', ascending=False)
    defenders_15 = best_15_df[best_15_df['element_type'] == DEFENDER].sort_values(by='predicted_total_points', ascending=False)
    midfielders_15 = best_15_df[best_15_df['element_type'] == MIDFIELDER].sort_values(by='predicted_total_points', ascending=False)
    forwards_15 = best_15_df[best_15_df['element_type'] == FORWARD].sort_values(by='predicted_total_points', ascending=False)

    best_11_players = None
    max_predicted_points_11 = -1

    for gk_count, def_count, mid_count, fwd_count in valid_formations:
        if (len(goalkeepers_15) >= gk_count and
            len(defenders_15) >= def_count and
            len(midfielders_15) >= mid_count and
            len(forwards_15) >= fwd_count):

            gk_combinations = list(combinations(goalkeepers_15.iterrows(), gk_count))
            def_combinations = list(combinations(defenders_15.iterrows(), def_count))
            mid_combinations = list(combinations(midfielders_15.iterrows(), mid_count))
            fwd_combinations = list(combinations(forwards_15.iterrows(), fwd_count))

            for gk_combo in gk_combinations:
                for def_combo in def_combinations:
                    for mid_combo in mid_combinations:
                        for fwd_combo in fwd_combinations:
                            current_11 = list(gk_combo) + list(def_combo) + list(mid_combo) + list(fwd_combo)
                            current_predicted_points = sum([player[1]['predicted_total_points'] for player in current_11])

                            if current_predicted_points > max_predicted_points_11:
                                max_predicted_points_11 = current_predicted_points
                                best_11_players = [player[1] for player in current_11]

    return pd.DataFrame(best_11_players) if best_11_players else pd.DataFrame(), max_predicted_points_11

best_11_df, max_predicted_points_11 = select_best_11(best_15_df)


# --- Dashboard Development ---
st.title("FPL Predicted Best Teams")

# Display the best 11 team
st.header("Predicted Best Starting 11")
if not best_11_df.empty:
    st.dataframe(best_11_df[['first_name', 'second_name', 'element_type', 'team', 'now_cost', 'predicted_total_points']])
    st.subheader(f"Total Predicted Points for Best 11: {max_predicted_points_11:.2f}")
else:
    st.write("Could not select a valid Best 11 based on the selected 15 players and formations.")


# Display the best 15 squad
st.header("Predicted Best 15 Squad")
if not best_15_df.empty:
    st.dataframe(best_15_df[['first_name', 'second_name', 'element_type', 'team', 'now_cost', 'predicted_total_points']])
    st.subheader(f"Remaining Budget (Best 15): {remaining_budget:.2f}")
    st.subheader(f"Position Counts (Best 15): Goalkeepers: {position_counts_15[GOALKEEPER]}, Defenders: {position_counts_15[DEFENDER]}, Midfielders: {position_counts_15[MIDFIELDER]}, Forwards: {position_counts_15[FORWARD]}")
else:
    st.write("Could not select a Best 15 squad.")

# --- Accuracy Tracking ---
st.header("Model Accuracy (on Test Data)")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.info("Note: These metrics are based on a historical test set and may not reflect real-time prediction accuracy.")


# --- Visualization ---
st.header("Best Starting 11 Visualization")

if not best_11_df.empty:
    # Define field dimensions and key areas (simplified)
    field_width = 105
    field_height = 68
    penalty_box_width = 16.5
    penalty_box_height = 40.3
    six_yard_box_width = 5.5
    six_yard_box_height = 18.3
    goal_width = 7.32
    centre_circle_radius = 9.15

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw the pitch
    ax.add_patch(patches.Rectangle((0, 0), field_width, field_height, color='green', alpha=0.5))

    # Draw pitch lines
    ax.plot([field_width/2, field_width/2], [0, field_height], color='white') # Halfway line
    ax.add_patch(patches.Circle((field_width/2, field_height/2), centre_circle_radius, color='white', fill=False)) # Centre circle
    ax.plot([0, field_width], [field_height/2, field_height/2], color='white', linestyle='dashed') # Midline (approx)

    # Draw penalty boxes
    ax.add_patch(patches.Rectangle((0, (field_height - penalty_box_height) / 2), penalty_box_width, penalty_box_height, color='white', fill=False)) # Left penalty box
    ax.add_patch(patches.Rectangle((field_width - penalty_box_width, (field_height - penalty_box_height) / 2), penalty_box_width, penalty_box_height, color='white', fill=False)) # Right penalty box

    # Draw six-yard boxes
    ax.add_patch(patches.Rectangle((0, (field_height - six_yard_box_height) / 2), six_yard_box_width, six_yard_box_height, color='white', fill=False)) # Left six-yard box
    ax.add_patch(patches.Rectangle((field_width - six_yard_box_width, (field_height - six_yard_box_height) / 2), six_yard_box_width, six_yard_box_height, color='white', fill=False)) # Right six-yard box

    # Draw goals (simplified)
    ax.plot([0, 0], [(field_height - goal_width) / 2, (field_height + goal_width) / 2], color='white', linewidth=3) # Left goal
    ax.plot([field_width, field_width], [(field_height - goal_width) / 2, (field_height + goal_width) / 2], color='white', linewidth=3) # Right goal

    # Set axis limits and remove ticks
    ax.set_xlim(-5, field_width + 5)
    ax.set_ylim(-5, field_height + 5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Define player positions (approximate coordinates based on element_type)
    player_positions_viz = {
        GOALKEEPER: (5, field_height / 2),
        DEFENDER: (20, None),
        MIDFIELDER: (50, None),
        FORWARD: (80, None)
    }

    # Separate players by position for visualization
    gk_players_viz = best_11_df[best_11_df['element_type'] == GOALKEEPER]
    def_players_viz = best_11_df[best_11_df['element_type'] == DEFENDER]
    mid_players_viz = best_11_df[best_11_df['element_type'] == MIDFIELDER]
    fwd_players_viz = best_11_df[best_11_df['element_type'] == FORWARD]

    # Function to distribute players vertically for visualization
    def distribute_players_vertically_viz(players_df, base_y_start):
        num_players = len(players_df)
        if num_players == 0:
            return []
        spacing = field_height / (num_players + 1)
        return [(base_y_start + i * spacing) for i in range(num_players)]

    # Plot players and annotate
    for index, player in best_11_df.iterrows():
        pos_type = player['element_type']
        x_pos, y_pos_base = player_positions_viz[pos_type]

        if pos_type == GOALKEEPER:
            y_pos = y_pos_base
        elif pos_type == DEFENDER:
            y_positions = distribute_players_vertically_viz(def_players_viz, 0)
            y_pos = y_positions[def_players_viz.index.get_loc(index)]
        elif pos_type == MIDFIELDER:
            y_positions = distribute_players_vertically_viz(mid_players_viz, 0)
            y_pos = y_positions[mid_players_viz.index.get_loc(index)]
        elif pos_type == FORWARD:
            y_positions = distribute_players_vertically_viz(fwd_players_viz, 0)
            y_pos = y_positions[fwd_players_viz.index.get_loc(index)]

        ax.plot(x_pos, y_pos, 'o', color='red', markersize=10)
        annotation_text = f"{player['second_name']}\n({player['predicted_total_points']:.1f})"
        ax.annotate(annotation_text, (x_pos, y_pos), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='white')

    ax.set_title("Predicted Best Starting 11", color='white', fontsize=14)
    fig.patch.set_facecolor('#005C53')

    st.pyplot(fig)
else:
    st.write("Cannot visualize the Best 11 as no valid team was selected.")
