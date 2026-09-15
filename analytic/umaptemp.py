import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bottle import request, response
import plotly.graph_objects as go
from datetime import datetime
import umap

RANDOM_STATE = 14  #22

# Enable CORS decorator
def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept'
        if request.method != 'OPTIONS':
            return fn(*args, **kwargs)
    return _enable_cors

# Efficient numpy to list conversion
def convert_numpy_to_list(data):
    if isinstance(data, (np.ndarray, np.float32)):
        return data.tolist() if isinstance(data, np.ndarray) else float(data)
    if isinstance(data, dict):
        return {k: convert_numpy_to_list(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_to_list(i) for i in data]
    return data

def preprocess_data(databaseName):
    """Load and preprocess the data into a DataFrame."""
    try:
        with open(databaseName) as f:
            data = json.load(f)
    except Exception as e:
        return None, f"Error loading data: {e}"

    trajectories, line_ids = [], []
    for line in data:
        # Safely check if 'line' and 'properties' exist
        line_info = line.get('line')
        if line_info and 'properties' in line_info:
            properties = line_info['properties']
            trajectories.append(properties)
            line_ids.append(line_info.get('tid', None))
        else:
            # Skip entries that don't have the expected structure
            continue

    # Ensure that data is not empty after filtering
    if not trajectories:
        return None, "No valid trajectory data found"

    # Create DataFrame and filter numeric features
    df = pd.DataFrame(trajectories)
    df['TID'] = line_ids
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop('TID', errors='ignore')
    return df, numeric_features

def showUmap(databaseName):
    """Generate UMAP projection and Plotly visualization."""
    start_time = datetime.now()
    print(f'Generate UMAP Plot <Started> at {start_time}')

    # Load and preprocess data
    df_trajectory, numeric_features_trajectory = preprocess_data(databaseName)
    if df_trajectory is None:
        return json.dumps({"error": numeric_features_trajectory})

    # Handle missing values and normalization efficiently
    df_numeric = df_trajectory[numeric_features_trajectory].fillna(0)
    df_normalized = StandardScaler().fit_transform(df_numeric)

    # Using UMAP with a random state for reproducibility
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    #reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(df_normalized)

    # Calculate only the necessary columns and avoid reprocessing
    df_trajectory[['D1', 'D2']] = embedding
    df_trajectory['Distance'] = np.linalg.norm(embedding, axis=1)

    df_trajectory['HoverText'] = (
        "<span style='color:white;'>TID: " + df_trajectory['TID'].astype(str) +
        "<br>Distance: " + df_trajectory['Distance'].round(2).astype(str) + "</span>"
    )

    # Create scatter plot without unnecessary colorbar computation
    fig = go.Figure(data=go.Scatter(
        x=df_trajectory['D1'],
        y=df_trajectory['D2'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_trajectory['Distance'],  # You can keep color but remove colorbar for performance
            showscale=False,
            opacity=0.7,
        ),
        text=df_trajectory['HoverText'],
        hoverinfo='text'
    ))

    fig.update_layout(
        title='UMAP Projection',
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
    )

    # Collect only max and min distances for client-side use
    UMAP_distance_min = df_trajectory['Distance'].min()
    UMAP_distance_max = df_trajectory['Distance'].max()

    result = {
        "trajectory_data": fig.to_dict(),
        "line_ids": df_trajectory['TID'].tolist(),
        "features_trajectory": numeric_features_trajectory.tolist(),
        "trajectory_count": len(df_trajectory),
        "UMAP_distance_min": f"{UMAP_distance_min:.2f}",
        "UMAP_distance_max": f"{UMAP_distance_max:.2f}"
    }

    end_time = datetime.now()
    print(f'Generate UMAP Plot <Finished> at {end_time}')
    print(f'Total processing time: {end_time - start_time}')

    return json.dumps(convert_numpy_to_list(result))
