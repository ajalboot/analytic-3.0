import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bottle import route, run, request, response, static_file
import plotly.graph_objects as go
from datetime import datetime
import umap

#RANDOM_STATE = 42

def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type, Accept'
        if request.method != 'OPTIONS':
            return fn(*args, **kwargs)
    return _enable_cors


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
        line_info = line.get('line')
        if line_info and 'properties' in line_info:
            properties = line_info['properties']
            trajectories.append(properties)
            line_ids.append(line_info.get('tid', None))
        else:
            continue

    if not trajectories:
        return None, "No valid trajectory data found"

    df = pd.DataFrame(trajectories)
    df['TID'] = line_ids
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop('TID', errors='ignore')
    return df, numeric_features

def showUmap(databaseName):
    start_time = datetime.now()

    df_trajectory, numeric_features_trajectory = preprocess_data(databaseName)
    if df_trajectory is None:
        return json.dumps({"error": numeric_features_trajectory})

    df_numeric = df_trajectory[numeric_features_trajectory].fillna(0)
    df_normalized = StandardScaler().fit_transform(df_numeric)

    #reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(df_normalized)

    df_trajectory[['D1', 'D2']] = embedding
    df_trajectory['Distance'] = np.linalg.norm(embedding, axis=1)

    df_trajectory['HoverText'] = (
        "<span style='color:white;'>TID: " + df_trajectory['TID'].astype(str) +
        "<br>Distance: " + df_trajectory['Distance'].round(2).astype(str) + "</span>"
    )

    fig = go.Figure(data=go.Scatter(
        x=df_trajectory['D1'],
        y=df_trajectory['D2'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_trajectory['Distance'],
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

    UMAP_distance_min = df_trajectory['Distance'].min()
    UMAP_distance_max = df_trajectory['Distance'].max()
    UMAP_distance_median = df_trajectory['Distance'].median()

    sorted_distances = df_trajectory['Distance'].sort_values().reset_index(drop=True)
    middle_index = len(sorted_distances) // 2
    UMAP_distance_middle = sorted_distances[middle_index]

    result = {
        "trajectory_data": fig.to_dict(),
        "line_ids": df_trajectory['TID'].tolist(),
        "features_trajectory": numeric_features_trajectory.tolist(),
        "trajectory_count": len(df_trajectory),
        "UMAP_distance_min": f"{UMAP_distance_min:.2f}",
        "UMAP_distance_max": f"{UMAP_distance_max:.2f}",
        "UMAP_distance_median": f"{UMAP_distance_median:.2f}",
        "UMAP_distance_middle": f"{UMAP_distance_middle:.2f}"
    }

    end_time = datetime.now()
    print('Platform(B):----------------------------------------------------------')
    print(f'Platform(B): UMAP Plot <Started>  : {start_time}')
    print(f'Platform(B): UMAP Plot <Finished> : {end_time}')
    print(f'Platform(B): UMAP processing time : {end_time - start_time}')
    print('Platform(B):----------------------------------------------------------')
    return json.dumps(convert_numpy_to_list(result))

