#!/usr/bin/python

import os

# Limit the number of CPU cores used by joblib/scikit-learn
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


# ============================================================
# Import necessary libraries
# ============================================================

from bottle import (
    route,
    request,
    response,
    get,
    static_file,
    default_app,
    error,
    run
)

from analytic import (
    al_strategies,
    classify_all,
    trajectory_manager,
    http_get_solr_data,
    umaptemp
)

import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import f1_score

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

from matplotlib.figure import Figure

from datetime import datetime


# ============================================================
# Function to enable Cross-Origin Resource Sharing (CORS)
# ============================================================

def enable_cors(fn):

    def _enable_cors(*args, **kwargs):

        response.headers['Access-Control-Allow-Origin'] = '*'

        response.headers['Access-Control-Allow-Methods'] = (
            'GET, POST, PUT, OPTIONS'
        )

        response.headers['Access-Control-Allow-Headers'] = (
            'Origin, Accept, Content-Type, X-Requested-With, '
            'X-CSRF-Token'
        )

        if request.method != 'OPTIONS':
            return fn(*args, **kwargs)

    return _enable_cors


# ============================================================
# Application configuration
# ============================================================

application = default_app()

WorkDir = './'
DataDir = '../data/'

# WorkDir = 'C:/_Portable/_Dev/analytic-2.0-main/V15/'
# DataDir = 'C:/_Portable/_Dev/analytic-2.0-main/Data/'


# ============================================================
# Serve the main platform page
# ============================================================

@route('/')
@enable_cors
def send_index():
    return static_file(
        'home.html',
        WorkDir
    )


@route('/index')
@enable_cors
def send_index():
    return static_file(
        'index.html',
        WorkDir
    )


# ============================================================
# Serve CSS
# ============================================================

@route('/css')
@enable_cors
def send_css():

    response.body = static_file(
        'style.min.css',
        WorkDir + 'dist/css'
    )

    response.content_type = 'text/css'

    return response


# ============================================================
# Serve JavaScript
# ============================================================

@route('/analytic_js')
@enable_cors
def send_analytic_js():

    response.body = static_file(
        'analytic.min.js',
        WorkDir + 'dist/js'
    )

    response.content_type = 'text/javascript'

    return response


# ============================================================
# Serve images
# ============================================================

@get('/images/<image_name>', method='GET')
@enable_cors
def images(image_name):

    response.body = static_file(
        image_name,
        root=WorkDir + 'images'
    )

    return response


# ============================================================
# Dataset configuration
# ============================================================

BagSize = 20


# ============================================================
# Retrieve GeoJSON data for the Geolife dataset
# ============================================================

@get('/data/geolife', method='GET')
@enable_cors
def geolife_geojson():

    response.headers['Access-Control-Allow-Origin'] = '*'

    out = trajectory_manager.get_random_trajectories(
        DataDir + 'geolife.json',
        BagSize
    )

    response.body = out

    return response


# ============================================================
# Retrieve GeoJSON data for the fishing vessels dataset
# ============================================================

@get('/data/fishingvessels', method='GET')
def fishing_geojson():

    response.headers['Access-Control-Allow-Origin'] = '*'

    out = trajectory_manager.get_random_trajectories(
        DataDir + 'fishingvessels.json',
        BagSize
    )

    response.body = out

    return response


# ============================================================
# Retrieve GeoJSON data for the hurricanes dataset
# ============================================================

@get('/data/hurricanes', method='GET')
def hurricanes_geojson():

    response.headers['Access-Control-Allow-Origin'] = '*'

    out = trajectory_manager.get_random_trajectories(
        DataDir + 'hurricanes.json',
        BagSize
    )

    response.body = out

    return response


# ============================================================
# Retrieve GeoJSON data for the animals dataset
# ============================================================

@get('/data/animals', method='GET')
@enable_cors
def animals_geojson():

    response.headers['Access-Control-Allow-Origin'] = '*'

    out = trajectory_manager.get_random_trajectories(
        DataDir + 'animals.json',
        BagSize
    )

    response.body = out

    return response


# ============================================================
# Run active learning strategy to label trajectories
# ============================================================

@route('/al_run', method='POST')
def al_strategy():

    response.headers['Access-Control-Allow-Origin'] = '*'

    database = ''
    trajectories_to_be_labeled = []

    global classificationStart_time

    classificationStart_time = datetime.now()

    print(
        'Platform(B): User Classification Start Time :',
        classificationStart_time
    )

    for l in request.body:

        str_l = str(l, 'utf-8')

        data_to_parse = json.loads(str_l)

        database = (
            DataDir +
            data_to_parse["dataset"] +
            ".json"
        )

        trajectories_to_be_labeled = (
            al_strategies.run_al_strategy(
                data_to_parse["strategy"],
                database,
                data_to_parse["classifier"],
                data_to_parse["labeled_trajectories"],
                int(data_to_parse["time_step"])
            )
        )

    out = http_get_solr_data.get_trajectories_geojson(
        database,
        trajectories_to_be_labeled
    )

    response.body = out

    return response


# ============================================================
# Classify all trajectories
# ============================================================

@route('/classify_all', method='POST')
@enable_cors
def classify():

    response.headers['Access-Control-Allow-Origin'] = '*'

    dict_tid_label = None
    data_to_parse = None

    global classificationEnd_time

    classificationEnd_time = datetime.now()

    print(
        'Platform(B): User Classification End Time   :',
        classificationEnd_time
    )

    for l in request.body:

        str_l = str(l, 'utf-8')

        data_to_parse = json.loads(str_l)

        dict_tid_label = (
            classify_all.run_classification(
                DataDir +
                data_to_parse["dataset"] +
                ".json",
                data_to_parse["classifier"],
                data_to_parse["labeled_trajectories"]
            )
        )

    if data_to_parse is None:
        response.status = 400

        return json.dumps(
            {
                'error': 'No classification request data received'
            }
        )

    with open(
        WorkDir +
        data_to_parse["dataset"] +
        "_label.json",
        'w'
    ) as json_file:

        json.dump(
            dict_tid_label,
            json_file
        )

    try:

        print_classification_times(
            classificationStart_time,
            classificationEnd_time
        )

    except NameError:

        print(
            "Platform(B): Total Classification time is not available"
        )

    out = json.dumps(dict_tid_label)

    response.body = out

    return response


# ============================================================
# Print classification times
# ============================================================

def print_classification_times(Start_time, End_time):

    log_path = "./ClassificationTime.log"

    with open(log_path, 'a') as log_file:

        log_file.write(
            'Platform(B):-------------------------------------\n'
        )

        log_file.write(
            'Platform(B): User Classification Start Time : {}\n'
            .format(Start_time)
        )

        log_file.write(
            'Platform(B): User Classification End Time   : {}\n'
            .format(End_time)
        )

        log_file.write(
            'Platform(B): Total Classification time      : {}\n'
            .format(End_time - Start_time)
        )

        log_file.write(
            'Platform(B):-------------------------------------\n'
        )

        log_file.write('\n')


# ============================================================
# Retrieve trajectories as GeoJSON
# ============================================================

@route('/trajectories_geojson', method='POST')
@enable_cors
def get_trajectories_geojson():

    response.headers['Access-Control-Allow-Origin'] = '*'

    geo_out = ''

    for l in request.body:

        str_l = str(l, 'utf-8')

        data_to_parse = json.loads(str_l)

        database = (
            DataDir +
            data_to_parse["dataset"] +
            ".json"
        )

        for tid in data_to_parse['bag']:

            geojson = http_get_solr_data.get_geojson_layer(
                database,
                int(tid)
            )

            geojson['label'] = data_to_parse['bag'][tid]

            geo_out += (
                json.dumps(geojson) +
                '\n'
            )

    response.body = geo_out

    return response


# ============================================================
# Perform dimensionality reduction
# ============================================================

@route('/perform_dimensionality_reduction', method='GET')
@enable_cors
def dimensionality_reduction_endpoint():

    database = request.query.key

    file_path = (
        WorkDir +
        database +
        "_label.json"
    )

    # --------------------------------------------------------
    # Read label JSON
    # --------------------------------------------------------

    with open(file_path) as file:

        json_str = file.read()

    json_data = json.loads(json_str)

    df_labels = pd.DataFrame(
        list(json_data.items()),
        columns=['id', 'label']
    )

    # Convert ID column to integer
    df_labels['id'] = df_labels['id'].astype(int)

    # --------------------------------------------------------
    # Read trajectory data
    # --------------------------------------------------------

    with open(
        DataDir +
        database +
        ".json"
    ) as file:

        data = json.load(file)

    features = [
        item["line"]
        for item in data
    ]

    df_all = pd.json_normalize(features)

    # Remove GeoJSON fields
    df_all = df_all.drop(
        columns=[
            "type",
            "geometry.type",
            "geometry.coordinates"
        ],
        errors='ignore'
    )

    # Rename columns
    df_all = df_all.rename(
        columns=lambda x: x.replace(
            "properties.",
            ""
        )
    )

    # Remove rows containing NaN
    df_all = df_all.dropna()

    # --------------------------------------------------------
    # Merge trajectory data with labels
    # --------------------------------------------------------

    merged_df = df_all.merge(
        df_labels,
        left_on='tid',
        right_on='id'
    )

    # Sort by trajectory ID
    merged_df = merged_df.sort_values('tid')

    # Save CSV
    merged_df.to_csv(
        WorkDir +
        database +
        ".csv",
        index=False
    )

    # --------------------------------------------------------
    # Prepare X and Y
    # --------------------------------------------------------

    df = merged_df.drop(
        columns=[
            'id',
            'tid'
        ]
    )

    # Split into labels and features
    y = df['label'].values

    x = df.drop(
        columns=['label']
    )

    # --------------------------------------------------------
    # Standardize the data
    # --------------------------------------------------------

    x = StandardScaler().fit_transform(x)

    # --------------------------------------------------------
    # PCA - 2 dimensions
    # --------------------------------------------------------

    # PCA requires at least 2 features
    if x.shape[1] < 2:

        raise ValueError(
            "PCA requires at least 2 features. "
            "Only {} feature(s) are available."
            .format(x.shape[1])
        )

    pca = PCA(
        n_components=2
    )

    principalComponents = (
        pca.fit_transform(x)
        .tolist()
    )

    # --------------------------------------------------------
    # PCA before t-SNE
    #
    # Use maximum 50 components.
    # Never use more components than available features.
    # --------------------------------------------------------

    n_pca_components = min(
        50,
        x.shape[1],
        x.shape[0]
    )

    pca_50 = PCA(
        n_components=n_pca_components
    )

    pca_result_50 = (
        pca_50.fit_transform(x)
    )

    # --------------------------------------------------------
    # t-SNE
    #
    # IMPORTANT:
    # Modern scikit-learn uses max_iter instead of n_iter.
    # --------------------------------------------------------

    n_samples = pca_result_50.shape[0]

    # t-SNE requires at least 3 samples
    if n_samples < 3:

        raise ValueError(
            "t-SNE requires at least 3 samples. "
            "Only {} samples are available."
            .format(n_samples)
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # t-SNE requires perplexity < number of samples.
    #
    # Original value = 40
    # For small datasets, automatically reduce it.
    # --------------------------------------------------------

    perplexity = min(
        40,
        n_samples - 1
    )

    print(
        'Platform(B): t-SNE samples      :',
        n_samples
    )

    print(
        'Platform(B): t-SNE PCA features :',
        n_pca_components
    )

    print(
        'Platform(B): t-SNE perplexity    :',
        perplexity
    )

    # --------------------------------------------------------
    # FIX:
    # n_iter was removed/replaced in newer scikit-learn.
    # Use max_iter instead.
    # --------------------------------------------------------

    tsne = TSNE(
        random_state=42,
        n_components=2,
        verbose=0,
        perplexity=perplexity,
        max_iter=400
    ).fit_transform(
        pca_result_50
    )

    # --------------------------------------------------------
    # Prepare response
    # --------------------------------------------------------

    response_data = {

        'reduced_data':
            principalComponents,

        'label':
            y.tolist(),

        'tsne':
            tsne.tolist()
    }

    response.content_type = (
        'application/json'
    )

    return json.dumps(
        response_data
    )


# ============================================================
# Train machine learning models
# ============================================================

@route('/train_model', method='GET')
@enable_cors
def train_model():

    database = request.query.key

    print(
        WorkDir +
        database +
        '.csv'
    )

    df = pd.read_csv(
        WorkDir +
        database +
        '.csv'
    )

    unique_labels = df['label'].unique()

    if len(unique_labels) != 2:

        raise ValueError(
            "The dataset must contain exactly 2 unique labels. "
            "Found {} labels."
            .format(len(unique_labels))
        )

    df['label'] = df['label'].map(
        {
            unique_labels[0]: 0,
            unique_labels[1]: 1
        }
    )

    seed = 7

    names = [
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting"
    ]

    models = [

        RandomForestClassifier(
            random_state=seed
        ),

        ExtraTreesClassifier(
            random_state=seed
        ),

        GradientBoostingClassifier(
            n_estimators=10,
            learning_rate=.3,
            max_depth=1,
            random_state=seed
        )
    ]

    # Convert dataframe and answer value to numpy
    X = df.drop(
        columns='label'
    )

    Y = df['label'].to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        random_state=seed,
        shuffle=True
    )

    scores_arr = []

    for i, name, model in zip(
        range(0, len(models)),
        names,
        models
    ):

        scores = cross_val_score(
            model,
            X,
            Y,
            scoring='f1_weighted',
            cv=skf,
            n_jobs=1
        )

        scores_arr.append(
            scores.tolist()
        )

    # Prepare response data
    response_data = {
        'scores': scores_arr
    }

    response.content_type = (
        'application/json'
    )

    return json.dumps(
        response_data
    )


# ============================================================
# Predict labels for test data
# ============================================================

@route('/predict', method='GET')
@enable_cors
def predict():

    database = request.query.key

    df = pd.read_csv(
        WorkDir +
        database +
        '.csv'
    )

    unique_labels = df['label'].unique()

    if len(unique_labels) != 2:

        raise ValueError(
            "The dataset must contain exactly 2 unique labels. "
            "Found {} labels."
            .format(len(unique_labels))
        )

    df['label'] = df['label'].map(
        {
            unique_labels[0]: 0,
            unique_labels[1]: 1
        }
    )

    X = df.drop(
        'label',
        axis=1
    )

    Y = df['label']

    # Normalize X
    seed = 34

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    X = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(
        n_components=2
    )

    Xt = pca.fit_transform(
        X=X
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        Xt,
        Y,
        test_size=0.2,
        random_state=seed
    )

    data = []

    data.append(
        {
            'name': "1. Forest data",
            'points': Xt.tolist(),
            'labels': Y.tolist()
        }
    )

    response.content_type = (
        'application/json'
    )

    return json.dumps(
        data
    )


# ============================================================
# Generate plots
# ============================================================

@route('/getPlots', method='GET')
@enable_cors
def get_plots():

    database = request.query.key

    colorScale = json.loads(
        request.query.colorScale
    )

    df = pd.read_csv(
        WorkDir +
        database +
        '.csv'
    )

    unique_labels = df['label'].unique()

    label_mapping = {
        unique_labels[i]: i
        for i in range(len(unique_labels))
    }

    reverse_label_mapping = {
        v: k
        for k, v in label_mapping.items()
    }

    df['label'] = df['label'].map(
        label_mapping
    )

    reversed_labels = sorted(
        reverse_label_mapping.values(),
        reverse=True
    )

    reversed_color_mapping = {
        label: colorScale[label]
        for label in reversed_labels
        if label in colorScale
    }

    X = df.drop(
        'label',
        axis=1
    )

    Y = df['label']

    seed = 34

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    X = scaler.fit_transform(X)

    names = [
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting"
    ]

    models = [

        RandomForestClassifier(
            random_state=seed
        ),

        ExtraTreesClassifier(
            random_state=seed
        ),

        GradientBoostingClassifier(
            n_estimators=10,
            learning_rate=.3,
            max_depth=1,
            random_state=seed
        )
    ]

    # PCA
    pca = PCA(
        n_components=2
    )

    Xt = pca.fit_transform(
        X=X
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        Xt,
        Y,
        test_size=0.2,
        random_state=seed
    )

    # Create plots
    fig2, ax2 = plt.subplots(
        4,
        1,
        figsize=(6, 14)
    )

    fig2.set_facecolor(
        'white'
    )

    ax2 = ax2.flatten()

    h = 0.02

    x_min = (
        Xt[:, 0].min() -
        .5
    )

    x_max = (
        Xt[:, 0].max() +
        .5
    )

    y_min = (
        Xt[:, 1].min() -
        .5
    )

    y_max = (
        Xt[:, 1].max() +
        .5
    )

    xx, yy = np.meshgrid(
        np.arange(
            x_min,
            x_max,
            h
        ),
        np.arange(
            y_min,
            y_max,
            h
        )
    )

    ax2[0].set_title(
        "Scatter plot"
    )

    sns.scatterplot(
        x=Xt[:, 0],
        y=Xt[:, 1],
        hue=Y.map(
            reverse_label_mapping
        ),
        ax=ax2[0],
        palette=reversed_color_mapping
    )

    ax2[0].set_xlim(
        xx.min(),
        xx.max()
    )

    ax2[0].set_ylim(
        yy.min(),
        yy.max()
    )

    handles, labels = (
        ax2[0].get_legend_handles_labels()
    )

    ax2[0].legend(
        handles,
        unique_labels,
        loc='lower right'
    )

    image_paths = []

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for i, name, model in zip(
        range(len(models)),
        names,
        models
    ):

        model.fit(
            X_train,
            y_train
        )

        hue = model.predict(
            X_test
        )

        score = f1_score(
            y_test,
            hue,
            average='weighted'
        )

        fig = Figure(
            figsize=(6, 3.5)
        )

        ax = fig.subplots()

        Z = model.predict(
            np.c_[
                xx.ravel(),
                yy.ravel()
            ]
        )

        Z = Z.reshape(
            xx.shape
        )

        ax.contourf(
            xx,
            yy,
            Z,
            cmap=mcolors.ListedColormap(
                colorScale.values()
            ),
            alpha=0.75
        )

        sns.scatterplot(
            x=X_test[:, 0],
            y=X_test[:, 1],
            hue=pd.Series(
                hue
            ).map(
                reverse_label_mapping
            ),
            ax=ax,
            palette=reversed_color_mapping
        )

        ax.set_title(
            '{}. {}, F-Score: {}'.format(
                i + 1,
                name,
                round(score, 2)
            )
        )

        image_file = (
            WorkDir +
            "images/test_{}.png".format(
                i + 1
            )
        )

        fig.savefig(
            image_file
        )

        plt.close(fig)

        image_files = (
            "images/test_{}.png".format(
                i + 1
            )
        )

        image_paths.append(
            image_files
        )

    plt.close(fig2)

    return json.dumps(
        {
            'image_paths':
                image_paths
        }
    )


# ============================================================
# Bar chart data
# ============================================================

@route('/barchart_data', method='GET')
@enable_cors
def get_barchart_data():

    """Route to serve bar chart data."""

    dataset = request.query.get(
        'dataset'
    )

    try:

        with open(
            DataDir +
            dataset +
            '.json'
        ) as f:

            data = json.load(f)

        return json.dumps(
            data
        )

    except Exception as e:

        response.status = 500

        return json.dumps(
            {
                "error": str(e)
            }
        )


# ============================================================
# Download files
# ============================================================

@route('/download/<filename>')
def download_file(filename):

    return static_file(
        filename,
        root=WorkDir,
        download=filename
    )


# ============================================================
# Serve static files
# ============================================================

@route('/<filename:path>')
@enable_cors
def serve_image(filename):

    return static_file(
        filename,
        root='images'
    )


# ============================================================
# UMAP
# ============================================================

@route('/umap_plot', method='POST')
@enable_cors
def get_umap():

    dataset = request.forms.get(
        'dataset'
    )

    umap_json = umaptemp.showUmap(
        DataDir +
        dataset +
        '.json'
    )

    response.content_type = (
        'application/json'
    )

    return umap_json


# ============================================================
# Error 404
# ============================================================

@error(404)
def error404(error):

    response.content_type = (
        'application/json'
    )

    return json.dumps(
        {
            'error':
                'Resource not found'
        }
    )


# ============================================================
# Error 500
# ============================================================

@error(500)
def error500(error):

    response.content_type = (
        'application/json'
    )

    return json.dumps(
        {
            'error':
                'Internal server error'
        }
    )


# ============================================================
# Run local server
# ============================================================

if __name__ == "__main__":

    run(
        host='127.0.0.1',
        port=8081,
        debug=True,
        server='cheroot'
    )