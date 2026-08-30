#!/usr/bin/python
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Import necessary libraries
from bottle import route, request, response, get, static_file, default_app, error, run
from analytic import al_strategies, classify_all, trajectory_manager, http_get_solr_data,umaptemp
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from datetime import datetime



task_start_time = None
task_end_time = None


# Function to enable Cross-Origin Resource Sharing (CORS)
#def enable_cors(fn):
#    def decorated(*args, **kwargs):
#        response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:8080'
#        response.headers['Access-Control-Allow-Headers'] = 'Origin, X-Requested-With, Content-Type, Accept'
#        return fn(*args, **kwargs)
#    return decorated


### For online hosting
#application = default_app()
#WorkDir='/home/abdalmunaem86/analyticV3/'
#DataDir='/home/abdalmunaem86/data/'

WorkDir='C:/_Portable/_Dev/analytic-2.0-main/V15/'
DataDir='C:/_Portable/_Dev/analytic-2.0-main/Data/'

def enable_cors(fn):
    def _enable_cors(*args, **kwargs):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'
        if request.method != 'OPTIONS':
            return fn(*args, **kwargs)
    return _enable_cors


# Serve the main platform page (index.html)
@route('/')
@enable_cors
def send_index():
    return static_file('home.html', WorkDir)

# Serve the main platform page (index.html)
@route('/index')
@enable_cors
def send_index():
    return static_file('index.html', WorkDir)

# Serve the CSS file
@route('/css')
@enable_cors
def send_css():
    response.body = static_file('style.min.css', WorkDir + 'dist/css')
    response.content_type = 'text/css'
    return response

# Serve the JavaScript file
@route('/analytic_js')
@enable_cors
def send_analytic_js():
    response.body = static_file('analytic.min.js', WorkDir + 'dist/js')
    response.content_type = 'text/javascript'
    return response

# Serve images
@get('/images/<image_name>', method='GET')
@enable_cors
def images(image_name):
    response.body = static_file(image_name, root=WorkDir + 'images')
    return response

BagSize=10

# Retrieve GeoJSON data for the Geolife dataset
@get('/data/geolife', method='GET')
@enable_cors
def geolife_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    out = trajectory_manager.get_random_trajectories(DataDir + 'geolife.json',BagSize)
    response.body = out
    return response

# Retrieve GeoJSON data for the fishing vessels dataset
@get('/data/fishingvessels', method='GET')
def fishing_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    out = trajectory_manager.get_random_trajectories(DataDir + 'fishingvessels.json',BagSize)
    response.body = out
    return response

# Retrieve GeoJSON data for the hurricanes dataset
@get('/data/hurricanes', method='GET')
def hurricanes_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    out = trajectory_manager.get_random_trajectories(DataDir + 'hurricanes.json',BagSize)
    response.body = out
    return response


# Retrieve GeoJSON data for the animals dataset
@get('/data/animals', method='GET')
@enable_cors
def animals_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    out = trajectory_manager.get_random_trajectories(DataDir + 'animals.json',BagSize)
    response.body = out
    return response

# Run active learning strategy to label trajectories
@route('/al_run', method='POST')
def al_strategy():
    response.headers['Access-Control-Allow-Origin'] = '*'
    database = ''
    for l in request.body:
        str_l = str(l, 'utf-8')
        data_to_parse = json.loads(str_l)
        database = DataDir + data_to_parse["dataset"] + ".json"
        trajectories_to_be_labeled = al_strategies.run_al_strategy(
            data_to_parse["strategy"],
            database,
            data_to_parse["classifier"],
            data_to_parse["labeled_trajectories"],
            int(data_to_parse["time_step"]))
    out = http_get_solr_data.get_trajectories_geojson(database, trajectories_to_be_labeled)
    response.body = out
    return response

# Classify all trajectories
@route('/classify_all', method='POST')
@enable_cors
def classify():
    response.headers['Access-Control-Allow-Origin'] = '*'
    global task_end_time
    dict_tid_label = None
    for l in request.body:
        str_l = str(l, 'utf-8')
        data_to_parse = json.loads(str_l)
        dict_tid_label = classify_all.run_classification(DataDir + data_to_parse["dataset"] + ".json",
                                                         data_to_parse["classifier"],
                                                         data_to_parse["labeled_trajectories"])
    with open(WorkDir + data_to_parse["dataset"] + "_label.json", 'w') as json_file:
        json.dump(dict_tid_label, json_file)
    
    task_end_time = datetime.now()
    out = json.dumps(dict_tid_label)
    response.body = out
    return response

# Retrieve trajectories as GeoJSON
@route('/trajectories_geojson', method='POST')
@enable_cors
def get_trajectories_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    geo_out = ''
    for l in request.body:
        str_l = str(l, 'utf-8')
        data_to_parse = json.loads(str_l)
        database = DataDir + data_to_parse["dataset"] + ".json"
        for tid in data_to_parse['bag']:
            geojson = http_get_solr_data.get_geojson_layer(database, int(tid))
            geojson['label'] = data_to_parse['bag'][tid]
            geo_out += json.dumps(geojson) + '\n'
    response.body = geo_out
    return response

# Perform dimensionality reduction
@route('/perform_dimensionality_reduction', method='GET')
@enable_cors
def dimensionality_reduction_endpoint():
    database = request.query.key
    file_path = WorkDir + database + "_label.json"

    # Read the JSON data from the file
    with open(file_path) as file:
        json_str = file.read()

    # Parse the JSON data into a dictionary
    json_data = json.loads(json_str)

    df_labels = pd.DataFrame(list(json_data.items()), columns=['id', 'label'])

    # Converting 'id' column to int
    df_labels['id'] = df_labels['id'].astype(int)

    with open(DataDir + database + ".json") as file:
        data = json.load(file)

    features = [item["line"] for item in data]

    df_all = pd.json_normalize(features)

    df_all = df_all.drop(columns=["type", "geometry.type", "geometry.coordinates"])

    # Rename columns
    df_all = df_all.rename(columns=lambda x: x.replace("properties.", ""))

    df_all = df_all.dropna()

    merged_df = df_all.merge(df_labels, left_on='tid', right_on='id')

    # Now sort the DataFrame by 'tid'
    merged_df = merged_df.sort_values('tid')
    merged_df.to_csv(WorkDir + database + ".csv", index=False)
    df = merged_df.drop(columns=['id', 'tid'])

    # Split into y (label) and x (rest of the columns)
    y = df['label'].values
    x = df.drop(columns=['label'])
    comp = (x.shape[1]) // 2

    # Standardizing the data
    x = StandardScaler().fit_transform(x)

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x).tolist()

    pca_50 = PCA(n_components=comp)
    pca_result_50 = pca_50.fit_transform(x)

    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(random_state=42, n_components=2, verbose=0, perplexity=40, n_iter=400).fit_transform(pca_result_50)

    # Prepare response data
    response_data = {
        'reduced_data': principalComponents,
        'label': y.tolist(),
        'tsne': tsne.tolist()
    }

    response.content_type = 'application/json'
    return json.dumps(response_data)


# Train machine learning models
@route('/train_model', method='GET')
@enable_cors
def train_model():
    database = request.query.key
    print (WorkDir + database + '.csv')
    df = pd.read_csv(WorkDir + database + '.csv')
    unique_labels = df['label'].unique()
    df['label'] = df['label'].map({unique_labels[0]: 0, unique_labels[1]: 1})

    seed = 7
    names = ["Random Forest", "Extra Trees", "Gradient Boosting"]
    models = [RandomForestClassifier(random_state=seed),
              ExtraTreesClassifier(random_state=seed),
              GradientBoostingClassifier(n_estimators=10, learning_rate=.3, max_depth=1, random_state=seed)]

    # Convert the dataframe and its answer value to numpy.
    X = df.drop(columns='label')
    Y = df['label'].to_numpy()

    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

    scores_arr = []
    for i, name, model in zip(range(0, 4), names, models):
        scores = cross_val_score(model, X, Y,
                                 scoring='f1_weighted', cv=skf, n_jobs=1)
        scores_arr.append(scores.tolist())

    # Prepare response data
    response_data = {
        'scores': scores_arr
    }
    response.content_type = 'application/json'

    return json.dumps(response_data)



# Predict labels for test data
@route('/predict', method='GET')
@enable_cors
def predict():
    database = request.query.key
    df = pd.read_csv(WorkDir + database + '.csv')
    unique_labels = df['label'].unique()
    df['label'] = df['label'].map({unique_labels[0]: 0, unique_labels[1]: 1})

    X = df.drop('label', axis=1)
    Y = df['label']

    # Normalize X
    seed = 34
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    Xt = pca.fit_transform(X=X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(Xt, Y, test_size=0.2, random_state=seed)

    data = []
    data.append({
        'name': "1. Forest data",
        'points': Xt.tolist(),
        'labels': Y.tolist()
    })

    print(f'\nPlatform(B):')
    print(f'Task start  : {task_start_time}')
    print(f'Task End    : {task_end_time}')
    print(f'Total Task processing time: {int((task_end_time - task_start_time).total_seconds())} seconds\n')

    response.content_type = 'application/json'
    return json.dumps(data)

@route('/getPlots', method='GET')
@enable_cors
def get_plots():
    database = request.query.key
    colorScale = json.loads(request.query.colorScale) 
    df = pd.read_csv(WorkDir + database + '.csv')

    unique_labels = df['label'].unique()
    label_mapping = {unique_labels[i]: i for i in range(len(unique_labels))}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    df['label'] = df['label'].map(label_mapping)

    reversed_labels = sorted(reverse_label_mapping.values(), reverse=True)
    reversed_color_mapping = {label: colorScale[label] for label in reversed_labels if label in colorScale}

    X = df.drop('label', axis=1)
    Y = df['label']
    seed = 34
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    names = ["Random Forest", "Extra Trees", "Gradient Boosting"]
    models = [RandomForestClassifier(random_state=seed),
              ExtraTreesClassifier(random_state=seed),
              GradientBoostingClassifier(n_estimators=10, learning_rate=.3, max_depth=1, random_state=seed)]

    pca = PCA(n_components=2)
    Xt = pca.fit_transform(X=X)
    X_train, X_test, y_train, y_test = train_test_split(Xt, Y, test_size=0.2, random_state=seed)

    fig2, ax2 = plt.subplots(4, 1, figsize=(6, 14))
    fig2.set_facecolor('white')
    ax2 = ax2.flatten()

    h = 0.02
    x_min, x_max = Xt[:, 0].min() - .5, Xt[:, 0].max() + .5
    y_min, y_max = Xt[:, 1].min() - .5, Xt[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax2[0].set_title("Scatter plot")
    
    sns.scatterplot(x=Xt[:, 0], y=Xt[:, 1], hue=Y.map(reverse_label_mapping), ax=ax2[0],
                    palette=reversed_color_mapping)
    ax2[0].set_xlim(xx.min(), xx.max())
    ax2[0].set_ylim(yy.min(), yy.max())

    handles, labels = ax2[0].get_legend_handles_labels()
    ax2[0].legend(handles, unique_labels, loc='lower right')

    image_paths = []

    for i, name, model in zip(range(len(models)), names, models):
        model.fit(X_train, y_train)
        hue = model.predict(X_test)
        score = f1_score(y_test, hue, average='weighted')

        fig = Figure(figsize=(6, 3.5))
        ax = fig.subplots()

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=mcolors.ListedColormap(colorScale.values()), alpha=0.75)

        sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=pd.Series(hue).map(reverse_label_mapping),
                        ax=ax, palette=reversed_color_mapping)

        ax.set_title('{}. {}, F-Score: {}'.format(i + 1, name, round(score, 2)))

        image_file = WorkDir + "images/test_{}.png".format(i + 1)
        fig.savefig(image_file)
        plt.close(fig)
        image_files = "images/test_{}.png".format(i + 1)
        image_paths.append(image_files)
    return json.dumps({'image_paths': image_paths})



@route('/barchart_data', method='GET')
@enable_cors
def get_barchart_data():
    """Route to serve bar chart data."""
    dataset = request.query.get('dataset') 
    try:
        with open(DataDir + dataset + '.json') as f:
            data = json.load(f)
        return json.dumps(data)
    except Exception as e:
        #print(f"Error loading barchart data for {dataset}: {str(e)}")
        response.status = 500
        return json.dumps({"error": str(e)})
    

@route('/download/<filename>')
def download_file(filename):
    return static_file(filename, root=WorkDir, download=filename)

# Serve static files
@route('/<filename:path>')
@enable_cors
def serve_image(filename):
    return static_file(filename, root='images')

@route('/umap_plot', method='POST')
@enable_cors
def get_umap():
    global task_start_time
    task_start_time = datetime.now()
    dataset = request.forms.get('dataset')
    umap_json = umaptemp.showUmap(DataDir + dataset + '.json')
    response.content_type = 'application/json'
    return umap_json

@error(404)
def error404(error):
    response.content_type = 'application/json'
    return json.dumps({'error': 'Resource not found'})

@error(500)
def error500(error):
    response.content_type = 'application/json'
    return json.dumps({'error': 'Internal server error'})


# Run the server
if __name__ == "__main__":
    run(host='127.0.0.1', port=8081, debug=True, server='cheroot')
