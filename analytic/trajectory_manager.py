import json
import numpy as np
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from analytic import al_strategies
from analytic import http_get_solr_data


# def get_random_trajectories(dataset):
#     bag_size = 20
#     all_tids = http_get_solr_data.get_tids_in_database(dataset)
#     randgen = np.random.RandomState()
#     all_tids = randgen.permutation(all_tids)
#     tids = all_tids[:bag_size]
#     out = http_get_solr_data.get_trajectories_geojson(dataset, tids)
#     return out


def get_random_trajectories(dataset,bag_size):
    # all_data_dict = load_data(trajectory_filename)
    # all_tids = lines.keys()
    # print("before tid: " + dataset)
    all_tids = http_get_solr_data.get_tids_in_database(dataset)
    # print("after tid")
    # print all_tids
    randgen = np.random.RandomState()
    all_tids = randgen.permutation(all_tids)
    tids = all_tids[:bag_size]
    print("TID")
    # print(len(all_tids))
    print(tids)
    # out = get_trajectories_geojson(trajectory_filename, point_file_name, tids)
    out = http_get_solr_data.get_trajectories_geojson(dataset, tids)
    # print(out)
    return out
    # return tids

def get_file_name(dataset_name):

    if (dataset_name=='fishingvessels'):
        file_name = '../data/fishing_vessels.geojson'
        point_file_name = '../data/fishing_vessels_points.geojson'

    elif (dataset_name=='geolife'):
        file_name = '../data/geolife.geojson'
        point_file_name = '../data/geolife_points.geojson'

    elif (dataset_name=='hurricanes'):
        file_name = '../data/hurricanes.geojson'
        point_file_name = '../data/hurricanes_points.geojson'

    elif (dataset_name=='animals'):
        file_name = '../data/animals_lines.json'
        point_file_name = '../data/animals_points.json'

    return file_name, point_file_name


def get_classifier_model(classifier_name):
    model = None
    if classifier_name=='Logistic Regression':
        model = LogisticRegression()
    elif classifier_name=='Random Forest':
        model = RandomForestClassifier()
    elif classifier_name=='KNN':
        model = KNeighborsClassifier()
    elif classifier_name=='Decision Tree':
        model = DecisionTreeClassifier()
    elif classifier_name == 'Ada Boost':
        model = AdaBoostClassifier()
    elif classifier_name == 'Gaussian Naive Bayes':
        model = GaussianNB()

    return model


def get_al_strategy(strategy_name, model, classifier_name):
    active_s = None
    t = 1
    if strategy_name == 'Random Sampling':
        active_s = al_strategies.RandomStrategy(seed=t)
    elif strategy_name == 'Query-by-committee':
        active_s = al_strategies.QBCStrategy(classifier_name=classifier_name)
    elif strategy_name == 'Uncertain Sampling':
        active_s = al_strategies.UncStrategy(seed=t)

    return active_s


def add_label_to_geojson(json_out, dict_tid_label):
    new_json_out = ''
    lines = json_out.split('\n')
    total = 0
    for line in lines:
        if len(line) > 1:
            data_to_parse = json.loads(line)
            tid = data_to_parse['tid']
            data_to_parse['label'] = dict_tid_label[tid]
            final_line = json.dumps(data_to_parse)+'\n'
            new_json_out += final_line
            total += 1

    return new_json_out