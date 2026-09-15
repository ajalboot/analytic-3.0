import math
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from analytic import trajectory_manager
from analytic import http_get_solr_data

def get_inner_list_lengths(arr):
    lengths = []
    for inner_list in arr:
        lengths.append(len(inner_list))
    return lengths

def check_for_none_values(arr):
    result = []
    for inner_list in arr:
        has_none = any(element is None for element in inner_list)
        result.append(has_none)
    return result
def find_none_indices(arr):
    none_indices = []
    for i, inner_list in enumerate(arr):
        if None in inner_list:
            none_indices.append(i)
    return none_indices

def remove_inner_lists(arr, indices):
    indices.sort(reverse=True)  # Sort indices in descending order

    for index in indices:
        if 0 <= index < len(arr):
            del arr[index]

    return arr

# Example usage
two_d_list = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
indices_to_remove = [0, 2]
result = remove_inner_lists(two_d_list, indices_to_remove)
print(result)


def run_classification(dataset, classifier_name, labeled_data):
    labeled_data_dict = {}

    for label in labeled_data:
        labeled_data_dict[label["tid"]] = label["label_value"]

    all_data_tids = http_get_solr_data.get_tids_in_database(dataset)
    all_data_traj_feats = http_get_solr_data.get_all_trajectory_features(dataset)

    model = trajectory_manager.get_classifier_model(classifier_name)

    data_to_train_array = []
    provided_labels = []
    tids_labeled = []
    tids_not_labeled = []
    map_tid_to_index = {}

    index = 0
    for tid_key in all_data_tids:
        data_line = []
        map_tid_to_index[index] = tid_key
        if tid_key in labeled_data_dict:
            tids_labeled.append(index)
            label = str(labeled_data_dict[tid_key])
            provided_labels.append(label)

        else:
            tids_not_labeled.append(index)

        for feature in all_data_traj_feats[tid_key]:
            data_line.append(all_data_traj_feats[tid_key][feature])

        data_to_train_array.append(data_line)
        index += 1

    possible_labels = np.unique(provided_labels)
    map_label_bin = {}
    numeric_label = 0
    numeric_labels = []

    for l in possible_labels:
        map_label_bin[l] = numeric_label
        numeric_label += 1

    for l in provided_labels:
        numeric_labels.append(map_label_bin[l])

    data_to_train_array = np.array(data_to_train_array)

    train_data = data_to_train_array[tids_labeled]
    # print(train_data)
    values = [list(inner_list[0].values()) for inner_list in data_to_train_array[tids_labeled]]

    model.fit(values, provided_labels)

    test_data = data_to_train_array[tids_not_labeled]
    test_values = [list(inner_list[0].values()) for inner_list in test_data]
    index = find_none_indices(test_values)
    # test_values.pop(18995)
    # tids_not_labeled.pop(18995)
    test_values = remove_inner_lists(test_values, index)
    tids_not_labeled = remove_inner_lists(tids_not_labeled, index)
    # print("Test Values")
    # print(test_values)
    # print(len(test_values))
    predicted = model.predict(test_values)
    final_indices = [map_tid_to_index[i] for i in tids_not_labeled]
    final_labeled_indices = [map_tid_to_index[i] for i in tids_labeled]
    final_dict = {}
    index = 0
    for i in final_indices:
        final_dict[i] = predicted[index]
        index += 1
    index = 0
    for i in final_labeled_indices:
        final_dict[i] = provided_labels[index]
        index += 1

    return final_dict


# labeled_data = [{'tid': 38, 'label_value': 'q'}, {'tid': 39, 'label_value': 'q'}, {'tid': 40, 'label_value': 't'},
#                 {'tid': 1099, 'label_value': 'q'}, {'tid': 1906, 'label_value': 'q'}, {'tid': 2689, 'label_value': 't'},
#                 {'tid': 2714, 'label_value': 't'}, {'tid': 3653, 'label_value': 't'}, {'tid': 4071, 'label_value': 't'},
#                 {'tid': 4122, 'label_value': 't'}, {'tid': 4612, 'label_value': 'q'}, {'tid': 5750, 'label_value': 'q'},
#                 {'tid': 5769, 'label_value': 'q'}, {'tid': 6543, 'label_value': 'q'}, {'tid': 7106, 'label_value': 'q'},
#                 {'tid': 7155, 'label_value': 'q'}, {'tid': 7212, 'label_value': 't'}, {'tid': 8395, 'label_value': 't'},
#                 {'tid': 8546, 'label_value': 'q'}, {'tid': 9638, 'label_value': 't'},
#                 {'tid': 10271, 'label_value': 't'}, {'tid': 10813, 'label_value': 'q'},
#                 {'tid': 11221, 'label_value': 'q'}, {'tid': 12789, 'label_value': 'q'},
#                 {'tid': 13215, 'label_value': 'q'}, {'tid': 13260, 'label_value': 'q'},
#                 {'tid': 13266, 'label_value': 't'}, {'tid': 13843, 'label_value': 'q'},
#                 {'tid': 14382, 'label_value': 'q'}, {'tid': 14728, 'label_value': 'q'},
#                 {'tid': 15114, 'label_value': 't'}, {'tid': 15961, 'label_value': 't'},
#                 {'tid': 16283, 'label_value': 't'}, {'tid': 16448, 'label_value': 't'},
#                 {'tid': 18540, 'label_value': 't'}, {'tid': 19118, 'label_value': 'q'},
#                 {'tid': 20431, 'label_value': 'q'}, {'tid': 21432, 'label_value': 't'},
#                 {'tid': 21858, 'label_value': 't'}, {'tid': 21925, 'label_value': 'q'},
#                 {'tid': 22210, 'label_value': 't'}, {'tid': 22463, 'label_value': 't'},
#                 {'tid': 22490, 'label_value': 't'}, {'tid': 22693, 'label_value': 'q'},
#                 {'tid': 22962, 'label_value': 'q'}, {'tid': 23427, 'label_value': 't'},
#                 {'tid': 23593, 'label_value': 't'}, {'tid': 24029, 'label_value': 't'},
#                 {'tid': 24431, 'label_value': 't'}, {'tid': 24646, 'label_value': 't'}]
#
# run_classification("./animals.json", 'Logistic Regression', labeled_data)
