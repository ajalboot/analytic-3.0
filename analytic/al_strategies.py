import math
import numpy as np
from collections import defaultdict
import scipy.sparse as ss
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from analytic import http_get_solr_data
from analytic import trajectory_manager


# This class is used if the AL strategy selected is Random
class RandomBootstrap(object):
    # Parameters
    # seed(int) - a seed to manage the random values.
    def __init__(self, seed):
        self.randomSeed = RandomStrategy(seed)

    def bootstrap(self, pool, y=None, k=1):
        # Parameters
        # pool (int) - range of numbers within length of pool
        # y - None or possible pool
        # k (int) - 1 or possible bootstrap size
        # choose next pool

        return self.randomSeed.chooseNext(pool, k=k)


# This class is used if not bootstrapped
class BootstrapFromEach(object):

    def __init__(self, seed):
        self.randomSeed = RandomStrategy(seed)

    def bootstrap(self, pool, y, k=1):
        # pool (*int*) - range of numbers within length of pool
        # y - None or possible pool
        # k (*int*) - 1 or possible bootstrap size
        # chosen array of indices

        data = defaultdict(lambda: [])
        for i in pool:
            data[y[i]].append(i)
        chosen = []
        num_classes = len(data.keys())
        for label in data.keys():
            candidates = data[label]
            indices = self.randomSeed.chooseNext(candidates, k=k / num_classes)
            chosen.extend(indices)
        return chosen


# Class Base strategy for all AL
class BaseStrategy(object):

    def __init__(self, seed=0):
        self.randgen = np.random.RandomState(seed)

    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices=None, current_train_y=None):
        pass


# Class is used if strategy is rand, inherits from BaseStrategy
class RandomStrategy(BaseStrategy):
    # Overide method BaseStrategy.chooseNext
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices=None, current_train_y=None):
        # pool (*int*) - range of numbers within length of pool
        # X - None or pool.toarray()
        # model - None
        # k (*int*) - 1 or step size
        # current_train_indices - None or array of trained indices
        # current_train_y - None or train_indices specific to y_pool
        # returns [list_pool[i] for i in rand_indices[:k]] - array of random permutations given pool

        list_pool = list(pool)
        rand_indices = self.randgen.permutation(len(pool))
        # print(f'rand_indices: {rand_indices} \t k: {k} \t list_pool{list_pool} \t pool {pool}')
        return [list_pool[i] for i in rand_indices[:int(k)]]


# Class - used if strategy selected is unc, inherits from BaseStrategy
class UncStrategy(BaseStrategy):

    def __init__(self, seed=0, sub_pool=None):
        # Instantiate :mod:`al.instance_strategies.UncStrategy`

        # seed (*int*) - 0 or trial number.
        # sub_pool - None or sub_pool parameter

        super(UncStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool

    # Overide method BaseStrategy.chooseNext
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices=None, current_train_y=None):
        # Parameters
        # pool (int) - range of numbers within length of pool
        # X - None or pool.toarray()
        # model - None
        # k (int) - 1 or step size
        # current_train_indices - None or array of trained indices
        # current_train_y - None or train_indices specific to y_pool
        # Returns
        # [candidates[i] for i in uis[:k]]

        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
        else:
            candidates = list(pool)

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()
        # Assemble _X properly
        _X = []        
        for sublist in X[candidates]:
            row = [sublist[0][key] for key in sublist[0]]
            _X.append(row)        
        _X = np.array(_X)

        # Use _X for prediction
        ##########################################
        # The next line only works for 2 classes #
        ##########################################
        probs = model.predict_proba(_X)
        uncerts = np.min(probs, axis=1)
        # print(f'------- \n UNCERTS{uncerts} \n size {len(uncerts)} \n -------')
        uis = np.argsort(uncerts)[::-1]
        # print(f'------- \n UIS{uis} \n size {len(uis)} \n -------')
        # print(f'------- \n candidates{candidates} \n size {len(candidates)} \n -------')

        # Ensure candidates is within the bounds of the data
        k = min(k, len(uis))  # Ensure k doesn't exceed the number of available indices
        chosen = [candidates[i] for i in uis[:k]]
        return chosen




# Class - used if strategy selected is qbc, inherits from BaseStrategy
class QBCStrategy(BaseStrategy):
    def __init__(self, classifier_name, seed=0, sub_pool=None, num_committee=10):
        # classifier - Represents the classifier that will be used (default: MultinomialNB).
        # seed (*int*) - 0 or trial number.
        # sub_pool - None or sub_pool parameter
        # num_committee
        super(QBCStrategy, self).__init__(seed=seed)
        self.sub_pool = sub_pool
        self.num_committee = num_committee
        self.classifier = classifier_name

    def vote_entropy(self, sample):
        # Parameters
        # sample
        # Returns
        # out (int)
        votes = defaultdict(lambda: 0.0)
        size = float(len(sample))

        for i in sample:
            votes[i] += 1.0

        out = 0
        for i in votes:
            aux = (float(votes[i] / size))
            out += ((aux * math.log(aux, 2)) * -1.)

        return out

    # Overide method BaseStrategy.chooseNext
    def chooseNext(self, pool, X=None, model=None, k=1, current_train_indices=None, current_train_y=None):
        # Parameters
        # pool (*int*) - range of numbers within length of pool
        # X - None or pool.toarray()
        # model - None
        # k (*int*) - 1 or step size
        # current_train_indices - None or array of trained indices
        # current_train_y - None or train_indices specific to y_pool
        # Returns
        # [candidates[i] for i in dis[:k]]

        if not self.sub_pool:
            rand_indices = self.randgen.permutation(len(pool))
            array_pool = np.array(list(pool))
            candidates = array_pool[rand_indices[:self.sub_pool]]
        else:
            candidates = list(pool)

        if ss.issparse(X):
            if not ss.isspmatrix_csr(X):
                X = X.tocsr()

        # Create bags
        comm_predictions = []

        for c in range(self.num_committee):
            # Make sure that we have at least one of each label in each bag
            bfe = BootstrapFromEach(seed=c)
            num_labels = len(np.unique(current_train_y))
            initial = bfe.bootstrap(range(len(current_train_indices)), current_train_y, num_labels)

            r_inds = self.randgen.randint(0, len(current_train_indices), size=len(current_train_indices) - num_labels)
            r_inds = np.hstack((r_inds, np.array(initial)))

            bag = [current_train_indices[i] for i in r_inds]
            bag_y = [current_train_y[i] for i in r_inds]

            new_classifier = trajectory_manager.get_classifier_model(self.classifier)
            
            # Assemble _X properly
            _X_train = []        
            for sublist in X[bag]:
                row = [sublist[0][key] for key in sublist[0]]
                _X_train.append(row)        
            _X_train = np.array(_X_train)
            new_classifier.fit(_X_train, bag_y)

            _X = []        
            for sublist in X[candidates]:
                row = [sublist[0][key] for key in sublist[0]]
                _X.append(row)        
            _X = np.array(_X)
            predictions = new_classifier.predict(_X)

            comm_predictions.append(predictions)

        # Compute disagreement for com_predictions

        disagreements = []
        for i in range(len(comm_predictions[0])):
            aux_candidates = []
            for prediction in comm_predictions:
                aux_candidates.append(prediction[i])
            disagreement = self.vote_entropy(aux_candidates)
            disagreements.append(disagreement)

        dis = np.argsort(disagreements)[::-1]
        chosen = [candidates[i] for i in dis[:k]]

        return chosen


def run_al_strategy(strategy, dataset, classifier_name, labeled_data, time_step):
    labeled_data_dict = {}

    for label in labeled_data:
        labeled_data_dict[label["tid"]] = label["label_value"]

    all_data_tids = http_get_solr_data.get_tids_in_database(dataset)
    all_data_traj_feats = http_get_solr_data.get_all_trajectory_features(dataset)

    model = trajectory_manager.get_classifier_model(classifier_name)
    active_s = trajectory_manager.get_al_strategy(strategy, model, classifier_name)

    data_to_train_array = []
    provided_labels = []
    tids_labeled = []
    tids_not_labeled = set()
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
            tids_not_labeled.add(index)

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

    # Extract values from the 2D array
    values = [list(inner_list[0].values()) for inner_list in data_to_train_array[tids_labeled]]
    model.fit(values, provided_labels)

    k = time_step

    newIndices = active_s.chooseNext(pool=tids_not_labeled, X=data_to_train_array, model=model, k=k,
                                     current_train_indices=tids_labeled, current_train_y=provided_labels)
    final_indices = [map_tid_to_index[i] for i in newIndices]
    return final_indices
