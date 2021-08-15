# Author: Yunus Karatepe
# This file includes some functions to use in the future. 
# Basiclly this file is a library.
from collections import Counter, defaultdict
from math import sqrt, erf, log, exp
from functools import partial
from numpy.lib.function_base import copy
import csv

import numpy.random as r

def make_dict(keys, values):
    return {key: value for key, value in zip(keys, values)}

def euclidean(x, y):
    try:
        return sqrt(sum([(xi - yi) ** 2 for xi, yi in zip(x, y)]))
    except:
        print("Usage: euclidean(vector1, vector2), Ex: euclidean([0, 0], [3, 4]) -> returns 5.0")


def get_column(matrix, column):
    return [i[column] for i in matrix]


def mean(array):
    return sum(array) / len(array)


def median(array):
    length = len(array)
    sorted_arr = sorted(array)
    if length % 2 == 0:
        return (sorted_arr[length // 2] + sorted_arr[length // 2 - 1]) / 2
    else:
        return sorted_arr[length // 2]


def quantile(array, p):
    """returns the pth-percentile value in x"""
    p_index = int(p * len(array))
    return sorted(array)[:p_index]


def mode(array):
    counts = Counter(array)
    return counts.most_common(1)[0][0]


def data_range(array):
    return max(array) - min(array)


def sum_of_squares(array):
    return sum([xi ** 2 for xi in array])

def total_sum_of_squares(y):
    return sum(v ** 2 for v in de_mean(y))

def de_mean(array):
    mean_of_array = mean(array)
    return [xi - mean_of_array for xi in array]


def variance(array, population=False):
    de_means = de_mean(array)
    if population == False:
        return sum_of_squares(de_means) / (len(array) - 1)
    else:
        return sum_of_squares(de_means) / len(array)


def standard_deviation(array, population=False):
    return sqrt(variance(array, population))


def interquarile_range(array):
    return quantile(array, 0.75) - quantile(array, 0.25)


def covariance(x, y, population=False):
    n = 0
    if population:
        n = len(x)
    else:
        n = len(x) - 1
    return dot_product(de_mean(x), de_mean(y)) / n


def correlation(x, y, population=False):
    return covariance(x, y) / (standard_deviation(x, population) * standard_deviation(y, population))


def make_matrix(num_rows, num_cols, entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def magnitude(vector):
    return euclidean(vector, [0 for _ in range(len(vector))])


def vector_add(x, y):
    return [xi + yi for xi, yi in zip(x, y)]


def vector_substract(x, y):
    return [xi - yi for xi, yi in zip(x, y)]


def vector_sum(vectors):
    vectors = list(vectors)
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot_product(x, y):
    return sum([xi * yi for xi, yi in zip(x, y)])


def dot_sum(x, y):
    return sum([xi + yi for xi, yi in zip(x, y)])


def direction(w):
    mag = magnitude(w)
    return [wi / mag for wi in w]


def directional_variance_i(xi, w):
    return dot_product(xi, direction(w))


def directional_variance(x, w):
    return sum(directional_variance_i(xi, w) for xi in x)


def directional_variance_gradient_i(xi, w):
    projection_length = dot_product(xi, direction(w))
    return [2 * projection_length * xij for xij in xi]


def directional_variance_gradient(x, w):
    return vector_sum(directional_variance_gradient_i(xi, w) for xi in x)


def sum_of_squares(v):
    return sum(vi ** 2 for vi in v)


def scalar_multiply(skalar, vector):
    return [skalar * vi for vi in vector]

def squared_distance(v, w):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(vector_substract(v, w))



# ---- GRADIEND DESCENT FUNCTIONS ---- #
def difference_quotient(f, x, h):
    return (f(x + h) - f(x) / h)

def partial_difference_quotient(f, v, i, h):
    w = [vj + (h if j == i else 0)
    for j, vj in enumerate(v)]

    return (f(w) - f(v)) / h

def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]

def step(v, direction, step_size):
    return [vi + step_size * direction_i for vi, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * vi for vi in v]

def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f

def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)

    while(True):
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]

        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)

        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value

def negate(f):
    return lambda *args, **kwargs: -f(*args, **kwargs)

def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(negate(target_fn),
                            negate_all(gradient_fn),
                            theta_0,
                            tolerance)

def in_random_order(data):
    indexes = [i for i, _ in enumerate(data)]
    r.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = list(zip(x, y))

    # print(data)

    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float('inf')
    iterations_with_no_improvement = 0

    while(iterations_with_no_improvement < 100):
        value = sum( target_fn(xi, yi, theta) for xi, yi in data )

        if value < min_value:
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            iterations_with_no_improvement += 1
            alpha *= 0.9

        for xi, yi in in_random_order(data):
            gradient_i = gradient_fn(xi, yi, theta)
            theta = vector_substract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0 = 0.01):
    return minimize_stochastic(negate(target_fn),
                                negate_all(gradient_fn),
                                x, y, theta_0, alpha_0)
# ---- GRADIEND DESCENT FUNCTIONS end ---- #

def scale(data_matrix):
    # returns means, stdevs for each column of given matrix
    _, num_cols = data_matrix.shape
    means = [mean(get_column(data_matrix, j)) for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix, j)) for j in range(num_cols)]

    return means, stdevs

def rescale(data_matrix):
    # rescales each column of given matrix as mean=0 and stdev=1
    means, stdevs = scale(data_matrix)

    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - means[j]) / stdevs[j]
        else:
            return data_matrix[i][j]

    num_rows, num_cols = data_matrix.shape
    return make_matrix(num_rows, num_cols, rescaled)

# principal component analysis ->

def first_principal_component(x):
    guess = [1 for _ in x[0]]
    unscaled_maximizer = maximize_batch(
        partial(directional_variance, x),
        partial(directional_variance_gradient, x),
        guess
    )
    return direction(unscaled_maximizer)

def first_principal_component_sgd(x):
    guess = [1 for _ in range(x[0])]
    unscaled_maximizer = maximize_stochastic(
        lambda x, _, w: directional_variance_i(x, w),
        lambda x, _, w: directional_variance_gradient_i,
        x,
        [0 for _ in x],
        guess
    )
    return direction(unscaled_maximizer)

def project_fpc(v, w):
    projection_length = dot_product(v, w)
    return scalar_multiply(projection_length, w)

def remove_projection_from_vector(v, w):
    return vector_substract(v, project_fpc(v, w))

def remove_projection(x, w):
    return [remove_projection_from_vector(xi, w) for xi in x]

def principal_component_analysis(x, num_components=3):
    components = []

    for _ in range(num_components):
        component = first_principal_component(x)
        components.append(component)
        x = remove_projection(x, component)

    return components

def matrix_product_entry(A, B, i, j):
    return dot_product(A[i], get_column(B, j))


def matrix_multiply(A, B):
    n1, k1 = len(A), len(A[0])
    n2, k2 = len(B), len(B[0])

    if k1 != n2:
        raise ArithmeticError('incompatible shapes!')
    else:
        return make_matrix(n1, k2, partial(matrix_product_entry, A, B))

def vector_to_matrix(v):
    """converts list to nx1 matrix"""
    return [[vi] for vi in v]

def vector_from_matrix(mat):
    """converts nx1 matrix to list"""
    return [row[0] for row in mat]

def cosine_similarity(v, w):
    """this basically measures the angle between v and w"""
    return dot_product / sqrt(dot_product(v, v) * dot_product(w, w))

def transpose_matrix(A):
    return [get_column(A, i) for i in range(len(A[0]))]

class file_ops:

    def read_csv(file_path, sep=',', keys=None, as_array=False):
        """Keys must be given as ordered as in file,
        if keys=None then it assumes keys are exists at start of the file.\n
        If as_array=True it returns list of list(row) and it does not use keys."""
        data = []
        if as_array:
            with open(file_path, 'rt') as f:
                reader = csv.reader(f, delimiter=sep)
                for row in reader:
                    data.append(row)
            return data
        elif keys:
            with open(file_path, 'rt') as f:
                reader = csv.reader(f, delimiter=sep)
                if keys:
                    for row in reader:
                        data.append(make_dict(keys, row))
            return data
        else:
            with open(file_path, 'rt') as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    data.append(row)
            return data



class parser:
    def parse_vector2float(vector):
        """returns list of values which are parsed to float"""
        try:
            return [float(vi) for vi in vector]
        except:
            print('Input must be a list which includes float parsable elements.')



# Generating Random Data ->
class random:

    def random():
        # return a number between [0, 1)
        return r.random()

    def normal(mean=0, st_dev=1, size=1000):
        return r.normal(mean, st_dev, size)

    def shuffle(data):
        """returns a shuffled copy of given data"""
        new_data = copy(list(data))
        r.shuffle(new_data)
        return new_data

    def split_data(data, test_proportion=0.2):
        """splits data into 2 parts -> [prob, 1 - prob]"""
        new_data = random.shuffle(data)
        length = len(new_data)
        train = new_data[:int(length * (1 - test_proportion))]
        test = new_data[int(length * (1 - test_proportion)):]
        return train, test


    def train_test_split(x, y, test_proportion=0.2):
        """returns x_train, x_test, y_train, y_test after shuffling data."""
        data = zip(x, y)
        train, test = random.split_data(data, test_proportion)
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        return x_train, x_test, y_train, y_test

    def sample(data, num_samples=None, replace=True):
        return r.choice(data, size=num_samples, replace=replace)

    def rand_index(max_number):
        """returns a number between [0, max_number)"""
        return r.randint(low=0, high=max_number)

    def normal_cdf(x, mu=0, sigma=1):
        return (1 + erf((x - mu) / sqrt(2) / sigma)) / 2



class ml:

    def accuracy(tp, fp, fn, tn):
        return (tp + tn) / (tp + fp + fn + tn)

    def precision(tp, fp, fn, tn):
        return tp / (tp + fp)

    def recall(tp, fp, fn, tn):
        return tp / (tp + fn)

    def f1_score(tp, fp, fn, tn):
        p = ml.precision(tp, fp, fn, tn)
        r = ml.recall(tp, fp, fn, tn)

        if p == r == 0:
            print('P = 0, R = 0')
            return

        return 2 * p * r / (p + r)

    class knn:
        def raw_majority_vote(labels):
            votes = Counter(labels)
            winner, _ = votes.most_common(1)[0]
            return winner

        def majority_vote(labels):
            vote_counts = Counter(labels)
            winner, winner_count = vote_counts.most_common(1)[0]
            num_winners = len([count for count in vote_counts.values() if count == winner_count])

            if num_winners == 1:
                return winner
            else:
                return ml.knn.majority_vote(labels[:-1])

        def knn_classify(k, labelled_points, new_point):
            """each labeled point should be a pair (point, label)"""
            def distance(point):
                try:
                    return euclidean(parser.parse_vector2float(point[0]), parser.parse_vector2float(new_point))
                except:
                    print('Float parsing error!')

            by_distance = sorted(labelled_points, key=distance)

            k_nearest_labels = [label for _, label in by_distance[:k]]

            return ml.knn.majority_vote(k_nearest_labels)

    class NaiveBayesClassifier:
        def __init__(self, k=0.5):
            self.k = k
            self.word_probs = []

        def tokenize(self, message: str):
            import re
            message = message.lower()
            all_words = re.findall("[a-z0-9]+", message)
            return set(all_words)

        def spam_ham_count_from_data(self, data):
            """returns spam_count, ham_count from given data"""
            spam_count = ham_count = 0
            for _, label in data:
                if label == 'spam':
                    spam_count += 1
                else:
                    ham_count += 1

            return spam_count, ham_count

        def count_words(self, training_set):
            from collections import defaultdict
            """training set consists of pairs (message, is_spam)"""
            counts = defaultdict(lambda: [0, 0])

            for message, is_spam in training_set:
                tokenized = self.tokenize(message)
                for word in tokenized:
                    counts[word][0 if is_spam == 'spam' else 1] += 1
            return counts

        def word_probabilities(self, counts, total_spams, total_non_spams, k=0.5):
            """turn the word_counts into a list of triplets w, p(w | spam) and p(w | ~spam) """
            return [(w,
                    (spam + k) / (total_spams + 2 * k),
                    (non_spam + k) / (total_non_spams + 2 * k))
                    for w, (spam, non_spam) in counts.items()]

        def spam_probability(self, word_probs, message):
            from math import log, exp
            message_words = self.tokenize(message)
            log_prob_spam = log_prob_nospam = 0.0

            for word, prob_spam, prob_nospam in word_probs:
                if word in message_words:
                    log_prob_spam += log(prob_spam)
                    log_prob_nospam += log(prob_nospam)
                    # print(word, prob_spam, prob_nospam)
                else:
                    log_prob_spam += log(1.0 - prob_spam)
                    log_prob_nospam += log(1.0 - prob_nospam)

                prob_spam = exp(log_prob_spam)
                prob_nospam = exp(log_prob_nospam)

            # print(prob_spam, prob_nospam)
            predicted = 0.0
            try:
                predicted = prob_spam / (prob_spam + prob_nospam)
            except:
                pass
            return predicted

        def train(self, training_data):
            num_spams, num_hams = self.spam_ham_count_from_data(training_data)

            word_counts = self.count_words(training_data)

            self.word_probs = self.word_probabilities(word_counts, num_spams, num_hams, self.k)

        def classify(self, message, treshold=None):
            """if treshold is given returns True-False, if treshold is not given returns probability of being spam"""
            if treshold:
                return "spam" if self.spam_probability(self.word_probs, message) >= treshold else "ham"
            else:
                return self.spam_probability(self.word_probs, message)

        def test(self, test_data, treshold=0.5):
            """returns (true_positive_count, false_positive_count, true_negative_count, false_negative_count)"""
            classified = [self.classify(test_i_message, treshold) for test_i_message, _ in test_data]

            tp_counter = 0
            tn_counter = 0
            fp_counter = 0
            fn_counter = 0

            for i in range(len(test_data)):
                if test_data[i][1] == "spam":
                    if test_data[i][1] == classified[i]:
                        tp_counter += 1
                    else:
                        fn_counter += 1
                elif test_data[i][1] == "ham":
                    if test_data[i][1] == classified[i]:
                        tn_counter += 1
                    else:
                        fp_counter += 1

            return tp_counter, fp_counter, tn_counter, fn_counter

    class MultiLinearRegressionClassifier():

        def __init__(self, alpha=0.001) -> None:
            self.beta = 0.0
            self.alpha = alpha

        def predict(self, xi, beta=None):
            if beta is None:
                beta = self.beta
            return dot_product(xi, beta)

        def error(self, xi, yi, beta):
            return yi - self.predict(xi, beta)

        def squared_error(self, xi, yi, beta):
            return self.error(xi, yi, beta) ** 2

        def squared_error_gradient(self, xi, yi, beta):
            return [-2 * xij * self.error(xi, yi, beta) for xij in xi]

        def estimate_beta(self, x, y):
            beta_initial = [random.random() for _ in x[0]]
            return minimize_stochastic(self.squared_error,
                self.squared_error_gradient,
                x, y,
                beta_initial,
                self.alpha
            )

        def train(self, train_x, train_y):
            self.beta = self.estimate_beta(train_x, train_y)

        def test(self, test_features, test_labels, allowed_error=0.25):
            """if actual + actual * (allowed_error / 2) > predicted > actual - actual * (allowed_error / 2): 
                correct += 1\n
                returns correct / total"""
            correct = 0
            total = len(test_labels)
            for feature, label in zip(test_features, test_labels):
                predicted = self.predict(feature)
                actual = label


                if actual + actual * allowed_error > predicted > actual - actual * allowed_error:
                    correct += 1

            return correct / total

        def r_squared(self, x, y):
            sum_of_sq_errs = sum(self.error(xi, yi, self.beta) ** 2 for xi, yi in zip(x, y))
            return 1 - sum_of_sq_errs / total_sum_of_squares(y)

    class LogisticRegressionClassifier():

        def __init__(self):
            pass

        def logistic(self, x):
            return 1.0 / (1 + exp(-x))


        def logistic_prime(self, x):
            return self.logistic(x) * (1 - self.logistic(x))

        def logistic_log_likelihood_i(self, xi, yi, beta):
            if yi == 1:
                return log(self.logistic(dot_product(xi, beta)))
            else:
                return log(1.0 - self.logistic(dot_product(xi, beta)))


        def logistic_log_likelihood(self, x, y, beta):
            return sum([self.logistic_log_likelihood_i(xi, yi, beta) for xi, yi in zip(x, y)])

        def logistic_log_partial_ij(self, xi, yi, beta, j):
            return (yi - self.logistic(dot_product(xi, beta))) * xi[j]

        def logistic_log_gradient_i(self, xi, yi, beta):
            return [self.logistic_log_partial_ij(xi, yi, beta, j) for j in range(len(beta))]

        def logistic_log_gradient(self, x, y, beta):
            return [self.logistic_log_gradient_i(xi, yi, beta) for xi, yi in zip(x, y)]

        def predict(self, xi):
            return self.logistic(dot_product(self.beta_hat, xi))

        def test(self, x, y, treshold=0.5):
            """returns (tp, fp, fn, tn)"""
            tp = fp = tn = fn = 0
            for xi, yi in zip(x, y):
                predicted = self.predict(xi)

                if yi == 1:
                    if predicted >= treshold:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predicted >= treshold:
                        fp += 1
                    else:
                        tn += 1

            return tp, fp, fn, tn

        def train(self, train_features, train_labels):
            beta_0 = [random.random() for _ in range(len(train_features[0]))]

            self.beta_hat = maximize_stochastic(
                self.logistic_log_likelihood_i, self.logistic_log_gradient_i,
                train_features, train_labels, beta_0)

    class DecisionTreeClassifier:

        def entropy(self, class_probabilities):
            return sum([-p * log(p, 2) for p in class_probabilities if p != 0])

        def class_probabilities(self, labels):
            total = len(labels)
            return [count / total for count in Counter(labels).values()]

        def data_entropy(self, labeled_data):
            labels = [label for _, label in labeled_data]
            class_probs = self.class_probabilities(labels)
            return self.entropy(class_probs)

        def partition_entropy(self, subsets):
            total_count = sum([len(subset) for subset in subsets])
            return sum([self.data_entropy(subset) * len(subset) / total_count
                for subset in subsets])

        def partition_by(self, inputs, attribute):
            groups = defaultdict(list)
            for input in inputs:
                key = input[0][attribute]
                groups[key].append(input)
            return groups

        def partition_entropy_by(self, inputs, attribute):
            partitions = self.partition_by(inputs, attribute)

            return self.partition_entropy(partitions.values())

        def __classify__(self, tree, input):
            """Classify the input using the decision tree."""
            if tree in [True, False]:
                return tree

            attribute, subtree_dict = tree

            subtree_key = input.get(attribute)

            if subtree_key not in subtree_dict:
                subtree_key = None

            subtree = subtree_dict[subtree_key]
            return self.__classify__(subtree, input)

        def classify(self, input):
            return self.__classify__(self.tree, input)

        def build_tree_id3(self, inputs, split_candidates=None):
            # if this is our first pass,
            # all keys of the first input are split candidates
            if split_candidates is None:
                split_candidates = inputs[0][0].keys()

            # count Trues and Falses in the inputs
            num_inputs = len(inputs)
            num_trues = len([label for _, label in inputs if label])
            num_falses = num_inputs - num_trues

            if num_trues == 0: return False
            if num_falses == 0: return True

            # otherwise, split on the best attribute
            if not split_candidates:
                return num_trues >= num_falses

            best_attribute = min(split_candidates, key=partial(self.partition_entropy_by, inputs))
            partitions = self.partition_by(inputs, best_attribute)
            new_candidates = [a for a in split_candidates if a != best_attribute]

            # recursively build subtrees
            subtrees = {attribute_value: self.build_tree_id3(subset, new_candidates)
                        for attribute_value, subset in partitions.items()}

            subtrees[None] = num_trues > num_falses # default case

            return (best_attribute, subtrees)

        def train(self, inputs):
            self.tree = self.build_tree_id3(inputs)

    class PerceptronClassifier:

        def __init__(self, input_size, number_of_hidden_layers, output_size):
            self.network = [
                [[random.random() for _ in range(input_size + 1)] for _ in range(number_of_hidden_layers)],
                [[random.random() for _ in range(number_of_hidden_layers + 1)] for _ in range(output_size)]
            ]
            self.targets = [[1.0 if i == j else 0.0 for i in range(output_size)]
                for j in range(output_size)]

        def sigmoid(self, t):
            return 1 / (1 + exp(-t))

        def neuron_output(self, weights, inputs):
            neuron_value = dot_product(weights, inputs)
            return self.sigmoid(neuron_value)

        def feed_forward(self, input_vector):
            """takes in a network (represented as a list of lists of lists of weights)
            and returns the output from forward-propagating the input"""

            outputs = []

            for layer in self.network:
                input_with_bias = input_vector + [1]
                output = [self.neuron_output(neuron, input_with_bias) for neuron in layer]
                outputs.append(output)

                input_vector = output # output of this layer is input for next layer

            return outputs

        def backpropagate(self, input_vector, target_vector):

            hidden_outputs, outputs = self.feed_forward(input_vector)

            # the output * (1 - output) is from the derivative of sigmoid
            output_deltas = [output * (1 - output) * (output - target) for output, target in zip(outputs, target_vector)]

            # adjust weights for output layer, one neuron at a time
            for i, output_neuron in enumerate(self.network[-1]): # network[-1] is the output layer
                # focus on the ith output layer neuron
                for j, hidden_output in enumerate(hidden_outputs + [1]):
                    # adjust the jth weight based on both
                    # this neuron's delta and its jth input
                    output_neuron[j] -= output_deltas[i] * hidden_output

            # back-propagate errors to hidden layer
            hidden_deltas = [hidden_output * (1 - hidden_output) * dot_product(output_deltas, [n[i] for n in self.network[-1]])
                for i, hidden_output in enumerate(hidden_outputs)]

            for i, hidden_neuron in enumerate(self.network[0]):
                for j, input in enumerate(input_vector + [1]):
                    hidden_neuron[j] -= hidden_deltas[i] * input

        def train(self, train_data, number_of_epochs=30000):
            for _ in range(number_of_epochs):
                for input_vector, target_vector in zip(train_data, self.targets):
                    self.backpropagate(input_vector, target_vector)

        def predict(self, input):
            """returns (class_number, probabilities)"""
            res = self.feed_forward(input)[-1]
            return res.index(max(res)), res

    class KMeans:

        def __init__(self, k):
            self.k = k
            self.means = None

        def classify(self, input):
            """return the index of the cluster closest to the input"""
            return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))

        def train(self, inputs):
            # choose k random points as the initial means
            self.means = []
            selected_indexes = []
            for _ in range(self.k):
                random_index = random.rand_index(len(inputs))
                while random_index in selected_indexes:
                    random_index = random.rand_index(len(inputs))
                self.means.append(inputs[random_index])
                selected_indexes.append(random_index)
                
            assignments = None

            while True:
                # Find new assignments
                new_assignments = list(map(self.classify, inputs))

                # if no assignments have changed, we're done
                if assignments and squared_distance(assignments, new_assignments) == 0:
                    return
                # otherwise keep new assignments.

                # compute new means based on the new assignments.
                assignments = new_assignments
                for i in range(self.k):
                    # find all the points assigned to cluster i
                    i_points = [p for p, a in zip(inputs, assignments) if a == i]

                    # make sure i_points is not empty so don't divide by 0
                    if len(i_points) > 0:
                        self.means[i] = vector_mean(i_points)


class Table:
    def __init__(self, columns) -> None:
        self.columns = columns
        self.rows = []

    def __repr__(self) -> str:
        return str(self.columns) + "\n" + "\n".join(map(str, self.rows))

    def insert(self, row_values):
        if len(row_values) != len(self.columns):
            raise TypeError('wrong number of elements')
        row_dict = dict(zip(self.columns, row_values))
        self.rows.append(row_dict)

    def update(self, updates, predicate):
        for row in self.rows:
            if predicate(row):
                for column, new_value in updates.items():
                    row[column] = new_value

    def delete(self, predicate=lambda row: True):
        """deletes all rows matching predicate, if predicate is not given deletes all rows"""
        self.rows = [row for row in self.rows if not(predicate(row))]

    def select(self, keep_columns=None, additional_columns=None):

        if keep_columns is None:
            keep_columns = self.columns

        if additional_columns is None:
            additional_columns = {}

        # new table for results
        result_table = Table(keep_columns + list(additional_columns.keys()))
    
        for row in self.rows:
            new_row = [row[column] for column in keep_columns]
            for _, calculation in additional_columns.items():
                new_row.append(calculation(row))
            result_table.insert(new_row)
            
        return result_table

    def where(self, predicate=lambda row: True):
        """return only the rows that satisfy the supplied predicate"""
        where_table = Table(self.columns)
        where_table.rows = filter(predicate, self.rows)
        return where_table

    def limit(self, num_rows):
        """return only the first num_rows rows"""
        limit_table = Table(self.columns)
        limit_table.rows = self.rows[:num_rows]
        return limit_table

    
    
























