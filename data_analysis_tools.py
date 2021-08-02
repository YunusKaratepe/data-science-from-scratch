# Author: Yunus Karatepe
# This file includes some functions to use in the future. 
# Basiclly this file is a library.
from collections import defaultdict
from math import sqrt
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


def variance(array, population=False):
    array_mean = mean(array)
    if population == False:
        return sum([(xi - array_mean) ** 2 for xi in array]) / (len(array) - 1)
    else:
        return sum([(xi - array_mean) ** 2 for xi in array]) / len(array)


def standard_deviation(array, population=False):
    return sqrt(variance(array, population))


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
            return (data_matrix[i][j] - means[j] / stdevs[j])
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

class file_ops:
    from collections import defaultdict
            

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
    


    

    






