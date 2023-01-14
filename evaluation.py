import random
from typing import Union, List

from treepredict import *


def train_test_split(dataset, test_size: Union[float, int], seed=None):
    if seed:
        random.seed(seed)

    # If test size is a float, use it as a percentage of the total rows
    # Otherwise, use it directly as the number of rows in the test dataset
    n_rows = len(dataset)
    if float(test_size) != int(test_size):
        test_size = int(n_rows * test_size)  # We need an integer number of rows

    # From all the rows index, we get a sample which will be the test dataset
    choices = list(range(n_rows))
    test_rows = random.choices(choices, k=test_size)

    test = [row for (i, row) in enumerate(dataset) if i in test_rows]
    train = [row for (i, row) in enumerate(dataset) if i not in test_rows]

    return train, test


def get_accuracy(tree: DecisionNode, dataset, label_col=-1):
    correct = 0
    for row in dataset:
        result = classify(tree, row, label_col)
        if list(result.keys())[0] == row[label_col]:
            correct += 1
    return correct / len(dataset)


def mean(values: List[float]):
    return sum(values) / len(values)


def cross_validation(dataset, k, agg, seed, scoref=entropy, beta=0, threshold=0.0):
    random.seed(seed)
    random.shuffle(dataset)
    partitions = partition_data(dataset, k)
    accuracy = []
    for i in range(k):
        train = [row for j in range(k) if j != i for row in partitions[j]]
        test = partitions[i]
        tree = buildtree(train, scoref, beta)
        prune(tree, threshold)
        accuracy.append(get_accuracy(tree, test))
    return agg(accuracy)

def partition_data(dataset, k):
    partitions = []
    n = len(dataset)
    partition_size = n//k
    for i in range(k):
        # Calculate the start and end index of each partition
        start_index = i * partition_size
        end_index = start_index + partition_size
        # Slice the dataset and append the partition to the partitions list
        partitions.append(dataset[start_index: end_index])
    return partitions

def main():
    filename = 'iris.csv'
    headers, data = read(filename)
    train, test = train_test_split(data, test_size=0.2, seed=random.seed(5))

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    accuracies = []

    for threshold in thresholds:
        print(f'Pruning with threshold {threshold}')
        accuracy = cross_validation(train, k=5, agg=mean, seed=1, scoref=entropy, beta=0, threshold=threshold)
        print(f'Accuracy: {accuracy:.2f}')
        accuracies.append(accuracy)

    max_accuracy = max(accuracies)
    best_threshold = thresholds[accuracies.index(max_accuracy)]

    print(f'Best threshold: {best_threshold} with accuracy {max_accuracy:.2f}')

    print('Building tree with best threshold')
    tree = buildtree(train, scoref=entropy, beta=0)
    prune(tree, best_threshold)

    test_accuracy = get_accuracy(tree, test)

    print(f'Accuracy on test set: {test_accuracy:.2f}')

if __name__ == '__main__':
    main()