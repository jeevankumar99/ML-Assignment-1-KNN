# Predict Benign or Malignant Breast cancer using KNN

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    2: "Benign",
    4: "Malignant"
}

TEST_SIZE = 0.2
# K varies from 1 to 20


def load_dataset(filename):
    # each value is split by a comma, so we use that as delimiter.
    data = np.genfromtxt(filename, delimiter=",", dtype=int)
    print ("\nDataset shape: ", data.shape)
    print ("First row from the dataset:-\n", data[0])
    
    # get the last column as labels, remove 1st and last column from data.
    labels = [x[-1] for x in data]
    dataset = np.delete(data, [0, -1], axis=1)

    return dataset, labels


def get_ideal_k_value(K_accuracy):
    highest_accuracy = 0
    ideal_k = []
    # get highest accuracy with least K value.
    for i in range (len(K_accuracy)):
        if K_accuracy[i] > highest_accuracy:
            highest_accuracy = K_accuracy[i]
            ideal_k = []
            ideal_k.append(i)
        # if more than one value K has same accuracy    
        if (K_accuracy[i] == highest_accuracy):
            ideal_k.append(i)
        
    return ideal_k[0]


def predict_accuracy(predicted_labels, labels):
    correct_predictions = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == labels[i]:
            correct_predictions += 1

    return correct_predictions / len(predicted_labels)


def classify_sample(sample, dataset, labels, k):
    # get diff matrix, and calculate distances.
    diff_matrix = np.tile(sample, (dataset.shape[0], 1)) - dataset
    sq_diff_matrix = diff_matrix ** 2
    sq_distances = sq_diff_matrix.sum(axis=1)
    distances = sq_distances ** 0.5
    
    # sort indexes based on distance from sample
    sorted_dist_indexes = distances.argsort()
    
    #get max class count of top k samples.
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indexes[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # sort dict by value in descending order
    sorted_class_count = sorted(
        class_count.items(),
        key=lambda item: item[1],
        reverse=True)

    return sorted_class_count[0][0]



if __name__ == "__main__":
    dataset, labels = load_dataset("dataset")
    print ("\nFiltered Dataset (first row):-\n", dataset[0])
    print ("Labels (first 10):-\n", labels[:10])
    
    #split dataset
    train_size = int((1-TEST_SIZE) * dataset.shape[0])
    train_data, train_labels = dataset[:train_size], labels[:train_size]
    test_data, test_labels = dataset[train_size + 1:], labels[train_size + 1:]
    print ("\nTrain shape: ", train_data.shape, "\nTest shape: ", test_data.shape)
    
    K_accuracy = []
    for K in range(1, 21):
        predicted_labels = []
        for sample in test_data:
            predicted_sample = classify_sample(sample, train_data, train_labels, K)
            predicted_labels.append(predicted_sample)
        K_accuracy.append(predict_accuracy(predicted_labels, test_labels))
    ideal_k = get_ideal_k_value(K_accuracy)

    print ("\nIdeal K value = ", ideal_k)
    print ("Accuracy: ", K_accuracy[ideal_k]*100, "%")

    print ("\nCustom Input Test:-\n")

    sample = np.array([1,1,1,1,2,1,2,1,1])
    print ("Input Sample: ", sample)
    sample_label = classify_sample(sample, dataset, labels, ideal_k)
    print ("Predicted Label: ", sample_label, " (", LABELS[sample_label], ")")
    print ("\n")


    





