
import numpy as np
from classify_iris import *
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    print ("\n ------IRIS using SKLearn -------")
    dataset, labels = load_dataset("iris.csv")
    print ("\nFiltered Dataset (first row):-\n", dataset[0])
    print ("Labels (first 10):-\n", labels[:10])

    #split dataset
    train_size = int((1-TEST_SIZE) * dataset.shape[0])
    train_data, train_labels = dataset[:train_size], labels[:train_size]
    test_data, test_labels = dataset[train_size + 1:], labels[train_size + 1:]
    print ("\nTrain shape: ", train_data.shape, "\nTest shape: ", test_data.shape)
    
    
    # 
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(train_data, train_labels)
    predicted_labels = []
    for sample in test_data:
        predicted_labels.append(
            knn_classifier.predict([sample])[0]
        )
    accuracy = predict_accuracy(predicted_labels, test_labels)
    print ("\nK is taken as 1, ideal value from previous program")
    print ("Accuracy: ", accuracy)

    sample = np.array([6.5, 2.8, 4.6, 1.5])
    print ("\nInput Sample: ", sample)
    sample_label = knn_classifier.predict([sample])[0]
    print ("Predicted Label: ", sample_label, " (", CLASSES[sample_label], ")")
    print ("\n")