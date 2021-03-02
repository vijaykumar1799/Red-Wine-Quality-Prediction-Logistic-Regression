import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_data(file_path):
    df = pd.read_csv(file_path)
    X = np.array([np.append(1, row) for row in df.values[:, :-1]])
    Y = np.array(df.values[:, -1])

    return df, X, Y


def normalize_data(data):
    # using min-max approach
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    norm_data = data.copy()
    for i, row in enumerate(norm_data):
        for j in range(len(row)):
            if j is not 0:
                norm_data[i, j] = (norm_data[i, j] - min_val[j]) / (max_val[j] - min_val[j])

    return norm_data


def standardize_data(data):
    standard_data = data.copy()
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    for i in range(len(standard_data)):
        for j in range(len(standard_data[i])):
            if j is not 0:
                standard_data[i, j] = (standard_data[i, j] - mean[j]) / std[j]

    return standard_data


def correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap='BrBG', robust=True, linewidths=0.5)
    plt.title("Correlation Heatmap", fontdict={'fontsize': 18}, pad=12)
    plt.savefig('heatmap.png')
    plt.show()


def save_weights(weights):
    with open(file='./weights.csv', mode='w') as f:
        string_weight = ""
        for i, theta in enumerate(weights):
            if i != 11:
                seperator = ','
            else:
                seperator = '\n'

            string_weight += str(theta) + seperator
        f.writelines(string_weight)


def compute_cost(data, labels, parameters):
    m = len(labels)
    h_x = np.dot(data, parameters)
    J = (1 / (2 * m)) * sum(np.square(h_x - labels))

    return J


def gradient_descent(data, labels, epochs, learning_rate):
    J = []
    m = len(labels)
    thetas = np.random.random(size=data[0, :].shape)
    for _ in range(epochs):
        h_x = np.dot(data, thetas)
        for i in range(len(thetas)):
            thetas[i] -= (learning_rate / m) * sum((h_x - labels) * data[:, i])
        J.append(compute_cost(data=data, labels=labels, parameters=thetas))
        print("Cost= {}".format(J[-1]))

    save_weights(weights=thetas)

    return thetas, J


def RMSE(data, labels, weights):
    # Root Mean Square Error to compute accuracy
    predictions = [np.dot(wine, weights) for wine in data]
    rmse = sum(np.square(labels - predictions)) / len(labels)
    return np.sqrt(rmse)


def accuracy(data, labels, weights):
    predictions = [np.dot(wine, weights) for wine in data]
    count = 0.0
    for i, pred in enumerate(predictions):
        if np.round(pred) == labels[i]:
            count += 1

    return (count / len(labels)) * 100.0


def main():
    data_frame, x, y = get_data(file_path='./winequality-red.csv')
    # correlation_matrix(df=data_frame)
    std_x = standardize_data(data=x)

    # Using 85% of data for training, 15% for testing
    x_train, x_test, y_train, y_test = train_test_split(std_x, y, test_size=0.15, random_state=42, shuffle=True)

    epochs = 5000
    alpha = 0.01
    optimized_theta, J_train = gradient_descent(data=x_train, labels=y_train, epochs=epochs, learning_rate=alpha)

    rmse = RMSE(x_test, y_test, optimized_theta)

    print("Root Mean Square Error on Test data: {}".format(rmse))
    print("Training Accuracy: {}%".format(accuracy(x_train, y_train, optimized_theta)))
    print("Testing Accuracy: {}%".format(accuracy(x_test, y_test, optimized_theta)))

    # continue project by implementing MultiClass Logistic Regression

    
if __name__ == '__main__':
    main()
