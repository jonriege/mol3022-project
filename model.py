from sklearn.ensemble import RandomForestClassifier
from preprocessing import read_data


def run():
    print("Preprocessing the training data.")
    X_train, y_train = read_data('data/train_set.fasta')
    print("Preprocessing the test data.")
    X_test, y_test = read_data('data/benchmark_set.fasta')

    print("Running model 1.")
    model_1 = RandomForestClassifier(n_estimators=10, criterion='gini')
    model_1.fit(X_train, y_train)
    accuracy_1 = model_1.score(X_test, y_test)
    print(f"Model 1 accuracy: {accuracy_1}\n")

    print("Running model 2.")
    model_2 = RandomForestClassifier(n_estimators=100, criterion='gini')
    model_2.fit(X_train, y_train)
    accuracy_2 = model_2.score(X_test, y_test)
    print(f"Model 2 accuracy: {accuracy_2}\n")

    print("Running model 3.")
    model_3 = RandomForestClassifier(n_estimators=10, criterion='entropy')
    model_3.fit(X_train, y_train)
    accuracy_3 = model_3.score(X_test, y_test)
    print(f"Model 3 accuracy: {accuracy_3}\n")

    print("Running model 4.")
    model_4 = RandomForestClassifier(n_estimators=100, criterion='entropy')
    model_4.fit(X_train, y_train)
    accuracy_4 = model_4.score(X_test, y_test)
    print(f"Model 4 accuracy: {accuracy_4}\n")


if __name__ == '__main__':
    run()
