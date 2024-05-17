from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import time

def train_random_forest(X_train, y_train):
    start_time = time.time()
    
    # Initialize the Random Forest classifier
    print('Model training initialized')
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    
    # Train the classifier
    rf_classifier.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Plot one of the trees
    plt.figure(figsize=(20, 10))
    plot_tree(rf_classifier.estimators_[0], filled=True)
    plt.show()

    return rf_classifier

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    
    # Predict on the test set
    print("Model evaluation started")
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluation elapsed time: {elapsed_time:.2f} seconds")

    return accuracy
