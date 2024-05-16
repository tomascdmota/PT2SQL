from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time

def train_random_forest(X_train, y_train):
    start_time = time.time()
    # Initialize the Random Forest classifier
    print('Model training initialized')
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return rf_classifier

def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
