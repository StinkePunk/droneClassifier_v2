import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from training import train_classifier
from plotting import plot_training_history, plot_confusion_matrix, plot_probability_distribution

class DroneClassifier:
    """
    Handles training, loading, prediction, and evaluation of a classification model.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.history = None

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model and save it to disk.
        """
        self.model, self.history = train_classifier(X_train, y_train, X_val, y_val)
        self.model.save(self.model_path)

    def load(self):
        """chunked_data_train
        Load a pre-trained model from disk.
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def predict(self, X):
        """
        Predict probabilities for input features.
        """
        return self.model.predict(X)

    def evaluate(self, X_val, y_val):
        """
        Evaluate model with confusion matrix and prediction histogram.
        """
        y_pred_prob = self.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        plot_confusion_matrix(cm)
        plot_probability_distribution(y_pred_prob)

    def summary(self):
        """
        Print the model architecture summary.
        """
        if self.model:
            self.model.summary()
        else:
            print("No model loaded.")