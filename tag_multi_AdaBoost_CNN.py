import numpy as np
from numpy.core.umath_tests import inner1d
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class AdaBoostClassifier(object):
    '''
    Custom AdaBoost Classifier

    Parameters
    -----------
    base_estimator: object  # Base estimator (weak classifier)
    n_estimators: integer, optional (default=50)  # Maximum number of estimators (weak classifiers)
    learning_rate: float, optional (default=1)  # Learning rate
    algorithm: {'SAMME','SAMME.R'}, optional (default='SAMME.R')  # Type of boosting algorithm
    random_state: int or None, optional (default=None)  # Random state

    Attributes
    -------------
    estimators_: list of base estimators  # List to store base estimators
    estimator_weights_: array of floats  # Array to store weights of each base estimator
    estimator_errors_: array of floats  # Array to store classification errors of each estimator

    Input Data Description: Includes feature data and label data.
    1. Feature Data (X):
        Feature data is typically a 2D array (or similar structure), where each row represents a sample and each column represents a feature.
        Features should be numeric, as most machine learning models (including AdaBoost) operate on numerical data.
        If the feature data contains non-numeric data (e.g., categorical data), it usually needs to be converted to numeric, e.g., using one-hot encoding or label encoding.
        The scale of feature data (i.e., the number of samples and features) depends on the specific application, but there should be enough samples for the model to learn patterns in the data.
    2. Label Data (y):
        Label data is a 1D array, where each element corresponds to a row in the feature data (a sample) and indicates the class or output of that sample.
        In classification tasks, labels are typically categorical and can be strings or integers. For example, in a binary classification problem, labels might be [0, 1]; in a multi-class problem, labels could be [0, 1, 2, ..., n].
        When using AdaBoost for training, if the labels are strings, they usually need to be converted to integers or one-hot encoded.
    3. Data Preprocessing:
        Data preprocessing is an important step before training the model. It includes standardizing or normalizing feature data, handling missing values, converting categorical features, etc.
        In tree-based models (like decision trees, often used as base estimators for AdaBoost), feature standardization or normalization is not necessary, as tree models are not distance-based algorithms. However, if the base estimator is a distance-based model (like SVM or KNN), feature standardization or normalization becomes important.

    Function Description:
    1. Initialization (__init__ method):
        First, when creating an instance of the AdaBoost classifier, the __init__ method is called. It initializes the classifier's parameters, such as base estimator, learning rate, number of iterations, etc.
    2. Training the Model (fit method):
        Next, use the fit method to train the model. In this method, the model iterates over the training data multiple times.
        In each iteration, the fit method calls the boost method to perform boosting.
    3. Boosting Process (boost method):
        The boost method calls either the discrete_boost or real_boost method based on the selected algorithm version (SAMME or SAMME.R).
        In these methods, the base estimator (e.g., CNN) is trained and sample weights are updated.
        These methods also call deepcopy_CNN to create a new instance of the base estimator to ensure independent training.
    4. Prediction (predict and predict_proba methods):
        1. SAMME + predict:
            In this case, the predict method combines the weighted votes of all base estimators to determine the class of each sample. Each base estimator's contribution to the final classification is determined by its performance during training.
        2. SAMME + predict_proba:
            Although the SAMME algorithm is primarily based on class labels, the predict_proba method still calculates the probability for each class. This is done by combining the weighted predictions of each base estimator and applying an appropriate probability calculation method.
        3. SAMME.R + predict:
            In the SAMME.R algorithm, even though the algorithm uses class probabilities internally, the predict method still returns a definite class label based on the weighted probabilities.
        4. SAMME.R + predict_proba:
            In this case, the predict_proba method uses the probability information in the SAMME.R algorithm to calculate the probability for each class. Since SAMME.R itself is based on probability, the probability estimates may be more accurate in this case.

    '''

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0,
                 algorithm='SAMME.R', random_state=None, epochs=10,
                 copy_previous_estimator=True, callbacks=None, batch_size=32):  # , *args, **kwargs

        # Initialize class attributes
        self.base_estimator_ = base_estimator  # Base estimator
        self.n_estimators_ = n_estimators  # Number of weak classifiers
        self.learning_rate_ = learning_rate  # Learning rate
        self.algorithm_ = algorithm  # Algorithm used
        self.random_state_ = random_state  # Random state
        self.epochs = epochs  # Number of epochs for CNN training
        self.copy_previous_estimator = copy_previous_estimator  # New parameter to control whether to copy the estimator
        self.callbacks = callbacks if callbacks is not None else []  # Add callbacks parameter defaulting to an empty list
        self.batch_size = batch_size
        self.estimators_ = list()  # List to store trained estimators
        self.estimator_weights_ = np.zeros(self.n_estimators_)  # Initialize weights of estimators
        self.estimator_errors_ = np.ones(self.n_estimators_)  # Initialize error rates of estimators

    def _samme_proba(self, estimator, n_classes, X):
        """Calculate probabilities according to Algorithm 4, step 2, equation c) in Zhu et al.'s paper.

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        # Use estimator to predict probabilities for each sample in X
        proba = estimator.predict(X)

        # Adjust probabilities to ensure valid logarithm calculations.
        # Replace all probabilities smaller than the smallest positive float to avoid invalid values or negatives in log calculations.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        # Compute log probabilities
        log_proba = np.log(proba)

        # Adjust probabilities according to SAMME.R algorithm to calculate final weighted probability for each sample.
        # This adjustment is the core of the SAMME.R algorithm, emphasizing misclassified samples by adjusting the original log probabilities.
        return (n_classes - 1) * (log_proba - (1. / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Calculate the number of samples in input data X
        self.n_samples = X_train.shape[0]
        # Sort class labels to ensure consistent class order in predictions
        self.classes_ = np.array(sorted(list(set(y_train))))
        # Calculate the number of classes
        self.n_classes_ = len(self.classes_)

        # Iterate over each estimator for training
        for iboost in range(self.n_estimators_):
            # Initialize sample weights to equal values in the first iteration
            if iboost == 0:
                sample_weights = np.ones(self.n_samples) / self.n_samples
                self.sample_weights_ = sample_weights
            else:
                sample_weights = self.sample_weights_
            # Call boost method to train the current estimator and update sample weights
            sample_weights, estimator_weight, estimator_error = self.boost(X_train, y_train, X_val, y_val, sample_weights)

            # Stop training if early stopping condition (e.g., estimator error is None) is detected
            if estimator_error is None:
                break

            # Record the current estimator's error rate and weight
            self.sample_weights_ = sample_weights
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            # Stop iteration if the estimator's error rate is less than or equal to 0
            if estimator_error <= 0:
                break

        # Return the current object to allow method chaining
        return self

    def boost(self, X_train, y_train, X_val, y_val, sample_weight):
        # Execute different boosting methods based on the selected AdaBoost algorithm version
        if self.algorithm_ == 'SAMME':
            # Call discrete_boost method if SAMME algorithm is selected
            return self.discrete_boost(X_train, y_train, X_val, y_val, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            # Call real_boost method if SAMME.R algorithm is selected
            return self.real_boost(X_train, y_train, X_val, y_val, sample_weight)

    def real_boost(self, X_train, y_train, X_val, y_val, sample_weight):
        # Check if there are already trained estimators
        if self.copy_previous_estimator and len(self.estimators_) > 0:
            # Copy the last trained estimator if the option is enabled
            estimator = self.deepcopy_CNN(self.estimators_[-1])
        else:
            # Otherwise, create a new instance from the base estimator each time
            estimator = self.deepcopy_CNN(self.base_estimator_)

        # Set random state in the estimator if specified
        if self.random_state_:
            estimator.set_params(random_state=self.random_state_)

        # Convert labels to one-hot encoding format for CNN training
        lb = OneHotEncoder(sparse=False)
        y_train_encoded = y_train.reshape(len(y_train), 1)
        y_train_encoded = lb.fit_transform(y_train_encoded)

        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.reshape(len(y_val), 1)
            y_val_encoded = lb.fit_transform(y_val_encoded)

        # Train the estimator with the given data and sample weights
        if X_val is not None and y_val is not None:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, validation_data=(X_val, y_val_encoded),
                          callbacks=self.callbacks)
        else:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, callbacks=self.callbacks)

        # Predict training data
        y_pred = estimator.predict(X_train)
        # Convert CNN output to class labels
        y_pred_l = np.argmax(y_pred, axis=1)
        # Calculate incorrect predictions
        incorrect = y_pred_l != y_train

        # Calculate the estimator's error rate
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # Stop boosting if the error rate is worse than random guessing
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        # Predict probabilities again
        y_predict_proba = estimator.predict(X_train)  # In Keras, usually use predict method to get predicted probabilities
        # Replace zeros in probabilities to avoid issues in log calculations
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        # Prepare to update sample weights
        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y_train[:, np.newaxis])
        # Calculate intermediate variable for updating sample weights
        intermediate_variable = (-1. * self.learning_rate_ * (
                    ((self.n_classes_ - 1) / self.n_classes_) * inner1d(y_coding, np.log(y_predict_proba))))

        # Update sample weights
        sample_weight *= np.exp(intermediate_variable)
        # Calculate the total sum of sample weights
        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # Normalize sample weights
        sample_weight /= sample_weight_sum

        # Add the current estimator to the list
        self.estimators_.append(estimator)

        # Return updated sample weights, estimator weight (set to 1 here), and estimator error
        return sample_weight, 1, estimator_error

    def deepcopy_CNN(self, base_estimator0):
        # Create a deep copy of the base estimator (CNN)
        # Get the configuration of the base estimator
        config = base_estimator0.get_config()

        # Create a new Sequential model based on the configuration
        estimator = Sequential.from_config(config)

        # Get the weights of the base estimator
        weights = base_estimator0.get_weights()
        # Set these weights in the new model
        estimator.set_weights(weights)

        # Decide which learning rate configuration to use based on use_initial_optimizer_config
        if self.copy_previous_estimator:
            # Use the current optimizer configuration (including current learning rate)
            original_optimizer_config = base_estimator0.optimizer.get_config()
            current_learning_rate = K.get_value(base_estimator0.optimizer.lr)
            original_optimizer_config['learning_rate'] = current_learning_rate
        else:
            # Use the initial learning rate (assumed here or obtained from somewhere)
            original_optimizer_config = base_estimator0.optimizer.get_config()
            initial_learning_rate = 0.01  # Assumed initial learning rate
            original_optimizer_config['learning_rate'] = initial_learning_rate

        # Compile the new model using the updated optimizer configuration
        estimator.compile(loss='categorical_crossentropy',
                          optimizer=Adam(**original_optimizer_config),
                          metrics=['accuracy'])

        # Return the newly created and compiled model
        return estimator

    def discrete_boost(self, X_train, y_train, X_val, y_val, sample_weight):
        # Check if there are already trained estimators
        if self.copy_previous_estimator and len(self.estimators_) > 0:
            # Copy the last trained estimator if the option is enabled
            estimator = self.deepcopy_CNN(self.estimators_[-1])
        else:
            # Otherwise, create a new instance from the base estimator each time
            estimator = self.deepcopy_CNN(self.base_estimator_)

        # Set random state in the estimator if specified
        if self.random_state_:
            estimator.set_params(random_state=1)

        # Convert labels to one-hot encoding format for CNN training
        lb = OneHotEncoder(sparse=False)
        y_train_encoded = y_train.reshape(len(y_train), 1)
        y_train_encoded = lb.fit_transform(y_train_encoded)

        if X_val is not None and y_val is not None:
            y_val_encoded = y_val.reshape(len(y_val), 1)
            y_val_encoded = lb.fit_transform(y_val_encoded)

        # Train the estimator
        if X_val is not None and y_val is not None:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, validation_data=(X_val, y_val_encoded),
                          callbacks=self.callbacks)
        else:
            estimator.fit(X_train, y_train_encoded, sample_weight=sample_weight, epochs=self.epochs,
                          batch_size=self.batch_size, callbacks=self.callbacks)

        # Predict training data
        y_pred = estimator.predict(X_train)
        # Convert CNN output to class labels
        y_pred_l = np.argmax(y_pred, axis=1)
        # Calculate incorrect predictions
        incorrect = y_pred_l != y_train

        # Calculate the estimator's error rate
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # Stop boosting if the error rate is zero
        if estimator_error == 0:
            return sample_weight, 0, 0

        # Update the estimator's weight
        estimator_weight = self.learning_rate_ * (
                np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        # Stop boosting if the estimator's weight is less than or equal to 0
        if estimator_weight <= 0:
            return None, None, None

        # Update sample weights
        sample_weight *= np.exp(estimator_weight * incorrect)

        # Correct sample_weight to ensure no invalid values or values less than or equal to zero
        sample_weight[sample_weight <= 0] = np.finfo(float).eps

        # Calculate the total sum of sample weights
        sample_weight_sum = np.sum(sample_weight, axis=0)

        # Stop boosting if the total sum of sample weights is less than or equal to 0
        if sample_weight_sum <= 0:
            return None, None, None

        # Normalize sample weights
        sample_weight /= sample_weight_sum

        # Add the current estimator to the list of estimators
        self.estimators_.append(estimator)

        # Return updated sample weights, estimator weight, and estimator error
        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        # Get the number of classes and class labels
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        # Initialize prediction result variable
        pred = None

        # If using SAMME.R algorithm
        if self.algorithm_ == 'SAMME.R':
            # In SAMME.R, all estimator weights are considered to be 1
            # Calculate the sum of weighted prediction probabilities from all estimators
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # If using SAMME algorithm
            # Calculate the sum of weighted predictions from all estimators
            # Use the CNN output to determine the predicted class for each sample
            pred = sum((estimator.predict(X).argmax(axis=1) == classes).T * w
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))

        # Divide the sum of weighted predictions by the total sum of estimator weights to get the average effect
        pred /= self.estimator_weights_.sum()

        # For binary classification
        if n_classes == 2:
            # Negate the first column (representing negative class predictions)
            pred[:, 0] *= -1
            # Calculate the sum of predictions to determine the final predicted class
            pred = pred.sum(axis=1)
            # Select the class based on the sign of the prediction
            return self.classes_.take(pred > 0, axis=0)

        # For multi-class classification, select the class with the highest average weighted prediction probability
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def predict_proba(self, X):
        # If using SAMME.R algorithm
        if self.algorithm_ == 'SAMME.R':
            # In SAMME.R, all estimator weights are considered to be 1
            # Calculate the sum of weighted prediction probabilities from all estimators
            proba = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)
        else:  # If using SAMME algorithm
            # Calculate the sum of weighted prediction probabilities from all estimators
            # Use the predict_proba method of each estimator to get predicted probabilities
            proba = sum(
                estimator.predict_proba(X) * w for estimator, w in zip(self.estimators_, self.estimator_weights_))

        # Divide the sum of weighted prediction probabilities by the total sum of estimator weights to get the average effect
        proba /= self.estimator_weights_.sum()

        # Exponentially scale the probabilities to bring them within a reasonable range
        # This is a feature of the SAMME.R algorithm to adjust the predicted probabilities
        proba = np.exp((1. / (self.n_classes_ - 1)) * proba)

        # Normalize the predicted probabilities to ensure the sum of class probabilities for each sample is 1
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0  # Avoid division by zero
        proba /= normalizer

        # Return the predicted probabilities
        return proba
