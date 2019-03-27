"""QSVM Kernel Classifier
Simple sklearn interface for the Qiskit QSVMKernel.
"""

from common import utils
from qiskit import Aer
from qiskit_aqua import QuantumInstance, run_algorithm, set_aqua_logging
from qiskit_aqua.algorithms import QSVMKernel
from qiskit_aqua.components.feature_maps import SecondOrderExpansion
from qiskit_aqua.input import SVMInput
from qiskit_aqua.utils import (map_label_to_class_name,
                               split_dataset_to_data_and_labels)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class QSVMKernelClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        # Validate the shape of X and y
        X, y = check_X_y(X, y)

        # Map the features to qubits.
        feature_map = SecondOrderExpansion(
            num_qubits=X.shape[1], depth=2, entanglement='linear')

        # Build the qsvm.
        training_dataset = utils.make_qiskit_style_dataset(X, y)
        self.impl_ = QSVMKernel(feature_map, training_dataset)

        # Run the qsvm on the backend (for now, a simulator).
        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(
            backend, shots=1024, seed=self.seed, seed_mapper=self.seed)
        self.impl_.run(quantum_instance, print_info=True)
        return self

    def decision_function(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['impl_'])

        return self.impl_.instance.get_predicted_confidence(X)

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['impl_'])

        return self.impl_.predict(X)

    def ret(self):
        return self.impl_.ret
