import numpy as np
from common import utils
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RYRZ
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class QSVMVariationalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, seed):
        self.seed = seed

    def _build(self, X, y):
        X, y = check_X_y(X, y)
        self.feature_dim_ = X.shape[1]

        # Map the features to qubits.
        feature_map = SecondOrderExpansion(
            feature_dimension=self.feature_dim_, depth=2, entanglement='linear')

        # Setup the optimizer (use fixed settings for now).
        optimizer = SPSA(max_trials=100, c0=4.0, skip_calibration=True)
        optimizer.set_options(save_steps=1)

        # Setup the variational form.
        var_form = RYRZ(num_qubits=self.feature_dim_, depth=3)

        # Build the qsvm.
        training_dataset = utils.make_qiskit_style_dataset(X, y)
        self.impl_ = VQC(
            optimizer, feature_map, var_form, training_dataset)

        # Set the quantum instance.
        backend = Aer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(
            backend, shots=1024, seed=self.seed, seed_transpiler=self.seed)
        self.impl_._quantum_instance = quantum_instance

    def fit(self, X, y):
        # Build the model.
        self._build(X, y)

        # Run the model.
        self.impl_.run(self.impl_._quantum_instance)
        return self

    def decision_function(self, X):
        return self.impl_.predict(X)[0][:, 1]

    def predict(self, X):
        return self.impl_.predict(X)[1]

    def ret(self):
        return self.impl_.ret

    def save_model(self, file_path):
        self.impl_.save_model(file_path)

    def load_model(self, file_path, dim):
        X = np.zeros((2, dim))
        y = np.array([0, 1])
        self._build(X, y)
        self.impl_.load_model(file_path)
