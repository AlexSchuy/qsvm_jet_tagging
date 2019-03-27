from common import utils
from qiskit_aqua.algorithms import SVM_Classical
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SVMClassicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma):
        self.gamma = gamma

    def ret(self):
        return self.impl_.ret

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        training_dataset = utils.make_qiskit_style_dataset(X, y)
        self.impl_ = SVM_Classical(training_dataset, gamma=self.gamma)
        self.impl_.instance.test_dataset = None

        self.impl_._run()

    def decision_function(self, X):
        check_is_fitted(self, ['impl_'])

        ret = self.ret()
        alphas = ret['svm']['alphas']
        bias = ret['svm']['bias']
        svms = ret['svm']['support_vectors']
        yin = ret['svm']['yin']
        kernel_matrix = self.impl_.instance.construct_kernel_matrix(
            X, svms, self.impl_.instance.gamma)

        total_num_points = X.shape[0]
        lconf = np.zeros(total_num_points)
        for tin in range(total_num_points):
            ltot = 0
            for sin in range(len(svms)):
                l = yin[sin] * alphas[sin] * kernel_matrix[tin][sin]
                ltot += l
            lconf[tin] = ltot + bias
        return lconf

    def predict(self, X):
        return self.impl_.predict(X)
