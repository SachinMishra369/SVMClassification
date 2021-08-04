#import  libraries
import pandas as pd
import numpy as np

#Reading the dataset
dataset = pd.read_csv('social_ads.csv')

#dependent variable
y = dataset.iloc[:, -1].values

#independent variable
x = dataset.iloc[:, :-1].values

#Diving the dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=0, test_size=0.25)

#feature scaling scaling the data into a scale so that none of feature get dominant by other features
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#SVM classifier
from sklearn.svm import SVC

#kerne could be linear with accuracy_score 88,rbf with accuracy_score 93
'''
    Cfloat, default=1.0

        Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
    kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’

        Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
    degreeint, default=3

        Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    gamma{‘scale’, ‘auto’} or float, default=’scale’

        Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.

            if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,

            if ‘auto’, uses 1 / n_features.

        Changed in version 0.22: The default value of gamma changed from ‘auto’ to ‘scale’.
    coef0float, default=0.0

        Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    shrinkingbool, default=True

        Whether to use the shrinking heuristic. See the User Guide.
    probabilitybool, default=False

        Whether to enable probability estimates. This must be enabled prior to calling fit, will slow down that method as it internally uses 5-fold cross-validation, and predict_proba may be inconsistent with predict. Read more in the User Guide.
    tolfloat, default=1e-3

        Tolerance for stopping criterion.
    cache_sizefloat, default=200

        Specify the size of the kernel cache (in MB).
    class_weightdict or ‘balanced’, default=None

        Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
    verbosebool, default=False

        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
    max_iterint, default=-1

        Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’

        Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one (‘ovo’) is always used as multi-class strategy. The parameter is ignored for binary classification.

        Changed in version 0.19: decision_function_shape is ‘ovr’ by default.

        New in version 0.17: decision_function_shape=’ovr’ is recommended.

        Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.
    break_tiesbool, default=False

        If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. Please note that breaking ties comes at a relatively high computational cost compared to a simple predict.

        New in version 0.22.
    random_stateint or RandomState instance, default=None

        Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when probability is False. Pass an int for reproducible output across multiple function calls. See Glossary.

Attributes

    support_ndarray of shape (n_SV,)

        Indices of support vectors.
    support_vectors_ndarray of shape (n_SV, n_features)

        Support vectors.
    n_support_ndarray of shape (n_class,), dtype=int32

        Number of support vectors for each class.
    dual_coef_ndarray of shape (n_class-1, n_SV)

        Dual coefficients of the support vector in the decision function (see Mathematical formulation), multiplied by their targets. For multiclass, coefficient for all 1-vs-1 classifiers. The layout of the coefficients in the multiclass case is somewhat non-trivial. See the multi-class section of the User Guide for details.
    coef_ndarray of shape (n_class * (n_class-1) / 2, n_features)

        Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.

        coef_ is a readonly property derived from dual_coef_ and support_vectors_.
    intercept_ndarray of shape (n_class * (n_class-1) / 2,)

        Constants in decision function.
    fit_status_int

        0 if correctly fitted, 1 otherwise (will raise warning)
    classes_ndarray of shape (n_classes,)

        The classes labels.
    probA_ndarray of shape (n_class * (n_class-1) / 2)
    probB_ndarray of shape (n_class * (n_class-1) / 2)

        If probability=True, it corresponds to the parameters learned in Platt scaling to produce probability estimates from decision values. If probability=False, it’s an empty array. Platt scaling uses the logistic function 1 / (1 + exp(decision_value * probA_ + probB_)) where probA_ and probB_ are learned from the dataset [R20c70293ef72-2]. For more information on the multiclass case and training procedure see section 8 of [R20c70293ef72-1].
    class_weight_ndarray of shape (n_class,)

        Multipliers of parameter C for each class. Computed based on the class_weight parameter.
    shape_fit_tuple of int of shape (n_dimensions_of_X,)

        Array dimensions of training vector X.

kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
'''

classifer = SVC(kernel='linear')
classifer.fit(x_train, y_train)
#Logistic regressiomn predict the result
y_pred = classifer.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Visualising the Training set results
# import matplotlib.pyplot  as plt
# from matplotlib.colors import ListedColormap
# X_set, y_set = x_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[: , 0].max() + 1, step = 0.01),
#   np.arange(start = X_set[: , 1].min() - 1, stop = X_set[: , 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#   alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#   plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#     c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('KNN Classifier (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()

# Visualising the testing set results
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(
    np.arange(
        start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(
        start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
print(X1, X2)
plt.contourf(
    X1,
    X2,
    classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i),
        label=j)
plt.title('KNN Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
