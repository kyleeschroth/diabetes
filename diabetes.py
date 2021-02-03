
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Put this when it's called
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
#from sklearn.model_selection import plot_validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

#Importing data from downloads.
df = pd.read_csv('/Users/kyleeschroth/Downloads/diabetes_data_upload.csv')
df_raw = df.copy()

# yes = 1, no = 0; male = 1, female = 0; positive = 1, negative = 0
df['Gender'].replace(['Male', 'Female'], [1,0], inplace=True)
df['Polyuria'].replace(['Yes','No'], [1,0], inplace = True)
#df['Polyuria'].replace({'Yes':1, 'No':0})
df['Polydipsia'].replace(['Yes','No'], [1,0], inplace = True)
df['sudden weight loss'].replace(['Yes','No'], [1,0], inplace = True)
df['weakness'].replace(['Yes','No'], [1,0], inplace = True)
df['Polyphagia'].replace(['Yes','No'], [1,0], inplace = True)
df['Genital thrush'].replace(['Yes','No'], [1,0], inplace = True)
df['visual blurring'].replace(['Yes','No'], [1,0], inplace = True)
df['Itching'].replace(['Yes','No'], [1,0], inplace = True)
df['Irritability'].replace(['Yes','No'], [1,0], inplace = True)
df['delayed healing'].replace(['Yes','No'], [1,0], inplace = True)
df['partial paresis'].replace(['Yes','No'], [1,0], inplace = True)
df['muscle stiffness'].replace(['Yes','No'], [1,0], inplace = True)
df['Alopecia'].replace(['Yes','No'], [1,0], inplace = True)
df['Obesity'].replace(['Yes','No'], [1,0], inplace = True)
df['class'].replace(['Positive','Negative'], [1,0], inplace = True)

#Making each attribute Categorical.
df['Gender'] = pd.Categorical(df['Gender'])
df['Polyuria'] = pd.Categorical(df['Polyuria'])
df['Polydipsia'] = pd.Categorical(df['Polydipsia'])
df['sudden weight loss'] = pd.Categorical(df['sudden weight loss'])
df['weakness'] = pd.Categorical(df['weakness'])
df['Polyphagia'] = pd.Categorical(df['Polyphagia'])
df['Genital thrush'] = pd.Categorical(df['Genital thrush'])
df['visual blurring'] = pd.Categorical(df['visual blurring'])
df['Itching'] = pd.Categorical(df['Itching'])
df['Irritability'] = pd.Categorical(df['Irritability'])
df['delayed healing'] = pd.Categorical(df['delayed healing'])
df['partial paresis'] = pd.Categorical(df['partial paresis'])
df['muscle stiffness'] = pd.Categorical(df['muscle stiffness'])
df['Alopecia'] = pd.Categorical(df['Alopecia'])
df['Obesity'] = pd.Categorical(df['Obesity'])
df['class'] = pd.Categorical(df['class'])

#Subsetting the dataset to only include those chosen by PCA.
df = df[['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'Itching', 'Irritability', 'muscle stiffness', 'Obesity', 'class']]

#Creating the predictor variable as an integer instead of categorical.
#df['class'] = df['class'].eq('yes').mul(1)
df['class'] = df['class'].astype('int16')

#Create dummy variables.
df_dum = pd.get_dummies(df, drop_first=True)

X = df_dum[df_dum.loc[:, df_dum.columns != 'class'].columns]
y = df_dum['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debug
print('Inputs: \n', X_train.head())
print('Outputs: \n', y_train.head())

# Fit logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.predict(X)
y_pred = logreg.predict(X)

#compute classification accuracy for the logistic regression model
from sklearn import metrics

print(metrics.accuracy_score(y, y_pred))

# Model performance
scores = cross_val_score(logreg, X_train, y_train, cv=10)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curves
title = "Learning Curves (Logistic Regression)"
cv = 10
plot_learning_curve(logreg, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);

#plt.show()

# Plot validation curve
# title = 'Validation Curve (Logistic Regression)'
# param_name = 'C'
# param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# cv = 10
# plot_validation_curve(estimator=logreg, title=title, X=X_train, y=y_train, param_name=param_name,
#                       ylim=(0.5, 1.01), param_range=param_range);
# plt.show()

from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))

#K nearest neigbors

import random
from numpy.random import permutation
import math

#randomly shuffle the indices
random_indices = permutation(df.index)
#set a cutoff for how many items we want to test.
test_cutoff = math.floor(len(df)/5)
#generate the test set by takijng hte first 1/5 of the radomly shuffled indices
test = df.loc[random_indices[1:test_cutoff]]
#generate the train set with the rest of the data
train = df.loc[random_indices[test_cutoff:]]
#columns we are making predictions with
x_columns = ['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'Itching', 'Irritability', 'muscle stiffness', 'Obesity']
#predictor column
y_column = ['class']
from sklearn.neighbors import KNeighborsRegressor
#create the knn model; looking at the five closest neighbors
knn = KNeighborsRegressor(n_neighbors=5)
#fit the model on the training set
knn.fit(train[x_columns],train[y_column])
#make point predictions on the test set using the fit model
predictions = knn.predict(test[x_columns])
#get the actual values for the test set
actual = test[y_column]
#compute the average error of our predictions
error = ((abs(predictions-actual)).sum())/(len(predictions))
print('Error: %f' % error)
#

#KNN SIMPLE WAY  -------------------------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X = df[df.loc[:, df.columns != 'class'].columns]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
#
# markers = ('s', 'x', 'o')
# colors = ('red', 'blue', 'lightgreen')
# cmap = ListedColormap(colors[:len(np.unique(y_test))])
# for idx, cl in enumerate(np.unique(y)):
#     plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), marker=markers[idx], label=cl)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5
                           , p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(X_train_std, y_train)))
print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(X_test_std, y_test)))


# import warnings
#
# def versiontuple(v):
#     return tuple(map(int, (v.split("."))))
#
#
# def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#
#     # setup marker generator and color map
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
#
#     # plot the decision surface
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                            np.arange(x2_min, x2_max, resolution))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())
#
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#                     alpha=0.8, c=cmap(idx),
#                     marker=markers[idx], label=cl)
#
# plot_decision_regions(X_test_std, y_test, knn)
# plt.show()


# #KNN ANOTHER WAY ---------------------------------------------------------------------------
# knnX = df[df.loc[:, df.columns != 'class'].columns]
# knny = df['class']
#
# knnX_train, knnX_test, knny_train, knny_test = train_test_split(knnX, knny, test_size=0.2, random_state=0)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# knnX_train = sc.fit_transform(knnX_train)
# knnX_test = sc.transform(knnX_test)
#
# #fitting Classifier to the training set
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors= 2)
# classifier.fit(knnX_train, knny_train)
#
# #Predicting the test set results
# knny_pred = classifier.predict(knnX_test)
#
# #Mkaing the confusion matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(knny_test, knny_pred)
#
# # Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = knnX_test, knny_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Classifier (Training set)')
# plt.legend()
# plt.show()

# training the model on training set
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

from sklearn.svm import SVC

svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(X_train_std, y_train)

print('The accuracy of the svm classifier on training data is {:.2f} out of 1'.format(svm.score(X_train_std, y_train)))

print('The accuracy of the svm classifier on test data is {:.2f} out of 1'.format(svm.score(X_test_std, y_test)))