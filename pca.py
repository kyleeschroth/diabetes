
import pandas as pd
import numpy as np

df = pd.read_csv('/Users/kyleeschroth/Downloads/diabetes_data_upload.csv')
print(df.isnull().values.any())
df.dropna(how="all", inplace=True)

#Converting each attribute into 0s and 1s
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

X = df.iloc[:,0:15].values
y = df.iloc[:,15].values

#1. Standardization
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
print(X_std)

#2.1 Covariance Matrix
#mean_vec = np.mean(X_std, axis=0)
#cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#print('Covariance matrix \n%s' %cov_mat)
#print('Covariance matrix \n')
cov_mat= np.cov(X_std, rowvar=False)
#print(cov_mat)

#2.2 Eigenvectors and eigenvalues computation from the covariance matrix
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#2.3 EigenVectors verification
sq_eig = []
for i in eig_vecs:
    sq_eig.append(i**2)
print(sq_eig)
sum(sq_eig)
print("sum of squares of each values in an eigen vector is \n", 0.27287211+ 0.13862096+0.51986524+ 0.06864169)
for ev in eig_vecs: np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

#3 Selecting the principal components
#3.1 Sorting eigenvalues
#Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print(type(eig_pairs))
#Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()
print("\n",eig_pairs)
#Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('\n\n\nEigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#3.2 Explained variance
tot = sum(eig_vals)
print("\n",tot)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print("\n\n1. Variance Explained\n",var_exp)
cum_var_exp = np.cumsum(var_exp)
print("\n\n2. Cumulative Variance Explained\n",cum_var_exp)
print("\n\n3. Percentage of variance the first two principal components each contain\n ",var_exp[0:10])
print("\n\n4. Percentage of variance the first two principal components together contain\n",sum(var_exp[0:10]))

#4 Construct the projection matrix W from the selected k eigenvectors

matrix_w = np.hstack((eig_pairs[0][1].reshape(15,1), eig_pairs[1][1].reshape(15,1), eig_pairs[2][1].reshape(15,1),
                      eig_pairs[3][1].reshape(15,1), eig_pairs[4][1].reshape(15,1), eig_pairs[5][1].reshape(15,1),
                      eig_pairs[6][1].reshape(15,1), eig_pairs[7][1].reshape(15,1), eig_pairs[8][1].reshape(15,1),
                      eig_pairs[9][1].reshape(15,1), eig_pairs[10][1].reshape(15,1)))
#hstack: Stacks arrays in sequence horizontally (column wise).
#print('Matrix W:\n', matrix_w)

#5 Projection Onto the New Feature Spec
Y = X_std.dot(matrix_w)
principalDf = pd.DataFrame(data = Y , columns = ['principal component 1', 'principal component 2', 'principal component 3',
                                                 'principal component 4', 'principal component 5', 'principal component 6',
                                                 'principal component 7', 'principal component 8', 'principal component 9',
                                                 'principal component 10', 'principal component 11'])

finalDf = pd.concat([principalDf,pd.DataFrame(y,columns = ['class'])], axis = 1)

#Subsetting the dataset to only include those chosen by PCA.
df = df[['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'Itching', 'Irritability', 'muscle stiffness', 'Obesity', 'class']]