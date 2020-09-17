# Common imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from mpl_toolkits.mplot3d import Axes3D

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

def MSE(y, y_tilde):
    n = len(y)
    return np.sum((y-y_tilde)**2)/n

def R2(y, y_tilde):
    return 1- np.sum((y-y_tilde)**2)/np.sum((y-np.mean(y))**2)


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def create_X(x, y, n):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


# Making meshgrid of datapoints and compute Franke's function
np.random.seed(4155)
n = 5
N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
X, Y = np.meshgrid(x, y)
z_matrix = FrankeFunction(X, Y)
z = np.ravel(z_matrix)

X_D = create_X(X, Y, n=n)
print(X_D.shape, z.shape)

# split in training and test data
X_train, X_test, y_train, y_test = train_test_split(X_D,z,test_size=0.25)
print(X_train.shape, y_train.shape)

OLSbeta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
ytilde_OLS = X_train@OLSbeta

print("MSE train before scale: ", MSE(y_train, ytilde_OLS))
print("R2 train before scale: ", R2(y_train, ytilde_OLS))

ypred_OLS = X_test@OLSbeta
print("------------")

print("MSE test before scale: ", MSE(y_test, ypred_OLS))
print("R2 test before scale: ", R2(y_test, ypred_OLS))

scaler = StandardScaler()
scaler.fit(X_train[:, 1:])
X_train_scaled = scaler.transform(X_train[:, 1:])
X_test_scaled = scaler.transform(X_test[:, 1:])

ones_train = np.ones((len(y_train), 1))
ones_test = np.ones((len(y_test), 1))
X_train_scaled = np.hstack((ones_train, X_train_scaled))
X_test_scaled = np.hstack((ones_test, X_test_scaled))

OLSbeta_scale = np.linalg.inv(X_train_scaled.T@X_train_scaled)@X_train_scaled.T@y_train
ytilde_scale = X_train_scaled@OLSbeta_scale

print("MSE train after scale: ", MSE(y_train, ytilde_scale))
print("R2 train after scale: ", R2(y_train, ytilde_scale))

ypred_scale = X_test_scaled@OLSbeta_scale
print("------------")

print("MSE test after scale: ", MSE(y_test, ypred_scale))
print("R2 test after scale: ", R2(y_test, ypred_scale))


"""
print("Sklearn fitting: ")
clf = skl.LinearRegression().fit(X_train, y_train)

# The mean squared error and R2 score
print("MSE before scaling: {:.5f}".format(mean_squared_error(clf.predict(X_test), y_test)))
print("R2 score before scaling {:.5f}".format(clf.score(X_test,y_test)))

print("Feature min values before scaling:\n {}".format(X_train.min(axis=0)))
print("Feature max values before scaling:\n {}".format(X_train.max(axis=0)))

print("Feature min values after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("Feature max values after scaling:\n {}".format(X_train_scaled.max(axis=0)))

clf = skl.LinearRegression().fit(X_train_scaled, y_train)


print("MSE after  scaling: {:.5f}".format(mean_squared_error(clf.predict(X_test_scaled), y_test)))
print("R2 score for  scaled data: {:.5f}".format(clf.score(X_test_scaled,y_test)))"""

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X,Y,z_matrix)
plt.show()
