import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

np.random.seed(4155)
N = 1000
x = np.random.rand(N)

x_sort = np.sort(x)
y = 2.0 + 5*x_sort*x_sort + 0.5*np.random.randn(N)
yreal = 2.0 + 5*x_sort*x_sort

dim = np.linspace(2,19,18)
Test_err = np.zeros(len(dim))
Train_err = np.zeros(len(dim))
i = 0

for p in range(2,20):

    X = np.ones((len(x_sort), p))

    for j in range(p):
        X[:,j] = (x_sort)**j

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    OLSbeta = np.linalg.inv(x_train.T@x_train)@x_train.T@y_train

    ytilde_OLS = x_train@OLSbeta

    Train_err[i] = MSE(y_train, ytilde_OLS)
    print("polynomial of degree ", p)
    print("training R2 for OLS: ", R2(y_train, ytilde_OLS))
    print("training MLS for OLS: ", MSE(y_train, ytilde_OLS))
    ypred_OLS = x_test@OLSbeta
    Test_err[i] = MSE(y_test, ypred_OLS)
    print("test R2 for OLS: ", R2(y_test, ypred_OLS))
    print("test MLS for OLS: ", MSE(y_test, ypred_OLS))
    print(" ")
    i += 1

plt.plot(dim, Test_err, label='test error')
plt.plot(dim, Train_err, label='train_err')
plt.legend()
plt.show()
