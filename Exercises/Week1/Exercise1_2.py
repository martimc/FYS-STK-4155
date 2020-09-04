import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
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

x = np.random.rand(100,1)
y = 2.0 + 5*x*x + 0.5*np.random.randn(100,1)

x_sort = np.sort(x[:,0])
y_sort = np.sort(y[:,0])

p = 3
X = np.zeros((len(x_sort), p))

for i in range(p):
    X[:,i] = x_sort**i

DesignMatrix = pd.DataFrame(X)
DesignMatrix.index = x_sort
DesignMatrix.columns = ['1', 'x', 'x^2']
print("Exercise 1.2:")
display(DesignMatrix)

x_train, x_test, y_train, y_test = train_test_split(X, y_sort, test_size = 0.2)

beta = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train
print('beta = ', beta)

y_pred = beta[0]*np.ones(len(x_sort)) + beta[1]*x_sort + beta[2]*x_sort**2
y_real = 2*np.ones(len(x_sort)) + 5*x_sort**2

plt.plot(x_sort, y_sort, 'ro', label='data')
plt.plot(x_sort, y_pred, 'b-', label='pred')
plt.plot(x_sort, y_real, 'g-', label='real')
plt.legend()
save_fig('exercise1_2')
plt.show()

ytilde = x_train @ beta
ypred = x_test @ beta
MSE_train = MSE(y_train, ytilde)
MSE_test = MSE(y_test, ypred)
R2_train = R2(y_train, ytilde)
R2_test = R2(y_test, ypred)

print('training values, MSE: %.4f, R2: %.4f' % (MSE_train, R2_train))
print('test values, MSE: %.4f, R2: %.4f' % (MSE_test, R2_test))
