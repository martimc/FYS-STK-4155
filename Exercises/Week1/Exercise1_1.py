import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
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


infile = open(data_path('EoS.csv'), 'r')

EoS = pd.read_csv(infile, names=("Density", 'Energy'))
EoS['Energy'] = pd.to_numeric(EoS['Energy'], errors='coerce')
EoS = EoS.dropna()
Energies = EoS['Energy']
Density = EoS['Density']

X = np.zeros((len(Density), 4))
for i in range(4):
    X[:,i] = Density**i

DesignMatrix = pd.DataFrame(X)
DesignMatrix.index = Density
DesignMatrix.columns = ['1', 'x', 'x^2', '$x^3$']
print("Exercise 1.1:")
display(DesignMatrix)
