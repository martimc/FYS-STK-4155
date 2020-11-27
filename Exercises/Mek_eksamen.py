from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def E1(x, y):
    return 1 + y + np.sqrt(3)*x
def E2(x,y):
    return 1 + y - np.sqrt(3)*x
def E3(x,y):
    return 1-2*y

def S(x,y):
    s = -(E1(x,y)*np.log(E1(x,y)) + E2(x,y)*np.log(E2(x,y)) + E3(x,y)*np.log(E3(x,y)))
    return s

x1 = np.linspace(-np.sqrt(3)/2, np.sqrt(3)/2, 1001)
x = np.linspace(-1,1,1001)
y = np.linspace(-1,1,1001)
X,Y = np.meshgrid(x,y)

Line1 = 1/2
Line2 = -1+np.sqrt(3)*x1
Line3 = -1-np.sqrt(3)*x1
Line4 = np.maximum(Line2, Line3)
#Line5 = np.maximum(Line1, Line2)
#Line6 = np.maximum(Line5, Line3)
F = (1+Y+np.sqrt(3)*X)*(1+Y-np.sqrt(3)*X)*(1-2*Y)
Circ = X**2 + Y**2 - 1

#plt.contour(X,Y,F,[0])
plt.contour(X,Y,Circ,[0], label='$m_1^2+m_8^2=1')
#plt.contourf(X,Y,S(X,Y))
#plt.colorbar()
plt.fill_between(x1,Line4, Line1, color='green', alpha=0.5)
plt.axis((-1.2, 1.2, -1.2, 1.2))
plt.ylabel("$m_8$")
plt.xlabel("$m_1$")
plt.grid()
plt.legend()
plt.savefig("1_f.pdf")
plt.show()

plt.contour(X,Y,Circ,[0], label='$m_1^2+m_8^2=1')
plt.contourf(X,Y,S(X,Y), label='entropy')
plt.colorbar()
plt.axis((-1.2, 1.2, -1.2, 1.2))
plt.ylabel("$m_8$")
plt.xlabel("$m_1$")
plt.grid()
plt.legend()
plt.savefig("1_g.pdf")
plt.show()
