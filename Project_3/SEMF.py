import numpy as np
import matplotlib.pyplot as plt

a_v = 15.56
a_s = 17.23
ac = 0.697
aa = 93.14
ap = 12

Mp = 938.27
Mn = 939.57
Me = 0.511

def semf(Z,A):
    if (np.mod(Z,2) == 0) and (np.mod(A-Z,2) == 0):
        mass = Z*(Mp+Me) + (A-Z)*Mn - a_v*A + a_s*(A**(2/3)) + ac*(Z*(Z-1))/(A**(1/3)) + aa*((Z-A/2)**2)/A - ap/(A**0.5)
    elif (np.mod(Z,2) == 1) and (np.mod(A-Z,2) == 1):
        mass = Z*(Mp+Me) + (A-Z)*Mn - a_v*A + a_s*(A**(2/3)) + ac*(Z*(Z-1))/(A**(1/3)) + aa*((Z-A/2)**2)/A + ap/(A**0.5)
    else:
        mass = Z*(Mp+Me) + (A-Z)*Mn - a_v*A + a_s*(A**(2/3)) + ac*(Z*(Z-1))/(A**(1/3)) + aa*((Z-A/2)**2)/A
    return mass

def neutron_removal(Z,A):
    energy = semf(Z,A-1) + Mn - semf(Z,A)
    return energy

def stable_nucleus(A):
    beta = aa + (Mn - Mp - Me)
    gamma = aa/A + ac/(A**(1/3))
    return beta/(2*gamma)

print(semf(20,48))

print(neutron_removal(20,44))

print(neutron_removal(50,118), neutron_removal(50,119), neutron_removal(8,16))

print(stable_nucleus(136))

Zs = np.linspace(45,70,70-45+1)
mass = np.zeros(len(Zs))

for i in range(len(Zs)):
    mass[i] = semf(Zs[i], 136)

plt.plot(Zs, mass)
plt.show()
