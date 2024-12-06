import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def funcbrekingsindex(theta1, n2):
    c1 = 2 * n2 * d *(1/np.sqrt(1-n1**2/n2**2 * np.sin(theta1)**2)-1)
    c2 = (2*d * n1**2 * np.sin(theta1)**2)/np.sqrt(n2**2 - n1**2 * np.sin(theta1)**2)
    c3 = 2*n1*d * (np.cos(theta1) - 1)
    return (c1 - c2 - c3)/l


def steekproefgemendev(data):
    mu = sum(data)/len(data)
    if type(data) == list:
        data = np.array(data)
    dev = np.sqrt(sum((data - mu)**2) / (len(data) - 1))
    return mu, dev


d = 2e-3
l = 532e-9
n1 = 1.00029

N = [[0,0,0,0,1,0,0,0], [1,1,2,1], [4,6,6,4], [8,6,8,6], [7,8,9,8], 
     [9,11,9,11,11], [10,14,13,12,10,12,11,11], [13,15,13,12,15,15,12,15],
     [19,21,22,21,21,20,22,19], [29,31,30,39,35,37,34], [49,48,49,49,49,47],
     [50,49,50,56,50,51], [68,65,60,60,59,64,66,60], [72,71,68,70,70,68,72,67],
     [82,77,78,77,80,81], [106,104,100,103], [103,80,76,84], [103,78,85,112],
     [123,118,124,119], [124,120,117,118]]

Ngem = []
Ndev = []


for i in range(len(N)):
    gem, dev = steekproefgemendev(N[i])
    Ngem.append(gem)
    Ndev.append(dev)

gem = sum(Ngem)/len(Ngem)
SStotal = sum((np.array(Ngem) - gem)**2)
    
theta1 = np.linspace(np.radians(1),np.radians(len(N)), len(N))
    
popt, pcov, infodict, mesg, ier = sp.optimize.curve_fit(funcbrekingsindex, theta1, Ngem, 1.49, Ndev, full_output = True, absolute_sigma = False)
perr = np.sqrt(np.diag(pcov))
SSresidual = sum(infodict['fvec']**2)
R2 = 1 - SSresidual/SStotal
RMSE = np.sqrt(sum(infodict['fvec']**2) / len(infodict['fvec']))

plt.errorbar(theta1, Ngem, yerr=Ndev, fmt='o')
plt.plot(theta1, funcbrekingsindex(theta1, popt), label = 'n₂ = ' + str(round(popt[0],2)))
plt.xlabel("θ₁ [rad]")
plt.ylabel('N [aantal]')
plt.title('data met de fitvergelijking')
plt.legend()
plt.show()


print("n₂ = " + str(round(popt[0], 2)) + ' ± ' + str(round(perr[0], 2)) + "\n" + str(100* round(R2, 3)) + '% wordt verklaard door de lijn.')
