# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:38:46 2018

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy import asarray as ar, exp

z = 0.043143
k = 1 + z
print ('k')

data = np.loadtxt('HE0232-0900_ASCII')
wavelength = data[:,0]
flux_density = data[:,1]

plt.figure(1)
select4 = (wavelength>4940*k) & (wavelength<5023*k)

#n = len(wavelength[select])
#mean = sum(wavelength[select]*flux_density[select])/n
#sigma = sum(flux_density[select]*(wavelength[select]-mean)**2)/n

def gaus4(wavelength, amp1,amp2,cent1,cent2,cent3,cent4,sigma1,sigma2,c):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) + amp2*exp(-(wavelength-cent2)**2/(2*sigma2**2)) + 0.33*amp1*exp(-(wavelength-cent3)**2/(2*sigma1**2)) + 0.33*amp2*exp(-(wavelength-cent4)**2/(2*sigma2**2))+c
 
popt4, pcov4= curve_fit(gaus4,wavelength[select4],flux_density[select4], p0=[40,8,5007*k,5004*k,4959*k,4955*k,5,10,5])
print("Popt4 = \n"+str(popt4))
print("\n")
plt.plot(wavelength[select4],gaus4(wavelength[select4],popt4[0],popt4[1],popt4[2],popt4[3],
       popt4[4],popt4[5],popt4[6],popt4[7],popt4[8]),'r-',label='Fit') 
plt.plot(wavelength[select4],flux_density[select4],'k-',label='Actual')
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')
plt.legend()
plt.show()
# =============================================================================
# 
# =============================================================================
plt.figure(2)
select12 = (wavelength>4750*k) & (wavelength<5090*k)

def gaus12(wavelength,amp1,amp2,cent1,cent2,cent3,cent4,sigma1,sigma2,c,d):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) + amp2*exp(-(wavelength-cent2)**2/(2*sigma2**2)) + 0.33*amp1*exp(-(wavelength-cent3)**2/(2*sigma1**2)) + 0.33*amp2*exp(-(wavelength-cent4)**2/(2*sigma2**2))+(c*wavelength+d)

popt12, pcov12= curve_fit(gaus12,wavelength[select12],flux_density[select12], p0=[40,8,5007*k,5004*k,4959*k,4955*k,5,10,-0.001,0.01], maxfev = 10000000)
print("Popt12 = \n"+str(popt12))
print("\n")
plt.plot(wavelength[select12],flux_density[select12],label='Actual')
fig2 = plt.plot(wavelength[select12],gaus12(wavelength[select12],popt12[0],popt12[1],popt12[2],popt12[3],popt12[4],popt12[5],popt12[6],popt12[7],popt12[8],popt12[9]),'r-',label='Fit')
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')
plt.legend()
plt.show()
# =============================================================================
# 
# =============================================================================
plt.figure(3)
select = (wavelength>4745*k) & (wavelength<4937*k)

def gaus(wavelength, amp1, cent1, sigma1,c):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) +  c

def gaus3(wavelength, a1, c1, s1, a2, c2, s2, a3, c3, s3, c):
    return (gaus(wavelength, a1, c1, s1, c=0) +
            gaus(wavelength, a2, c2, s2, c=0) +
            gaus(wavelength, a3, c3, s3, c=0) + c)

popt, pcov = curve_fit(gaus3,wavelength[select],flux_density[select], p0=[5,5040,5,10,5070,5,7,5088,10,10])
print("popt-3gaus = \n"+str(popt))
print("\n")
plt.plot(wavelength[select],gaus3(wavelength[select],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9]),label='fit')
plt.plot(wavelength[select],flux_density[select],label='actual')
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')
plt.legend()
plt.show()
# =============================================================================
# 
# =============================================================================
plt.figure(4)
select12 = (wavelength>4750*k) & (wavelength<5090*k)

def gauss(wavelength, amp1, cent1, sigma1,c):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) +  c

def gauss3(wavelength, a1, c1, s1, a2, c2, s2, a3, c3, s3, a4, c4, s4, a5, c5, s5, c):
    return (gauss(wavelength, a1, c1, s1, c=0) +
            gauss(wavelength, a2, c2, s2, c=0) +
            gauss(wavelength, a3, c3, s3, c=0) +
            gauss(wavelength, a4, c4, s4, c=0) +
            gauss(wavelength, a5, c5, s5, c=0) + c)

poptAll, pcov = curve_fit(gauss3,wavelength[select12],flux_density[select12], p0=[6,5056,14,10,5070,5,7,5088,10,50,5220,10,17,5170,10,10])
print("PoptAll = \n"+str(poptAll))
print("\n")
plt.plot(wavelength[select12],gauss3(wavelength[select12],poptAll[0],poptAll[1],poptAll[2],poptAll[3],poptAll[4],poptAll[5],poptAll[6],poptAll[7],poptAll[8],poptAll[9], 
         poptAll[10],poptAll[11],poptAll[12],poptAll[13],poptAll[14],poptAll[15]),label='fit')
plt.plot(wavelength[select12],flux_density[select12],label='actual')
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')

residuals = flux_density[select12] - gauss3(wavelength[select12], *poptAll)
plt.plot(wavelength[select12],residuals,'ko',markersize=1)
plt.xlabel('Wavelength(Ang)')
plt.ylabel('Residuals')
plt.legend()
plt.show()
#different redshifts. narrow = core, near the black hole. 4923 and 5018

