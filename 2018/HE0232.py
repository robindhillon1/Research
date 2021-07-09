# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:38:46 2018

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp

z = 0.043143
k = 1 + z
print ('k')

data = np.loadtxt('HE0232-0900_ASCII')
wavelength = data[:,0]
flux_density = data[:,1]
#plt.plot(wavelength,flux_density)
#plt.xlabel('Wavelength (Ang)')
#plt.ylabel('Flux Density')

select = (wavelength>4984*k) & (wavelength<5023*k)
#plt.plot(wavelength[select],flux_density[select])

n = len(wavelength[select])
mean = sum(wavelength[select]*flux_density[select])/n
sigma = sum(flux_density[select]*(wavelength[select]-mean)**2)/n

def gaus(wavelength, amp1, cent1, sigma1,c):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) +  c

popt, pcov = curve_fit(gaus,wavelength[select],flux_density[select], p0=[48,5007*k,10,10])
print("popt: \n",popt)
plt.plot(wavelength[select],gaus(wavelength[select],popt[0],popt[1],popt[2],popt[3]),label='fit')
plt.plot(wavelength[select],flux_density[select],label='actual')
plt.legend()

plt.figure(2)
def gaus2(wavelength, amp1, amp2, cent1, cent2, sigma1, sigma2, c):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) + amp2*exp(-(wavelength-cent2)**2/(2*sigma2**2)) + c

popt2, pcov2 = curve_fit(gaus2,wavelength[select],flux_density[select], p0=[40,10,5007*k,5007*k,5,10,10])
print("popt2: \n",popt2)
plt.plot(wavelength[select],gaus2(wavelength[select],popt2[0],popt2[1],popt2[2],popt2[3],popt2[4],popt2[5],popt2[6]),'r-',label='Best-Fit')
plt.plot(wavelength[select],flux_density[select],'ko')
plt.plot(wavelength[select],gaus(wavelength[select],popt2[0],popt2[2],popt2[4],popt2[6]),'b-',label='Core')
plt.plot(wavelength[select],gaus(wavelength[select],popt2[1],popt2[3],popt2[5],popt2[6]),'g-',label='Wing')#Wing is blueshifted. 
plt.legend()
plt.figure(3)

#from IPython import get_ipython

#get_ipython().run_line_magic('matplotlib', 'inline')

def gaus2_wo(wavelength, amp1, amp2, cent1, cent2, sigma1, sigma2):
    return amp1*exp(-(wavelength-cent1)**2/(2*sigma1**2)) + amp2*exp(-(wavelength-cent2)**2/(2*sigma2**2))

#ax = plt.axes([0,0,2,2])
plt.xlim([4999*k,5011*k])
wavelength = np.arange(5200,5400,0.02)
#plt.plot(x,y, 'bo:',label='data')
#plt.plot(x,gaus1(x,*([popt3[0],popt3[2],popt3[4],popt3[6]])),'b--')
#plt.plot(x,gaus1(x,*([popt3[1],popt3[3],popt3[5],popt3[6]])),'g--')
#plt.plot(wave,gaus3(wave,*popt3),'r-',label='fit')



cor_popt2 = np.array(popt2)
cor_popt2[4] = np.sqrt(cor_popt2[4]**2-(2.5/2.354)**2)
cor_popt2[5] = np.sqrt(cor_popt2[5]**2-(2.5/2.354)**2)
cumsum = np.cumsum(gaus2_wo(wavelength,*cor_popt2[:-1]))
norm_sum=cumsum/cumsum[-1]
select = (norm_sum>0.1) & (norm_sum<0.9)
w80 = wavelength[select][-1]-wavelength[select][0]#velocity at 80% of the Flux
print("\n\nw80: "+str(w80))
plt.plot(wavelength[select],norm_sum[select],'-k')
select =  (norm_sum>0.05) & (norm_sum<0.5)
v5 = wavelength[select][0]-wavelength[select][-1]
print('v5: '+str(v5))        
plt.legend
plt.title('Fitted')
plt.xlabel('wavelength')
plt.ylabel('flux')
plt.show()

c = 3*(10**10)
w80_actual = ((w80)/5007)*(c/(1+z))
print("w80_actual (cm/s): "+str(w80_actual))

F_total = (2.507*(popt2[0]*popt2[4]) + 2.507*(popt2[1]*popt2[5]))*(10**-16)
print("F_total: "+str(F_total))

F_narrow = 2.507*(popt2[0]*popt2[4])*(10**-16)
print("F_narrow: "+str(F_narrow))

F_broad = 2.507*(popt2[1]*popt2[5])*(10**-16)
print("F_broad: "+str(F_broad))

D_L = 7.0*(10**27)
print("D_L: "+str(D_L))


L_total = 31.5*((D_L)**2)*(F_total)/(2.507)
print("L_total: "+str(L_total))


L_narrow = 31.5*((D_L)**2)*(F_narrow)/(2.507)
print("L_narrow: "+str(L_narrow))


L_broad = 31.5*((D_L)**2)*(F_broad)/(2.507)
print("L_broad: "+str(L_broad))


velshift_angstrom = popt2[2]- popt2[3]
print("velshift_angstrom: "+str(velshift_angstrom))


velshift_actual = ((velshift_angstrom)/popt2[3])*(c/(1+z))
print("velshift_actual: "+str(velshift_actual))