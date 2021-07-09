# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:00:49 2018

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import ndimage
from scipy import stats

import profile_fitting_HE0232_0900
from profile_fitting_HE0232_0900 import full_gauss2,Hb_O3_gauss,Hb_Fe_doublet_gauss

hdu = fits.open('HE0232-0900.binned.fits')
qso_data=hdu[0].data
qso_error=hdu[1].data
qso_header=hdu[0].header
wavestart=qso_header['CRVAL3']
try:
    wavint = qso_header['CD3_3']
except KeyError:
    wavint = qso_header['CDELT_3']

wave=np.arange(wavestart,(wavestart+(wavint*qso_data.shape[0])),wavint)

z = 0.043143
k = 1+z

select_bp = (wave>5006*k)&(wave<5008*k)
qso_data_bp = qso_data[select_bp]
qso_2D = qso_data_bp[0,:,:]

[bp_y,bp_x] = ndimage.measurements.maximum_position(qso_2D)
print("Brightest Pixel: {} {}".format(bp_y,bp_x))
qso_data_agn = qso_data[:,(bp_y)-1,(bp_x)-1]
fluxden = qso_data_agn

select = (wave>4750*k)&(wave<5090*k)
y = fluxden[select]
x = wave[select]



popt,pcov = curve_fit(full_gauss2,x,y,p0=[0.6,16,12942,160,2,30,12642,280,4.7,0.2,12942,1500,7,0.2,12942,1500,10,10],maxfev=10000000)
print("Popt = \n"+str(popt))
print("\n")
residuals = y - full_gauss2(x, *popt)

(amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c) = popt
fixed_parameters = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2]

#print (fixed_parameters)
#Kinematic paramters are constant, and amplitudes are then fixed(no more 
#negative amplitudes). This is spectroastrometry; it holds the kinmeatic 
#parameters constant while varying amplitudes, fitting to the data.

def full_gauss2_fixkin(wave,amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c):
    (vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2) = fixed_parameters
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br) 
  #  broad_OIII = 0
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = 0       
    cont = (wave/1000.0)*m+c
    return narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont

popt,pcov = curve_fit(full_gauss2_fixkin,x,y,p0=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,-0.001,0.07],maxfev=10000000)
print (popt)
plt.plot(x,full_gauss2_fixkin(x,*popt),'g-',label='fit')
plt.plot(x,y,'ko',markersize=1.2,label='actual')
plt.plot(x,residuals)
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')
plt.title('Central Spectrum of Galaxy: HE0232-0900')
plt.legend()
plt.show()

print(stats.ks_2samp(y,full_gauss2_fixkin(x,*popt)))

