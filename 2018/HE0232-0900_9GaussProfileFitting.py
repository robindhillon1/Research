# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 07:59:59 2018

@author: robin
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import ndimage
from scipy import stats

from profile_fitting_HE0232_0900 import full_gauss2

hdu = fits.open('HE0232-0900.binned.fits')
#hdu.info()
qso_data= hdu[0].data
qso_error = hdu[1].data

qso_header= hdu[0].header
wavstart = qso_header['CRVAL3']
try: 
   wavint = qso_header['CD3_3']#
except KeyError:
   wavint = qso_header['CDELT_3']#
wave = np.arange(wavstart,(wavstart+(wavint*qso_data.shape[0])),wavint)#start,stop,step

z = 0.043143
k = 1+z

select_bp = (wave>5006*k) & (wave<5008*k)#brightest pixel. OIII brightest
qso_data_bp = qso_data[select_bp]
qso_2D = qso_data_bp[0,:,:]

[brightest_pixel_y,brightest_pixel_x]= ndimage.measurements.maximum_position(qso_2D)#Find the positions of the maximums of the values of an array at labels.
print (brightest_pixel_y,brightest_pixel_x)
qso_data_agn = qso_data[:,brightest_pixel_y,brightest_pixel_x]
spectrum = qso_data_agn #flux density

select = (wave>4750*k) & (wave<5090*k)
y= spectrum[select]
x = wave[select]


popt,pcov = curve_fit(full_gauss2,x,y,p0=[0.6,16,12942,140,5,35,12642,280,4.7,0.2,12942,1500,7,0.2,12942,1500,10,10],maxfev=10000000)
print("Popt = \n"+str(popt))
print("\n")
residuals = y - full_gauss2(x, *popt)
plt.plot(x,full_gauss2(x,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7],popt[8],popt[9], 
         popt[10],popt[11],popt[12],popt[13],popt[14],popt[15],popt[16],popt[17]),'r-',label='fit')
plt.plot(x,y,'ko',markersize=1.2,label='actual')
plt.plot(x,residuals)
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')
plt.title('Central Spectrum of Galaxy: HE0232-0900')
plt.legend()
plt.show()

print(stats.ks_2samp(y,full_gauss2(x,*popt)))


