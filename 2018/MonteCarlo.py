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

import profile_fitting_HE1330_1013
from profile_fitting_HE1330_1013 import full_gauss2

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
x = np.arange(wavstart,(wavstart+(wavint*qso_data.shape[0])),wavint)#start,stop,step

z = 0.043143
k = 1+z

select_bp = (x>5006*k) & (x<5008*k)#brightest pixel. OIII brightest
qso_data_bp = qso_data[select_bp]
qso_2D = qso_data_bp[0,:,:]

[brightest_pixel_y,brightest_pixel_x]= ndimage.measurements.maximum_position(qso_2D)#Find the positions of the maximums of the values of an array at labels.
print (brightest_pixel_y,brightest_pixel_x)
y = qso_data[:,brightest_pixel_y,brightest_pixel_x]
error = qso_error[:,brightest_pixel_y,brightest_pixel_x]

select = (x>4750*k) & (x<5090*k)

popt,pcov = curve_fit(full_gauss2,x[select],y[select],p0=[0.6,16,12942,140,5,35,12642,280,4.7,0.2,12942,1500,7,0.2,12942,1500,10,10],sigma=error[select],maxfev=10000000)
residuals = y[select] - full_gauss2(x[select], *popt)

plt.plot(x[select],full_gauss2(x[select],*popt),'r-',label='fit')
plt.plot(x[select],y[select],'ko',markersize=1.2,label='actual')
plt.plot(x[select],residuals)
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Flux Density')
plt.title('Central Spectrum of Galaxy: HE0232-0900')
plt.legend()
plt.show()

Monte_Carlo_loops = 10
parameters_MC = np.zeros((len(popt),Monte_Carlo_loops))
for l in range(Monte_Carlo_loops):
        iteration_data = np.random.normal(y[select],error[select]) 
        popt_MC,pcov_MC = curve_fit(full_gauss2,x[select],iteration_data,p0=popt,sigma=error[select], maxfev = 1000000)
        parameters_MC[:,l]=popt_MC
        parameters_err = np.std(parameters_MC,1)  
(amp_Hb_error,amp_OIII5007_error,vel_OIII_error,vel_sigma_OIII_error,amp_Hb_br_error,amp_OIII5007_br_error,vel_OIII_br_error,vel_sigma_OIII_br_error,amp_Hb1_error,amp_Fe5018_1_error,vel_Hb1_error,vel_sigma_Hb1_error,amp_Hb2_error,amp_Fe5018_2_error,vel_Hb2_error,vel_sigma_Hb2_error,m_error,c_error) = parameters_err
    
column_names={'amp_Hb':0,'amp_OIII5007':1,'vel_OIII':2,'vel_sigma_OIII':3,'amp_Hb_br':4,'OIII5007_br':5,'vel_OIII_br':6,
              'vel_sigma_OIII_br':7,'amp_Hb1':8,'amp_Fe5018_1':9,'vel_Hb1':10,'vel_sigma_Hb1':11,'amp_Hb2':12,
              'amp_Fe5018_2':13,'vel_Hb2':14,'vel_sigma_Hb2':15,'m':16,'c':17}
# =============================================================================
# columns=[]
# for key in column_names.keys():
#         columns.append(fits.Column(name=key,format='E',array=[popt[column_names[key]]]))
#         columns.append(fits.Column(name=key+'_err',format='E',array=[parameters_err[column_names[key]]]))
# coldefs = fits.ColDefs(columns)
# hdu = fits.BinTableHDU.from_columns(coldefs)
# hdu.writeto('HE11_central_fit.fits',overwrite=True)
# =============================================================================

print (column_names.keys())
#print(stats.ks_2samp(y,full_gauss2(x,*popt)))


