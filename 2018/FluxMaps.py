# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:59:03 2018

@author: robin
"""

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import ndimage
from scipy import stats

import profile_fitting_HE1330_1013
from profile_fitting_HE1330_1013 import full_gauss2,Hb_O3_gauss,Hb_Fe_doublet_gauss

hdu = fits.open('HE0232-0900.binned.fits')
#hdu.info()

qso_data = hdu[0].data
qso_error = hdu[1].data
qso_header = hdu[0].header

[central_x,central_y]= [67,51]#Qfitsview shows [52,68]. We swap and subtract 1 from each

mini_data = qso_data[:,central_y - 5:central_y + 6,central_x - 5:central_x + 6]
mini_error = qso_error[:,central_y - 5:central_y + 6,central_x - 5:central_x + 6]
qso_header['CRPIX1'] = qso_header['CRPIX1'] - (central_x - 5)
qso_header['CRPIX2'] = qso_header['CRPIX2'] - (central_y - 5)
new_hdu = fits.HDUList([fits.PrimaryHDU(mini_data),fits.ImageHDU(mini_error)])
new_hdu[0].header = qso_header
new_hdu.writeto('minicube_HE022222222_FluxMaps.fits',overwrite=True)


z =0.043143 
k = 1+z


hdu = fits.open('minicube_HE022222222_FluxMaps.fits')
mini_data = hdu[0].data
mini_error = hdu[1].data
mini_header = hdu[0].header

wavstart = mini_header['CRVAL3']
wavint = mini_header['CD3_3']
wave = np.arange(wavstart,(wavstart+(wavint*mini_data.shape[0])),wavint)#start,stop,step

select = (wave > 4750*k) & (wave < 5090*k) 

dat = mini_data[select]
wav = wave[select]
err = mini_error[select]

par = np.zeros((10,mini_data.shape[1],mini_data.shape[2]),dtype=np.float32) #dtype = datatype. the type of the output array
err = np.zeros((10,mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
fitted = np.zeros((np.shape(wav)[0],mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
residuals = np.zeros((np.shape(wav)[0],mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)


for i in range(mini_data.shape[1]):
    for j in range(mini_data.shape[2]):
        y = dat[:,i,j]
        y_err = err[:,i,j]
        x = wav
        
        
        popt,pcov = curve_fit(full_gauss2,x,y,p0=[0.6,16,12942,160,2,30,12642,280,4.7,0.2,12942,1500,7,0.2,12942,1500,10,10],maxfev=10000000)

        #residuals = y - full_gauss2(x, *popt)

        (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c) = popt
        fixed_parameters = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]

        #Refer to HE0232 9Gauss fitting part 2.

        def full_gauss2_fixkin(wave,amp_Hb,amp_OIII5007,amp_Hb_br,amp_OIII_br,amp_Hb1,amp_Fe5018_1,amp_Hb2,amp_Fe5018_2,m,c):
            (vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_OIII_sigma_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2) = fixed_parameters
            narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
            broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_OIII_sigma_br)
            Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
            Hb_broad2 = 0
            cont = m*(wave/1000)+c
            return narrow_OIII + broad_OIII + Hb_broad1 + Hb_broad2 + cont

        p0 = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        popt1,pcov1 = curve_fit(full_gauss2_fixkin,x,y,p0,maxfev=10000000)
        fitted[:,i,j] = full_gauss2_fixkin(x, *popt1)
        residuals[:,i,j] = y - fitted[:,i,j] 
        Monte_Carlo_loops = 10
        parameters_MC = np.zeros((len(popt1),Monte_Carlo_loops))
        
        for l in range(Monte_Carlo_loops):
            iteration_data = np.random.normal(y,y_err) 
            popt_MC,pcov_MC = curve_fit(full_gauss2_fixkin,wav,iteration_data,p0=popt1,maxfev = 1000000)
            parameters_MC[:,l]=popt_MC
            parameters_err = np.std(parameters_MC,1)  

        