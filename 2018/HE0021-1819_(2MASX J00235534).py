# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 08:40:33 2018

@author: robin
"""

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

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
new_hdu.writeto('minicube_HE0232_0900.fits',overwrite=True)
new_hdu.close()

z =0.043143 
k = 1+z


hdu = fits.open('minicube_HE0232_0900.fits')
min_data = hdu[0].data
min_error = hdu[1].data
mini_header = hdu[0].header
wavstart = mini_header['CRVAL3']
wavint = mini_header['CD3_3']
wave = np.arange(wavstart,(wavstart+(wavint*mini_data.shape[0])),wavint)#start,stop,step

select = (wave > 4750*k) & (wave < 5090*k) 

y = min_data[select]
x = wave[select]
err = mini_error[select]

par = np.zeros((10,mini_data.shape[1],mini_data.shape[2]),dtype=np.float32) #dtype = datatype. the type of the output array
err = np.zeros((10,mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)

w80 = np.zeros((mini_data.shape[1],mini_data.shape[2]),dtype=np.float32) #dtype = datatype. the type of the output array
vel_diff = np.zeros((mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
flux_narrow = np.zeros((mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
flux_broad = np.zeros((mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
flux_total = np.zeros((mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
v5 = np.zeros((mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
#outflow is not spherical, but conical
np.seterr(divide='ignore', invalid='ignore')#setting error
for i in range(mini_data.shape[1]):
    for j in range(mini_data.shape[2]):
        spectrum = mini_data[:,i,j]
        wave = np.arange(wavstart,(wavstart+(wavint*mini_data.shape[0])),wavint)
        select = (wave > 4967*k) & (wave < 5037*k) 
        #select1 = (spectrum> 0.7)#amplitude masking
        n = len(wave[select])
        mean = sum(wave[select]*spectrum[select])/n
        sigma = sum(spectrum[select]*(wave[select]-mean)**2)/n
        def gaus(wave, amp1, cent1, sigma1,c):
                return amp1*exp(-(wave-cent1)**2/(2*sigma1**2)) +  c
        def gaus2(wave, amp1, amp2, cent1, cent2, sigma1, sigma2, c):
                return amp1*exp(-(wave-cent1)**2/(2*sigma1**2)) + amp2*exp(-(wave-cent2)**2/(2*sigma2**2)) + c
        def gaus2_wo(wave, amp1, amp2, cent1, cent2, sigma1, sigma2):
                return amp1*exp(-(wave-cent1)**2/(2*sigma1**2)) + amp2*exp(-(wave-cent2)**2/(2*sigma2**2))  
        #print('Mean: \n',+str(mean))
        #print('Sigma: \n',+str(sigma))

        if spectrum[select].max() > 0.1:#amplitude masking
                popt2,pcov2 = curve_fit(gaus2,wave[select],spectrum[select],p0=[1.0,1.0,5007*k,5007*k,1.0,10.0,0.01], maxfev =100000000)
                plt.plot(wave[select],spectrum[select])
                plt.plot(wave[select],gaus2(wave[select],*popt2),'r-',label='fit')
                plt.show()
                cor_popt2 = np.array(popt2)
                cor_popt2[4] = np.sqrt(np.abs(cor_popt2[4]**2-(2.5/2.354)**2))
                cor_popt2[5] = np.sqrt(np.abs(cor_popt2[5]**2-(2.5/2.354)**2))
                wave = np.arange(5200,5400,0.02)
                cumsum = np.cumsum(gaus2_wo(wave,*cor_popt2[:-1]))
                norm_sum=cumsum/cumsum[-1]
                select = (norm_sum>0.1) & (norm_sum<0.9)
                try:
                    w80_spec = wave[select][-1]- wave[select][0]
                except IndexError:
                    continue
                w80_spec = wave[select][-1]- wave[select][0]
                w80_actual = ((w80_spec)/cor_popt2[3])*(c/(1+z))
                w80[i,j]= w80_actual/100000
                print ('w80 is',w80[i,j])
                #plt.plot(wave[select],norm_sum[select],'-k')
                #plt.show()
                select =  (norm_sum>0.05) & (norm_sum<0.5)
                #try:
                #    v5_spec = wave[select][-1]- wave[select][0]
                #except IndexError:
                #print wave[select]
                #    continue
                v5_spec = wave[select][-1]-wave[select][0]
                v5[i,j]= v5_spec
                print ('v5 is',v5[i,j])
                print(np.sum(w80))
                #plt.legend
                F_total = (2.507*(cor_popt2[0]*cor_popt2[4]) + 2.507*(cor_popt2[1]*cor_popt2[5]))*(10**-16)
                flux_total[i,j] =  F_total
                F_narrow = 2.507*(cor_popt2[0]*cor_popt2[4])*(10**-16)
                F_broad = 2.507*(cor_popt2[1]*cor_popt2[5])*(10**-16)
                velshift_angstrom = (cor_popt2[2]-cor_popt2[3])
                #print cor_popt2[3]
                velshift_actual = (velshift_angstrom)/(cor_popt2[3])*(c/(1+z)) 
                vel_diff[i,j] = velshift_actual/100000 
                print('velocity shift is', vel_diff[i,j])
        else:
             exit

from matplotlib.colors import LogNorm
from matplotlib import cm
cmap=cm.gray
plt.imshow(w80,origin='lower',interpolation='nearest', cmap='gray', norm=LogNorm())
cbar = plt.colorbar()
plt.show()

fig = plt.figure()

cmap = cm.plasma
cmap.set_bad('white',1)

frame = plt.imshow(w80, origin='lower', cmap=cmap, norm=LogNorm(), interpolation='nearest')
cbar = plt.colorbar()
cbar.set_label(r"w80")

plt.imshow(vel_diff,origin='lower',interpolation='nearest',cmap='gray')
cbar = plt.colorbar()
cmap.set_bad('white',1)
plt.show()

fig = plt.figure()

cmap = cm.RdBu
cmap.set_bad('white',1)

frame = plt.imshow(vel_diff, origin='lower', cmap=cmap, norm=LogNorm(), interpolation='nearest')
cbar = plt.colorbar()
cbar.set_label(r"velocity shift")

w80_hdu = fits.PrimaryHDU(data=w80)
w80_hdu.writeto('w80_myfit.fits',clobber=True)
plt.imshow(v5,origin='lower',interpolation='nearest',cmap='gray', norm=LogNorm())
cbar = plt.colorbar()
plt.show()

plt.imshow(flux_total,origin='lower',interpolation='nearest',cmap='gray',norm=LogNorm())
cbar = plt.colorbar()
plt.show()

vel_diff_hdu = fits.PrimaryHDU(data=vel_diff)
vel_diff_hdu.writeto('vel_diff_myfit.fits',clobber=True)
flux_total_hdu = fits.PrimaryHDU(data=flux_total)
flux_total_hdu.writeto('flux_total_myfit.fits',clobber=True)

Flux= np.sum(flux_total)
D_L = 7.31*(10**26)
L_total = 31.5*((D_L)**2)*(Flux)/(2.507)
velocity_shift = velshift_actual/100000
v5_final = v5_spec
np.median(w80)
#velocity_dispersion = w80_final/3.29

#wavelength>4940*k & wavelength<5023*k