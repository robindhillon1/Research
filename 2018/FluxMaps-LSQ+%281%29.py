
# coding: utf-8

# In[1]:


import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import random
from scipy.optimize import leastsq


# In[2]:


hdu = fits.open('HE0232-0900.wo_absorption.fits')
#hdu.info()
qso_data = hdu[0].data
qso_error = hdu[1].data
qso_header = hdu[0].header
wavestart = qso_header['CRVAL3']
wavint = qso_header['CD3_3']
#wave = wavestart+np.arange(qso_data.shape[0])*wavint. This is the same as the one below. 
[central_x,central_y]= [67,51]#


# In[3]:


mini_cube = qso_data[:,central_y - 7:central_y + 8,central_x - 7:central_x + 8]
mini_cube_error = qso_error[:,central_y - 7:central_y + 8,central_x - 7:central_x + 8]
qso_header['CRPIX1'] = qso_header['CRPIX1'] - (central_x - 7)
qso_header['CRPIX2'] = qso_header['CRPIX2'] - (central_y - 7)
new_hdu = fits.HDUList([fits.PrimaryHDU(mini_cube),fits.ImageHDU(mini_cube_error)])
new_hdu[0].header = qso_header
wave = np.arange(wavestart,(wavestart+(wavint*mini_cube.shape[0])),wavint)#start,stop,step


# In[4]:


z =0.043143 
k = 1+z
mini_data = mini_cube
mini_error = mini_cube_error


# In[5]:


select = (wave > 4750*k) & (wave < 5090*k) 
par = np.zeros((10,mini_data.shape[1],mini_data.shape[2]),dtype=np.float32) #dtype = datatype. the type of the output array
err = np.zeros((10,mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
fitted = np.zeros((np.shape(wave[select])[0],mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)
residuals = np.zeros((np.shape(wave[select])[0],mini_data.shape[1],mini_data.shape[2]),dtype=np.float32)


# In[6]:


from profile_fitting_HE0232_0900_LSQ import *


# In[7]:


for i in range(mini_data.shape[1]):
    for j in range(mini_data.shape[2]):
        y = mini_data[:,i,j][select]
        y_err = mini_error[:,i,j][select]
        x = wave[select]    
        
        popt1,pcov1 = leastsq(full_gauss2_fixkin,x0=[0.1,0.1,0.1,0.1,0.01,0.01,0.01,0.01,-0.7,0.001],args = (x,y,y_err,fixed_parameters),maxfev=10000000)
        par[:,i,j] = popt1
        model = full_gauss2_fixkin(popt1,x,y,y_err,fixed_parameters)*(y_err)+y
        fitted[:,i,j] = model
        
        #plt.plot(x,y)
        #plt.plot(x,model)
        #plt.show()
        
        residuals[:,i,j] = mini_data[:,i,j][select] - fitted[:,i,j]

        Monte_Carlo_loops = 5
        parameters_MC = np.zeros((len(popt1),Monte_Carlo_loops))
        
        for l in range(Monte_Carlo_loops):
            iteration_data = np.random.normal(y,y_err) 
            popt_MC,pcov_MC = leastsq(full_gauss2_fixkin,x0=popt1,args=(x,iteration_data,y_err,fixed_parameters),maxfev = 1000000)
            parameters_MC[:,l]=popt_MC
            
        parameters_err = np.std(parameters_MC,1)
        err[:,i,j]=parameters_err


# In[8]:


hdus=[]
hdus.append(fits.PrimaryHDU())
hdus.append(fits.ImageHDU(par[0,:,:],name='amp_Hb'))
hdus.append(fits.ImageHDU(par[1,:,:],name='amp_OIII5007'))
hdus.append(fits.ImageHDU(par[2,:,:],name='amp_OIII5007_br'))
hdus.append(fits.ImageHDU(par[3,:,:],name='amp_Hb_br'))
hdus.append(fits.ImageHDU(par[4,:,:],name='amp_Hb1'))
hdus.append(fits.ImageHDU(par[5,:,:],name='amp_Hb2'))
hdus.append(fits.ImageHDU(par[6,:,:],name='amp_Fe5018_1'))
hdus.append(fits.ImageHDU(par[7,:,:],name='amp_Fe5018_2'))
hdus.append(fits.ImageHDU(par[8,:,:],name='m'))
hdus.append(fits.ImageHDU(par[9,:,:],name='c'))
hdu = fits.HDUList(hdus)

s = 'subcube_par_HE12_LSQ'
x = random.randint(1,101)
s += str(x)+'.fits'
print("Filename: "+str(s))
hdu.writeto(s,overwrite=True)


# In[9]:


hdus=[]
hdus.append(fits.PrimaryHDU())
hdus.append(fits.ImageHDU(err[0,:,:],name='amp_Hb'))
hdus.append(fits.ImageHDU(err[1,:,:],name='amp_OIII5007'))
hdus.append(fits.ImageHDU(err[2,:,:],name='amp_OIII5007_br'))
hdus.append(fits.ImageHDU(err[3,:,:],name='amp_Hb_br'))
hdus.append(fits.ImageHDU(err[4,:,:],name='amp_Hb1'))
hdus.append(fits.ImageHDU(err[5,:,:],name='amp_Hb2'))
hdus.append(fits.ImageHDU(err[6,:,:],name='amp_Fe5018_1'))
hdus.append(fits.ImageHDU(err[7,:,:],name='amp_Fe5018_2'))
hdus.append(fits.ImageHDU(err[8,:,:],name='m'))
hdus.append(fits.ImageHDU(err[9,:,:],name='c'))
hdu = fits.HDUList(hdus)

s = 'subcube_par_err_HE12_LSQ'
s += str(x)+'.fits'
print("Filename: "+str(s))
hdu.writeto(s,overwrite=True)

