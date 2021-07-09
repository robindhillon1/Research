# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:22:30 2018

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
from profile_fitting_HE1330_1013 import full_gauss2,Hb_O3_gauss,Hb_Fe_doublet_gauss

hdu = fits.open('HE0232-0900.binned.fits')
qso_data=hdu[0].data
qso_error=hdu[1].data
qso_header=hdu[0].header
wavestart=qso_header['CRVAL3']
try:
    wavint = qso_header['CD3_3']
except KeyError:
    wavint = qso_header['CDELT_3']

wave = np.arange(wavestart,(wavestart+(wavint*qso_data.shape[0])),wavint)

z = 0.043143
k = 1+z

select_bp = (wave>5006*k)&(wave<5008*k)
qso_data_bp = qso_data[select_bp]
qso_2D = qso_data[0,:,:]

[bp_y,bp_x]=ndimage.measurements.maximum_position(qso_2D)
print("Brightest Pixel: ({},{})".format(bp_y,bp_x))
qso_data_agn = qso_data[:,(bp_y)+1,(bp_x)+1]
fluxden = qso_data_agn

select = (wave>4750*k)&(wave<5090*k)
y = fluxden[select]
x = wave[select]

popt,pcov = curve_fit(full_gauss2,x,y,p0=[0.6,16,12942,160,2,30,12642,280,4.7,0.2,12942,1500,7,0.2,12942,1500,10,10],maxfev=10000000)
print("\nPopt = \n"+str(popt)+"\n")

residuals = y - full_gauss2(x, *popt)

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
popt,pcov = curve_fit(full_gauss2_fixkin,x,y,p0,maxfev=10000000)
print("Popt (SpectroAstrometry): \n"+str(popt))
plt.plot(x,y,'ko',markersize=1.2,label='Actual Data')
plt.plot(x,full_gauss2_fixkin(x,*popt),'r-',label='Fit')
plt.plot(x,residuals,label='Residuals')
plt.xlabel('Wavelength(Ang)')
plt.ylabel('Flux Density')
plt.title('Central Spectrum of Galaxy: HE0232-0900')
plt.legend()
plt.show()

print(stats.ks_2samp(y,full_gauss2_fixkin(x,*popt)))

#KS Test says that there are x% chances the two samples come from the same
#distribution. Purpose is to test for differences in the shape of two sample 
#distributions (or to compare to expected statistical distribution). Compares 
#overall shape of distributions, not specifically central tendency, dispersion,
#or other parameters. 

#Test statistic (D) is simply the maximum absolute difference (supremum) between the 
#two cumulative distribution functions (CDFs). 

#first value is the test statistics, and second value is the p-value. if the
#p-value is less than 95 (for a level of significance of 5%), this means that 
#you cannot reject the Null-Hypothese that the two sample distributions are 
#identical.  Null Hypothesis says that there is no significant difference betweenhesis says that there is no significant difference betweenhesis says that there is no significant difference between 
#specified populations, any observed difference being due to sampling or 
#experimental error.The null-hypothesis for the KT test is that the distributions are the same. 
#Thus, the lower your p value the greater the statistical evidence you have to
#reject the null hypothesis and conclude the distributions are different.

#For ks_2samp, we cannot reject the null hypothesis if p value is more than 10%. 
#If the K-S statistic is small or the p-value is high, then we cannot reject 
#the hypothesis that the distributions of the two samples are the same.

# If you fit a Gaussian then that would be the PDF, so the residual works with 
# the PDF while the KS checks on the CDF. More important, though, is the fact
# that scipy.stats.ks_2samp takes sample data, neither the PDF nor the CDF.

#The area under the pdf up to x (i.e. the area to the left of x) is the cdf at x.
 
# =============================================================================
# a probability density function (PDF), or density of a continuous random variable, 
# is a function, whose value at any given sample (or point) in the sample space 
# (the set of possible values taken by the random variable) can be interpreted as
# providing a relative likelihood that the value of the random variable would 
# equal that sample.[citation needed] In other words, while the absolute likelihood 
# for a continuous random variable to take on any particular value is 0 (since there
# are an infinite set of possible values to begin with), the value of the PDF at two 
# different samples can be used to infer, in any particular draw of the random variable,
# how much more likely it is that the random variable would equal one sample compared 
# to the other sample.
# =============================================================================

# =============================================================================
# the cumulative distribution function (CDF, also cumulative density function) of
# a real-valued random variable X, or just distribution function of X, evaluated 
# at x, is the probability that X will take a value less than or equal to x. In 
# the case of a continuous distribution, it gives the area under the probability 
# density function from minus infinity to x. 
# =============================================================================
