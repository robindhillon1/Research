# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 12:12:05 2018

@author: robin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:07:41 2018

@author: robin
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

def redshift(vel):
    return vel/300000.0

#This function also represent the line dispersion in A through a velocity dispersion in km/s also taking into account 
# that the spectrograph itself already broadens the emission lines. This way you automatically fit for the intrinsic line dispersion
def line_width(vel_sigma,rest_line,inst_res_fwhm=2.4):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

def gauss(wave,amplitude,vel,vel_sigma, rest_wave):
    line = (amplitude)*exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width(vel_sigma, rest_wave))**2))
    return line                           

def Ha_gauss(wave,amp_Ha,vel,vel_sigma):
    narrow_Ha = gauss(wave,amp_Ha,vel,vel_sigma,6562.8)
    broad_Ha = 
    return Ha

def NII_doublet_gauss(wave,amp_N6583,vel,vel_sigma):
    N_6548 = 0.33*gauss(wave,amp_N6583,vel,vel_sigma,6548)
    N_6583 = gauss(wave,amp_N6583,vel,vel_sigma,6583)
    return N_6548+N_6583

def SII_doublet_gauss(wave,amp_S6716,amp_S6731,vel,vel_sigma):
    S_6716 = gauss(wave,amp_S6716,vel,vel_sigma,6716)
    S_6731 = gauss(wave,amp_S6731,vel,vel_sigma,6731)
    return S_6716+S_6731

def full_gauss3(p,wave,data,error):    
    (amp_Ha,vel_Ha,vel_sigma_Ha,amp_NI,vel_NI,vel_sigma_NI,amp_N6583,vel_N6583,vel_N6583_sigma,amp_N6583_br,vel_N6583_br,vel_N6583_sigma_br,amp_S6716,amp_S6731,vel_S6716,vel_S6716_sigma,amp_S6716_br,amp_S6731_br,vel_S6716_br,vel_S6716_sigma_br,m,c) = p 
    Ha = Ha_gauss(wave,amp_Ha,vel_Ha,vel_sigma_Ha)
    NI = NI_gauss(wave,amp_NI,vel_NI,vel_sigma_NI)
    narrow_NII = NII_doublet_gauss(wave,amp_N6583,vel_N6583,vel_N6583_sigma)
    broad_NII = NII_doublet_gauss(wave,amp_N6583_br,vel_N6583_br,vel_N6583_sigma_br)
    narrow_SII = SII_doublet_gauss(wave,amp_S6716,amp_S6731,vel_S6716,vel_S6716_sigma)
    broad_SII = SII_doublet_gauss(wave,amp_S6716_br,amp_S6731_br,vel_S6716_br,vel_S6716_sigma_br)
    cont = (wave/1000.0)*m+c
    return (Ha+NI+narrow_NII+broad_NII+narrow_SII+broad_SII+cont-data)/error


# =============================================================================
# hdu = fits.open('HE0232-0900_central_fit.fits')
# central_tab = hdu[1].data
# central_columns = hdu[1].header
# 
# vel_OIII = central_tab.field('vel_OIII')
# vel_sigma_OIII = central_tab.field('vel_sigma_OIII')
# vel_OIII_br = central_tab.field('vel_OIII_br')
# vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')
# vel_Hb1 = central_tab.field('vel_Hb1')
# vel_Hb2 = central_tab.field('vel_Hb2')
# vel_sigma_Hb1 = central_tab.field('vel_sigma_Hb1')
# vel_sigma_Hb2 = central_tab.field('vel_sigma_Hb2')
# 
# fixed_parameters = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]  
# 
# # =============================================================================
# # def fixed_parameters(vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2):
# #     return (vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2)
# # 
# # =============================================================================
# 
# def full_gauss2_fixkin(p,wave,data,error,fixed_parameters):
#     (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
#     [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]=fixed_parameters
#     narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
#     broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
#     Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
#     Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2)
#     cont = (wave/1000.0)*m+c
#     return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error
# =============================================================================


