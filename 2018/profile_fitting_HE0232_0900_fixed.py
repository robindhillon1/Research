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
                           
# Here we couple the HB and OIII doublet together using the gaussian function defined before
def Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33)
    OIII_4959 = (0.33)*gauss(wave,amp_OIII5007,vel,vel_sigma,4958.9)
    OIII_5007 = gauss(wave,amp_OIII5007,vel,vel_sigma,5006.8)
    return Hb + OIII_4959 + OIII_5007

# Same as before but fore the Fe doublet
def Fe_doublet_gauss(wave,amp_Fe4923,amp_Fe5018,vel,vel_sigma):
    Fe_4923 = gauss(wave,amp_Fe4923,vel,vel_sigma,4923)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018)
    return Fe_4923+Fe_5018

def Hb_Fe_doublet_gauss(wave,amp_Hb,amp_Fe5018,vel,vel_sigma):
    Hb = gauss(wave,amp_Hb,vel,vel_sigma,4861.33)
    Fe_4923 = 0.81*gauss(wave,amp_Fe5018,vel,vel_sigma,4923)
    Fe_5018 = gauss(wave,amp_Fe5018,vel,vel_sigma,5018)
    return Hb+Fe_4923+Fe_5018

def full_gauss2(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c):
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    #broad_OIII = 0
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    #Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    Hb_broad2 = 0
    cont = (wave/1000.0)*m+c
    return narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont

hdu = fits.open('HE0232-0900_central_fit.fits')
central_tab = hdu[1].data
central_columns = hdu[1].header

vel_OIII = central_tab.field('vel_OIII')
vel_sigma_OIII = central_tab.field('vel_sigma_OIII')
vel_OIII_br = central_tab.field('vel_OIII_br')
vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')
vel_Hb1 = central_tab.field('vel_Hb1')
vel_Hb2 = central_tab.field('vel_Hb2')
vel_sigma_Hb1 = central_tab.field('vel_sigma_Hb1')
vel_sigma_Hb2 = central_tab.field('vel_sigma_Hb2')

fixed_param = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]  
fixed_param

# =============================================================================
# def fixed_parameters(vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2):
#     return (vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2)
# =============================================================================
# =============================================================================
# 
# def full_gauss2_fixkin(wave,amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c):
#     fixed_parameters = (vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_sigma_Hb2)
#     narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
#     broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
#     #broad_OIII = 0
#     Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
#     Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
#     #Hb_broad2 = 0
#     cont = (wave/1000.0)*m+c
#     return narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont
# 
# =============================================================================

def full_gauss2_fixkin(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error