import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import curve_fit
from numpy import exp
from astropy.modeling import models, fitting
from scipy import ndimage


def loadCube(filename):
    hdu = fits.open(filename)
    cube = hdu[0].data
    global err
    try:
        err = hdu[1].data
    except IndexError:
        err = 0
    header = hdu[0].header
    hdu.close()
    wavestart = header['CRVAL3']
    try:
        wavint = header['CD3_3']
    except KeyError:
        wavint = header['CDELT3']  
    wave = wavestart+np.arange(cube.shape[0])*wavint
    return cube,err,wave,header

#If you do define store_cube as 
def store_cube(filename,mini_cube_data,wave,mini_cube_err=None,header=None): #
    if mini_cube_err is None:
        hdu_out = fits.PrimaryHDU(mini_cube_data)
    else:
        hdu_out = fits.HDUList([fits.PrimaryHDU(mini_cube_data),fits.ImageHDU(mini_cube_err)])
    if header is not None:
        hdu_out[0].header = header
    hdu_out[0].header['CRPIX3'] = wave[1]
    hdu_out[0].header['CRVAL3'] = wave[0]
    hdu_out[0].header['CDELT3'] = (wave[1]-wave[0])
    hdu_out.writeto(filename,overwrite=True)
    
def loadmap(filename):
    hdu = fits.open(filename)
    (OIII_nr,OIII_br,Hb1_blr_br,Hb2_blr_br) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    (amp_OIII_nr,amp_OIII_br,amp_Hb1_blr_br,amp_Hb2_blr_br) = (np.max(OIII_nr),np.max(OIII_br),np.max(Hb1_blr_br),np.max(Hb2_blr_br))
    if amp_Hb1_blr_br > amp_Hb2_blr_br:
        (Hb_blr_br,amp_Hb_blr_br) = (Hb1_blr_br,amp_Hb1_blr_br)
    else:
        (Hb_blr_br,amp_Hb_blr_br) = (Hb2_blr_br,amp_Hb2_blr_br)
    return Hb_blr_br,OIII_br,OIII_nr,amp_Hb_blr_br,amp_OIII_br,amp_OIII_nr
    
   
def loadplot(filename):
    hdu = fits.open(filename)
    (Hb_data,OIII_br_data,OIII_nr_data)=(hdu[1].data,hdu[2].data,hdu[3].data)
    (Hb_model,OIII_br_model,OIII_nr_model) = (hdu[4].data,hdu[5].data,hdu[6].data)
    (Hb_res,OIII_br_res,OIII_nr_res) = (hdu[7].data,hdu[8].data,hdu[9].data)
    return Hb_data,Hb_model,Hb_res,OIII_br_data,OIII_br_model,OIII_br_res,OIII_nr_data,OIII_nr_model,OIII_nr_res

def loadblr(filename):
    hdu = fits.open(filename)
    (Hb1_blr_br_data,Hb2_blr_br_data) = (hdu[5].data,hdu[6].data)
    return Hb1_blr_br_data,Hb2_blr_br_data

def redshift(vel):
    return vel/300000.0
    #This function also represent the line dispersion in A through a velocity dispersion in km/s also taking into account 
    # that the spectrograph itself already broadens the emission lines. This way you automatically fit for the intrinsic line dispersion
def line_width(vel_sigma,rest_line,inst_res_fwhm=2.4):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

def line_width_recons(vel_sigma,rest_line,inst_res_fwhm=0):
    sigma = vel_sigma/(300000.0-vel_sigma)*rest_line
    return np.sqrt(sigma**2+(inst_res_fwhm/2.354)**2)

def gauss(wave,amplitude,vel,vel_sigma, rest_wave):
    line = (amplitude)*exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width(vel_sigma, rest_wave))**2))
    return line

def gauss_recons(wave,amplitude,vel,vel_sigma, rest_wave):
    line = (amplitude)*exp(-(wave-(rest_wave*(1+redshift(vel))))**2/(2*(line_width_recons(vel_sigma, rest_wave))**2))
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

def OIII_wo_cont(wave,amp_OIII5007,amp_OIII5007_br,vel,vel_sigma,vel_br,vel_sigma_br):  
    OIII_5007_core = gauss_recons(wave,amp_OIII5007,vel,vel_sigma,5006.8)
    OIII_5007_wing = gauss_recons(wave,amp_OIII5007_br,vel_br,vel_sigma_br,5006.8)
    return OIII_5007_core + OIII_5007_wing
 

def full_gauss1(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error


def full_gauss2(p,wave,data,error):
    (amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2,m,c)=p
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error


def full_gauss1_fixkin(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1)
    Hb_broad2 = 0 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def full_gauss2_fixkin(p,wave,data,error,fixed_param):
    (amp_Hb,amp_OIII5007,amp_OIII5007_br,amp_Hb_br,amp_Hb1,amp_Hb2,amp_Fe5018_1,amp_Fe5018_2,m,c) = p 
    [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2] = fixed_param 
    narrow_OIII = Hb_O3_gauss(wave,amp_Hb,amp_OIII5007,vel_OIII,vel_sigma_OIII)
    broad_OIII = Hb_O3_gauss(wave,amp_Hb_br,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br)
    Hb_broad1 = Hb_Fe_doublet_gauss(wave,amp_Hb1,amp_Fe5018_1,vel_Hb1,vel_sigma_Hb1) 
    Hb_broad2 = Hb_Fe_doublet_gauss(wave,amp_Hb2,amp_Fe5018_2,vel_Hb2,vel_sigma_Hb2) 
    cont = (wave/1000.0)*m+c
    return (narrow_OIII+broad_OIII+Hb_broad1+Hb_broad2+cont-data)/error

def difference_in_wavelength_dimension(orig_cube,cont_cube):
    difference_in_wavelength_dimension_source = np.shape(orig_cube)[0] - np.shape(cont_cube)[0]
    return difference_in_wavelength_dimension_source

def brightest_pixel(QSO_cube,wo_cube,wo_wave,z):
    k = 1 + z
    QSO_slice = QSO_cube[0,:,:]
    [guess_y,guess_x] = ndimage.measurements.maximum_position(QSO_slice)
    test_cube = wo_cube[:,guess_y-5:guess_y+5,guess_x-5:guess_x+5]
    select = (wo_wave >5006*k) & (wo_wave<5009*k) 
    test_cube = test_cube[select]
    (y0,x0) = ndimage.measurements.maximum_position(test_cube[0,:,:])
    (brightest_pixel_y,brightest_pixel_x) = (y0+guess_y-5,x0+guess_x-5)
    return brightest_pixel_x,brightest_pixel_y
    
def fixed_parameters(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    vel_OIII = central_tab.field('vel_OIII')[0]
    vel_sigma_OIII = central_tab.field('vel_sigma_OIII')[0]
    vel_OIII_br = central_tab.field('vel_OIII_br')[0]
    vel_sigma_OIII_br = central_tab.field('vel_sigma_OIII_br')[0]
    vel_Hb1 = central_tab.field('vel_Hb1')[0]
    vel_Hb2 = central_tab.field('vel_Hb2')[0]
    vel_sigma_Hb1 = central_tab.field('vel_sigma_Hb1')[0]
    vel_sigma_Hb2 = central_tab.field('vel_sigma_Hb2')[0]
    fixed_param = [vel_OIII,vel_sigma_OIII,vel_OIII_br,vel_sigma_OIII_br,vel_Hb1,vel_sigma_Hb1,vel_Hb2,vel_sigma_Hb2]  
    return fixed_param
    
def light_weighted_centroid(obj):
    hdu = fits.open('subcube_par_%s.fits'%(obj))
    (OIII_nr_data,OIII_br_data,Hb1_br_data,Hb2_br_data) = (hdu[2].data,hdu[3].data,hdu[5].data,hdu[6].data)
    centroid_OIII_nr = ndimage.measurements.center_of_mass(OIII_nr_data)
    centroid_OIII_br = ndimage.measurements.center_of_mass(OIII_br_data)
    centroid_Hb1_br = ndimage.measurements.center_of_mass(Hb1_br_data)
    centroid_Hb2_br = ndimage.measurements.center_of_mass(Hb2_br_data)
    if np.max(Hb1_br_data) > np.max(Hb2_br_data):
        return centroid_Hb1_br,centroid_OIII_br,centroid_OIII_nr
    else:
        return centroid_Hb2_br,centroid_OIII_br,centroid_OIII_nr

def brightest_pixel_flux_map(Hb_blr_br_data,OIII_br_data,OIII_nr_data):
    [brightest_pixel_Hb_blr_br_y,brightest_pixel_Hb_blr_br_x] = ndimage.measurements.maximum_position(Hb_blr_br_data)
    [brightest_pixel_OIII_br_y,brightest_pixel_OIII_br_x] = ndimage.measurements.maximum_position(OIII_br_data)
    [brightest_pixel_OIII_nr_y,brightest_pixel_OIII_nr_x] = ndimage.measurements.maximum_position(OIII_nr_data)
    return brightest_pixel_Hb_blr_br_x,brightest_pixel_Hb_blr_br_y,brightest_pixel_OIII_br_x,brightest_pixel_OIII_br_y,brightest_pixel_OIII_nr_x,brightest_pixel_OIII_nr_y
    
def sampling_size(cont_cube):
    single_dimension_shape = np.shape(cont_cube)[1] 
    if single_dimension_shape > 250:
        sampling_size = 0.2
    else:
        sampling_size = 0.4
    return sampling_size

def centers(obj):
    hdu = fits.open('moffat_table_%s.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    (Hb_x,Hb_y) = (central_tab.field('x0_Hb_Blr')[0],central_tab.field('y0_Hb_Blr')[0])
    (OIII_br_x,OIII_br_y) = (central_tab.field('x0_OIII_br')[0],central_tab.field('y0_OIII_br')[0])
    (OIII_nr_x,OIII_nr_y) = (central_tab.field('x0_OIII_nr')[0],central_tab.field('y0_OIII_nr')[0])
    return Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y  

def offset(Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y,muse_sampling_size):
    offset_OIII_br_pixel= [(OIII_br_x - Hb_x),(OIII_br_y - Hb_y)]
    offset_OIII_nr_pixel= [(OIII_nr_x - Hb_x),(OIII_nr_y - Hb_y)]
    offset_OIII_br_arcsec = np.asarray(offset_OIII_br_pixel)*muse_sampling_size 
    offset_OIII_nr_arcsec = np.asarray(offset_OIII_nr_pixel)*muse_sampling_size 
    return offset_OIII_br_pixel,offset_OIII_nr_pixel,offset_OIII_br_arcsec,offset_OIII_nr_arcsec
    
def ranges(Hb_x,Hb_y,muse_sampling_size,asymmetry=False):
    if asymmetry:
        size = 14
    else:
        size = 15
    (x_min,x_max) = (-(Hb_x+0.5)*muse_sampling_size,(size-1-Hb_x+0.5)*muse_sampling_size)
    (y_min,y_max) = (-(Hb_y+0.5)*muse_sampling_size,(size-1-Hb_y+0.5)*muse_sampling_size)
    return x_min,x_max,y_min,y_max

def par(obj):
    hdu = fits.open('%s_central_fit.fits'%(obj))
    central_tab = hdu[1].data
    central_columns = hdu[1].header
    
    (amp_Hb1,amp_Hb2) = (central_tab.field('amp_Hb1'),central_tab.field('amp_Hb2'))
    (vel_sigma_Hb1,vel_sigma_Hb2) = (central_tab.field('vel_sigma_Hb1'),central_tab.field('vel_sigma_Hb2'))
    (amp_OIII5007,amp_OIII5007_br) = (central_tab.field('amp_OIII5007'),central_tab.field('amp_OIII5007_br'))
    (vel_sigma_OIII,vel_sigma_OIII_br) = (central_tab.field('vel_sigma_OIII'),central_tab.field('vel_sigma_OIII_br'))
    (vel_OIII,vel_OIII_br) = (central_tab.field('vel_OIII'),central_tab.field('vel_OIII_br'))
     
    vel_offset = vel_OIII - vel_OIII_br
    
    return amp_Hb1,amp_Hb2,vel_sigma_Hb1,vel_sigma_Hb2,amp_OIII5007,vel_OIII,vel_sigma_OIII,amp_OIII5007_br,vel_OIII_br,vel_sigma_OIII_br,vel_offset


def wavlim(vel_OIII,vel_OIII_br):
    c = 300000 # km/s
    k_OIII = 1+(vel_OIII/c)
    k_OIII_br = 1+(vel_OIII_br/c)
    wav_min = (5007*k_OIII_br) - 100
    wav_max = (5007*k_OIII) + 100
    return wav_min,wav_max


