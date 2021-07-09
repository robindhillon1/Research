
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.optimize import leastsq
from scipy import ndimage
from func import *
from astropy.table import Table, hstack
import os
import glob
from astropy.table import Table, vstack, join


# In[2]:


def tab(obj,par):
    column_names={'Hb_x':0,'Hb_y':1,'OIII_br_x':2,'OIII_br_y':3,'OIII_nr_x':4,'OIII_nr_y':5,'off_OIII_br_pix_x':6,'off_OIII_br_pix_y':7,'off_OIII_nr_pix_x':8,'off_OIII_nr_pix_y':9,'off_OIII_br_arc_x':10,'off_OIII_br_arc_y':11,'off_OIII_nr_arc_x':12,'off_OIII_nr_arc_y':13,'off_wing_arc':14,'off_wing_parsec':15}
    columns=[]
    for key in column_names.keys():
        columns.append(fits.Column(name=key,format='E',array=[par[column_names[key]]]))
    coldefs = fits.ColDefs(columns)
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto('%s_center_table.fits'%(obj),overwrite=True)   

def source_tab(obj):
    a1 = np.array(['%s'%(obj)])
    col1 = fits.Column(name='Source', format='20A', array=a1)
    cols = fits.ColDefs([col1])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto('Source_name_%s.fits'%(obj),overwrite=True)

def create_center_table(obj):
    t1 = Table.read('Source_name_%s.fits'%(obj),format='fits')
    t2 = Table.read('%s_center_table.fits'%(obj),format='fits')
    new = hstack([t1, t2])
    new.write('%s_Moffat_centroids_with_source.fits'%(obj),overwrite=True)
    
def vel_table(obj):
    t1 = Table.read('Source_name_%s.fits'%(obj),format='fits')
    t2 = Table.read('%s_central_fit.fits'%(obj),format='fits')
    new = hstack([t1, t2])
    new.write('%s_central_tab.fits'%(obj),overwrite=True)


# In[3]:


def algorithm_script(obj,d_a,prefix_path_cube="/home/mainak/xdata/ftp.hidrive.strato.com/users/login-carsftp"):
    (cont_cube,cont_err,cont_wave,cont_header) = loadCube('%s/MUSE/%s/%s.cont_model.fits'%(prefix_path_cube,obj,obj))
    muse_sampling_size = sampling_size(cont_cube)
   
    (Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y) = centers(obj)
    (off_OIII_br_pix,off_OIII_nr_pix,off_OIII_br_arc,off_OIII_nr_arc) = offset(Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y,muse_sampling_size)
    (off_OIII_br_pix_x,off_OIII_br_pix_y) = off_OIII_br_pix
    (off_OIII_nr_pix_x,off_OIII_nr_pix_y) = off_OIII_nr_pix
    (off_OIII_br_arc_x,off_OIII_br_arc_y) = off_OIII_br_arc
    (off_OIII_nr_arc_x,off_OIII_nr_arc_y) = off_OIII_nr_arc
    off_wing_arc = np.sqrt((off_OIII_br_arc_x)**2 + (off_OIII_br_arc_y)**2)
    off_wing_parsec = 4.848*d_a*off_wing_arc
    #print '%s'%(obj),off_wing_arc,d_a,off_wing_parsec
    
    par = [Hb_x,Hb_y,OIII_br_x,OIII_br_y,OIII_nr_x,OIII_nr_y,off_OIII_br_pix_x,off_OIII_br_pix_y,off_OIII_nr_pix_x,off_OIII_nr_pix_y,off_OIII_br_arc_x,off_OIII_br_arc_y,off_OIII_nr_arc_x,off_OIII_nr_arc_y,off_wing_arc,off_wing_parsec]
    tab(obj,par)
    source_tab(obj)
    create_center_table(obj)
    vel_table(obj)


# In[4]:


z = {"HE0021-1819":0.053197,"HE0040-1105":0.041692 #,"HE0108-4743":0.02392,"HE0114-0015":0.04560
    ,"HE0119-0118":0.054341,"HE0224-2834":0.059800,"HE0227-0913":0.016451,"HE0232-0900":0.043143,"HE0253-1641":0.031588
    ,"HE0345+0056":0.031,"HE0351+0240":0.036,"HE0412-0803":0.038160,"HE0429-0247":0.042009,"HE0433-1028":0.035550
    ,"HE0853+0102":0.052,"HE0934+0119":0.050338,"HE1011-0403":0.058314,"HE1017-0305":0.049986,"HE1029-1831":0.040261
    ,"HE1107-0813":0.058,"HE1108-2813":0.024013,"HE1126-0407":0.061960,"HE1330-1013":0.022145,"HE1353-1917":0.035021
    ,"HE1417-0909":0.044,"HE2211-3903":0.039714,"HE2222-0026":0.059114,"HE2233+0124":0.056482,"HE2302-0857":0.046860}


d_a = {"HE0021-1819":213.8,"HE0040-1105":170.9 #,"HE0108-4743":0.02392,"HE0114-0015":0.04560
    ,"HE0119-0118":218.1,"HE0224-2834":238.5,"HE0227-0913":69.07,"HE0232-0900":175.5,"HE0253-1641":130.2
    ,"HE0345+0056":127.9,"HE0351+0240":147.7,"HE0412-0803":156.1,"HE0429-0247":171.1,"HE0433-1028":145.9
    ,"HE0853+0102":209.3,"HE0934+0119":203.0,"HE1011-0403":233.0,"HE1017-0305":201.7,"HE1029-1831":164.3
    ,"HE1107-0813":231.8,"HE1108-2813":99.91,"HE1126-0407":246.5,"HE1330-1013":92.344,"HE1353-1917":143.8
    ,"HE1417-0909":178.8,"HE2211-3903":162.2,"HE2222-0026":236.0,"HE2233+0124":226.2,"HE2302-0857":189.8}

#z_remaining = {"HE2128-0221":0.05248,"HE1248-1356":0.01465}

objs = z.keys()

for obj in objs:
    algorithm_script(obj,d_a[obj])


# In[22]:


infiles = sorted(glob.glob('./*source.fits'))

# Instantiate an empty dictionary from which we will make our final table
tabledict = {}

# Populate the dictionary with Astropy Tables from each FITS files
for i, file in enumerate(infiles):
    hdulist = fits.open(file)
    bintab = hdulist[1].data
    table = Table(bintab)
    name = table['Source'][0]  
    tabledict[name] = table

master_table = vstack(list(tabledict.values()))
master_table.write('final_table.fits', format='fits', overwrite=True)


# In[8]:


hdu = fits.open('final_table.fits')
central_tab = hdu[1].data
central_columns = hdu[1].header
offset_size = central_tab.field('off_wing_parsec')
dont_need = np.array(offset_size[-1]) #It's HE1353-1917velocity offset and distance
offset_wo_HE1353 = np.setdiff1d(offset_size,dont_need)


# In[9]:


plt.hist(offset_wo_HE1353,bins=10)
plt.xlabel('Outflow Size (parsec)')
plt.ylabel('Number')
plt.show()


# In[10]:


infiles = sorted(glob.glob('./*tab.fits'))

# Instantiate an empty dictionary from which we will make our final table
tabledict = {}

# Populate the dictionary with Astropy Tables from each FITS files
for i, file in enumerate(infiles):
    hdulist = fits.open(file)
    bintab = hdulist[1].data
    table = Table(bintab)
    name = table['Source'][0]  
    tabledict[name] = table
    
master_table = vstack(list(tabledict.values()))
master_table.write('central_table.fits', format='fits', overwrite=True)


# In[11]:


hdu = fits.open('central_table.fits')
central_tab = hdu[1].data
central_columns = hdu[1].header

vel_core = central_tab.field('vel_OIII')
vel_wing = central_tab.field('vel_OIII_br')
vel_offset = vel_core - vel_wing


# In[16]:


plt.plot(vel_offset,offset_size,'ko')
plt.xlabel('vel_offset(km)')
plt.ylabel('offset_size(parsec)')
plt.show()


# In[20]:


from scipy import stats
stats.wilcoxon(vel_offset,offset_size)


# In[24]:


plt.hist(vel_offset,bins=10)
plt.xlabel('Outflow velocity (km/s)')
plt.ylabel('Number')
plt.show()


# In[44]:


x=np.log10(np.abs(offset_size))


# In[50]:


plt.hist(x,bins=10)
plt.xlabel('log(outflow_size)')
plt.ylabel('Number')
#plt.xlim(0.5,1.9)
plt.show()


# In[72]:


import numpy as np
import seaborn as sns
sns.kdeplot(vel_offset)
plt.show()

