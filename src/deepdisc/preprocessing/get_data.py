
import numpy as np
import os
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import scarlet
import pandas as pd

from scarlet.display import AsinhMapping
stretch = 1
Q = 5
NORM = AsinhMapping(minimum=0, stretch=stretch, Q=Q)


def get_DC2_data(dirpath, filters=['u','g','r','i','z','y'], tract=10054, patch=[0,0], coord=None, cutout_size=[128, 128]):
    """
    Get HSC data given tract/patch info or SkyCoord
    
    Parameters
    ----------
    dirpath : str
        Path to HSC image file directory
    filters : list 
        A list of filters for your images. Default is ['g', 'r', 'i'].
    tract  : int
        An integer used for specifying the tract. Default is 10054|
    patch : [int, int]
        Patch #,#. Default is [0,0]
    coord  : SkyCoord
        Astropy SkyCoord, when specified, overrides tract/patch info and attempts to lookup HSC filename from ra, dec. 
        Default is None
    cutout_size: [int, int]
        Size of cutout to use (set to None for no cutting). Default is [128, 128]
        
    The image filepath is in the form:
        {dirpath}/deepCoadd/HSC-{filter}/{tract}/{patch[0]},{patch[1]}/calexp-HSC-{filter}-{tract}-{patch[0]},{patch[1]}.fits
    
    Returns
    -------
    data : ndarray
        HSC data array with dimensions [filters, N, N]
    """

    
    datas = []

    for f in filters:
        filepath = os.path.join('/',*[dirpath,f,tract,patch,f'calexp-{f}-{tract}-{patch}.fits'])
        
        #print(f'Loading "{filepath}".')
        #try:
        
        with fits.open(filepath) as obs_hdul:
        #obs_hdul = fits.open(filepath)
            data = obs_hdul[1].data
            wcs = WCS(obs_hdul[1].header)
        
        cutout =None
        
        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0]/2, shape[1]/2)
            else:
                position = coord
            #data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            cutout = Cutout2D(data, position=position, size=cutout_size, wcs=wcs)
            data = cutout.data

        datas.append(data)
        #except:
        #    print('Missing filter ', f)
            

    return np.array(datas), cutout


def get_centers(sub_shape,n):
    centers=[]
    for i in range(n):
        for j in range(n):
            #print(sub_shape[1]*i)
            s=np.array(sub_shape)/2 + (sub_shape[0]*j, sub_shape[1]*i)
            centers.append(s)
            
    return centers


def get_cutout(dirpath,tract,patch,sp,nblocks=4,filters=['u','g','r','i','z','y'],plot=False):

    dat,cutout = get_DC2_data(dirpath,filters=filters,tract=tract,patch=patch,coord=None,cutout_size=None)
    print(dat.shape)
    
    block_size = [dat.shape[1]//nblocks, dat.shape[2]//nblocks]
    print(block_size)

    
    sub_shape =[dat.shape[1]//nblocks,dat.shape[2]//nblocks]
    centers = get_centers(sub_shape[::-1],nblocks)

    coord=centers[sp]
    
    datsm,cutout = get_DC2_data(dirpath,tract=tract,patch=patch,coord=coord,cutout_size=sub_shape)
    if plot:
        fig,ax = plt.subplots(1,2,figsize=(10,10))
        img_rgb = scarlet.display.img_to_rgb(dat, norm=NORM)
        img_rgbsm = scarlet.display.img_to_rgb(datsm, norm=NORM)

        ax[0].imshow(img_rgb,origin='lower')
        cutout.plot_on_original(ax[0],color='white')
        ax[1].imshow(img_rgbsm,origin='lower')

        ax[0].axis('off')
        ax[1].axis('off')
        plt.tight_layout()
    
    return cutout,datsm


def get_cutout_cat(dirpath,dall,tract,patch,sp,nblocks=4,filters=['u','g','r','i','z','y']):
    '''
        WARNING!!!!!
        It is not efficient to have the full catalog as input when doing multiprocesing.  
        Keep it in the top level process
    
    '''
    cutout,datsm= get_cutout(tract=tract,patch=patch,sp=sp,nblocks=nblocks, filters=filters,plot=False)
    xs,ys = cutout.wcs.world_to_pixel(allcatalog)
    inds = np.where((xs>=0) & (xs<cutout.shape[1]-1) & (ys>=0) & (ys<cutout.shape[0]-1))[0]
    
    dcut = dall.iloc[inds]

    dcut['new_x'] = xs[inds]
    dcut['new_y'] = ys[inds]

    column_to_move = dcut.pop("objectId")

    # insert column with insert(location, column_name, column_value)
    dcut.insert(0, "objectId", column_to_move)
    dcut.sort_values(by='objectId')
    
    return datsm, dcut