from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scarlet
from astropy.io import fits


def get_DC2_data(
    filters=["u", "g", "r", "i", "z", "y"], tract=10054, patch=[0, 0], coord=None, cutout_size=[128, 128]
):
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

    # tract = str(tract)
    # patch = (str(patch[0],str(patch[1])))
    # if coord is not None:
    #    print("Overriding tract/patch info and looking for HSC file at requested coordinates.")
    #    tract, patch = get_tract_patch_from_coord(coord)

    # print(f'Loading "{filepath}".')
    # try:

    dirpath = "/home/g4merz/DC2/coadd-t3828-t3829/deepCoadd-results/"
    # fp = os.path.join('/',*[dirpath,filters[i],tract,p,f'calexp-{filters[i]}-{tract}-{p}.fits'])

    # with fits.open(filepath) as obs_hdul:
    # obs_hdul = fits.open(filepath)
    #    alldata = obs_hdul[1].data
    #    wcs = WCS(obs_hdul[1].header).dropaxis(2)

    datas = []

    for f in filters:
        filepath = os.path.join("/", *[dirpath, f, tract, patch, f"calexp-{f}-{tract}-{patch}.fits"])

        # print(f'Loading "{filepath}".')
        # try:

        # with fits.open(filepath) as obs_hdul:
        # obs_hdul = fits.open(filepath)
        #    data = obs_hdul[1].data
        #    wcs = WCS(obs_hdul[1].header)

        data, hdr = fits.getdata(filepath, 1, header=True, memmap=False)
        wcs = WCS(hdr)

        cutout = None

        # Cutout data at center of patch (coord=None) or at coord (if specified)
        if cutout_size is not None:
            # Use coord for center position if specified
            if coord is None:
                shape = np.shape(data)
                position = (shape[0] / 2, shape[1] / 2)
            else:
                position = coord
            # data = Cutout2D(data, position=position, size=cutout_size, wcs=wcs).data
            cutout = Cutout2D(data, position=position, size=cutout_size, wcs=wcs)
            data = cutout.data

        datas.append(data)
        # except:
        #    print('Missing filter ', f)

    return np.array(datas), cutout


def get_centers(sub_shape, n):
    centers = []
    for i in range(n):
        for j in range(n):
            # print(sub_shape[1]*i)
            s = np.array(sub_shape) / 2 + (sub_shape[0] * j, sub_shape[1] * i)
            centers.append(s)

    return centers


# def get_cutout(tract,patch,sp,plot=True):
def get_cutout(tract, patch, sp, plot=True):
    nblocks = 8
    nfilters = 6

    # Loading "/home/g4merz/deblend/data/raw_HSC_DR3/HSC-G/calexp-HSC-G-8765-2,3.fits".
    # hsc_dirpath = '/home/g4merz/deblend/data/raw_HSC_DR3/'

    dat, cutout = get_DC2_data(tract=tract, patch=patch, coord=None, cutout_size=None)

    block_size = [dat.shape[1] // nblocks, dat.shape[2] // nblocks]
    # datas_blocks = view_as_blocks(dat, block_shape=(nfilters, block_size[0], block_size[1]))
    # sub_shape = datas_blocks.shape[-2:]

    sub_shape = [dat.shape[1] // nblocks, dat.shape[2] // nblocks]
    # review this code
    centers = get_centers(sub_shape[::-1], nblocks)
    # centers = get_centers(sub_shape)

    coord = centers[sp]

    # coord = [dat.shape[2]/2,dat.shape[1]/2]
    # datsm,cutout = get_DC2_data(tract=tract,patch=patch,coord=coord,cutout_size=(dat.shape[1]//2,dat.shape[2]//2))
    datsm, cutout = get_DC2_data(tract=tract, patch=patch, coord=coord, cutout_size=sub_shape)
    # datas = datas_blocks[sp,:,:,:]
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        img_rgb = scarlet.display.img_to_rgb(dat, norm=norm)
        img_rgbsm = scarlet.display.img_to_rgb(datsm, norm=norm)

        ax[0].imshow(img_rgb, origin="lower")
        cutout.plot_on_original(ax[0], color="white")
        ax[1].imshow(img_rgbsm, origin="lower")

        ax[0].axis("off")
        ax[1].axis("off")
        plt.tight_layout()

    return cutout, datsm
