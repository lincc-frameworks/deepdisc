import os
import sys
import time

import astropy.io.fits as fits
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scarlet
import scarlet.wavelet
import sep
from astropy.io import ascii
from astropy.stats import median_absolute_deviation as astromad
from astropy.visualization.lupton_rgb import AsinhMapping, LinearMapping
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from scipy.stats import median_abs_deviation as mad


def write_scarlet_results(
    datas,
    observation,
    starlet_sources,
    model_frame,
    catalog_deblended,
    segmentation_masks,
    dirpath,
    filters,
    s,
):
    """
    Saves images in each channel, with headers for each source in image,
    such that the number of headers = number of sources detected in image.

    Parameters
    ----------
    datas: array
        array of Data objects
    observation: scarlet function
        Scarlet observation objects
    starlet_sources: list
        List of ScarletSource objects
    model_frame: scarlet function
        Image frame of source model
    catalog_deblended: list
        Deblended source detection catalog
    segmentation_masks: list
        List of segmentation mask of each object in image
    dirpath : str
        Path to HSC image file directory
    filters : list
        A list of filters for your images. Default is ['g', 'r', 'i'].
    s : str
        File basename string


    Returns
    -------
    filename : dict
        dictionary of all paths to the saved scarlet files for the particular dataset.
        Saved image and model files for each filter, and one total segmentation mask file for all filters.
    """

    def _make_hdr(starlet_source, cat):
        """
        Helper function to make FITS header and insert metadata.
        Parameters
        ----------
        starlet_source: starlet_source
            starlet_source object for source k
        cat: dict
            catalog data for source k

        Returns
        -------
        model_hdr : Astropy fits.Header
            FITS header for source k with catalog metadata
        """
        # For each header, assign descriptive data about each source
        # (x0, y0, w, h) in absolute floating pixel coordinates
        bbox_h = starlet_source.bbox.shape[1]
        bbox_w = starlet_source.bbox.shape[2]
        bbox_y = starlet_source.bbox.origin[1] + int(np.floor(bbox_w / 2))  # y-coord of the source's center
        bbox_x = starlet_source.bbox.origin[2] + int(np.floor(bbox_w / 2))  # x-coord of the source's center

        # Ellipse parameters (a, b, theta) from deblend catalog
        e_a, e_b, e_theta = cat["a"], cat["b"], cat["theta"]
        ell_parm = np.concatenate((cat["a"], cat["b"], cat["theta"]))

        # Add info to header
        model_hdr = fits.Header()
        model_hdr["bbox"] = ",".join(map(str, [bbox_x, bbox_y, bbox_w, bbox_h]))
        model_hdr["area"] = bbox_w * bbox_h
        model_hdr["ell_parm"] = ",".join(map(str, list(ell_parm)))
        model_hdr["cat_id"] = 1  # Category ID #TODO: set categor_id based on if the source is extended or not

        return model_hdr

    # Create dict for all saved filenames
    segmask_hdul = []
    model_hdul = []
    filenames = {}

    # Filter loop
    for i, f in enumerate(filters):  # datas is HSC data array with dimensions [filters, N, N]
        f = f.upper()

        # Primary HDU is full image
        img_hdu = fits.PrimaryHDU(data=datas[i])

        # Create header entry for each scarlet source
        for k, (src, cat) in enumerate(zip(starlet_sources, catalog_deblended)):
            # Get each model, make into image
            model = starlet_sources[k].get_model(frame=model_frame)
            model = observation.render(model)
            model = src.bbox.extract_from(model)

            model_hdr = _make_hdr(starlet_sources[k], cat)

            model_hdu = fits.ImageHDU(data=model[i], header=model_hdr)
            model_primary = fits.PrimaryHDU()

            model_hdul.append(model_hdu)

        # Write final fits file to specified location
        # Save full image and then headers per source w/ descriptive info
        save_img_hdul = fits.HDUList([img_hdu])
        save_model_hdul = fits.HDUList([model_primary, *model_hdul])

        # GM-11/29/22
        # There's some bug here that causes the R band model.fits file to copy all the catalog sources a second time
        # and the I band model.fits file to copy all the sources a third time
        # I.e., All bands should have the same N sources, but R band has 2N and I band has 3N
        # I'm not tracking this down as of now because the G band is fine and
        # the only downstream thing the model files are used for is determining the source class from the catalog cross-matching

        # Save list of filenames in dict for each band
        filenames[f"img_{f}"] = os.path.join(dirpath, f"{f}-{s}_scarlet_img.fits")
        save_img_hdul.writeto(filenames[f"img_{f}"], overwrite=True)

        filenames[f"model_{f}"] = os.path.join(dirpath, f"{f}-{s}_scarlet_model.fits")
        save_model_hdul.writeto(filenames[f"model_{f}"], overwrite=True)

    # If we have segmentation mask data, save them as a separate fits file
    # Just using the G band for the segmentation mask
    if segmentation_masks is not None:
        for i, f in enumerate(filters[0]):
            # Create header entry for each scarlet source
            for k, (src, cat) in enumerate(zip(starlet_sources, catalog_deblended)):
                segmask_hdr = _make_hdr(starlet_sources[k], cat)

                # Save each model source k in the image
                segmask_hdu = fits.ImageHDU(data=segmentation_masks[k], header=segmask_hdr)
                segmask_primary = fits.PrimaryHDU()

                segmask_hdul.append(segmask_hdu)

            save_segmask_hdul = fits.HDUList([segmask_primary, *segmask_hdul])

            # Save list of filenames in dict for each band
            filenames["segmask"] = os.path.join(dirpath, f"{f}-{s}_scarlet_segmask.fits")
            save_segmask_hdul.writeto(filenames["segmask"], overwrite=True)

    return filenames


def plot_stretch_Q(datas, stretches=[0.01, 0.1, 0.5, 1], Qs=[1, 10, 5, 100]):
    """
    Plots different normalizations of your image using the stretch, Q parameters.

    Parameters
    ----------
    stretches : array
        List of stretch params you want to permutate through to find optimal image normalization.
        Default is [0.01, 0.1, 0.5, 1]
    Qs : array
        List of Q params you want to permutate through to find optimal image normalization.
        Default is [1, 10, 5, 100]

    Code adapted from:
        https://pmelchior.github.io/scarlet/tutorials/display.html

    Returns
    -------
    fig : Figure object

    """

    fig, ax = plt.subplots(len(stretches), len(Qs), figsize=(9, 9))
    for i, stretch in enumerate(stretches):
        for j, Q in enumerate(Qs):
            asinh = scarlet.display.AsinhMapping(minimum=0, stretch=stretch, Q=Q)
            # Scale the RGB channels for the image
            img_rgb = scarlet.display.img_to_rgb(datas, norm=asinh)
            ax[i][j].imshow(img_rgb)
            ax[i][j].set_title("Stretch {}, Q {}".format(stretch, Q))
            ax[i][j].axis("off")

    return fig


def make_catalog(
    datas,
    lvl=4,
    wave=True,
    segmentation_map=False,
    maskthresh=10.0,
    object_limit=100000,
):
    """
    Creates a detection catalog by combining low and high resolution data

    Parameters
    ----------
    datas: array
        array of Data objects
    lvl: int
        detection lvl
    wave: Bool
        set to True to use wavelet decomposition of images before combination
    subtract_background : Bool
        if you want to subtract the background and retrieve an estimate, change to True. But default
        is False because HSC images are already background subtracted.
    segmentation_map : Bool
        Whether to run sep segmentation map
    maskthresh : float
        Mask threshold for sep segmentation
    object_limit : int
        Limit on number of objects to detect in image

    Code adapted from https://pmelchior.github.io/scarlet/tutorials/wavelet_model.html

    Returns
    -------
    catalog: sextractor catalog
        catalog of detected sources (use 'catalog.dtype.names' for info)
    bg_rms: array
        background level for each data set (set to None if subtract_background is False)
    """

    if type(datas) is np.ndarray:
        hr_images = datas / np.sum(datas, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        # detect_image = np.sum(hr_images, axis=0)
        detect_image = np.sum(datas, axis=0)

    else:
        data_lr, data_hr = datas
        # Create observations for each image
        # Interpolate low resolution to high resolution
        interp = interpolate(data_lr, data_hr)
        # Normalization of the interpolate low res images
        interp = interp / np.sum(interp, axis=(1, 2))[:, None, None]
        # Normalisation of the high res data
        hr_images = data_hr.images / np.sum(data_hr.images, axis=(1, 2))[:, None, None]
        # Detection image as the sum over all images
        detect_image = np.sum(interp, axis=0) + np.sum(hr_images, axis=0)
        detect_image *= np.sum(data_hr.images)
    if np.size(detect_image.shape) == 4:
        if wave:
            # Wavelet detection in the first three levels
            wave_detect = scarlet.Starlet(detect_image.mean(axis=0), lvl=5).coefficients
            wave_detect[:, -1, :, :] = 0
            detect = scarlet.Starlet(coefficients=wave_detect).image
        else:
            # Direct detection
            detect = detect_image.mean(axis=0)
    else:
        if wave:
            wave_detect = scarlet.Starlet.from_image(detect_image).coefficients
            detect = wave_detect[0] + wave_detect[1] + wave_detect[2]
        else:
            detect = detect_image

    bkg = sep.Background(detect)
    # Set the limit on the number of sub-objects when deblending.
    sep.set_sub_object_limit(object_limit)

    # Extract detection catalog with segmentation maps!
    # Can use this to retrieve ellipse params
    catalog = sep.extract(
        detect,
        lvl,
        err=bkg.globalrms,
        segmentation_map=segmentation_map,
        maskthresh=maskthresh,
    )

    # Estimate background
    # Have to include because will no longer take ndarray
    if type(datas) is np.ndarray and len(datas > 2):
        bkg_rms = np.zeros(len(datas))
        for i, data in enumerate(datas):
            bkg_rms[i] = mad_wavelet_own(data)

    else:
        bkg_rms = []
        for data in datas:
            bkg_rms.append(scarlet.wavelet.mad_wavelet(data.images))

    return catalog, bkg_rms


def mad_wavelet_own(image):
    """image: Median absolute deviation of the first wavelet scale.
    (WARNING: sorry to disapoint, this is not a wavelet for mad scientists)

    Have to use astropy mad as scipy mad does not like ignoring NaN and computing over multiple axes

    Parameters
    ----------
    image: array
        An image or cube of images
    Returns
    -------
    mad: array
        median absolute deviation for each image in the cube
    """

    # Scarlet seems to no longer take ndarrays, so have to go through each channel
    # Scale =1/1.4826 to replicate older scipy MAD behavior
    scale = 1 / 1.4826
    sigma = astromad(
        scarlet.Starlet.from_image(image, scales=2).coefficients[0, ...],
        axis=(-2, -1),
        ignore_nan=True,
    )
    return sigma / scale


def fit_scarlet_blend(
    starlet_sources,
    observation,
    catalog,
    max_iters=15,
    e_rel=1e-4,
    plot_likelihood=True,
    savefigs=False,
    figpath="",
):
    """
    Creates a detection catalog by combining low and high resolution data

    Parameters
    ----------
    datas: array
        array of Data objects

    Will end early if likelihood and constraints converge

    Returns
    -------
    """
    # Create and fit Blend model. Go for 200 iterations,
    # but will end early if likelihood and constraints converge
    print(f"Fitting Blend model.")
    try:
        starlet_blend = scarlet.Blend(starlet_sources, observation)
        it, logL = starlet_blend.fit(max_iters, e_rel=e_rel)
        print(f"Scarlet ran for {it} iterations to logL = {logL}")

    # Catch any exceptions like no detections
    except AssertionError as e1:
        print(f"Length of detection catalog is {len(catalog)}.")
        raise

    if plot_likelihood == True:
        scarlet.display.show_likelihood(starlet_blend)
        plt.ylabel("log-Likelihood", fontsize=15)
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.xlabel("Iteration", fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.subplots_adjust(left=0.2)
        if savefigs:
            plt.savefig(figpath + "scarlet_likelihood.png")

        plt.show()

    return starlet_blend, logL


def _plot_wavelet(datas):
    """
    Helper function to plot wavelet transformation diagnostic figures with scarlet

    Parameters
    ----------
    datas: array
        array of Data objects

    Returns
    -------
    """

    # Declare a starlet object (and performs the transform)
    Sw = scarlet.Starlet.from_image(datas)  # , lvl=5, direct=True)

    # This is the starlet transform as an array
    w = Sw.coefficients
    # The inverse starlet transform of w (new object otherwise, the tranform is not used)
    iw = Sw.image

    # TODO: Clean this code up using plt.subplots()
    # The wavelet transform of the first slice of images in pictures
    lvl = w.shape[1]
    plt.figure(figsize=(lvl * 5 + 5, 5))
    plt.suptitle("Wavelet coefficients")
    for i in range(lvl):
        plt.subplot(1, lvl, i + 1)
        plt.title("scale" + str(i + 1))
        plt.imshow(w[0, i], cmap="inferno")
        plt.colorbar()
    plt.show()

    # Making sure we recover the original image
    plt.figure(figsize=(30, 10))
    plt.subplot(131)
    plt.title("Original image", fontsize=20)
    plt.imshow(datas[0], cmap="inferno")
    plt.colorbar()
    plt.subplot(132)
    plt.title("Starlet-reconstructed image", fontsize=20)
    plt.imshow(iw[0], cmap="inferno")
    plt.colorbar()
    plt.subplot(133)
    plt.title("Absolute difference", fontsize=20)
    plt.imshow((np.abs(iw[0] - datas[0])), cmap="inferno")
    plt.colorbar()
    plt.show()

    return


def _plot_scene(
    starlet_sources,
    observation,
    norm,
    catalog,
    show_model=True,
    show_rendered=True,
    show_observed=True,
    show_residual=True,
    add_labels=True,
    add_boxes=True,
    add_ellipses=True,
    savefigs=False,
    figpath="",
):
    """
    Helper function to plot scene with scarlet

    Parameters
    ----------
    starlet_sources: List
        List of ScarletSource objects
    observation:
        Scarlet observation objects
    norm:
        Scarlet normalization for plotting
    catalog: list
        Source detection catalog
    show_model: bool
        Whether to show model
    show_rendered: bool
        Whether to show rendered model
    show_observed: bool
        Whether to show observed
    show_residual: bool
        Whether to show residual
    add_labels: bool
        Whether to add labels
    add_boxes: bool
        Whether to add bounding boxes to each panel
    add_ellipses: bool
        Whether to add ellipses to each panel


    Returns
    -------
    fig : matplotlib Figure
        Figure object
    """

    fig = scarlet.display.show_scene(
        starlet_sources,
        observation=observation,
        norm=norm,
        show_model=show_model,
        show_rendered=show_rendered,
        show_observed=show_observed,
        show_residual=show_residual,
        add_labels=add_labels,
        add_boxes=add_boxes,
    )

    for ax in fig.axes:
        # Plot sep ellipse around all sources from the detection catalog
        if add_ellipses == True:
            for k, src in enumerate(catalog):
                # See https://sextractor.readthedocs.io/en/latest/Position.html
                e = Ellipse(
                    xy=(src["x"], src["y"]),
                    width=6 * src["a"],
                    height=6 * src["b"],
                    angle=np.rad2deg(src["theta"]),
                )

                e.set_facecolor("none")
                e.set_edgecolor("white")

                ax.add_artist(e)

        ax.axis("off")

    fig.subplots_adjust(wspace=0.01)

    if savefigs:
        plt.savefig(figpath + "scarlet_out.png")

    plt.show()

    return fig


def run_scarlet(
    datas,
    filters,
    stretch=0.1,
    Q=5,
    sigma_model=1,
    sigma_obs=5,
    psf=None,
    subtract_background=False,
    max_chi2=5000,
    max_iters=15,
    morph_thresh=0.1,
    starlet_thresh=0.1,
    lvl=5,
    lvl_segmask=2,
    maskthresh=0.025,
    segmentation_map=True,
    plot_wavelet=False,
    plot_likelihood=True,
    plot_scene=False,
    plot_sources=False,
    add_ellipses=True,
    add_labels=False,
    add_boxes=False,
    percentiles=(1, 99),
    savefigs=False,
    figpath="",
    weights=None,
):
    """Run P. Melchior's scarlet (https://github.com/pmelchior/scarlet) implementation
    for source separation. This function will create diagnostic plots, a source detection catalog,
    and fit a model for all sources in the observation scene (image).

    Parameters
    ----------
    subtract_background : boolean
        Whether or not to estimate and subtract the background (often background is already subtracted)
        Detault is False
    plot_wavelet_transform : boolean
        Plot starlet wavelet transform and inverse transform at different scales.
        NOTE: Not really useful at large image sizes (> ~few hundred pixels length/height)
        Default is False
    plot_detections : boolean
        Plot detection catalog results. Default is False
    plot_likelihood : boolean
        Plot likelihood as function of iterations from Blend fit function. Default is True
    plot_full_scene : boolean
        Plot full scene with the model, rendered model, observation, and residual. Default is False.
    plot_all_sources : boolean
        Plot the model, rendered model, observation, and spectrum across channels for each object.
        WARNING: dumb to do this with a large image with many sources! Default is False
    plot_first_isolated_comp : boolean
        Plot the subtracted and isolated first (or any) starlet component. Recommended for finding a bright
        component. Default is False.


    Return
    -------
    FITS file with...
    TODO: fill this out once I get the exact fits file output generated to Colin's liking
    """

    # norm = scarlet.display.AsinhMapping(minimum=0, stretch=stretch, Q=Q)
    norm = scarlet.display.AsinhPercentileNorm(datas, percentiles=percentiles)

    # Generate source catalog using wavelets
    t0 = time.time()
    catalog, bg_rms_hsc = make_catalog(datas, lvl, wave=True)
    print(time.time() - t0)
    # If image is already background subtracted, weights are set to 1
    if subtract_background:
        weights = np.ones_like(datas) / (bg_rms_hsc**2)[:, None, None]
    else:
        weights = np.ones_like(datas)

    print("Source catalog found ", len(catalog), "objects")

    # Plot wavelet transform at different scales
    if plot_wavelet == True:
        _plot_wavelet(datas)

    # Define model frame and observations:
    model_psf = scarlet.GaussianPSF(sigma=sigma_model)  # , boxsize=100)
    model_frame = scarlet.Frame(datas.shape, psf=model_psf, channels=filters)

    # observation_psf = scarlet.GaussianPSF(sigma=sigma_obs)
    observation_psf = scarlet.ImagePSF(psf)
    observation = scarlet.Observation(datas, psf=observation_psf, weights=weights, channels=filters).match(
        model_frame
    )

    # Initialize starlet sources to be fit. Assume extended sources for all because
    # we are not looking at all detections in each image

    # TODO: Plot chi2 vs. binned size and mag. Implement conidition if chi2 > xxx then
    # add another component until larger sources are modeled well
    print("Initializing starlet sources to be fit.")

    # Compute radii and spread of sources
    Rs = np.sqrt(catalog["a"] ** 2 + catalog["b"] ** 2)
    spread = Rs / sigma_obs

    # Array of chi^2 residuals computed after fit on each model
    chi2s = np.zeros(len(catalog))
    t0 = time.time()
    # Loop through detections in catalog
    starlet_sources = []
    for k, src in enumerate(catalog):
        # Is the source compact relative to the PSF?
        if spread[k] < 1:
            compact = True
        else:
            compact = False

        # Try modeling each source as a single ExtendedSource first
        new_source = scarlet.ExtendedSource(
            model_frame,
            (src["y"], src["x"]),
            observation,
            K=1,
            thresh=morph_thresh,
            compact=compact,
        )

        starlet_sources.append(new_source)

    # Fit scarlet blend
    starlet_blend, logL = fit_scarlet_blend(
        starlet_sources,
        observation,
        catalog,
        max_iters=max_iters,
        plot_likelihood=plot_likelihood,
        savefigs=savefigs,
        figpath=figpath,
    )
    print(time.time() - t0)

    """
    print("Computing residuals.")

    # Compute reduced chi^2 for each rendered sources
    for k, src in enumerate(starlet_sources):

        model = src.get_model(frame=model_frame)
        model = observation.render(model)
        res = datas - model

        # Compute in bbox only
        res = src.bbox.extract_from(res)
        chi2s[k] = np.sum(res**2)
        # Replace models with poor fits with StarletSource models
        if chi2s[k] > max_chi2:
            starlet_sources[k] = scarlet.StarletSource(model_frame,
                                                       (catalog["y"][k], catalog["x"][k]), observation,
                                                       thresh=morph_thresh, starlet_thresh=starlet_thresh
                                                       )

    # If any chi2 residuals are flagged, re-fit the blend with a more complex model
    if np.any(chi2s > max_chi2):
        
        print("Re-fitting with Starlet models for poorly-fit sources.")
        
        starlet_blend, logL = fit_scarlet_blend(starlet_sources, observation, catalog, max_iters=max_iters, 
                                                plot_likelihood=plot_likelihood,savefigs=savefigs,figpath=figpath)

        
    """
    # Extract the deblended catalog and update the chi2 residuals
    print("Extracting deblended catalog.")

    catalog_deblended = []
    segmentation_masks = []

    for k, src in enumerate(starlet_sources):
        model = src.get_model(frame=model_frame)
        model = observation.render(model)
        # Compute in bbox only
        model = src.bbox.extract_from(model)

        bkgmod = sep.Background(np.sum(model, axis=0))

        # Run sep
        try:
            cat, _ = make_catalog(
                model,
                lvl_segmask,
                wave=False,
                segmentation_map=False,
                maskthresh=maskthresh,
            )
        except:
            print(f"Exception with source {k}")
            cat = []
        # if segmentation_map == True:
        #    cat, mask = cat
        # If more than 1 source is detected for some reason (e.g. artifacts)
        if len(cat) > 1:
            # keep the brightest
            idx = np.argmax([c["cflux"] for c in cat])
            cat = cat[idx]
        #    if segmentation_map == True:
        #        mask = mask[idx]
        # If failed to detect model source
        if len(cat) == 0:
            # Fill with nan
            cat = [np.full(catalog[0].shape, np.nan, dtype=catalog.dtype)]
        # Append to full catalog
        if segmentation_map == True:
            # For some reason sep doesn't like these images, so do the segmask ourselves for now
            # model_det = np.array(model[0,:,:])
            model_det = np.sum(model, axis=0)
            mask = np.zeros_like(model_det)
            # mask[model_det>lvl_segmask*bg_rms_hsc[0]] = 1
            mask[model_det > lvl_segmask * bkgmod.globalrms] = 1
            segmentation_masks.append(mask)
            # plt.imshow(mask)
            # plt.show()
        catalog_deblended.append(cat)

        """
        try:
            catalog,mask = sep.extract(model, lvl_segmask, err=bg_rms_hsc, segmentation_map=True)
        except:
            print('Exception with source {i} in file ')
            #exceptions.append(i)
            mask = np.zeros_like(model)
            mask[model>lvl_segmask*bg_rms_hsc] = 1
            catalog=[]

        # If more than 1 source is detected for some reason (e.g. artifacts)
        if len(catalog) > 1:
            print('More than 1 source for object ', i, ' in file ')
            # keep the brightest
            idx = np.argmax([c['cflux'] for c in catalog])
            catalog = catalog[idx]
            mask[mask!=idx]=0

        segmentation_masks.append(mask)
        catalog_deblended.append(catalog)
        """

    # Combine catalog named array
    catalog_deblended = np.vstack(catalog_deblended)

    # Plot scene: rendered model, observations, and residuals
    if plot_scene == True:
        _plot_scene(
            starlet_sources,
            observation,
            norm,
            catalog,
            show_model=False,
            show_rendered=True,
            show_observed=True,
            show_residual=True,
            add_labels=add_labels,
            add_boxes=add_boxes,
            add_ellipses=add_ellipses,
            savefigs=savefigs,
            figpath=figpath,
        )

    # Plot each for each source
    if plot_sources == True:
        scarlet.display.show_sources(
            starlet_sources,
            observation,
            norm=norm,
            show_rendered=True,
            show_observed=True,
            add_boxes=add_boxes,
        )
        if savefigs:
            plt.savefig(figpath + "sources.png")
        plt.show()

    return (
        observation,
        starlet_sources,
        model_frame,
        catalog,
        catalog_deblended,
        segmentation_masks,
    )


def overlapped_slices(bbox1, bbox2):
    """Slices of bbox1 and bbox2 that overlap

    Parameters
    ---Z-------
    bbox1: `~scarlet.bbox.Box`
    bbox2: `~scarlet.bbox.Box`

    Returns
    -------
    slices: tuple of slices
        The slice of an array bounded by `bbox1` and
        the slice of an array bounded by `bbox` in the
        overlapping region.
    """
    overlap = bbox1 & bbox2
    _bbox1 = overlap - bbox1.origin
    _bbox2 = overlap - bbox2.origin
    slices = (
        _bbox1.slices,
        _bbox2.slices,
    )
    return slices


def get_processed_hsc_DR3_data(
    filename,
    filters=["g", "r", "i"],
    dirpath="/home/g4merz/deblend/data/processed_HSC_DR3/lvl5/",
    stringcap=14,
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
        An integer used for specifying the tract. Default is 10054
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

    s = filename.split(f"G-")[1].split(".fits")[0]
    tract, patch, sp = s.split("-")
    tract = int(tract)
    patch = tuple(map(int, patch.split(",")))
    sp = int(sp[1:-stringcap])

    filters = [f.upper() for f in filters]

    # if coord is not None:
    #    print("Overriding tract/patch info and looking for HSC file at requested coordinates.")
    #    tract, patch = get_tract_patch_from_coord(coord)

    datas = []
    # models = []
    # print(dirpath,'dirpath')

    for f in filters:
        impath = os.path.join(dirpath, f"{f}-{tract}-{patch[0]},{patch[1]}-c{sp}_scarlet_img.fits")
        modpath = os.path.join(dirpath, f"{f}-{tract}-{patch[0]},{patch[1]}-c{sp}_scarlet_model.fits")
        # print(impath, 'impath')

        # print(f'Loading "{filepath}".')
        try:
            with fits.open(impath) as obs_hdul:
                data = obs_hdul[0].data
                wcs = WCS(obs_hdul[0].header)
                # fits.close(impath)

            # mod_hdul = fits.open(modpath)
            # model = mod_hdul[1].data

            datas.append(data)
        # models.append(model)

        except:
            print("Missing filter ", f)
            return None

    return np.array(datas)


def return_model_objects(
    fiG,
    luptonize=False,
    stringcap=14,
    dirpath="/home/shared/hsc/HSC/HSC_DR3/data/train/",
):
    s = fiG.split(f"G-")[1].split(".fits")[0]
    tract, patch, sp = s.split("-")
    tract = int(tract)
    patch = tuple(map(int, patch.split(",")))
    sp = int(sp[1:-stringcap])

    # print('Running on ', fiG)

    # fiR=f'/home/shared/hsc/HSC/HSC_DR3/data/train/R-{tract}-{patch[0]},{patch[1]}-c{sp}_scarlet_model.fits'

    # fiI=f'/home/shared/hsc/HSC/HSC_DR3/data/train/I-{tract}-{patch[0]},{patch[1]}-c{sp}_scarlet_model.fits'

    fiR = dirpath + f"R-{tract}-{patch[0]},{patch[1]}-c{sp}_scarlet_model.fits"

    fiI = dirpath + f"I-{tract}-{patch[0]},{patch[1]}-c{sp}_scarlet_model.fits"

    d = get_processed_hsc_DR3_data(fiG, dirpath=dirpath, stringcap=stringcap)
    fb = scarlet.bbox.Box(d.shape[1:])

    model = np.zeros(d.shape)
    with fits.open(fiG) as sourcesG:
        ls = len(sourcesG)

    objectsG = []
    objectsR = []
    objectsI = []

    for i, file in enumerate([fiG, fiR, fiI]):
        bandmodel = np.zeros(d.shape[1:])
        l = 1 + i * (ls - 1)
        u = l + ls - 1
        with fits.open(file) as sources:
            sources = fits.open(file)
            for src in sources[l:u]:
                srcmodel = src.data
                if i == 0:
                    objectsG.append(srcmodel)
                elif i == 1:
                    objectsR.append(srcmodel)
                elif i == 2:
                    objectsI.append(srcmodel)

                bb = src.header["BBOX"]
                bb = bb.split(",")
                bmin1 = int(bb[0])
                bmin2 = int(bb[1])
                bs1 = int(bb[2])
                bs2 = int(bb[3])

                shape = [bs2, bs1]
                origin = [
                    bmin2 - int(np.floor(bs1 / 2)),
                    bmin1 - int(np.floor(bs2 / 2)),
                ]
                # print(shape,origin)

                mb = scarlet.bbox.Box(shape, origin)
                frame_slices, model_slices = overlapped_slices(fb, mb)
                result = np.zeros(fb.shape)
                result[frame_slices] = srcmodel[model_slices]

                bandmodel += result
            model[i] = bandmodel

        # ps=PSNR(d,model,luptonize)

    return objectsG, objectsR, objectsI, model


def return_spliced_sources(sourceG, sourceR, sourceI):
    wmin = np.array([sourceG.shape[0], sourceR.shape[0], sourceI.shape[0]]).min()
    hmin = np.array([sourceG.shape[1], sourceR.shape[1], sourceI.shape[1]]).min()

    sources = []
    for source in [sourceG, sourceR, sourceI]:
        if source.shape[0] > wmin:
            source = source[(source.shape[0] - wmin) // 2 : -(source.shape[0] - wmin) // 2, :]
        if source.shape[1] > hmin:
            source = source[:, (source.shape[1] - hmin) // 2 : -(source.shape[1] - hmin) // 2]
        sources.append(source)

    return sources
