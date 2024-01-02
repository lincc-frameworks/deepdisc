import scarlet
import scarlet.wavelet
from astropy.stats import median_absolute_deviation as astromad
import numpy as np
import sep
import matplotlib.pyplot as plt
import time

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

    # Scale =1/1.4826 to replicate older scipy MAD behavior
    scale = 1 / 1.4826
    sigma = astromad(
        scarlet.Starlet.from_image(image, scales=2).coefficients[0, ...], axis=(-2, -1), ignore_nan=True
    )
    return sigma / scale


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


    return starlet_blend, logL



def run_scarlet(
    datas,
    filters,
    catalog=None,
    stretch=0.1,
    Q=5,
    sigma_model=1,
    sigma_obs=5,
    psf=None,
    subtract_background=False,
    max_chi2=5000,
    max_iters=15,
    morph_thresh=0.1,
    lvl=5,
    lvl_segmask=2,
    maskthresh=0.025,
    segmentation_map=True,
    wave_cat=False,
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

    datas: ndarray
        multichannel array of data, shape (Nfilters x N x N)
    filters: list
        str list of filters, e.g. ('u','g','r)
    source_catalog: pandas df
        external source catalog used to initialize source positions and get additional source information
    stretch: float
        plotting parameter
    Q: float
        plotting parameter
    sigma_model: float
        scarlet model psf
    sigma_obs: FWHM for observational psf.  Use psf argument instead if you have an image of the psf
    psf: ndarray
        psf image array
    max_iters: int
        max number of iterations for scarlet to fit
    morph_thresh: float
        parameter for source initialization
    lvl: int
        detection level for wavelet catalog detection
    lvl_segmask: int
        segmentation mask detection level

    subtract_background : boolean
        Whether or not to estimate and subtract the background (often background is already subtracted)
        Detault is False
    plot_wavelet : boolean
        Plot starlet wavelet transform and inverse transform at different scales.
        NOTE: Not really useful at large image sizes (> ~few hundred pixels length/height)
        Default is False
    plot_detections : boolean
        Plot detection catalog results. Default is False
    plot_likelihood : boolean
        Plot likelihood as function of iterations from Blend fit function. Default is True
    plot_scene : boolean
        Plot full scene with the model, rendered model, observation, and residual. Default is False.
    plot_sources : boolean
        Plot the model, rendered model, observation, and spectrum across channels for each object.
        WARNING: Not advised to do this with a large image with many sources! Default is False
    plot_first_isolated_comp : boolean
        Plot the subtracted and isolated first (or any) starlet component. Recommended for finding a bright
        component. Default is False.


    Return
    -------
    observation: scarlet function
        Scarlet observation objects
    starlet_sources: list
        List of ScarletSource objects
    model_frame: scarlet function
        Image frame of source model
    catalog_deblended: list
        Deblended source detection catalog
    source_catalog: pandas df
        External catalog of source detections
    segmentation_masks: list
        List of segmentation mask of each object in image


    """

    # norm = scarlet.display.AsinhMapping(minimum=0, stretch=stretch, Q=Q)
    norm = scarlet.display.AsinhPercentileNorm(datas, percentiles=percentiles)
    
    # Define model frame and observations:
    model_psf = scarlet.GaussianPSF(sigma=sigma_model)  # , boxsize=100)
    model_frame = scarlet.Frame(datas.shape, psf=model_psf, channels=filters)

    if catalog is None:
        wavecat=True
        print('Generate source catalog using wavelets')
        t0 = time.time()
        catalog, bg_rms_hsc = make_catalog(datas, lvl, wave=True)
        print(time.time() - t0)
        # If image is already background subtracted, weights are set to 1
        if subtract_background:
            weights = np.ones_like(datas) / (bg_rms_hsc**2)[:, None, None]
        else:
            weights = np.ones_like(datas)

        print("Sep found ", len(catalog), "objects")
        Rs = np.sqrt(catalog["a"] ** 2 + catalog["b"] ** 2)
        spread = Rs / sigma_obs
        chi2s = np.zeros(len(catalog))
        centers = [(src["y"], src["x"]) for src in catalog]  # y needs to be first

    else:
        wavecat=False
        print("Source catalog has ", len(catalog), "objects")

        chi2s = np.zeros(len(catalog))
        centers = [
            (catalog.iloc[i]["new_y"], catalog.iloc[i]["new_x"])
            for i in range(len(catalog))
        ]

    # Plot wavelet transform at different scales
    if plot_wavelet == True:
        _plot_wavelet(datas)



    if psf is None:
        observation_psf = scarlet.GaussianPSF(sigma=sigma_obs)
    else:
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
    t0 = time.time()
    # Loop through detections in catalog

    starlet_sources, skipped = scarlet.initialization.init_all_sources(
        model_frame,
        centers,
        observation,
        max_components=2,
        thresh=morph_thresh,
        fallback=True,
        silent=True,
        set_spectra=False,
    )

    """
    for k, src in enumerate(catalog):

        # Is the source compact relative to the PSF?
        if spread[k] < 1:
            compact = True
        else:
            compact = False

        # Try modeling each source as a single ExtendedSource first
        new_source = scarlet.ExtendedSource(model_frame, (src['y'], src['x']), observation,
                                            K=1, thresh=morph_thresh, compact=compact)

        
        starlet_sources.append(new_source)
    """
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
            cat = [np.full(model.shape, np.nan, dtype=model.dtype)]
        # Append to full catalog
        if segmentation_map == True:
            # For some reason sep doesn't like these images, so do the segmask ourselves for now
            model_det = np.sum(model, axis=0)
            mask = np.zeros_like(model_det)
            mask[model_det > lvl_segmask * bkgmod.globalrms] = 1
            segmentation_masks.append(mask)
            # plt.imshow(mask)
            # plt.show()
        catalog_deblended.append(cat)

    # Plot scene: rendered model, observations, and residuals
    if plot_scene == True:
        _plot_scene(
            starlet_sources,
            observation,
            norm,
            catalog=catalog,
            wave_cat=wave_cat,
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







def _plot_scene(
    starlet_sources,
    observation,
    norm,
    catalog,
    wave_cat=True,
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
            if wave_cat:
                for k, src in enumerate(catalog):
                    # See https://sextractor.readthedocs.io/en/latest/Position.html
                    e = Ellipse(xy=(src['x'], src['y']),
                                width=6*src['a'],
                                height=6*src['b'],
                                angle=np.rad2deg(src['theta']))

                    e.set_facecolor('none')
                    e.set_edgecolor('white')

                    ax.add_artist(e)
            
            else:
                for i in range(len(catalog)):
                    obj = catalog.iloc[i]
                    # See https://sextractor.readthedocs.io/en/latest/Position.html
                    e = Ellipse(xy=(obj["new_x"], obj["new_y"]), width=10, height=10, angle=0)

                    e.set_facecolor("none")
                    e.set_edgecolor("white")

                    ax.add_artist(e)

        ax.axis("off")

    fig.subplots_adjust(wspace=0.01)

    if savefigs:
        plt.savefig(figpath + "scarlet_out.png")

    plt.show()

    return fig

