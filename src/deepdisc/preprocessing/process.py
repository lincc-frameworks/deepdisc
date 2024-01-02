import numpy as np
import scarlet
import astropy.io.fits as fits
import os
import h5py

def write_scarlet_results(
    datas,
    observation,
    starlet_sources,
    model_frame,
    catalog_deblended,
    segmentation_masks,
    outdir,
    filters,
    s,
    source_catalog=None,
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
    source_catalog: pandas df
        External catalog of source detections
    segmentation_masks: list
        List of segmentation mask of each object in image
    outdir : str
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

    def _make_hdr(starlet_source, cat, source_cat=None):
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

        
        # Add info to header
        model_hdr = fits.Header()
        model_hdr["bbox"] = ",".join(map(str, [bbox_x, bbox_y, bbox_w, bbox_h]))
        model_hdr["area"] = bbox_w * bbox_h

        if source_cat is not None:
            catalog_redshift = source_cat["redshift_truth"]
            oid = source_cat["objectId"]
            imag = source_cat["mag_i"]
            if not np.isfinite(imag):
                imag = -1
            model_hdr["cat_id"] = source_cat['truth_type']  # Category ID
            model_hdr["redshift"] = catalog_redshift
            model_hdr["objid"] = oid
            model_hdr["mag_i"] = imag
            
        else:
            # Ellipse parameters (a, b, theta) from deblend catalog
            e_a, e_b, e_theta = cat["a"], cat["b"], cat["theta"]
            ell_parm = np.concatenate((cat["a"], cat["b"], cat["theta"]))
            model_hdr["ell_parm"] = ",".join(map(str, list(ell_parm)))
            model_hdr["cat_id"] = 1  # Category ID



        return model_hdr

    # Create dict for all saved filenames
    segmask_hdul = []
    model_hdul = []
    filenames = {}

    # Filter loop
    for i, f in enumerate(filters):
        f = f.upper()

        # Primary HDU is full image
        img_hdu = fits.PrimaryHDU(data=datas[i])

        # Create header entry for each scarlet source
        for k, (src, cat) in enumerate(zip(starlet_sources, catalog_deblended)):
            if source_catalog is not None:
                source_cat = source_catalog.iloc[k]
            else:
                source_cat=None
            # Get each model, make into image
            model = starlet_sources[k].get_model(frame=model_frame)
            model = observation.render(model)
            model = src.bbox.extract_from(model)

            model_hdr = _make_hdr(starlet_sources[k], cat, source_cat)

            model_hdu = fits.ImageHDU(data=model[i], header=model_hdr)
            model_primary = fits.PrimaryHDU()

            model_hdul.append(model_hdu)

        # Write final fits file to specified location
        # Save full image and then headers per source w/ descriptive info
        save_img_hdul = fits.HDUList([img_hdu])
        save_model_hdul = fits.HDUList([model_primary, *model_hdul])

        # Save list of filenames in dict for each band
        #filenames["img"] = os.path.join(outdir, f"{s}_images.npy")
        #np.save(filenames["img"],datas)
        filenames[f"img_{f}"] = os.path.join(outdir, f"{f}_{s}_scarlet_img.fits")
        save_img_hdul.writeto(filenames[f"img_{f}"], overwrite=True)
        
        filenames[f"model_{f}"] = os.path.join(outdir, f"{f}_{s}_scarlet_model.fits")
        save_model_hdul.writeto(filenames[f"model_{f}"], overwrite=True)

    # If we have segmentation mask data, save them as a separate fits file
    # Just using the first band for the segmentation mask
    if segmentation_masks is not None:
        for i, f in enumerate(filters[0]):
            # Create header entry for each scarlet source
            for k, (src, cat) in enumerate(zip(starlet_sources, catalog_deblended)):
                if source_catalog is not None:
                    source_cat = source_catalog.iloc[k]
                else:
                    source_cat=None

                segmask_hdr = _make_hdr(starlet_sources[k], cat, source_cat)

                # Save each model source k in the image
                segmask_hdu = fits.ImageHDU(data=segmentation_masks[k], header=segmask_hdr)
                segmask_primary = fits.PrimaryHDU()

                segmask_hdul.append(segmask_hdu)

            save_segmask_hdul = fits.HDUList([segmask_primary, *segmask_hdul])

            # Save list of filenames in dict for each band
            filenames["segmask"] = os.path.join(outdir, f"{s}_scarlet_segmask.fits")
            save_segmask_hdul.writeto(filenames["segmask"], overwrite=True)

    return filenames




