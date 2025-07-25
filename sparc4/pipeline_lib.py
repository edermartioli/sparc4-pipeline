"""
    Created on May 2 2022

    Description: Library of recipes for the SPARC4 pipeline

    @author: Eder Martioli <emartioli@lna.br>

    Laboratório Nacional de Astrofísica - LNA/MCTI
    """

import glob
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from astropop.astrometry import solve_astrometry_xy
from astropop.file_collection import FitsFileGroup
from astropop.image import imarith, imcombine, processing
from astropop.image.register import compute_shift_list, register_framedata_list
from astropop.math.array import trim_array
from astropop.math.physical import QFloat
from astropop.photometry import aperture_photometry, background, starfind
from astropop.photometry.detection import _fwhm_loop
from astropop.polarimetry import (SLSDualBeamPolarimetry, estimate_dxdy,
                                  halfwave_model, match_pairs,
                                  quarterwave_model)

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from astropy.modeling import fitting, models
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from uncertainties import ufloat, umath

from astroquery.simbad import Simbad

import sparc4.db as s4db
import sparc4.params as params
import sparc4.product_plots as s4plt
import sparc4.products as s4p
import sparc4.utils as s4utils
from sparc4.utils import timeout

import photutils
from photutils.psf import fit_fwhm

from astropy.stats import SigmaClip

#import astroalign as aa
from aafitrans import find_transform
import twirl
from astropy.wcs.utils import fit_wcs_from_points
from astropop.catalogs import gaia
from astroquery.vizier import Vizier
from astroquery.gaia import Gaia
from scipy import stats
from scipy import optimize

from astropy.coordinates import EarthLocation
from astroquery.jplhorizons import Horizons

import yaml

logger = s4utils.start_logger()

def build_target_list_from_data(object_list=[], skycoords_list=[], search_radius_arcsec=10, update_ids=False, target_list="", output="") :

    """ Pipeline module to build a target list based on a list of objects and coordinates obtained from the data
    Parameters
    ----------
    object_list : list, optional
        list of object names
    skycoords_list : list of tuples, optional
        list of (RA,DEC) sky coordinates in the format RA="HH:MM:SS.SS", DEC="+/-DD:MM:SS.SS"
    search_radius_arcsec: float
        search radius (in units of arcseconds) for matching SIMBAD sources around the input list of skycoords
    update_ids: bool, optional
        whether or not to update object IDs with the MAIN ID a matching SIMBAD source
    target_list: str, optional
        input target list file name
    output: str, optional
        otuput csv file name for target list
    Returns
        tbl : astropy.table.Table
        table of targets
    -------
    """
    
    # initialize output table of targets (OBJECT_ID, RA, DEC)
    tbl = Table()
    
    # initialize lists to feed output table
    ids, ras, decs = [], [], []
    
    if target_list != "" :
        # read a table of input targets
        intbl = ascii.read(target_list)
        logger.info("Reading input target list {} with {} objects".format(target_list,len(intbl)))

        for i in range(len(intbl)) :
            ids.append(intbl["OBJECT_ID"][i])
            ras.append(intbl["RA"][i])
            decs.append(intbl["DEC"][i])
    
    # loop over all observed objects given as a list
    for obj_id in object_list :
        try :
            logger.info("Querying SIMBAD database to match object {}".format(obj_id))
            
            # query SIMBAD repository to match object by name
            obj_match_simbad = Simbad.query_object(obj_id)
        
            if obj_match_simbad is None :
                continue
                
            # append
            if update_ids :
                ids.append(obj_match_simbad["MAIN_ID"][0])
            else :
                ids.append(obj_id)

            # cast coordinates into SkyCoord
            coord = SkyCoord(obj_match_simbad["RA"][0], obj_match_simbad["DEC"][0], unit=(u.hourangle, u.deg), frame='icrs')
            
            # append coordinates into arrays
            ras.append(coord.ra.deg)
            decs.append(coord.dec.deg)

        except Exception as e:
            logger.warn("Failed to retrieve SIMBAD info for object {} : {}".format(obj_id, e))
            continue
    
    # loop over all observed coordinates given as a list
    for coords in skycoords_list :
        try :
            logger.info("Querying SIMBAD database to match an object at RA={} DEC={}".format(coords[0], coords[1]))

            # cast input coordinates into SkyCoord
            coord = SkyCoord(coords[0], coords[1], unit=(u.hourangle, u.deg), frame='icrs')
        
            # query SIMBAD repository to match an object by coordinates
            results = Simbad.query_region(coord, radius = search_radius_arcsec * (1./3600.) * u.deg)
    
            if results is None :
                continue
                
            for i in range(len(results)) :
                id = results[i]["MAIN_ID"]
                res_coord = SkyCoord(results[i]["RA"], results[i]["DEC"], unit=(u.hourangle, u.deg), frame='icrs')

                if (id in ids) or (res_coord.ra.deg in ras and res_coord.dec.deg in decs):
                    continue
                else :
                    # append object id into array
                    ids.append(id)
                    # append coordinates into arrays
                    ras.append(res_coord.ra.deg)
                    decs.append(res_coord.dec.deg)
        except  Exception as e:
            logger.warn("Failed to retrieve SIMBAD info for coordinates RA={} DEC={}: {}".format(coords[0], coords[1], e))
            continue
    
    if len(ids) and len(ras) and len(decs) :
        tbl["OBJECT_ID"] = ids
        tbl["RA"], tbl["DEC"] = ras, decs
    else :
        tbl["OBJECT_ID"] = ["Vega"]
        tbl["RA"] = [279.23473458]
        tbl["DEC"] = [38.78368889]

    if output != "" :
        tbl.write(output, overwrite=True)
        
    return tbl
    
   
def astrometry_from_existing_wcs(wcs, img_data, pixel_coords=None, sky_coords=None, pixel_scale=0.335, fov_search_factor=2.0, max_number_of_catalog_sources=300, compute_wcs_tolerance=10, nsources_to_plot=30, twirl_find_peak_threshold=2.0, sip_degree=None, use_vizier=False, vizier_catalogs=["UCAC"], vizier_catalog_idx=2, plot_solution=False, ra_offset=0., dec_offset=0., plot_filename="") :

    """ Pipeline module to calcualte astrometric solution from an existing wcs
    Parameters
    ----------
    wcs : astropy.wcs.WCS
        wcs object
    img_data : numpy.ndarray (n x m)
        float array containing the image data (electrons)
    pixel_coords : np.array (optional)
        array of pixel coordinates np.array([x0,y0],[x1,y1],...,[xn,yn])
    sky_coords : np.array (optional)
        array of sky coordinates np.array([ra0,dec0],[ra1,dec1],...,[ran,decn])
    pixel_scale : float
        pixel scale in units of arc seconds
    ra_key : str
        header keyword to get Right Ascension
    dec_key : str
        header keyword to get Declination
    fov_search_factor : float
        size of FoV to search Gaia sources in units of the image FoV
    max_number_of_catalog_sources : int
        limit to truncate number of Gaia sources to be matched for astrometry
    compute_wcs_tolerance : float
        tolerance passed as a parameter to the function twirl.compute_wcs()
    plot_solution : bool
        to plot image and Gaia sources using new astrometric solution
    plot_filename : str
        output plot file name
    nsources_to_plot : int
        number of Gaia sources to plot
    sip_degree : int
        sip_degree to model geometric distortion
    ra_offset : float
        apply offset in right ascension (arcmin)
    dec_offset : float
        apply offset in declination (arcmin)

    Returns
        wcs : astropy.wcs.WCS
        Updated wcs object
    -------
    """

    loc = {}
    loc['ASTROMETRY_SOURCES_SKYCOORDS'] = None

    # detect stars in the image if a list of pixel coordinates is not provided
    if pixel_coords is None :
        logger.info("Detecting peaks using twirl")
        pixel_coords = twirl.find_peaks(img_data, threshold=twirl_find_peak_threshold)
        
    try :
        if sky_coords is None :
            
            logger.info("No sky coordinates given, searching sources in online catalogs")
        
            # get the center of the image
            ra, dec = wcs.wcs.crval[0]+ra_offset/60., wcs.wcs.crval[1]+dec_offset/60.
            wcs.wcs.crval = [ra, dec]
            #wcs_hdr = wcs.to_header(relax=True)
            #wcs = WCS(wcs_hdr, naxis=2)
            
            # set image center coordinates
            center = SkyCoord(ra, dec, unit=(u.deg,u.deg), frame='icrs')
            #print("Center: ", center)

            # get image shape
            shape = img_data.shape
    
            # set pixel scale
            pixel = pixel_scale * u.arcsec  # known pixel scale

            # set field of view
            fov = np.max(shape) * pixel.to(u.deg)
    
            # get RAs and Decs from online catalogs for a sky area of fov_search_factor x FoV
            if use_vizier :
                logger.info("Querying Vizier")
                
                result = Vizier.query_region(center, width=[fov_search_factor * fov,fov_search_factor * fov], catalog=vizier_catalogs)
                result[vizier_catalog_idx].sort(keys=['Vmag'])
                vizier_tbl = result[vizier_catalog_idx]
                
                sources_skycoords = []
                for i in range(len(vizier_tbl)) :
                    ra_i, dec_i = vizier_tbl['RAJ2000'][i], vizier_tbl['DEJ2000'][i]
                    sources_skycoords.append([ra_i,dec_i])
                sources_skycoords = np.array(sources_skycoords,dtype=float)
                
            else :
                try :
                    # set field of view
                    fov_radius_arcmin = fov.value * 60 / 2
                    logger.info("Querying Gaia DR3")
                    # query Gaia DR3 sources in the field
                    gaia_tbl = s4utils.gaiadr3_query(center.ra.deg, center.dec.deg, radius=fov_search_factor * fov_radius_arcmin, max_nsrcs=max_number_of_catalog_sources)
                    # grab ra and dec sky coords of returned gaia dr3 sources
                    sources_skycoords = np.array([gaia_tbl["ra"].value.data, gaia_tbl["dec"].value.data]).T
  
                except Exception as e :
                    logger.warn("Could not acesss Gaia, using Vizier -> {}:  {}".format(vizier_catalogs,e))
                    result = Vizier.query_region(center, width=[fov_search_factor * fov,fov_search_factor * fov], catalog=vizier_catalogs)
                    sources_skycoords = []
                    for i in range(len(result[vizier_catalog_idx])) :
                        ra_i, dec_i = result[vizier_catalog_idx]['RAJ2000'][i], result[vizier_catalog_idx]['DEJ2000'][i]
                        sources_skycoords.append([ra_i,dec_i])
                    sources_skycoords = np.array(sources_skycoords,dtype=float)
            
            # use input wcs to generate a "guess" for the set of pixel coordinates of Gaia sources
            sources_radecs_guess = np.array(wcs.world_to_pixel_values(sources_skycoords))

            #plt.imshow(img_data, vmin=np.median(img_data), vmax=3 * np.median(img_data), cmap="Greys_r")
            #_ = photutils.aperture.CircularAperture(pixel_coords, r=10.0).plot(color="y")
            #_ = photutils.aperture.CircularAperture(sources_radecs_guess[:max_number_of_catalog_sources], r=15.0).plot(color="r")
            #plt.show()
                                
            logger.info("Matching sources with catalog and solving astrometry to compute WCS")
                                
            # instead of astroalign, we use below the same function from aafitrans, which optimizes the use of astroalign
            T, (source_pos_array, target_pos_array) = find_transform(pixel_coords,
                                                                     sources_radecs_guess[:max_number_of_catalog_sources],
                                                                     max_control_points=max_number_of_catalog_sources,
                                                                     ttype='similarity',
                                                                     pixel_tolerance=2,
                                                                     min_matches=4,
                                                                     num_nearest_neighbors=8,
                                                                     kdtree_search_radius=0.02,
                                                                     n_samples=1,
                                                                     get_best_fit=True,
                                                                     seed=None)

            # Recover sky coordinates for the set of matched Gaia sources using original wcs
            matched_sky_coords = np.array(wcs.pixel_to_world_values(target_pos_array))

            # cast list of coordinates into SkyCoords
            all_sky_coords = SkyCoord(matched_sky_coords, unit='deg')
            
            # generate wcs from fit to array of sources
            wcs = fit_wcs_from_points(np.array([source_pos_array[:,0], source_pos_array[:,1]]), all_sky_coords, proj_point='center', projection='TAN', sip_degree=sip_degree)
            #print(repr(wcs.to_header()))
                
            loc['ASTROMETRY_SOURCES_SKYCOORDS'] = sources_skycoords[:max_number_of_catalog_sources]
        else :
            all_sky_coords = SkyCoord(sky_coords, unit='deg')
            wcs = fit_wcs_from_points(np.array([pixel_coords[:,0], pixel_coords[:,1]]), all_sky_coords, proj_point='center', projection='TAN', sip_degree=sip_degree)
            loc['ASTROMETRY_SOURCES_SKYCOORDS'] = sky_coords

    except Exception as e :
        loc['ASTROMETRY_SOURCES_SKYCOORDS'] = np.array(wcs.pixel_to_world_values(pixel_coords))
        logger.warn("Could not solve astrometry : {}".format(e))

    # plot Gaia sources using new wcs to visually check if solution is correct
    astrometry_sources_pixcoords = np.array(wcs.world_to_pixel_values(loc['ASTROMETRY_SOURCES_SKYCOORDS'][:nsources_to_plot]))

    if plot_solution :
        fig = plt.figure(figsize=(10,10))
        
        plt.imshow(img_data, vmin=np.median(img_data), vmax=3 * np.median(img_data), cmap="Greys_r")
        _ = photutils.aperture.CircularAperture(astrometry_sources_pixcoords, r=10.0).plot(color="y")
        
        if plot_filename != '' :
            fig.savefig(plot_filename, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()
        
    return wcs
    
@timeout(10)
def aperture_photometry_wrapper(img_data, x, y, err_data=None, aperture_radius=10, r_in=25, r_out=50, sigma_clip_threshold=3.0, read_noise=0.0, recenter=False, ap_phot=None, fwhm_from_fit=False, use_moffat=False, window_size=25, update_xycenter_from_profile_fit=False, update_xycenter_threshold=5, global_fit=True, calculate_fwhm=True) :

    """ Pipeline module to run aperture photometry of sources in an image
    Parameters
    ----------
    img_data : numpy.ndarray (n x m)
        float array containing the image data (electrons)
    x : list of floats
        list of x pixel coordinates of sources
    y : list of floats
        list of y pixel coordinates of sources
    err_data : numpy.ndarray (n x m)
        float array containing the error data (electrons)
    aperture_radius : float
        aperture radius within which to perform aperture photometry (pix)
    r_in : float
        sky annulus inner radius (pix)
    r_out : float
        sky annulus outer radius (pix)
    sigma_clip_threshold : float
        sigma clip threshold for sky measurements (in units of sigma)
    recenter : bool, optional
        to recenter sources by the centroid
    readnoise : float, optional
        set readout noise (electrons) for flux error calculation
    ap_phot : Table(), optional
        shared variable for multiprocessing
    fwhm_from_fit : bool
        calcualte fwhm from psf fit
    use_moffat : bool
        use Moffat function. If False, it adopts a Gaussian profile
    window_size : int
        square window size (pixels) for profile fit
    update_xycenter_from_profile_fit : bool
        to update x- and y-center positions using fitted values from 2D Gaussian fits
    update_xycenter_threshold : float
        threshold in units of pixels to limit update in xy center coordinates from fit
    global_fit : bool
        to use a single fit to the combined profile from all sources
    calculate_fwhm : bool, True
        to calculate fwhm
        
    Returns
        ap_phot : Table()
            ap_phot['original_x']: numpy.ndarray ()
                original x positions of sources (pix)
            ap_phot['original_y']: numpy.ndarray ()
                original y positions of sources (pix)
            ap_phot['x']: numpy.ndarray ()
                x positions of sources (pix)
            ap_phot['y']: numpy.ndarray ()
                y positions of sources (pix)
            ap_phot['aperture'] : numpy.ndarray ()
                aperture radius of sources (pix)
            ap_phot['flux']: numpy.ndarray ()
                sum flux of sources (electrons)
            ap_phot['flux_error'] : numpy.ndarray ()
                sum flux error of sources (electrons)
            ap_phot['aperture_area'] : numpy.ndarray ()
                aperture area of sources (pix*pix)
            ap_phot['bkg'] : numpy.ndarray ()
                total background flux of sources (electrons)
            ap_phot['bkg_stddev'] : numpy.ndarray ()
                total background flux error of sources (electrons)
            ap_phot['bkg_area'] : numpy.ndarray ()
                background aperture area of sources (pix*pix)
            ap_phot['flags'] : numpy.ndarray ()
                photometry flags
            ap_phot['fwhm'] : numpy.ndarray ()
                full width at half maximum of sources (pix)
    -------
     :
    """
    
    # when err_data array is not provided, then calculate it
    if err_data is None :
        err_data = np.sqrt(img_data + read_noise*read_noise)

    # set array of tuples with x,y positions of sources
    positions = [(x[i],y[i]) for i in range(len(x))]
    # set circular apertures for photometry
    apertures = photutils.aperture.CircularAperture(positions, r=aperture_radius)
    # calculate photometric quantities for all sources
    aper_stats = photutils.aperture.ApertureStats(img_data, apertures, sum_method='exact', error=err_data)
    
    # create a copy of x,y vectors
    new_x, new_y = deepcopy(x), deepcopy(y)
    
    # if recenter: run photometry again
    if recenter :
        # reset new x,y positions to x,y centroids
        new_x, new_y = aper_stats.xcentroid, aper_stats.ycentroid
        # reset array of tuples with x,y positions of sources
        positions = [(x[i],y[i]) for i in range(len(x))]
        # reset circular apertures for photometry
        apertures = photutils.aperture.CircularAperture(positions, r=aperture_radius)
        # recalculate photometric quantities for all sources
        aper_stats = photutils.aperture.ApertureStats(img_data, apertures, sum_method='exact', error=err_data)

    # set annulus aperture for background measurements
    annulus_apertures = photutils.aperture.CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    # set sigma clip for background measurements
    sigclip = SigmaClip(sigma=sigma_clip_threshold, maxiters=10)
    # calculate background photometric quantities for all sources
    bkg_stats = photutils.aperture.ApertureStats(img_data, annulus_apertures, sigma_clip=sigclip, error=err_data)
    
    # estimate total background flux
    total_bkg = bkg_stats.median * aper_stats.sum_aper_area.value
    # estimate total background flux error
    total_bkg_err = bkg_stats.std * np.sqrt(aper_stats.sum_aper_area.value)

    # calculate background subtracted flux
    apersum_bkgsub = aper_stats.sum - total_bkg
    # propagate uncertainties from background subtraction
    apersum_bkgsub_err = np.sqrt(aper_stats.sum_err**2 + total_bkg_err**2)

    ap_phot = Table()
    ap_phot['original_x'], ap_phot['original_y'] = x, y
    ap_phot['x'], ap_phot['y'] = new_x, new_y
    ap_phot['aperture'] = np.full_like(x,aperture_radius)
    ap_phot['flux'] = apersum_bkgsub
    ap_phot['flux_error'] = apersum_bkgsub_err
    ap_phot['aperture_area'] = aper_stats.sum_aper_area.value
    ap_phot['bkg'] = total_bkg
    ap_phot['bkg_stddev'] = total_bkg_err
    ap_phot['bkg_area'] = bkg_stats.sum_aper_area.value
    ap_phot['flags'] = np.full_like(x,0)

    fwhms = np.full_like(x, np.nan)
    xypos = []
    for i in range(len(x)) :
        xypos.append((x[i],y[i]))

    # create mask to identify NaNs
    mask = np.isfinite(aper_stats.fwhm.value)

    if calculate_fwhm :
        # Measure fwhms from data
        fwhms = measure_fwhm(img_data, xypos, window_size=window_size, plot=False, verbose=False)
        # create mask to identify NaNs
        mask = np.isfinite(fwhms)
    
    # Replace NaNs by default values
    #   the fwhm values below are not correct, but the factor 2 helps to bring them closer to the true fwhm.
    fwhms[~mask] = aper_stats.fwhm.value[~mask] / 2.
    
    # if performing a fit
    if fwhm_from_fit and calculate_fwhm :
        try :
            fwhms_x, fwhms_y, xc, yc = measure_fwhm_from_2DGaussianFit(img_data, xypos, err_data=err_data, window_size=window_size, global_fit=global_fit, plot=False, verbose=False)
            ap_phot['fwhm_x'] = fwhms_x
            ap_phot['fwhm_y'] = fwhms_y
            newmask = (np.isfinite(fwhms_x)) & (np.isfinite(fwhms_y))
            fwhms[newmask] = (fwhms_x[newmask] + fwhms_y[newmask]) / 2.
            
            if update_xycenter_from_profile_fit :
                keep = (np.abs(xc-ap_phot['x']) < update_xycenter_threshold) & (np.abs(yc-ap_phot['y']) < update_xycenter_threshold)
                ap_phot['x'][keep] = xc[keep]
                ap_phot['y'][keep] = yc[keep]
                logger.info("Updated x,y coordinates for {} of {} sources detected".format(len(xc[keep]),len(xc)))
            
        except Exception as e :
            logger.warn("Could not measure FWHM from 2D Gaussian fit; using values from standard method. {}".format(e))
        
    ap_phot['fwhm'] = fwhms

    return ap_phot


def measure_fwhm(img_data, xypos, window_size=24, plot=False, verbose=False) :

    """ Pipeline module to measure fwhm quickly, without fitting the data
    Parameters
    ----------
    img_data : numpy.ndarray (n x m)
        float array containing the image data (electrons)
    xypos : list of tuples (x,y)
        list of x,y coordinates
    window_size : int
        size of window to fit data
    plot : bool
        plot img data
    verbose : bool
        print fit data
    Returns
        fwhms: np.array(dtype=float)
            Full Width at Half Maximum (FWHM)
    """

    fwhms = np.array([])
    ny, nx = np.shape(img_data)

    for i in range(len(xypos)) :
        try :
            xc, yc = xypos[i][0], xypos[i][1]
        
            x1, y1 =int(np.floor((xc+0.5) - (window_size-1)/2)), int(np.floor((yc+0.5) - (window_size-1)/2))
            x2, y2 = x1 + window_size - 1, y1 + window_size - 1
       
            if x1 < 0 : x1, x2 = 0, window_size - 1
            if x2 > nx - 1: x1, x2 = nx - window_size, nx - 1
            if y1 < 0 : y1, y2 = 0, window_size - 1
            if y2 > ny - 1: y1, y2 = ny - window_size, ny - 1

            box_data = deepcopy(img_data[y1:y2,x1:x2])
            bkg_value = np.nanmedian(box_data)
            box_data -= bkg_value
            box_data[box_data < 0] = 0

            flux_max = np.nanmax(box_data)
            sum_flux = np.nansum(box_data)
            
            fwhm_value = 2.355 * np.sqrt( sum_flux / (2.*np.pi*flux_max) )
        
            if plot :
                plt.imshow(box_data)
                plt.show()
        
            fwhms = np.append(fwhms,fwhm_value)
            
            if verbose :
                print("Source index={} sum_flux={:.2f} flux_max={:.2f} window_size={}x{} pixels FHWM = {:.2f} pixels -> {:.2f} arcsec".format(i,sum_flux,flux_max,window_size,window_size,fwhm_value,fwhm_value*0.335))
                
        except Exception as e :
        
            fwhms = np.append(fwhms,np.nan)
            logger.warn("Could not measure FWHM for source index={}: {}".format(i,e))
            
    return fwhms



def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """ 2D Gaussian model
    Parameters
    ----------
    xy : (numpy.ndarray,numpy.ndarray)
        arrays of x and y coordinates in image
    amplitude : float
        amplitude of Gaussian
    amplitude : float
        amplitude of Gaussian
    xo : float
        x-center of Gaussian
    yo : float
        y-center of Gaussian
    sigma_x : float
        sigma in x-direction
    sigma_y : float
        sigma in y-direction
    offset : float
        baseline offset
    Returns
        residuals: numpy.ndarray
            flattened array of residuals given by the image array - 2D Gaussian model
    """
    theta = 0 # could become an input parameter
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g


def errfunc_2DGaussian(params, xy, img, errors=None) :
    """ Pipeline module to get error function for a 2D Gaussian fit
    Parameters
    ----------
    params : numpy.ndarray
        2D Gaussian fit parameters: (amp, xcenter, ycenter, sigmaX, sigmaY, offset)
    xy : (numpy.ndarray,numpy.ndarray)
        array of x and y source positions in the original image array
    img : numpy.ndarray(float)
        image array
    Returns
        residuals: numpy.ndarray
            flattened array of residuals given by the image array - 2D Gaussian model
    """
    if errors is not None :
        residuals = (img - twoD_Gaussian(xy, *params)) / errors
    else :
        residuals = img - twoD_Gaussian(xy, *params)

    
    return residuals.ravel()

def fit_2DGaussian(img, errors=None, plot=False):
    """ Pipeline module to get fwhm_x, fwhm_y, and xc, yc through a 2D Gaussian fit
    Parameters
    ----------
    img : numpy.ndarray (n x m)
        float array containing the image data (electrons)
    errors : numpy.ndarray (n x m)
        float array containing the errors data (electrons)
    plot : bool
        plot data and fit
    Returns
        popt, FWHM_x, FWHM_y, xcenter, ycenter: np.array(dtype=float), float, float, float, float
            popt : 2D Gaussian fit parameters
            FWHM_x: Full Width at Half Maximum (FWHM) fit value in x-direction
            FWHM_y: Full Width at Half Maximum (FWHM) fit value in y-direction
            xcenter: x-center fit value
            ycenter: y-center fit value
    """
    ny, nx = img.shape

    x = np.linspace(0.5, nx-0.5, nx)
    y = np.linspace(0.5, ny-0.5, ny)
    xy = np.meshgrid(x, y)
    
    win_cen_x = (x[-1]+x[0]+1)/2
    win_cen_y = (x[-1]+x[0]+1)/2

    #Parameters: amp, xpos, ypos, sigmaX, sigmaY, baseline
    initial_guess = [np.nanmax(img), win_cen_x, win_cen_y, 5., 5., 0.0001]

    # fit 2D Gaussian to the data
    popt, success = optimize.leastsq(errfunc_2DGaussian, initial_guess, args=(xy,img,errors))
        
    # get fitted parameters
    amp, xcenter, ycenter, sigmaX, sigmaY, offset = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]
    
    if plot :
        data_fitted = twoD_Gaussian(xy, *popt)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img, cmap=plt.cm.jet, origin='lower',extent=(0, nx, 0, ny))
        ax.contour(x, y, data_fitted, 8, colors='w')
        plt.plot(np.array([win_cen_x]),np.array([win_cen_y]),"y+",lw=2)
        plt.show()
    
    FWHM_x = np.abs(4*sigmaX*np.sqrt(-0.5*np.log(0.5)))
    FWHM_y = np.abs(4*sigmaY*np.sqrt(-0.5*np.log(0.5)))
    
    return popt, FWHM_x, FWHM_y, xcenter-win_cen_x, ycenter-win_cen_y


def measure_fwhm_from_2DGaussianFit(img_data, xypos, err_data=None, sigma_ini=3, window_size=24, global_fit=True, plot=False, verbose=False) :

    """ Pipeline module to measure fwhm through a Gaussian fit
    Parameters
    ----------
    img_data : numpy.ndarray (n x m)
        float array containing the image data (electrons)
    xypos : list of tuples (x,y)
        list of x,y coordinates in units of pixels
    err_data : numpy.ndarray (n x m)
        float array containing the error data (electrons)
    sigma_ini : float
        initial value of sigma in units of pixels
    window_size : int
        size of window to fit data in units of pixels
    global_fit : bool
        run global fit instead of individual fit
    plot : bool
        plot fit data
    verbose : bool
        print fit data
    Returns
        fwhmx, fwhmy, out_xc, out_yc: np.array(dtype=float)
            fwhmx, fwhmy: Full Width at Half Maximum (FWHM) fit values
            out_xc, out_yc: fit values of center of coordinates in pixel values
    """

    fwhmx, fwhmy = np.array([]), np.array([])
    out_xc, out_yc = np.array([]), np.array([])

    ny, nx = np.shape(img_data)

    if global_fit :
        cube = np.zeros((len(xypos), window_size, window_size))

    for i in range(len(xypos)) :
    
        xc, yc = xypos[i][0], xypos[i][1]
        
        x1, y1 =int(np.floor((xc+0.5) - (window_size-1)/2)), int(np.floor((yc+0.5) - (window_size-1)/2))
        x2, y2 = x1 + window_size - 1, y1 + window_size - 1
       
        if x1 < 0 : x1, x2 = 0, window_size - 1
        if x2 > nx - 1: x1, x2 = nx - window_size, nx - 1
        if y1 < 0 : y1, y2 = 0, window_size - 1
        if y2 > ny - 1: y1, y2 = ny - window_size, ny - 1
    
        box_data = deepcopy(img_data[y1:y2,x1:x2])
        bkg_value = np.nanmedian(box_data)
        box_data -= bkg_value
        box_data[box_data < 0] = 0

        if global_fit :
            # for global fit, normalize the data
            box_data /= np.nanmax(box_data)
            cube[i,:,:] = box_data
        else :
            try :
                box_errors = deepcopy(err_data[y1:y2,x1:x2])
                popt, fwhm_x, fwhm_y, fitted_xc, fitted_yc = fit_2DGaussian(box_data, errors=box_errors, plot=plot)
    
                fwhmx = np.append(fwhmx,fwhm_x)
                fwhmy = np.append(fwhmy,fwhm_y)
        
                if verbose :
                    print("Source index={} flux_max={} window_size={}x{} pixels FHWM_X = {:.2f} pix -> {:.2f} arcsec FHWM_Y = {:.2f} pix -> {:.2f} arcsec".format(i,np.nanmax(box_data),window_size,window_size,fwhm_x,fwhm_x*0.335,fwhm_y,fwhm_y*0.335))

                out_xc = np.append(out_xc,fitted_xc+xc)
                out_yc = np.append(out_yc,fitted_yc+yc)
            except Exception as e :
                logger.warn("Could not fit 2D Gaussian and measure FWHM for source index={} {}".format(i,e))
                fwhmx = np.append(fwhmx,np.nan)
                fwhmy = np.append(fwhmy,np.nan)
                out_xc = np.append(out_xc,np.nan)
                out_yc = np.append(out_yc,np.nan)


    if global_fit :
        median_slice = np.nanmedian(cube,axis=0)
        bkg_value = np.nanmedian(median_slice)
        median_slice -= bkg_value
        median_slice[median_slice < 0] = 0
        #sig_slice = np.nanmedian(np.abs(cube-median_slice),axis=0) / 0.67449
        try :
            popt, fwhm_x, fwhm_y, fitted_xc, fitted_yc = fit_2DGaussian(median_slice, plot=plot)
            if verbose :
                print("Global FIT: flux_max={} window_size={}x{} pixels FHWM_X = {:.2f} pix -> {:.2f} arcsec FHWM_Y = {:.2f} pix -> {:.2f} arcsec".format(np.nanmax(median_slice),window_size,window_size,fwhm_x,fwhm_x*0.335,fwhm_y,fwhm_y*0.335))
        except Exception as e :
            logger.warn("Could not fit 2D Gaussian and measure FWHM from global fit: {}".format(e))
            popt, fwhm_x, fwhm_y, fitted_xc, fitted_yc = None, np.nan, np.nan, 0., 0.
            
        xcs, ycs = np.array([]), np.array([])
        for i in range(len(xypos)) :
            xcs = np.append(xcs,xypos[i][0])
            ycs = np.append(ycs,xypos[i][1])
        fwhmx = np.full_like(xcs, fwhm_x)
        fwhmy = np.full_like(ycs, fwhm_y)
        out_xc = xcs + fitted_xc
        out_yc = ycs + fitted_yc
        
    return fwhmx, fwhmy, out_xc, out_yc


def init_s4_p(nightdir, datadir="", reducedir="", channels="", print_report=False, param_file=""):
    """ Pipeline module to initialize SPARC4 parameters
    Parameters
    ----------
    nightdir : str
        String to define the night directory name
    datadir : str, optional
        String to define the directory path to the raw data
    reducedir : str, optional
        String to define the directory path to the reduced data
    channels : str, optional
        String to define SPARC4 channels to be reduced, e.g. "1,2"
    print_report : bool, optional
        Boolean to print out report on all existing data for reduction

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # load pipeline parameters
    p = params.load_sparc4_parameters()
    
    # update parameters from user input file
    if param_file != "" :
        p = update_params(p, param_file)

    if datadir != "":
        p['ROOTDATADIR'] = datadir

    if reducedir != "":
        p['ROOTREDUCEDIR'] = reducedir

    # if reduced dir doesn't exist create one
    os.makedirs(p['ROOTREDUCEDIR'], exist_ok=True)

    p['SELECTED_CHANNELS'] = p['CHANNELS']
    if channels != "":
        p['SELECTED_CHANNELS'] = []
        chs = channels.split(",")
        for ch in chs:
            p['SELECTED_CHANNELS'].append(int(ch))

    # organize files to be reduced
    p = s4utils.identify_files(p, nightdir, print_report=print_report)

    p['data_directories'] = []
    p['reduce_directories'] = []
    p['s4db_files'] = []
    p['filelists'] = []
    p['excludedfilelists'] = []

    for j in range(len(p['CHANNELS'])):
        # figure out directory structures
        night_ch_data_dir = '{}/{}/sparc4acs{}/'.format(p['ROOTDATADIR'], nightdir, p['CHANNELS'][j])
        root_reduce_dir = '{}/{}/'.format(p['ROOTREDUCEDIR'], nightdir)
        reduce_dir = '{}/sparc4acs{}/'.format(root_reduce_dir, p['CHANNELS'][j])

        if p['RAW_NIGTHS_INSIDE_CHANNELS_DIR'] :
            night_ch_data_dir = '{}/sparc4acs{}/{}/'.format(p['ROOTDATADIR'], p['CHANNELS'][j], nightdir)
            
        if p['REDUCED_NIGTHS_INSIDE_CHANNELS_DIR'] :
            root_reduce_dir = '{}/sparc4acs{}/'.format(p['ROOTREDUCEDIR'],p['CHANNELS'][j])
            reduce_dir = '{}/{}/'.format(root_reduce_dir,nightdir)

        # produce lists of files for all channels
        channel_data_pattern = '{}/{}'.format(night_ch_data_dir,p["PATTERN_TO_INCLUDE_DATA"])

        # get full list using data wildcard pattern defined in parameters file:
        file_list = sorted(glob.glob(channel_data_pattern))
        exclfile_list = []
        
        # loop over exclude patterns to remove data from list
        for i in range(len(p['PATTERNS_TO_EXCLUDE_DATA'])) :
            exclude_data_pattern = '{}/{}'.format(night_ch_data_dir,p['PATTERNS_TO_EXCLUDE_DATA'][i])
            exclude_list = sorted(glob.glob(exclude_data_pattern))
            file_list = list(set(file_list) - set(exclude_list))
            exclfile_list += exclude_list
            
        p['excludedfilelists'].append(sorted(exclfile_list))
    
        p['filelists'].append(sorted(file_list))

        p['reduce_directories'].append(reduce_dir)
        p['data_directories'].append(night_ch_data_dir)

        # if reduced dir doesn't exist create one
        os.makedirs(root_reduce_dir, exist_ok=True)

        # if reduced dir doesn't exist create one
        os.makedirs(reduce_dir, exist_ok=True)

        db_file = '{}/{}_sparc4acs{}_db.csv'.format(reduce_dir, nightdir, p['CHANNELS'][j])
        
        if p['DB_FILE_FORMAT'] == 'FITS' :
            db_file = '{}/{}_sparc4acs{}_db.fits'.format(reduce_dir, nightdir, p['CHANNELS'][j])
        
        p['s4db_files'].append(db_file)

    if 'MEM_CACHE_FOLDER' not in p.keys() or not os.path.exists(p['MEM_CACHE_FOLDER']) :
        p['MEM_CACHE_FOLDER'] = None

    return p


def update_params(p, param_file) :
    """ Pipeline module to update SPARC4 parameters
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    param_file : str
        parameters file path
    
    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    with open(param_file, 'r') as f:
        params = yaml.safe_load(f)
    for key in params.keys() :
        if key in p.keys() :
            p[key] = params[key]
        else :
            logger.warn("Parameter {} is not a valid parameter, ignoring...".format(key))
            continue
    return p
    
    
def write_night_report(p, night_report_file_path, channel_index=0) :
    """ Pipeline module to write SPARC4 night report for a given channel
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    night_report_file_path : str
        night report file path
    channel_index : int
        channel index
    Returns
    -------
    """
    outfile = open(night_report_file_path,"w+")
    outfile.write(p["NIGHT_REPORT"])
    outfile.write("-------------------------------------------------------------------\n")
    outfile.write("INPUT VALID FILES (wildcard = {}):\n".format(p['PATTERN_TO_INCLUDE_DATA']))
    outfile.write("-------------------------------------------------------------------\n")
    for i in range(len(p['filelists'][channel_index])) :
        outfile.write("Included file {}/{}: {}\n".format(i+1, len(p['filelists'][channel_index]), p['filelists'][channel_index][i]))
    outfile.write("-------------------------------------------------------------------\n")
    outfile.write("EXCLUDED FILES (exclusion wildcard(s) = {}):\n".format(p['PATTERNS_TO_EXCLUDE_DATA']))
    outfile.write("-------------------------------------------------------------------\n")
    for i in range(len(p['excludedfilelists'][channel_index])) :
        outfile.write("Excluded file {}/{}: {}\n".format(i+1, len(p['excludedfilelists'][channel_index]), p['excludedfilelists'][channel_index][i]))
    outfile.write("-------------------------------------------------------------------\n")
    outfile.close()

    
def get_list_of_catalogs(apertures, inst_mode="PHOT", polar_beam="") :
    """ Pipeline module to create a list of catalog labels

    Parameters
    ----------
    apertures : list
        list of integers
    inst_mode : str, optional
        to set observations of a given instrument mode "PHOT" or "POL"
    polar_beam : str, optional
        to set beam "N" or "S"

    Returns
    -------
    list_of_catalogs : list
        list of catalog names
    """
    
    # reset polar inst_mode string to "POL" to match the catalog keys
    if inst_mode == "POLAR" :
        inst_mode = "POL"

    # set polar suffix with the beam key
    polar_suffix = ""
    if polar_beam != "":
        polar_suffix = "{}_".format(polar_beam)
    
    # initialize list of catalog keys
    list_of_catalogs = []

    # loop over all input aperture values
    for aperture_value in apertures:
    
        # construct catalog label
        cat_label = "CATALOG_{}_{}AP{:03d}".format(inst_mode, polar_suffix, aperture_value)

        # append to the output list
        list_of_catalogs.append(cat_label)
        
    return list_of_catalogs
    

def reduce_sci_data(db, p, channel_index, inst_mode, detector_mode, nightdir, reduce_dir, polar_mode=None, fit_zero=False, detector_mode_key="", obj=None, calw_modes=["OFF","None","CLEAR"], match_frames=True, force=False, plot_stack=False, plot_lc=False, plot_polar=False):
    """ Pipeline module to run the reduction of science data

    Parameters
    ----------
    db : astropy.table
        data base table
    p : dict
        dictionary to store pipeline parameters
    channel_index : int
        SPARC4 channel index
    inst_mode : str, optional
        to select observations of a given instrument mode
    detector_mode : dict
        to select observations of a given detector mode
    nightdir : str
        String to define the night directory name
    reduce_dir : str
        path to the reduce directory
    polar_mode : str, optional
        to select observations of a given polarimetric mode
    fit_zero : bool
        to fit zero of waveplate position angle
    aperture_index : int (optional)
        index to select aperture in catalog. Default is None and it will calculate best aperture
    detector_mode_key : str (optional)
        keyword name of detector mode
    obj : str (optional)
        object name to reduce only data for this object
    calw_modes : list of str (optional)
        define list of wheel modes to restrcit data reduction for these modes
    match_frames : bool
        match images using stack as reference
    force : bool
        force reduction even if product already exists
    plot_stack : bool
        turn on plotting stack product
    plot_lc : bool
        turn on plotting light curve product
    plot_polar : bool
        turn on plotting polarimetry product
        
    Returns
    -------
    p : dict
        updated dictionary to store pipeline parameters
    """

    # set suffix and switch for polarimetry mode
    polsuffix = ""
    polarimetry = False
    if inst_mode == p['INSTMODE_POLARIMETRY_KEYVALUE']:
        polarimetry = True
        polsuffix = "_{}_{}".format(inst_mode, polar_mode)

    # get list of objects observed
    objs = s4db.get_targets_observed(db, inst_mode=inst_mode, polar_mode=polar_mode, detector_mode=detector_mode)
    
    if obj is not None and len(objs):
        objs = objs[objs['OBJECT'] == obj]

    # if table of objects observed is empty, print a message and leave
    if len(objs) == 0 :
        if inst_mode == p['INSTMODE_POLARIMETRY_KEYVALUE'] :
            logger.warn("No objects observed in the {} mode, detector mode {} ".format(polsuffix.replace("_"," "), detector_mode_key))
        else :
            logger.warn("No objects observed in the {} mode, detector mode {} ".format(inst_mode, detector_mode_key))
        return p
    
    # cast output table of objects observed into a list
    objs = list(objs["OBJECT"])
    
    # get list of all calibration wheel modes found in observations
    calws_obs = s4db.get_calib_wheel_modes(db, polar_only=False)

    # initialize final list of calws
    calws = []
    # filter modes by input
    for i in range(len(calws_obs)) :
        calw = calws_obs[i][0]
        if calw not in calws :
            if calw in calw_modes :
                calws.append(calw)

    # loop over each object to run the reduction
    for k in range(len(objs)):
        obj = objs[k]
    
        # set solar system flag to False
        p['SOLAR_SYSTEM_OBJECT'] = False
        p['SOLAR_SYSTEM_OBJECT_ID'] = ""
        
        # when sinalized that observations has a Solar System object, it tests every object
        if p['HAS_SOLAR_SYSTEM_BODY'] :
        
            # query the JPL database to determine whether this is a Solar System object
            ssobj = Horizons(id=obj, location=p['JPL_HORIZONS_OBSERVATORY_CODE'], epochs=Time.now().jd)

            if 'No matches found' in ssobj.ephemerides_async().text :
                logger.info("Target {} is not a Solar System object, continue with normal reduction".format(obj))
            else :
                logger.info("Target {} is a Solar System object, continue with especial reduction".format(obj))
                # set Solar System flag to True when object matches a catalogued solar system object
                p['SOLAR_SYSTEM_OBJECT'] = True
                p['SOLAR_SYSTEM_OBJECT_ID'] = obj

        logger.info("Reducing data for object: {}".format(obj))
        
        # loop over calibration wheel modes: OFF, POLARIZER, or DEPOLARIZER
        for j in range(len(calws)) :
            calw = calws[j]
            calwsuffix = ""
            if calw != "OFF" and calw != "None" and calw != "CLEAR":
                calwsuffix = "_{}".format(calw)
                
            logger.info("Reducing data for calibration wheel mode: {}".format(calw))

            # set suffix for output stack filename
            stack_suffix = "{}_s4c{}{}_{}{}{}".format(nightdir, p['CHANNELS'][channel_index], detector_mode_key, obj.replace(" ", "").replace("/",""), polsuffix, calwsuffix)

            # get list of science files matching all selection criteria
            sci_list = s4db.get_file_list(db,
                                          object_id=obj,
                                          inst_mode=inst_mode,
                                          polar_mode=polar_mode,
                                          obstype=p['OBJECT_OBSTYPE_KEYVALUE'],
                                          calwheel_mode=calw,
                                          detector_mode=detector_mode)
                                          
            ################################################
            ## REDUCE, CREATE CATALOGS, and DO PHOTOMETRY ##
            ################################################
            # run stack and reduce individual science images (produce *_proc.fits)
            p = stack_and_reduce_sci_images(p,
                                            sci_list,
                                            reduce_dir,
                                            ref_img="",
                                            stack_suffix=stack_suffix,
                                            force=force,
                                            match_frames=match_frames,
                                            polarimetry=polarimetry,
                                            plot=plot_stack,
                                            plot_proc_frames=p['PLOT_PROC_FRAMES'])

            # set suffix for output time series filename
            ts_suffix = "{}_s4c{}_{}{}".format(nightdir, p['CHANNELS'][channel_index], obj.replace(" ", "").replace("/",""), polsuffix)

            logger.info("Start generating photometric time series products with suffix: {}".format(ts_suffix))
    
            #############################
            ## PHOTOMETRIC TIME SERIES ##
            #############################
            lists_of_catalogs = []
            addkeys = []
            if inst_mode == p['INSTMODE_PHOTOMETRY_KEYVALUE']:
                addkeys = p['PHOT_KEYS_TO_ADD_HEADER_DATA_INTO_TSPRODUCT']
                list_of_catalogs = get_list_of_catalogs(p['PHOT_APERTURES_FOR_LIGHTCURVES'], inst_mode)
                lists_of_catalogs.append(list_of_catalogs)
            
            elif inst_mode == p['INSTMODE_POLARIMETRY_KEYVALUE'] :
                addkeys = p['POLAR_KEYS_TO_ADD_HEADER_DATA_INTO_TSPRODUCT']
                for beam in p["CATALOG_BEAM_IDS"] :
                    list_of_catalogs = get_list_of_catalogs(p['PHOT_APERTURES_FOR_LIGHTCURVES'], inst_mode, polar_beam=beam)
                    lists_of_catalogs.append(list_of_catalogs)

            ts_products = []
            
            logger.info("Starting loop over sets of catalogs to create time series products")
            # loop over sets of catalogs to create time series products
            for kk in range(len(lists_of_catalogs)) :
                # add beam label to suffix
                ts_suffix_tmp = "{}".format(ts_suffix)
                if polarimetry :
                    ts_suffix_tmp = "{}_{}".format(ts_suffix, p["CATALOG_BEAM_IDS"][kk])
                
                logger.info("Running photometric time series")
                # run photometric time series
                phot_ts_product = phot_time_series(p['OBJECT_REDUCED_IMAGES'],
                                               ts_suffix=ts_suffix_tmp,
                                               reduce_dir=reduce_dir,
                                               time_key=p['TIME_KEYWORD_IN_PROC'],
                                               time_format=p['TIME_FORMAT_IN_PROC'],
                                               catalog_names=lists_of_catalogs[kk],
                                               time_span_for_rms=p['TIME_SPAN_FOR_RMS'],
                                               keys_to_add_header_data = addkeys,
                                               force=p['FORCE_TIME_SERIES_EXECUTION'])
                # append ts product to a list
                ts_products.append(phot_ts_product)
            
                if plot_lc and inst_mode == p['INSTMODE_PHOTOMETRY_KEYVALUE'] :
                    # plot light curve
                    
                    plot_coords_file, plot_rawmags_file, plot_lc_file = "", "", ""
                    if p['PLOT_TO_FILE'] :
                        plot_coords_file = phot_ts_product.replace(".fits","_coords{}".format(p['PLOT_FILE_FORMAT']))
                        plot_rawmags_file = phot_ts_product.replace(".fits","_rawmags{}".format(p['PLOT_FILE_FORMAT']))
                        plot_lc_file = phot_ts_product.replace(".fits",p['PLOT_FILE_FORMAT'])
                    try :
                        s4plt.plot_light_curve(phot_ts_product,
                                       target=p['TARGET_INDEX'],
                                       comps=p['COMPARISONS'],
                                       nsig=10,
                                       catalog_name=p['PHOT_REF_CATALOG_NAME'],
                                       plot_coords=True,
                                       plot_rawmags=True,
                                       plot_sum=True,
                                       plot_comps=True,
                                       output_coords=plot_coords_file,
                                       output_rawmags=plot_rawmags_file,
                                       output_lc=plot_lc_file)
                    except Exception as e:
                        logger.warn("Could not generate plot for product {} : {}".format(phot_ts_product, e))
                        
            ##################################################################
            ## POLARIMETRY AND TIME SERIES FOR POLARIMETRIC MODE (L2 OR L4) ##
            ##################################################################
            if inst_mode == p['INSTMODE_POLARIMETRY_KEYVALUE']:
        
                logger.info("Combining light curve data from the two beams")
                
                # combine light curve data from the two beams
                outlc = combine_ts_products_of_polar_beams(ts_products[0], ts_products[1], addkeys)

                # plot light curve
                if plot_lc:
                    plot_coords_file, plot_rawmags_file, plot_lc_file = "", "", ""
                    if p['PLOT_TO_FILE'] :
                        plot_coords_file = outlc.replace(".fits","_coords{}".format(p['PLOT_FILE_FORMAT']))
                        plot_rawmags_file = outlc.replace(".fits","_rawmags{}".format(p['PLOT_FILE_FORMAT']))
                        plot_lc_file = outlc.replace(".fits",p['PLOT_FILE_FORMAT'])
                        
                    try :
                        s4plt.plot_light_curve(outlc,
                                           target=p['TARGET_INDEX'],
                                           comps=p['COMPARISONS'],
                                           nsig=10,
                                           catalog_name=p['PHOT_REF_CATALOG_NAME'],
                                           plot_coords=True,
                                           plot_rawmags=True,
                                           plot_sum=True,
                                           plot_comps=True,
                                           output_coords=plot_coords_file,
                                           output_rawmags=plot_rawmags_file,
                                           output_lc=plot_lc_file)
                    except Exception as e:
                        logger.warn("Could not generate plot for product {} : {}".format(outlc, e))
                        
                compute_k, zero = p['COMPUTE_K_IN_L2'], 0
                wave_plate = 'halfwave'

                if polar_mode == 'L4':
                    wave_plate = 'quarterwave'
                    compute_k = p['COMPUTE_K_IN_L4']
                    zero = p['ZERO_OF_WAVEPLATE'][channel_index]

                logger.info("Selecting polarimetric sequences")

                # divide input list into a many sequences
                pol_sequences = s4utils.select_polar_sequences(p['OBJECT_REDUCED_IMAGES'], sortlist=True, npos_in_seq=p["MAX_NUMBER_OF_WPPOS_IN_SEQUENCE"], rolling_seq=p["ROLLING_POLAR_SEQUENCE"], nimages_per_seq_fixed=p["FIXED_NUMBER_OF_IMAGES_IN_POLAR_SEQUENCE"])

                p['PolarProducts'] = []

                for i in range(len(pol_sequences)):

                    if len(pol_sequences[i]) == 0:
                        continue

                    logger.info("Running {} polarimetry for sequence: {} of {}".format(wave_plate, i+1, len(pol_sequences)))
                       
                    # Compute polarimetry for one sequence
                    polarproduct = compute_polarimetry(pol_sequences[i],
                                                       wave_plate=wave_plate,
                                                       base_aperture=p['APERTURE_RADIUS_FOR_PHOTOMETRY_IN_POLAR'],
                                                       compute_k=compute_k,
                                                       fit_zero=fit_zero,
                                                       zero=zero,
                                                       force=p['FORCE_POLARIMETRY_COMPUTATION'])
                                                       
                    # Plot polarimetry results
                    if plot_polar :
                        # Get polarimetry from polarimetry product
                        pol_results = get_polarimetry_results(polarproduct,
                                                          source_index=p['TARGET_INDEX'],
                                                          min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                          max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                          compute_k=compute_k)
                        polar_plot_file = ""
                        if p['PLOT_TO_FILE'] :
                            polar_plot_file = polarproduct.replace(".fits",p['PLOT_FILE_FORMAT'])
                        if pol_results["POLARIMETRY_SUCCESS"] :
                            try :
                                s4plt.plot_polarimetry_results(pol_results, title_label=pol_results['TITLE_LABEL'], wave_plate=wave_plate, output=polar_plot_file)
                            except Exception as e:
                                logger.warn("Could not generate plot for product {} : {}".format(polarproduct, e))
                                
                    p['PolarProducts'].append(polarproduct)


                # create polar time series product
                p['PolarTimeSeriesProduct'] = polar_time_series(p['PolarProducts'],
                                                                reduce_dir=reduce_dir,
                                                                ts_suffix=ts_suffix,
                                                                aperture_radius=p['APERTURE_RADIUS_FOR_PHOTOMETRY_IN_POLAR'],
                                                                min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                                max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                                force=p['FORCE_TIME_SERIES_EXECUTION'])
                                                                
                # plot light curve
                if plot_lc:
                    polar_ts_plot_file = ""
                    if p['PLOT_TO_FILE'] :
                        polar_ts_plot_file = p['PolarTimeSeriesProduct'].replace(".fits",p['PLOT_FILE_FORMAT'])
                    try :
                        s4plt.plot_polar_time_series(p['PolarTimeSeriesProduct'],
                                                 target=p['TARGET_INDEX'],
                                                 comps=p['COMPARISONS'],
                                                 plot_total_polarization=p["PLOT_TOTAL_POLARIZATION"],
                                                 output=polar_ts_plot_file)
                    except Exception as e:
                        logger.warn("Could not generate plot for product {} : {}".format(p['PolarTimeSeriesProduct'], e))
                    
                logger.info("Running {} polarimetry for {} frames -- static polar".format(wave_plate, len(p['OBJECT_REDUCED_IMAGES'])))
                    
                output_static_polar = "{}/{}_polar.fits".format(reduce_dir, stack_suffix)
                
                # Calculate a static polar, which means a polarimetry product using all images available
                static_polar_product = compute_polarimetry(p['OBJECT_REDUCED_IMAGES'],
                                                           output_filename = output_static_polar,
                                                           wave_plate=wave_plate,
                                                           base_aperture=p['APERTURE_RADIUS_FOR_PHOTOMETRY_IN_POLAR'],
                                                           compute_k=compute_k,
                                                           fit_zero=fit_zero,
                                                           zero=zero,
                                                           force=p['FORCE_POLARIMETRY_COMPUTATION'])
                # plot static polar
                if plot_polar :
                    # get polarimetry from static polar product
                    static_polar_results = get_polarimetry_results(static_polar_product,
                                                                    source_index=p['TARGET_INDEX'],
                                                                    min_aperture=p['MIN_APERTURE_FOR_POLARIMETRY'],
                                                                    max_aperture=p['MAX_APERTURE_FOR_POLARIMETRY'],
                                                                    compute_k=compute_k)
                    static_polar_plot_file = ""
                    if p['PLOT_TO_FILE'] :
                        static_polar_plot_file = static_polar_product.replace(".fits",p['PLOT_FILE_FORMAT'])
                    if static_polar_results["POLARIMETRY_SUCCESS"] :
                        try :
                            s4plt.plot_polarimetry_results(static_polar_results, title_label=static_polar_results['TITLE_LABEL'], wave_plate=wave_plate, output=static_polar_plot_file)
                        except Exception as e:
                            logger.warn("Could not generate plot for product {} : {}".format(static_polar_product, e))
                        
    return p


def run_master_calibration(p, inputlist=[], output="", obstype='bias', data_dir="./", reduce_dir="./", normalize=False, force=False, plot=False):
    """ Pipeline module to run master calibration

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    obstype : str, optional
        String to define the type of observation.
        It accepts the following values: 'bias', 'flat', 'object'.
    data_dir : str, optional
        String to define the directory path to the raw data
    reduce_dir : str, optional
        String to define the directory path to the reduced data
    normalize : bool, optional
        Boolean to decide whether or not to normalize the data
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists
    plot : bool, optional
        plot results
        
    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    obstype_keyvalue = 'UNKNOWN'
    if obstype == 'bias':
        obstype_keyvalue = p['BIAS_OBSTYPE_KEYVALUE']
    elif obstype == 'flat':
        obstype_keyvalue = p['FLAT_OBSTYPE_KEYVALUE']
    elif obstype == 'object':
        obstype_keyvalue = p['OBJECT_OBSTYPE_KEYVALUE']
    else:
        logger.warn("obstype={} not recognized, setting to default = {}".format(obstype, obstype_keyvalue))

    if output == "":
        output = os.path.join(reduce_dir, "master_{}.fits".format(obstype))

    # set master calib keyword in parameters
    p["master_{}".format(obstype)] = output

    # Skip if product already exists and reduction is not forced
    if os.path.exists(output) and not force:
        return p

    # set method to combine images
    method = p['CALIB_IMCOMBINE_METHOD']

    if inputlist == []:
        # select FITS files in the data directory and build database
        main_fg = FitsFileGroup(
            location=data_dir, fits_ext=p['CALIB_WILD_CARDS'], ext=0)
        # print total number of files selected:
        logger.info(f'Total number of files selected : {len(main_fg)}')

        # Filter files by header keywords
        filter_fg = main_fg.filtered({'obstype': obstype_keyvalue})
    else:
        filter_fg = FitsFileGroup(files=inputlist)

    # print total number of bias files selected
    logger.info(f'{obstype} files: {len(filter_fg)}')

    # get frames
    frames = list(filter_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP'], cache_folder=p['MEM_CACHE_FOLDER']))

    # extract gain from the first image
    if float(frames[0].header['GAIN']) != 0:
        gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
    else:
        gain = 3.3*u.electron/u.adu
    logger.info('gain:{}'.format(gain))

    # Perform gain calibration
    for i, frame in enumerate(frames):
        logger.info(f'processing frame {i+1} of {len(frames)}')
        if p["DETECT_AND_REJECT_COSMIC_RAYS"] :
            processing.cosmics_lacosmic(frame, inplace=True)
        processing.gain_correct(frame, gain, inplace=True)

    # combine
    master = imcombine(frames, method=method, use_memmap_backend=p['USE_MEMMAP'], cache_folder=p['MEM_CACHE_FOLDER'])

    # get statistics
    stats = master.statistics()

    norm_mean_value = master.mean()
    logger.info('Normalization mean value:{}'.format(norm_mean_value))

    data_units = 'electron'

    if normalize:
        master = imarith(master, norm_mean_value, '/')
        data_units = 'dimensionless'

    # write information into an info dict
    info = {'INCOMBME': ('{}'.format(method), 'imcombine method'),
            'INCOMBNI': (len(filter_fg), 'imcombine nimages'),
            'BUNIT': ('{}'.format(data_units), 'data units'),
            'DRSINFO': ('astropop', 'data reduction software'),
            'DRSROUT': ('master image', 'data reduction routine'),
            'NORMALIZ': (normalize, 'normalized master'),
            'NORMMEAN': (norm_mean_value.value, 'normalization mean value in {}'.format(norm_mean_value.unit)),
            'MINVAL': (stats['min'].value, 'minimum value in {}'.format(stats['min'].unit)),
            'MAXVAL': (stats['max'].value, 'maximum value in {}'.format(stats['max'].unit)),
            'MEANVAL': (stats['mean'].value, 'mean value in {}'.format(stats['mean'].unit)),
            'MEDIANVA': (stats['median'].value, 'median value in {}'.format(stats['median'].unit)),
            'STDVALUE': (stats['std'].value, 'standard deviation in {}'.format(stats['std'].unit))
            }

    # get data arrays
    img_data = np.array(master.data)
    err_data = np.array(master.get_uncertainty())
    mask_data = np.array(master.mask)

    # call function masteZero from sparc4_products to generate final product
    mastercal = s4p.masterCalibration(sorted(filter_fg.files), img_data=img_data,err_data=err_data, mask_data=mask_data,info=info, filename=output)

    if plot :
        plot_file = ""
        if p["PLOT_TO_FILE"] :
            plot_file = output.replace(".fits",p['PLOT_FILE_FORMAT'])
            
        try :
            if obstype == 'bias':
                s4plt.plot_cal_frame(output, percentile=99.5, combine_rows=True, combine_cols=True, output=plot_file)
            else :
                s4plt.plot_cal_frame(output, percentile=99.5, xcut=512, ycut=512, output=plot_file)
        except Exception as e:
            logger.warn("Could not generate plot for product {} : {}".format(output, e))

    return p

def get_shuffled_short_list(inputlist, max_n_files=300) :

    """ Pipeline module to shuffle and limit number of files in an input list of files

    Parameters
    ----------
    inputlist : list
        list of file paths
    max_n_files : int, optional
        Maximum number of files to limit out list

    Returns
    -------
    outlist : list
        shuffled and limited out list of file paths
    """
    
    # get number of files from input list
    nfiles = len(inputlist)
    
    # prevent from going over the inputlist size
    if len(inputlist) < max_n_files :
        max_n_files = len(inputlist)
    
    # get list indexes
    idx = np.arange(nfiles)
    
    # shuffle indexes
    np.random.shuffle(idx)
    
    # build output list
    outlist = []
    for i in range(max_n_files) :
        outlist.append(inputlist[idx[i]])

    return outlist


def run_master_zero_calibration(p, db, nightdir, data_dir, reduce_dir, channel, detector_mode, detector_mode_key, force=True, plot=False) :

    """ Pipeline module to run master zero calibrations

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    db : astropy.table.Table
        database of observations in the nigth
    nightdir : str
        String to define the night directory
    data_dir : str
        String to define the directory path to the raw data
    reduce_dir : str
        String to define the directory path to the reduced data
    detector_mode : str
        detector mode
    detector_mode_key : str
        Keyword that identifies detector mode
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists
    plot : bool, optional
        plot results

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    if p["APPLY_BIAS_CORRECTION"] is False :
        p["master_bias"] = "None"
        return p

    # create a list of zeros for current detector mode
    zero_list = s4db.get_file_list(db, obstype=p['BIAS_OBSTYPE_KEYVALUE'], detector_mode=detector_mode)
    
    # truncate number of zero images to avoid memory issues
    zero_list = zero_list[:p['MAX_NUMBER_OF_ZERO_FRAMES_TO_USE']]
    
    if len(zero_list) :
        # calculate master bias
        p["master_bias"] = "{}/{}_s4c{}{}_MasterZero.fits".format(reduce_dir, nightdir, channel, detector_mode_key)
        
        # log messages:
        logger.info("Calculating master zero calibration and saving to file: {}".format(p["master_bias"]))
        
        p = run_master_calibration(p, inputlist=zero_list, output=p["master_bias"], obstype='bias', data_dir=data_dir, reduce_dir=reduce_dir, force=force, plot=plot)
    else :
        p["APPLY_BIAS_CORRECTION"] = False
        p["master_bias"] = "None"
        # log messages:
        logger.warn("No ZERO images for detector mode {}. Turning off zero correction.".format(detector_mode_key))
        
    return p


def run_master_flat_calibrations(p, db, nightdir, data_dir, reduce_dir, channel, detector_mode, detector_mode_key, force=True, plot=False) :

    """ Pipeline module to run master flat calibrations

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    db : astropy.table.Table
        database of observations in the nigth
    nightdir : str
        String to define the night directory
    data_dir : str
        String to define the directory path to the raw data
    reduce_dir : str
        String to define the directory path to the reduced data
    detector_mode : str
        detector mode
    detector_mode_key : str
        Keyword that identifies detector mode
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists
    plot : bool, optional
        plot results
        
    Returns
    -------
    p_phot, p_polarl2, p_polarl4 : dict, dict, dict
        dictionaries to store pipeline parameters for the three modes of operation
    """

    # create a deep copy of each parameter container
    p_phot = deepcopy(p)
    p_polarl2 = deepcopy(p)
    p_polarl4 = deepcopy(p)

    if p["APPLY_FLATFIELD_CORRECTION"] is False :
        p_phot["master_flat"] = "None"
        p_polarl2["master_flat"] = "None"
        p_polarl4["master_flat"] = "None"
        for wppos in range(1,17) :
            p_polarl2["wppos{:02d}_master_flat".format(wppos)] = "None"
            p_polarl4["wppos{:02d}_master_flat".format(wppos)] = "None"
        return p_phot, p_polarl2, p_polarl4

    # container to store flats
    flats = {}
    
    # create a list of sky flats
    #skyflat_list = s4db.get_file_list(db, detector_mode=detector_mode, skyflat=True)
    #if len(skyflat_list):
    #    # calculate master sky flat
    #    flats["skyflat"] = "{}/{}_s4c{}{}_MasterSkyFlat.fits".format(reduce_dir, nightdir, channel, detector_mode_key)
    #    p = run_master_calibration(p, inputlist=skyflat_list, output=flats["skyflat"], obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=force, plot=plot)

    # create a list of dome flats for current detector mode and for photometry mode
    phot_flat_list = s4db.get_file_list(db, inst_mode=p['INSTMODE_PHOTOMETRY_KEYVALUE'], obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_mode)
    # create a list of dome flats for current detector mode and for polarimetry L2 mode
    polar_l2_flat_list = s4db.get_file_list(db, inst_mode=p['INSTMODE_POLARIMETRY_KEYVALUE'], polar_mode=p['POLARIMETRY_L2_KEYVALUE'], obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_mode)
    # create a list of dome flats for current detector mode and for polarimetry L4 mode
    polar_l4_flat_list = s4db.get_file_list(db, inst_mode=p['INSTMODE_POLARIMETRY_KEYVALUE'], polar_mode=p['POLARIMETRY_L4_KEYVALUE'], obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_mode)

    if len(phot_flat_list):
        # calculate master dome flat for photometry mode
        flats["phot_master_flat"] = "{}/{}_s4c{}{}_MasterDomeFlat.fits".format(reduce_dir, nightdir, channel, detector_mode_key)

        # log messages:
        logger.info("Calculating master flat for PHOT mode and saving to file: {}".format(flats["phot_master_flat"]))

        p_phot = run_master_calibration(p_phot, inputlist=get_shuffled_short_list(phot_flat_list,max_n_files=p["MAX_NUMBER_OF_FLAT_FRAMES_TO_USE"]), output=flats["phot_master_flat"], obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=force, plot=plot)
        
        p_phot["master_flat"] = flats["phot_master_flat"]
    else :
        p_phot["APPLY_FLATFIELD_CORRECTION"] = False
        p_phot["master_flat"] = "None"
        
        # log messages:
        logger.warn("No flats for PHOT mode in detector mode {}. Turning off flat correction in PHOT mode.".format(detector_mode_key))
            
    if len(polar_l2_flat_list):
        # calculate master dome flat
        flats["polar_l2_master_flat"] = "{}/{}_s4c{}{}_POLAR_L2_MasterDomeFlat.fits".format(reduce_dir, nightdir, channel, detector_mode_key)
        # log messages:
        logger.info("Calculating master flat for POLAR L2 mode and saving to file: {}".format(flats["polar_l2_master_flat"]))

        p_polarl2 = run_master_calibration(p_polarl2, inputlist=get_shuffled_short_list(polar_l2_flat_list,max_n_files=p["MAX_NUMBER_OF_FLAT_FRAMES_TO_USE"]), output=flats["polar_l2_master_flat"], obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=force, plot=plot)
        
        master_flat_file = "None"
        for wppos in range(1,17) :
            # create a master flat for each waveplate position
            polar_l2_wppos_flat_list = s4db.get_file_list(db, inst_mode=p['INSTMODE_POLARIMETRY_KEYVALUE'], polar_mode=p['POLARIMETRY_L2_KEYVALUE'], obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_mode, wppos=str(wppos))
            
            if len(polar_l2_wppos_flat_list) :
                master_flat_file = "{}/{}_s4c{}{}_POLAR_L2_WPPOS{:02d}_MasterDomeFlat.fits".format(reduce_dir, nightdir, channel, detector_mode_key, wppos)
                # log messages:
                logger.info("Calculating master flat for POLAR L2 mode WPPOS={} and saving to file: {}".format(wppos,master_flat_file))
                _ = run_master_calibration(p_polarl2, inputlist=get_shuffled_short_list(polar_l2_wppos_flat_list,max_n_files=p["MAX_NUMBER_OF_FLAT_FRAMES_TO_USE"]), output=master_flat_file, obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=force, plot=plot)
            else :
                logger.info("No flats found in database for the following configuration: inst_mode={}, polar_mode={}, obstype={}, detector_mode={}, wppos={}. ".format(p['INSTMODE_POLARIMETRY_KEYVALUE'],p['POLARIMETRY_L2_KEYVALUE'],p['FLAT_OBSTYPE_KEYVALUE'],detector_mode,wppos))

            p_polarl2["wppos{:02d}_master_flat".format(wppos)] = master_flat_file
            master_flat_file = "None"
            
        p_polarl2["master_flat"] = flats["polar_l2_master_flat"]

        
    if len(polar_l4_flat_list):
        # calculate master dome flat
        flats["polar_l4_master_flat"] = "{}/{}_s4c{}{}_POLAR_L4_MasterDomeFlat.fits".format(reduce_dir, nightdir, channel, detector_mode_key)
        # log messages:
        logger.info("Calculating master flat for POLAR L4 mode and saving to file: {}".format(flats["polar_l4_master_flat"]))

        p_polarl4 = run_master_calibration(p_polarl4, inputlist=get_shuffled_short_list(polar_l4_flat_list,max_n_files=p["MAX_NUMBER_OF_FLAT_FRAMES_TO_USE"]), output=flats["polar_l4_master_flat"], obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=force, plot=plot)
        
        master_flat_file = "None"
        for wppos in range(1,17) :
            # create a master flat for each waveplate position
            polar_l4_wppos_flat_list = s4db.get_file_list(db, inst_mode=p['INSTMODE_POLARIMETRY_KEYVALUE'], polar_mode=p['POLARIMETRY_L4_KEYVALUE'], obstype=p['FLAT_OBSTYPE_KEYVALUE'], detector_mode=detector_mode, wppos=str(wppos))
            
            if len(polar_l4_wppos_flat_list) :
                master_flat_file = "{}/{}_s4c{}{}_POLAR_L4_WPPOS{:02d}_MasterDomeFlat.fits".format(reduce_dir, nightdir, channel, detector_mode_key, wppos)
                # log messages:
                logger.info("Calculating master flat for POLAR L4 mode WPPOS={} and saving to file: {}".format(wppos,master_flat_file))
                _ = run_master_calibration(p_polarl4, inputlist=get_shuffled_short_list(polar_l4_wppos_flat_list,max_n_files=p["MAX_NUMBER_OF_FLAT_FRAMES_TO_USE"]), output=master_flat_file, obstype='flat', data_dir=data_dir, reduce_dir=reduce_dir, normalize=True, force=force, plot=plot)
            else :
                logger.info("No flats found in database for the following configuration: inst_mode={}, polar_mode={}, obstype={}, detector_mode={}, wppos={}. ".format(p['INSTMODE_POLARIMETRY_KEYVALUE'],p['POLARIMETRY_L2_KEYVALUE'],p['FLAT_OBSTYPE_KEYVALUE'],detector_mode,wppos))
                
            p_polarl4["wppos{:02d}_master_flat".format(wppos)] = master_flat_file
            master_flat_file = "None"
            
        p_polarl4["master_flat"] = flats["polar_l4_master_flat"]


    if len(polar_l2_flat_list) == 0 :
        if p_polarl2["ALLOW_INTERCHANGE_L2L4_FLATS"] and "polar_l4_master_flat" in flats.keys() :
            p_polarl2["master_flat"] = flats["polar_l4_master_flat"]
            for wppos in range(1,17) :
                if "wppos{:02d}_master_flat".format(wppos) in p_polarl4.keys() :
                    p_polarl2["wppos{:02d}_master_flat".format(wppos)] = p_polarl4["wppos{:02d}_master_flat".format(wppos)]
            
            # log messages:
            logger.warn("Using flats in POLAR L4 mode to correct POLAR L2 mode; in detector mode {}".format(detector_mode_key))
        else :
            if p_polarl2["ALLOW_USING_PHOTFLAT_TO_CORRECT_POLAR"] and "phot_master_flat" in flats.keys() :
                p_polarl2["master_flat"] = flats["phot_master_flat"]
                for wppos in range(1,17) :
                    p_polarl2["wppos{:02d}_master_flat".format(wppos)] = flats["phot_master_flat"]
                        
                # log messages:
                logger.warn("Using flats in PHOT mode to correct POLAR L2 mode; in detector mode {}".format(detector_mode_key))
            else :
                p_polarl2["APPLY_FLATFIELD_CORRECTION"] = False
                p_polarl2["master_flat"] = "None"
                for wppos in range(1,17) :
                    p_polarl2["wppos{:02d}_master_flat".format(wppos)] = "None"
                # log messages:
                logger.warn("No flats for POLAR L2, L4 nor PHOT mode in detector mode {}. Turning off flat correction.".format(detector_mode_key))
                    
    if len(polar_l4_flat_list) == 0 :
        if p_polarl4["ALLOW_INTERCHANGE_L2L4_FLATS"] and "polar_l2_master_flat" in flats.keys() :
            p_polarl4["master_flat"] = flats["polar_l2_master_flat"]
            for wppos in range(1,17) :
                if "wppos{:02d}_master_flat".format(wppos) in p_polarl2.keys() :
                    p_polarl4["wppos{:02d}_master_flat".format(wppos)] = p_polarl2["wppos{:02d}_master_flat".format(wppos)]

            # log messages:
            logger.warn("Using flats in POLAR L2 mode to correct POLAR L4 mode; in detector mode {}".format(detector_mode_key))
        else :
            if p_polarl4["ALLOW_USING_PHOTFLAT_TO_CORRECT_POLAR"] and "phot_master_flat" in flats.keys() :
                p_polarl4["master_flat"] = flats["phot_master_flat"]
                for wppos in range(1,17) :
                    p_polarl4["wppos{:02d}_master_flat".format(wppos)] = flats["phot_master_flat"]

                # log messages:
                logger.warn("Using flats in PHOT mode to correct POLAR L4 mode; in detector mode {}".format(detector_mode_key))
            else :
                p_polarl4["APPLY_FLATFIELD_CORRECTION"] = False
                p_polarl4["master_flat"] = "None"
                for wppos in range(1,17) :
                    p_polarl4["wppos{:02d}_master_flat".format(wppos)] = "None"
                # log messages:
                logger.warn("No flats for POLAR L2, L4 nor PHOT mode in detector mode {}. Turning off flat correction.".format(detector_mode_key))

    return p_phot, p_polarl2, p_polarl4



def reduce_science_images(p, inputlist, data_dir="./", reduce_dir="./", match_frames=True, ref_img="", force=False, polarimetry=False, ra="", dec="", plot=False, animated_gif=""):
    """ Pipeline module to run the reduction of science images.

         The reduction consists of the following processing steps:
         1. Detect and mask cosmic rays
         2. Perform gain correction, convert units from ADU to electrons
         3. Subtract master bias
         4. Divide by a master flat field
         5. Calculate linear offset between images
         8. Calculate FWHM for all sources and set an aperture for photometry
         9. Perform aperture photometry for all images on all sources in the catalog
         10. Save reduced image, catalog data, and photometry into a S4 product

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    data_dir : str, optional
        String to define the directory path to the raw data
    reduce_dir : str, optional
        String to define the directory path to the reduced data
    match_frames : bool, optional
        Boolean to decide whether or not to match frames, usually for
        data taken as time series
    ref_img : str, optional
        reference image, if not given it selects an image in the sequence
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product
        already exists
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with
        duplicated sources
    ra : str, optional
        string to overwrite header RA (Right Ascension) keyword
    dec : str, optional
        string to overwrite header DEC (Declination) keyword
    plot : bool, optional
        plot processed frames
    animated_gif : str, optional
        give output file name to create an animated gif of all processed frames
    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # read master calibration files
    if p["APPLY_BIAS_CORRECTION"] :
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])

    if p["APPLY_FLATFIELD_CORRECTION"] :
        masterflat = s4p.getFrameFromMasterCalibration(p["master_flat"])

    # save original input list of files
    p['INPUT_LIST_OF_FILES'] = deepcopy(inputlist)
    # check whether the input reference image is in the input list
    #if ref_img not in inputlist:
    # add ref image to the first element of input list
    inputlist = [ref_img] + inputlist

    # make sure to get the correct index for the reference image in the new list
    p['REF_IMAGE_INDEX'] = inputlist.index(p['REFERENCE_IMAGE'])

    # select FITS files in the minidata directory and build database
    # main_fg = FitsFileGroup(location=data_dir, fits_ext=p['OBJECT_WILD_CARDS'], ext=p['SCI_EXT'])
    obj_fg = FitsFileGroup(files=inputlist)

    # print total number of object files selected
    logger.info(f'OBJECT files: {len(obj_fg)}')

    logger.info("Creating output list of processed science frames ... ")

    obj_red_images, obj_red_status = [], []

    for i in range(len(obj_fg.files)):
        # get basename
        basename = os.path.basename(obj_fg.files[i])
        # create output name in the reduced dir
        output = os.path.join(reduce_dir, basename.replace(".fits", "_proc.fits"))
        obj_red_images.append(output)

        red_status = False
        if not force:
            if os.path.exists(output):
                red_status = True
        obj_red_status.append(red_status)

        logger.info("{} of {} is reduced? {} -> {}".format(i + 1, len(obj_fg.files), red_status, output))

    if not all(obj_red_status) or force:
        logger.info("Loading science frames to memory ... ")
        # get frames
        frames = list(obj_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP'], cache_folder=p['MEM_CACHE_FOLDER']))

        # extract gain from the first image
        if float(frames[0].header['GAIN']) != 0:
            # using quantities is better for safety
            gain = float(frames[0].header['GAIN'])*u.electron/u.adu
        else:
            gain = 3.3*u.electron/u.adu

        # print gain value
        logger.info('gain:{}'.format(gain))

        # set units of reduced data
        data_units = 'electron'

        # write information into an info dict
        info = {'BUNIT': ('{}'.format(data_units), 'data units'),
                'DRSINFO': ('astropop', 'data reduction software'),
                'DRSROUT': ('science frame', 'data reduction routine'),
                'BIASSUB': (p["APPLY_BIAS_CORRECTION"] , 'bias subtracted'),
                'BIASFILE': (p["master_bias"], 'bias file name'),
                'FLATCORR': (p["APPLY_FLATFIELD_CORRECTION"], 'flat corrected'),
                'FLATWPOS': (p["APPLY_FLAT_PER_WPPOS"], 'flat corrected per WP pos'),
                'MFLAT': (p["master_flat"], 'master flat file name')
                }

        logger.info('Calibrating science frames (CR, gain, bias, flat) ... ')

        flat_applied = []

        # Perform calibration
        for i, frame in enumerate(frames):
            logger.info("Calibrating science frame {} of {} : {} ".format(i+1, len(frames), os.path.basename(obj_fg.files[i])))
            if not obj_red_status[i] or force:
                if p["DETECT_AND_REJECT_COSMIC_RAYS"] :
                    processing.cosmics_lacosmic(frame, inplace=True)
                processing.gain_correct(frame, gain, inplace=True)
                if p["APPLY_BIAS_CORRECTION"] :
                    logger.info("Subtracting bias using Master Zero frame : {} ".format(p["master_bias"]))
                    processing.subtract_bias(frame, bias, inplace=True)
                if p["APPLY_FLATFIELD_CORRECTION"] :
                    wppos = 0
                    if 'WPPOS' in frames[i].header  :
                        wppos = frames[i].header['WPPOS']
                        if wppos is not int :
                            try :
                                wppos = int(wppos)
                            except Exception as e:
                                wppos = 0
                                
                    if p["APPLY_FLAT_PER_WPPOS"] and polarimetry and wppos != 0:
                        master_flat = p["wppos{:02d}_master_flat".format(wppos)]
                        if os.path.exists(master_flat) :
                            logger.info("Applying flat-field correction using Master Flat frame : {} ".format(master_flat))
                            flat_pol_wppos = s4p.getFrameFromMasterCalibration(master_flat)
                            processing.flat_correct(frame, flat_pol_wppos, inplace=True)
                            flat_applied.append(master_flat)
                        else :
                            logger.info("Applying flat-field correction using Master Flat frame : {} ".format(p["master_flat"]))
                            processing.flat_correct(frame, masterflat, inplace=True)
                            flat_applied.append(p["master_flat"])
                    else :
                        logger.info("Applying flat-field correction using Master Flat frame : {} ".format(p["master_flat"]))
                        processing.flat_correct(frame, masterflat, inplace=True)
                        flat_applied.append(p["master_flat"])
                else :
                    flat_applied.append("None")
            else:
                flat_applied.append("")
                pass

        logger.info('Calculating offsets ... ')
        p = compute_offsets(p, frames, obj_fg.files, auto_ref_selection=False)

        # write ref image name to header
        info['REFIMG'] = (p['REFERENCE_IMAGE'], "reference image")

        # reset list of output plot data
        proc_plot_files = []
        
        # Perform aperture photometry and store reduced data into products
        for i, frame in enumerate(frames):

            if not obj_red_status[i] or force:
            
                info['FLATFILE'] = (flat_applied[i], 'flat file name')
            
                logger.info("Processing file: {}".format(os.path.basename(obj_fg.files[i])))
            
                info['XSHIFT'] = (0., "register x shift (pixel)")
                info['XSHIFTST'] = ("OK", "x shift status")
                info['YSHIFT'] = (0., "register y shift (pixel)")
                info['YSHIFTST'] = ("OK", "y shift status")
                if match_frames:
                    if np.isfinite(p["XSHIFTS"][i]):
                        info['XSHIFT'] = (p["XSHIFTS"][i], "register x shift (pixel)")
                    else:
                        info['XSHIFTST'] = ("UNDEFINED", "x shift status")

                    if np.isfinite(p["YSHIFTS"][i]):
                        info['YSHIFT'] = (p["YSHIFTS"][i], "register y shift (pixel)")
                    else:
                        info['YSHIFTST'] = ("UNDEFINED", "y shift status")
            
                # get data arrays
                img_data = np.array(frame.data)
                #err_data = np.array(frame.get_uncertainty())
                
                exptime = 1.0
                try :
                    exptime = s4utils.get_exptime(frames[i].header,exptimekey=p["EXPTIMEKEY"])
                except :
                    logger.warn("Exposure time could not be retrieved from header keyword {}".format(p["EXPTIMEKEY"]))
                
                # get readout noise from header
                readnoise = float(frames[i].header[p["READNOISEKEY"]])
                
                logger.info("Exposure time: {:.2f} s; Readout noise: {:.2f} e-".format(exptime,readnoise))
                try:
                    # make catalog
                    if match_frames and "CATALOGS" in p.keys():
                        p, frame_catalogs = build_catalogs(p, img_data, frames[i].header, deepcopy(p["CATALOGS"]), xshift=p["XSHIFTS"][i], yshift=p["YSHIFTS"][i], polarimetry=polarimetry, exptime=exptime, readnoise=readnoise, set_wcs_from_database=False)
                    else:
                        p, frame_catalogs = build_catalogs(p, img_data, frames[i].header, polarimetry=polarimetry, exptime=exptime, readnoise=readnoise, set_wcs_from_database=False)
                except Exception as e:
                    logger.warn("Could not build frame catalog: {}".format(e))
                    # set local
                    frame_catalogs = []

                logger.info("Saving frame {} of {}: {} -> {}".format(i+1, len(frames), obj_fg.files[i], obj_red_images[i]))
                
                frame_wcs_header = deepcopy(p['WCS_HEADER'])

                # call function to generate final product
                hdul_frame = s4p.scienceImageProduct(obj_fg.files[i], img_data=img_data, info=info, catalogs=frame_catalogs,polarimetry=polarimetry,filename=obj_red_images[i], catalog_beam_ids=p['CATALOG_BEAM_IDS'],wcs_header=frame_wcs_header,time_key=p["TIME_KEY"], ra=ra, dec=dec)
                
                # plot individual proc frames
                if plot :
                    frame_obstime = frames[i].header[p["TIME_KEY"]]
                    proc_plot_file = ""
                    if p['PLOT_TO_FILE'] :
                        proc_plot_file = obj_red_images[i].replace(".fits",p['PLOT_FILE_FORMAT'])
                        proc_plot_files.append(proc_plot_file)
                        try :
                            if polarimetry :
                                s4plt.plot_sci_polar_frame(obj_red_images[i], percentile=99.5, output=proc_plot_file, toplabel="UT {}".format(frame_obstime[:-3]), bottomlabel=os.path.basename(obj_fg.files[i]))
                            else :
                                s4plt.plot_sci_frame(obj_red_images[i], nstars=20, use_sky_coords=True, output=proc_plot_file, toplabel="UT {}".format(frame_obstime[:-3]), bottomlabel=os.path.basename(obj_fg.files[i]))
                        except Exception as e:
                            logger.warn("Could not generate plot for product {} : {}".format(obj_red_images[i], e))
                            
        if animated_gif != "" and p['PLOT_TO_FILE'] and p['PLOT_PROC_FRAMES']:
            command = "convert "
            for i in range(len(proc_plot_files)):
                command += "{} ".format(proc_plot_files[i])
            command += "{}".format(animated_gif)
            try :
                logger.info("Creating animated gif, executing command: \n {}".format(command))
                os.system(command)
            except Exception as e:
                logger.warn("Could not create animated gif {} : {}".format(animated_gif, e))
            
    # save as new or append list of reduced images to p dict, discarding the first element = redundant ref img
    if 'OBJECT_REDUCED_IMAGES' not in p.keys():
        p['OBJECT_REDUCED_IMAGES'] = obj_red_images[1:]
    else:
        for i in range(1,len(obj_red_images)):
            if obj_red_images[i] not in p['OBJECT_REDUCED_IMAGES']:
                p['OBJECT_REDUCED_IMAGES'].append(obj_red_images[i])
    return p


def stack_science_images(p, inputlist, reduce_dir="./", force=False, stack_suffix="", output_stack="", polarimetry=False):
    """ Pipeline module to run the stack of science images.

         The reduction consists of the following processing steps:
         1. Detect and mask cosmic rays
         2. Perform gain correction, convert units from ADU to electrons
         3. Subtract master bias
         4. Divide by a master flat field
         5. Calculate linear offset between images
         6. Select sub-set of images to calculate a stack
         7. Obtain a catalog of point-like sources from the stack image
         8. Calculate FWHM for all sources and set an aperture for photometry
         9. Perform aperture photometry for all images on all sources in the catalog
         10. Save reduced image, catalog data, and photometry into a S4 product

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters

    reduce_dir : str, optional
        String to define the directory path to the reduced data
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists
    output_stack : str, optional
        String to define the directory path to the output stack file
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # set output stack filename
    if output_stack == "":
        output_stack = os.path.join(reduce_dir, '{}_stack.fits'.format(stack_suffix))
    p['OBJECT_STACK'] = output_stack

    if os.path.exists(p['OBJECT_STACK']) and not force:
        logger.info("There is already a stack image :".format(p['OBJECT_STACK']))

        stack_hdulist = fits.open(p['OBJECT_STACK'])
        hdr = stack_hdulist[0].header
        p['REFERENCE_IMAGE'] = hdr['REFIMG']
        p['REF_IMAGE_INDEX'] = 0
        for i in range(len(inputlist)):
            if inputlist[i] == p['REFERENCE_IMAGE']:
                p['REF_IMAGE_INDEX'] = i
        p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

        p["CATALOGS"] = s4p.readScienceImageCatalogs(p['OBJECT_STACK'])

        return p

    # read master calibration files
    if p["APPLY_BIAS_CORRECTION"] :
        bias = s4p.getFrameFromMasterCalibration(p["master_bias"])

    if p["APPLY_FLATFIELD_CORRECTION"] :
        masterflat = s4p.getFrameFromMasterCalibration(p["master_flat"])

    # set base image as the reference image, which will be replaced
    # later if run "select_files_for_stack"
    p['REF_IMAGE_INDEX'] = 0
    p['REFERENCE_IMAGE'] = inputlist[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    # first select best files for stack
    p = select_files_for_stack(p, inputlist, saturation_limit=p['SATURATION_LIMIT'], imagehdu=0, max_number_of_files=p['SIMIL_MAX_NFILES'], shuffle=p['SIMIL_SHUFFLE'], max_n_sources=p['SIMIL_MAX_NSOURCES'], skip_n_brightest=p['SIMIL_SKIP_N_BRIGHTEST'])

    # select FITS files in the minidata directory and build database
    obj_fg = FitsFileGroup(files=p['SELECTED_FILES_FOR_STACK'])

    # print total number of object files selected
    logger.info(f'OBJECT files: {len(obj_fg)}')

    logger.info("Loading science frames to memory ... ")
    # get frames
    frames = list(obj_fg.framedata(unit='adu', use_memmap_backend=p['USE_MEMMAP'], cache_folder=p['MEM_CACHE_FOLDER']))

    # extract gain from the first image
    if float(frames[0].header['GAIN']) != 0:
        gain = float(frames[0].header['GAIN'])*u.electron/u.adu  # using quantities is better for safety
    else:
        gain = 3.3*u.electron/u.adu

    logger.info('gain:{}'.format(gain))

    # set units of reduced data
    data_units = 'electron'

    logger.info('Calibrating science frames (CR, gain, bias, flat) for stack ... ')

    # Perform calibration
    for i, frame in enumerate(frames):
        
        logger.info("Calibrating science frame {} of {} : {} ".format(i+1, len(frames), os.path.basename(obj_fg.files[i])))
        
        if p["DETECT_AND_REJECT_COSMIC_RAYS"] :
            processing.cosmics_lacosmic(frame, inplace=True)
        
        processing.gain_correct(frame, gain, inplace=True)
        
        if p["APPLY_BIAS_CORRECTION"] :
            processing.subtract_bias(frame, bias, inplace=True)
            
        if p["APPLY_FLATFIELD_CORRECTION"] :
            wppos = 0
            if 'WPPOS' in frames[i].header  and polarimetry:
                wppos = frames[i].header['WPPOS']
                if wppos is not int :
                    try :
                        wppos = int(wppos)
                    except Exception as e:
                        # It is commented below to avoid
                        #logger.warn("Could not convert the 'WPPOS' keyword value {} to an integer.".format(frames[i].header['WPPOS']))
                        wppos = 0
                        
            if p["APPLY_FLAT_PER_WPPOS"] and polarimetry and type(wppos) is int and wppos != 0:
                master_flat = p["wppos{:02d}_master_flat".format(wppos)]
                if os.path.exists(master_flat) :
                    flat_pol_wppos = s4p.getFrameFromMasterCalibration(master_flat)
                    processing.flat_correct(frame, flat_pol_wppos, inplace=True)
                else :
                    processing.flat_correct(frame, masterflat, inplace=True)
            else :
                processing.flat_correct(frame, masterflat, inplace=True)
        

    # write information into an info dict
    info = {'BUNIT': ('{}'.format(data_units), 'data units'),
            'DRSINFO': ('astropop', 'data reduction software'),
            'DRSROUT': ('science frame', 'data reduction routine'),
            'BIASSUB': (True, 'bias subtracted'),
            'BIASFILE': (p["master_bias"], 'bias file name'),
            'FLATCORR': (True, 'flat corrected'),
            'FLATWPOS': (p["APPLY_FLAT_PER_WPPOS"], 'flat corrected per WP pos'),
            'MFLAT': (p["master_flat"], 'master flat file name'),
            'REFIMG': (p['REFERENCE_IMAGE'], 'reference image for stack'),
            'NIMGSTCK': (p['FINAL_NFILES_FOR_STACK'], 'number of images for stack')
            }
            
    list_of_imgs = sorted(obj_fg.files)
    for i in range(len(list_of_imgs)) :
        # get file basename and add it to the info dict
        basename = os.path.basename(list_of_imgs[i])
        info['IN{:06d}'.format(i)] = (basename, 'input file {} of {}'.format(i, len(list_of_imgs)))

    logger.info('Registering science frames and stacking them ... ')

    p['SELECTED_FILE_INDICES_FOR_STACK'] = np.arange(p['FINAL_NFILES_FOR_STACK'])

    # Register images, generate global catalog and generate stack image
    p = run_register_frames(p, frames, obj_fg.files, info, output_stack=output_stack, force=force, polarimetry=polarimetry)

    return p


def select_files_for_stack(p, inputlist, saturation_limit=32768, imagehdu=0, max_number_of_files = 100, shuffle=True, max_n_sources = 8, skip_n_brightest = 3, src_detect_threshold=100, remove_background=False):
    """ Pipeline module to select a sub-set of frames for stack

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    inputlist : list of str len=N_sci_images
        list of file paths to the science object image files
    saturation_limit : int or float, optional
        saturation flux limit, images with any max value above or equal this value will be ignored
    imagehdu : int or string, optional
        HDU index/name containing the image data
    max_number_of_files : int, optional
        maximum number of files to compare similarity and select for stack
    shuffle : bool, optional
        to shuffle list of input images for similarity calculations
    max_n_sources : int, optional
        maximum number of sources to compare similarity
    skip_n_brightest : int, optional
        number of brigthest objects to skip for similarity calculations
    src_detect_threshold : int, optional
        number of sigmas threshold to detect sources for similarity calculations
    remove_background : bool
        to calculate and remove background in BY_SIMILARITY method

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    if p['METHOD_TO_SELECT_FILES_FOR_STACK'] == 'FIRST_FRAMES' :
    
        sort = np.arange(len(inputlist))
        
    elif p['METHOD_TO_SELECT_FILES_FOR_STACK'] == 'MAX_FLUXES' :
    
        peaks = []
        maxflux, ref_img_idx = 0, 0
        for i in range(len(inputlist)):
            img = fits.getdata(inputlist[i], imagehdu)
            keep = img < saturation_limit
            bkg = np.nanmedian(img[keep])
            maximgflux = np.nanmax(img[keep]) - bkg
            if maximgflux > maxflux:
                maxflux = maximgflux
                ref_img_idx = i
            peaks.append(maximgflux)
            # print(i, inputlist[i], '-> ', bkg, (maximgflux - bkg))
        peaks = np.array(peaks)
        sort = np.flip(np.argsort(peaks))
        
        p['REF_IMAGE_INDEX'] = ref_img_idx
   
    elif p['METHOD_TO_SELECT_FILES_FOR_STACK'] == 'BY_SIMILARITY' :
    
        nfiles = len(inputlist)

        meanflux = np.full(nfiles, np.nan)
        
        #bkgs, rmss, nsources = np.array([]), np.array([]), np.array([])
        xs,ys,fs = [], [], []
        
        for j in range(max_n_sources) :
            xs.append(np.full_like(meanflux,np.nan))
            ys.append(np.full_like(meanflux,np.nan))
            fs.append(np.full_like(meanflux,np.nan))
        
        idx = np.arange(nfiles)
        
        if shuffle :
            np.random.shuffle(idx)

        if len(inputlist) < max_number_of_files :
            max_number_of_files = len(inputlist)

        for i in range(max_number_of_files):
        
            # read image data
            img = fits.getdata(inputlist[idx[i]], imagehdu)
            
            # calculate background
            bkg, rms = background(img, global_bkg=False)
            mbkg = np.median(bkg)
            
            # cast data to float
            red_img = np.array(img,dtype=float)
                
            # Make sure to avoid images dominated by background, usually due to clouds
            keep = (red_img - bkg) > 5 * rms
            if len(red_img[keep]) < 100 :
                logger.info("STACK: skipping image {} of {}: i={} {} -> NSOURCES: {}  bkg: {} meanflux: {} ".format(i+1,max_number_of_files,idx[i],os.path.basename(inputlist[idx[i]]), 0, mbkg, 0))
                continue
             
            # remove background if requested
            if remove_background :
                red_img = red_img - bkg
                
            nsources = 0
            try :
                # detect sources
                sources = starfind(red_img, threshold=src_detect_threshold, background=0., noise=rms, sharp_limit=(0.2, 3.0))
                nsources = len(sources)
            except :
                logger.info("STACK: skipping image {} of {}: i={} {} -> NSOURCES: {}  bkg: {} meanflux: {} ".format(i+1,max_number_of_files,idx[i],os.path.basename(inputlist[idx[i]]), 0, mbkg, 0))
                continue
                
            # sort fluxes
            sortfluxmask = np.flip(np.argsort(sources['flux']))

            # reset max_n_sources to avoid iteration over index limit
            if len(sources) < max_n_sources :
                max_n_sources = len(sources)

            # select brightest sources
            selected_sources = sources[sortfluxmask][:max_n_sources]

            # append elements to vector
            meanflux[idx[i]] = np.nanmean(selected_sources['flux'])

            for j in range(max_n_sources) :
                if j < len(sources) :
                    xs[j][idx[i]] = selected_sources['x'][j]
                    ys[j][idx[i]] = selected_sources['y'][j]
                    fs[j][idx[i]] = selected_sources['flux'][j]
                
            logger.info("STACK: checking image {} of {}: i={} {} -> NSOURCES: {}  bkg: {} meanflux: {} ".format(i+1,max_number_of_files,idx[i],os.path.basename(inputlist[idx[i]]),len(sources), mbkg, meanflux[idx[i]]))

        global_median_flux = np.nanmedian(meanflux)
        global_mad_flux =  stats.median_abs_deviation(meanflux - global_median_flux, scale="normal")
        global_median_x, global_median_y, global_median_f = [], [], []
        global_std_x, global_std_y, global_std_f = [], [], []

        for j in range(max_n_sources) :
            global_median_x.append(np.nanmedian(xs[j]))
            global_median_y.append(np.nanmedian(ys[j]))
            global_median_f.append(np.nanmedian(fs[j]))
            
            mad_x, mad_y, mad_f = np.nan , np.nan, np.nan
            if global_median_x[j] != 0 and global_median_y[j] != 0 and global_median_f[j] != 0:
                mask = (np.isfinite(xs[j])) & (np.isfinite(ys[j])) & (np.isfinite(fs[j]))
                mad_x = stats.median_abs_deviation(xs[j][mask]-global_median_x[j], scale="normal")
                mad_y = stats.median_abs_deviation(ys[j][mask]-global_median_y[j], scale="normal")
                mad_f = stats.median_abs_deviation(fs[j][mask]-global_median_f[j], scale="normal")
                                
            global_std_x.append(mad_x)
            global_std_y.append(mad_y)
            global_std_f.append(mad_f)
                
        simi_fac = np.full_like(meanflux,np.nan)
        
        for i in range(max_number_of_files) :
            sf = 0
            
            #sf += ((bkgs[i] - global_median_bkg)/global_median_rms)**2
            sf += ((meanflux[idx[i]] - global_median_flux)/global_mad_flux)**2
            for j in range(skip_n_brightest,max_n_sources) :
                if np.isfinite(xs[j][idx[i]]) and np.isfinite(ys[j][idx[i]]) and np.isfinite(fs[j][idx[i]]) :
                    sf += ((xs[j][idx[i]] - global_median_x[j])/global_std_x[j])**2
                    sf += ((ys[j][idx[i]] - global_median_y[j])/global_std_y[j])**2
                    sf += ((fs[j][idx[i]] - global_median_f[j])/global_std_f[j])**2
            sf = np.sqrt(sf)
            
            simi_fac[idx[i]] = sf

        sort = np.argsort(simi_fac)
        p['REF_IMAGE_INDEX'] = 0
        
    else :
        logger.error("Sort_method = {} not recognized, select a valid method.".format(p['METHOD_TO_SELECT_FILES_FOR_STACK']))
        exit()
    
    p['REFERENCE_IMAGE'] = inputlist[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    logger.info("Reference image: {}".format(p['REFERENCE_IMAGE']))
    
    sorted_files = []
    # first add reference image
    sorted_files.append(p['REFERENCE_IMAGE'])
    # then add all valid images to the list of images for stack
    for i in sort:
        if inputlist[i] != p['REFERENCE_IMAGE']:
            sorted_files.append(inputlist[i])

    if p['SOLAR_SYSTEM_OBJECT'] :
        # when there is a solar system object, then use only the first image instead of a stack.
        sorted_files = sorted_files[:1]

    # Now select up to <N files for stack as defined in the parameters file and save list to the param dict
    if len(sorted_files) > p['NFILES_FOR_STACK'] :
        p['SELECTED_FILES_FOR_STACK'] = sorted_files[:p['NFILES_FOR_STACK']]
        p['FINAL_NFILES_FOR_STACK'] = p['NFILES_FOR_STACK']
    else:
        p['SELECTED_FILES_FOR_STACK'] = sorted_files
        p['FINAL_NFILES_FOR_STACK'] = len(sorted_files)

    return p


def compute_offsets(p, frames, obj_files, auto_ref_selection=False):
    """ Pipeline module to select a reference image and
    compute offsets between science images

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    frames : list of frame type len=N_sci_images
        list of frames containing the data of science images
    obj_files : list of str len=N_sci_images
        list of file paths to the science object image files
    auto_ref_selection : bool, optional
        to select a reference image automatically from peak flux

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    if auto_ref_selection:
        peaksnr = []
        for i, frame in enumerate(frames):
            bkg, rms = background(frame.data, global_bkg=False)
            bkgsubimg = frame.data - bkg
            snr = bkgsubimg / rms
            peaksnr.append(np.nanmax(snr))
        peaksnr = np.array(peaksnr, dtype=float)

        p['REF_IMAGE_INDEX'] = np.argmax(peaksnr)
        p['REFERENCE_IMAGE'] = obj_files[p['REF_IMAGE_INDEX']]
        p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    logger.info("Computing offsets with respect to the reference image: index={} -> {}".format(p['REF_IMAGE_INDEX'], obj_files[p['REF_IMAGE_INDEX']]))

    # get x and y shifts of all images with respect to the first image
    if p['SHIFT_ALGORITHM'] == 'cross-correlation':
        shift_list = compute_shift_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True, upsample_factor=p['UPSAMPLEFACTOR'])
    else :
        shift_list = compute_shift_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True)

    # store shifts in x and y np arrays
    x, y = np.array([]), np.array([])
    for i in range(len(shift_list)):
        x = np.append(x, shift_list[i][0])
        y = np.append(y, shift_list[i][1])
        # print(i,shift_list[i][0],shift_list[i][1])

    # store shifts
    p["XSHIFTS"] = x
    p["YSHIFTS"] = y

    return p


def select_files_for_stack_and_get_shifts(p, frames, obj_files, sort_method='MAX_FLUXES', correct_shifts=False, plot=False):
    """ Pipeline module to compute offset between all science images and
        select a sub-set of frames for stack

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    frames : list of frame type len=N_sci_images
        list of frames containing the data of science images
    obj_files : list of str len=N_sci_images
        list of file paths to the science object image files
    sort_method : 'str'
        method to sort frames and select best subset.
        possible values:
            'MAX_FLUXES' -> sort by flux in reverse order
            'MIN_SHIFTS' -> sort by offsets
    correct_shifts : bool, optional
        Boolean to decide whether or not to correct shifts
    plot : bool, optional
        Boolean to decide whether or not to plot

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    peaks, meanbkg, peaksnr = [], [], []
    for i, frame in enumerate(frames):
        bkg, rms = background(frame.data, global_bkg=False)
        bkgsubimg = frame.data - bkg
        snr = bkgsubimg / rms
        peaks.append(np.nanmax(bkgsubimg))
        meanbkg.append(np.nanmean(bkg))
        peaksnr.append(np.nanmax(snr))
    peaks = np.array(peaks, dtype=float)
    meanbkg = np.array(meanbkg, dtype=float)
    peaksnr = np.array(peaksnr, dtype=float)

    p["PEAKS"] = peaks
    p["MEAN_BKG"] = meanbkg
    p["PEAK_SNR"] = peaksnr

    p['REF_IMAGE_INDEX'] = np.argmax(peaksnr)
    p['REFERENCE_IMAGE'] = obj_files[p['REF_IMAGE_INDEX']]
    p['REF_OBJECT_HEADER'] = fits.getheader(p['REFERENCE_IMAGE'])

    logger.info(p['REF_IMAGE_INDEX'], "Reference image: {}".format(p['REFERENCE_IMAGE']))

    # get x and y shifts of all images with respect to the first image
    if p['SHIFT_ALGORITHM'] == 'cross-correlation':
        shift_list = compute_shift_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True, upsample_factor=p['UPSAMPLEFACTOR'])
    else :
        shift_list = compute_shift_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=p['REF_IMAGE_INDEX'], skip_failure=True)

    # store shifts in x and y np arrays
    x, y = np.array([]), np.array([])
    for i in range(len(shift_list)):
        x = np.append(x, shift_list[i][0])
        y = np.append(y, shift_list[i][1])
        # print(i,shift_list[i][0],shift_list[i][1])

    # store shifts
    p["XSHIFTS"] = x
    p["YSHIFTS"] = y

    if sort_method == 'MAX_FLUXES':
        sort = np.flip(np.argsort(peaksnr))

    elif sort_method == 'MIN_SHIFTS':
        # calculate median shifts
        median_x = np.nanmedian(x)
        median_y = np.nanmedian(y)
        # calculate relative distances to the median values
        dist = np.sqrt((x - median_x)**2 + (y - median_y)**2)
        # sort in crescent order
        sort = np.argsort(dist)

        if plot:
            indices = np.arange(len(sort))
            plt.plot(indices, dist, 'ko')
            if len(sort) > p['NFILES_FOR_STACK']:
                plt.plot(indices[sort][:p['NFILES_FOR_STACK']],
                         dist[sort][:p['NFILES_FOR_STACK']], 'rx')
            else:
                plt.plot(indices, dist, 'rx')
            plt.xlabel("Image index")
            plt.ylabel("Distance to median position (pixel)")
            plt.show()
    elif sort_method == 'FIRST_FRAMES':
        sort = np.arange(len(obj_files))
    else:
        logger.error("Sort_method = {} not recognized, select a valid method.".format(sort_method))
        exit()    # select files in crescent order based on the distance to the median position

    sorted_files = []
    newsort = []

    # first add reference image
    sorted_files.append(p['REFERENCE_IMAGE'])
    newsort.append(p['REF_IMAGE_INDEX'])

    # then add all valid images to the list of images for stack
    for i in sort:
        # print(obj_files[i],p["XSHIFTS"][i],p["YSHIFTS"][i],p["PEAKS"][i],p["PEAK_SNR"][i],p["MEAN_BKG"][i])
        if np.isfinite(p["XSHIFTS"][i]) and np.isfinite(p["YSHIFTS"][i]) and p["PEAK_SNR"][i] > 1. and i != p['REF_IMAGE_INDEX']:
            sorted_files.append(obj_files[i])
            newsort.append(i)
    newsort = np.array(newsort)

    # Now select up to <N files for stack as defined in the parameters file and save list to the param dict
    if len(sorted_files) > p['NFILES_FOR_STACK']:
        p['SELECTED_FILES_FOR_STACK'] = sorted_files[:p['NFILES_FOR_STACK']]
        p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort[:p['NFILES_FOR_STACK']]
        p['FINAL_NFILES_FOR_STACK'] = p['NFILES_FOR_STACK']
    else:
        p['FINAL_NFILES_FOR_STACK'] = len(sorted_files)
        p['SELECTED_FILES_FOR_STACK'] = sorted_files
        p['SELECTED_FILE_INDICES_FOR_STACK'] = newsort

    return p


def run_register_frames(p, inframes, inobj_files, info, output_stack="", force=False, maxnsources=0, polarimetry=False):
    """ Pipeline module to register frames

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    inframes : list of frame type len=N_sci_images
        list of frames containing the data of science images
    inobj_files : list of str len=N_sci_images
        list of file paths to the science object image files
    info : dict
        dictionary with additional header cards to include in the header of product
        The following format must be used:
            info = {key1: (value1, comment1), key2: (value2, comment2), ... }
    output_stack : str, optional
        String to define output file name for stacked image
    force : bool, default=False
        whether or not to force reduction if a product already exists
    maxnsources : int, default=0
        maximum number of sources to include in the catalog. Sources are sorted by brightness
        for maxnsources=0 it will include all detected sources
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    frames, obj_files = [], []
    meanexptime = 0
    for i in p['SELECTED_FILE_INDICES_FOR_STACK']:
        logger.info("Selected file for stack:{} {}".format(i, inobj_files[i]))
        frames.append(deepcopy(inframes[i]))
        obj_files.append(inobj_files[i])
        exptime = 1.0
        try :
            exptime = s4utils.get_exptime(frames[i].header,exptimekey=p["EXPTIMEKEY"])
        except :
            logger.warn("Exposure time could not be retrieved from header keyword {}".format(p["EXPTIMEKEY"]))
        meanexptime += exptime
    
    readnoise = float(frames[0].header[p["READNOISEKEY"]])
    
    meanexptime /= len(p['SELECTED_FILE_INDICES_FOR_STACK'])
    
    stack_method = p['SCI_STACK_METHOD']

    # shift_list = compute_shift_list(frames, algorithm='asterism-matching', ref_image=0)
    # print(shift_list)

    # register frames
    if p['SHIFT_ALGORITHM'] == 'cross-correlation':
        registered_frames = register_framedata_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=0, inplace=False, skip_failure=True, upsample_factor=p['UPSAMPLEFACTOR'])
    else :
        registered_frames = register_framedata_list(frames, algorithm=p['SHIFT_ALGORITHM'], ref_image=0, inplace=False, skip_failure=True)

    # stack all object files
    combined = imcombine(registered_frames, method=stack_method, sigma_clip=p['SCI_STACK_SIGMA_CLIP'], sigma_cen_func='median', sigma_dev_func='std')

    # get stack data
    img_data = np.array(combined.data, dtype=float)
    err_data = np.array(combined.get_uncertainty())
    mask_data = np.array(combined.mask)

    # get an aperture that's 2 x fwhm measure on the stacked image
    p = calculate_aperture_radius(p, img_data)

    # generate catalog
    p, stack_catalogs = build_catalogs(p, img_data, frames[0].header, maxnsources=maxnsources, polarimetry=polarimetry, stackmode=True, exptime=meanexptime, readnoise=readnoise, set_wcs_from_database=True)

    # set master catalogs
    p["CATALOGS"] = stack_catalogs

    # save stack product
    if output_stack != "":
        if not os.path.exists(output_stack) or force:
            s4p.scienceImageProduct(obj_files[0], img_data=img_data, info=info, catalogs=p["CATALOGS"], polarimetry=polarimetry,filename=output_stack, catalog_beam_ids=p['CATALOG_BEAM_IDS'], wcs_header=p['WCS_HEADER'], time_key=p["TIME_KEY"])

    return p


def uflux_to_magnitude(uflux):
    """ Pipeline tool to convert uncertain flux into uncertain magnitude

    Parameters
    ----------
    uflux : ufloat
        uncertain flux float value

    Returns
    -------
    umag : ufloat
        uncertain magnitude float value
    """

    try:
        return -2.5 * umath.log10(uflux)
    except:
        return ufloat(np.nan, np.nan)


def flux_to_magnitude(flux):
    """ Pipeline tool to convert flux into magnitude

    Parameters
    ----------
    flux : float
        flux float value

    Returns
    -------
    mag : float
        magnitude float value
    """
    try:
        return -2.5 * np.log10(flux)
    except:
        return np.nan


def calculate_aperture_radius(p, data):
    """ Pipeline tool to calculate aperture radius for photometry

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    # calculate background
    bkg, rms = background(data, global_bkg=False)

    try :
        # detect sources
        sources = starfind(data, threshold=p['PHOT_THRESHOLD'], background=bkg, noise=rms, sharp_limit=(0.2, 3.0))
        # get fwhm
        fwhm = sources.meta['astropop fwhm']
    except Exception as e:
        # set fwhm=3 by default
        fwhm = 3.
        logger.warn("Could not detect sources. Assuming FWHM = 3: error: {}".format(e))
    
    p["PHOT_APERTURE_RADIUS"] = p["PHOT_APERTURE_N_X_FWHM"] * fwhm
    p["PHOT_SKYINNER_RADIUS"] = p["PHOT_SKYINNER_N_X_FWHM"] * fwhm
    p["PHOT_SKYOUTER_RADIUS"] = p["PHOT_SKYOUTER_N_X_FWHM"] * fwhm

    return p


def run_aperture_photometry(img_data, x, y, aperture_radius, r_ann, err_data=None, output_mag=True, exptime=1.0, recenter=False, recenter_limit=5., readnoise=0., use_astropop=False, fwhm_from_fit=False, use_moffat=False, window_size=24, update_xycenter_from_profile_fit=False, global_fit=True, calculate_fwhm=True):
    """ Pipeline module to run aperture photometry of sources in an image
    Parameters
    ----------
    img_data : numpy.ndarray (n x m)
        float array containing the image data
    x : list of floats
        list of x coordinates for the sources in the image frame
    y : list of floats
        list of y coordinates for the sources in the image frame
    aperture_radius : float
        aperture radius within which to perform aperture photometry
    r_ann : tuple: (float,float)
        sky annulus inner and outer radii
    err_data : numpy.ndarray (n x m)
        float array containing the error data (electrons)
    output_mag : bool, optional
        to convert output flux into magnitude
    exptime : float, optional
        set exposure time (s) for flux calculation
    recenter : bool, optional
        to recenter sources
    recenter_limit : float, optional
        maximum accepted recenter offset in pixels
    readnoise : float, optional
        set readout noise (electrons) for flux error calculation
    use_astropop : bool, optional
        to use astropop photometry instead of phot utils wrapper
    fwhm_from_fit : bool
        calcualte fwhm from psf fit
    use_moffat : bool
        use Moffat function. If False, it adopts a Gaussian profile
    window_size : int
        square window size (pixels) for profile fit
    update_xycenter_from_profile_fit : bool
        to update x- and y-center positions using fitted values from 2D Gaussian fits
    global_fit : bool
        to use a single fit to the combined profile from all sources
    calculate_fwhm : bool
        to calculate fwhm
        
    Returns
        x, y, mag, mag_error, smag, smag_error, fwhm, flags
    -------
     :
    """

    ap_phot = Table()
    ap_phot['original_x'], ap_phot['original_y'] = x, y
    ap_phot['x'], ap_phot['y'] = x, y
    ap_phot['aperture'] = np.full_like(x,aperture_radius)
    ap_phot['flux'] = np.full_like(x,np.nan)
    ap_phot['flux_error'] = np.full_like(x,np.nan)
    ap_phot['aperture_area'] = np.full_like(x,np.nan)
    ap_phot['bkg'] = np.full_like(x,np.nan)
    ap_phot['bkg_stddev'] = np.full_like(x,np.nan)
    ap_phot['bkg_area'] = np.full_like(x,np.nan)
    ap_phot['flags'] = np.full_like(x,0)
    ap_phot['fwhm'] = np.full_like(x,np.nan)
    
    if use_astropop :
        recenter_method, recenter_limit = None, None
        if recenter :
            recenter_method='com' # com: center of mass method
            recenter_limit=recenter_limit # recenter accepted up to 5 pixels

        ap_phot = aperture_photometry(img_data, x, y, r=aperture_radius, r_ann=r_ann, gain=1.0, bkg_method='mmm', recenter_method=recenter_method, recenter_limit=recenter_limit)
        
        # E. Martioli Jul 8 2024 -- Below is a dirty fix since astropop does not return 'fwhm' values.
        ap_phot['fwhm'] = np.full_like(ap_phot['flux'],0.)
    else :
        try:
            ap_phot = aperture_photometry_wrapper(img_data, x, y,  err_data=err_data, aperture_radius=aperture_radius, r_in=r_ann[0], r_out=r_ann[1], read_noise=readnoise, recenter=recenter, fwhm_from_fit=fwhm_from_fit, use_moffat=use_moffat, window_size=window_size, update_xycenter_from_profile_fit=update_xycenter_from_profile_fit, global_fit=global_fit, calculate_fwhm=calculate_fwhm)
        except Exception as e:
            logger.warn("{}".format(e))
            pass
    
    if exptime == 0 :
        logger.warn("Invalid value of EXPTIME=0, resetting EXPTIME=1.0")
        exptime = 1.0
        
    x, y = ap_phot['x'], ap_phot['y']
    flux = ap_phot['flux'] / exptime
    flux_error = ap_phot['flux_error'] / exptime
    sky = np.array(ap_phot['bkg']) / exptime
    sky_error = np.array(ap_phot['bkg_stddev']) / exptime
    fwhm = np.array(ap_phot['fwhm'])
    flags = np.array(ap_phot['flags'])
    
    #for i in range(len(x)):
    #    print("*** Aperture={} Src_idx={} x,y={:.1f},{:.1f} flux={:.1f}+/-{:.1f} sky={:.1f}+/-{:.1f} fwhm={:.1f}".format(aperture_radius,i,x[i],y[i],flux[i],flux_error[i],sky[i],sky_error[i],fwhm[i]))
    
    if output_mag:
        mag, mag_error = np.full_like(flux, np.nan), np.full_like(flux, np.nan)
        smag, smag_error = np.full_like(flux, np.nan), np.full_like(flux, np.nan)
        for i in range(len(flux)):
            try:
                umag = uflux_to_magnitude(ufloat(flux[i], flux_error[i]))
                uskymag = uflux_to_magnitude(ufloat(sky[i], sky_error[i]))
                mag[i], mag_error[i] = umag.nominal_value, umag.std_dev
                smag[i], smag_error[i] = uskymag.nominal_value, uskymag.std_dev
            except Exception as e:
                logger.warn("{}".format(e))
                continue
                
        return x, y, mag, mag_error, smag, smag_error, fwhm, flags
    else:
        return x, y, flux, flux_error, sky, sky_error, fwhm, flags


def read_catalog_coords(catalog):
    """ Pipeline module to read coordinates from a catalog dict
    Parameters
    ----------
    catalog : dict
        catalog of sources containing photometry data
    Returns
        ra : numpy.ndarray (N sources)
            float array containing the right ascension [DEG] data for all sources
        dec : numpy.ndarray (N sources)
            float array containing the declination [DEG] data for all sources
        x : numpy.ndarray (N sources)
            float array containing the x-position [pix] data for all sources
        y : numpy.ndarray (N sources)
            float array containing the y-position [pix] data for all sources
    -------
     :
    """

    # initialize x and y coords of sources
    ra, dec = np.array([]), np.array([])
    x, y = np.array([]), np.array([])

    for star in catalog.keys():
        ra = np.append(ra, catalog[star][1])
        dec = np.append(dec, catalog[star][2])
        x = np.append(x, catalog[star][3])
        y = np.append(y, catalog[star][4])

    return ra, dec, x, y


def set_wcs_from_astrom_ref_image(ref_filename, header, ra_deg=None, dec_deg=None):
    """ Pipeline module to set WCS header parameters from an input header of a
            reference image. The reference image is usually an astrometric field.
    Parameters
    ----------
    ref_filename : str
        reference file name to get a guess of WCS
    header : fits.Header
        FITS image header
    ra_deg : float, optional
        right ascension in deg, overwrite header value
    dec_deg : float, optional
        declination in deg, overwrite header value
    Returns
        w :
        updated WCS object
    -------
     :
    """
    
    if (ra_deg is None) or (dec_deg is None) :
        # get ra and dec from current header
        ra, dec = header['RA'].split(":"), header['DEC'].split(":")
        ra_str = '{:02d}h{:02d}m{:.2f}s'.format(int(ra[0]), int(ra[1]), float(ra[2]))
        dec_str = '{:02d}d{:02d}m{:.2f}s'.format(int(dec[0]), int(dec[1]), float(dec[2]))
        #print("RA=",ra_str, "DEC=",dec_str)
    
        # set object coordinates as SkyCoords
        coord = SkyCoord(ra_str, dec_str, frame='icrs')
    
    if ra_deg is None :
        ra_deg = coord.ra.degree
    if dec_deg is None :
        dec_deg = coord.dec.degree

    crval1, crval2 = ra_deg, dec_deg

    # get image center in pixel coordinates
    crpix1 = (header['NAXIS1'] + 1) / 2
    crpix2 = (header['NAXIS2'] + 1) / 2

    #print("Getting WCS data from reference image:", p["ASTROM_REF_IMG"])
    w = WCS(fits.getheader(ref_filename, 0), naxis=2)
    
    # update values in WCS
    w.crval = [crval1, crval2]
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.crval = [crval1, crval2]
    
    # get header from updated wcs
    wcs_hdr = w.to_header(relax=True)
    
    # update date-obs in wcs
    wcs_hdr['DATE-OBS'] = header["DATE-OBS"]
        
    # update wcs object
    w = WCS(wcs_hdr,naxis=2)

    return w


def generate_catalogs(p, data, hdr, sources, fwhm, err_data=None, catalogs=[], catalogs_label='', aperture_radius=10, r_ann=(25, 50), sortbyflux=True, maxnsources=0, polarimetry=False, use_e_beam_for_astrometry=True, solve_astrometry=False, exptime=1.0, readnoise=0., ssobj_raw_index=None, update_xycenter_from_profile_fit=False, fwhm_from_global_fit=True, calculate_fwhm=False):
    """ Pipeline module to generate new catalogs and append it
    to a given list of catalogs
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data
    hdr : fits.PrimaryHDU().header
        FITS image header
    sources : dict
        container of sources data, returned from source detection routine
    fwhm : float
        full width at half maximum
    err_data : numpy.ndarray (n x m)
        float array containing the error data (electrons)
    catalogs : list of dicts, optional
        list of catalogs of sources. An empty list indicates a new catalog list to be generated
    catalogs_label : str
        catalogs label
    aperture_radius : float
        aperture radius within which to perform aperture photometry
    r_ann : tuple: (float,float)
        sky annulus inner and outer radii
    sortbyflux : bool
        sort sources by flux, starting with the brightest
    maxnsources : int
        maxinum number of sources to limit catalog size
    polarimetry : bool
        for polarimetry data
    use_e_beam_for_astrometry : bool
        whether to use the extraordinary beam as reference for astrometry; default is to use the ordinary beam.
    solve_astrometry : bool
        whether to solve astrometry
    exptime : float
        exposure time in seconds
    readnoise : float
        readout noise in e-
    ssobj_raw_index : int
        raw index of solar system object. In polarimetry the raw index is different than the final index because the two beams will share the same final index
    update_xycenter_from_profile_fit : bool
        to update x- and y-center positions using fitted values from 2D Gaussian fits
    fwhm_from_global_fit : bool
        to use a single fit to the combined profile from all sources
    calculate_fwhm : bool
        to calculate fwhm, if False it will try to use data in p['FHWMS']
    
    Returns
        catalogs: list of dicts
            returns a list of catalog dictionaries, where the new catalogs
            generatered within this method are appended to an input list
    -------
     :
    """
    current_catalogs_len = len(catalogs)

    if polarimetry:
        catalogs.append({})
        catalogs.append({})

        # no input catalogs, then create new ones
        dx, dy = estimate_dxdy(sources['x'], sources['y'])
        
        # match polarimetric pairs (dual beam polarimetry)
        pairs = match_pairs(sources['x'], sources['y'], dx, dy, tolerance=p["MATCH_PAIRS_TOLERANCE"])

        sources_table = Table()

        sources_table['star_index'] = np.arange(len(pairs))
        sources_table['x_o'] = sources['x'][pairs['o']]
        sources_table['y_o'] = sources['y'][pairs['o']]
        sources_table['x_e'] = sources['x'][pairs['e']]
        sources_table['y_e'] = sources['y'][pairs['e']]
        
        # set index for for solar system objects
        if p['SOLAR_SYSTEM_OBJECT'] and ssobj_raw_index is not None :
            index_corr = 0
            if sortbyflux :
                index_corr = 1
        
            if use_e_beam_for_astrometry:
                mask_e = pairs['e'] == ssobj_raw_index
                if len(pairs['e'][mask_e]) :
                    p['SOLAR_SYSTEM_OBJECT_INDEX'] = sources_table['star_index'][mask_e][0] - index_corr
            else :
                mask_o = pairs['o'] == ssobj_raw_index
                if len(pairs['o'][mask_o]) :
                    p['SOLAR_SYSTEM_OBJECT_INDEX'] = sources_table['star_index'][mask_o][0] - index_corr

        if use_e_beam_for_astrometry:
            dx = np.nanmedian(sources_table['x_o'] - sources_table['x_e'])
            dy = np.nanmedian(sources_table['y_o'] - sources_table['y_e'])
        else:
            dx = np.nanmedian(sources_table['x_e'] - sources_table['x_o'])
            dy = np.nanmedian(sources_table['y_e'] - sources_table['y_o'])

        p["POLAR_DUAL_BEAM_PIX_DISTANCE_X"] = dx
        p["POLAR_DUAL_BEAM_PIX_DISTANCE_Y"] = dy
        
        # s4plt.plot_sci_polar_frame(data, bkg, sources_table)
        # print("sources:\n",sources)
        # print("\n\nsources_table:\n",sources_table)

        xo, yo, mago, mago_error, smago, smago_error, fwhmso, flagso = run_aperture_photometry(data, sources_table['x_o'], sources_table['y_o'], aperture_radius, r_ann, err_data=None, output_mag=True, exptime=exptime, recenter=p["RECENTER_APER_FOR_PHOTOMETRY"], readnoise=readnoise, use_astropop=p["USE_ASTROPOP_PHOTOMETRY"], fwhm_from_fit=p['FWHM_FROM_FIT_IN_STACK'], window_size=p['WINDOW_SIZE_FOR_PROFILE_FIT'], update_xycenter_from_profile_fit=update_xycenter_from_profile_fit, global_fit=fwhm_from_global_fit, calculate_fwhm=calculate_fwhm)
            
        xe, ye, mage, mage_error, smage, smage_error, fwhmse, flagse = run_aperture_photometry(data, sources_table['x_e'], sources_table['y_e'], aperture_radius, r_ann, err_data=None, output_mag=True, exptime=exptime, recenter=p["RECENTER_APER_FOR_PHOTOMETRY"], readnoise=readnoise, use_astropop=p["USE_ASTROPOP_PHOTOMETRY"], fwhm_from_fit=p['FWHM_FROM_FIT_IN_STACK'], window_size=p['WINDOW_SIZE_FOR_PROFILE_FIT'], update_xycenter_from_profile_fit=update_xycenter_from_profile_fit, global_fit=fwhm_from_global_fit, calculate_fwhm=calculate_fwhm)

        if calculate_fwhm :
            # update FHWMs in the parameter dict
            p['FHWMS_O'] = fwhmso
            p['FHWMS_E'] = fwhmse
        else:
            # update FWHMs using values from parameter dict
            if 'FHWMS_O' in p.keys() and len(p['FHWMS_O']) == len(mago) :
                fwhmso = p['FHWMS_O']
            if 'FHWMS_E' in p.keys() and len(p['FHWMS_E']) == len(mage) :
                fwhmse = p['FHWMS_E']

        if 'SORTED_MASK' in p.keys() :
            sortedmask = deepcopy(p['SORTED_MASK'])
        else :
            # create a sorted mask to sort data by magnitude if requested
            sortedmask = np.full_like(xo, True, dtype=bool)

        if sortbyflux:
            sortedmask = np.argsort(mago)
            # save sortedmask into a globoal variable in the paramters file
        p['SORTED_MASK'] = sortedmask
            
        xo, yo = xo[sortedmask], yo[sortedmask]
        mago, mago_error = mago[sortedmask], mago_error[sortedmask]
        smago, smago_error = smago[sortedmask], smago_error[sortedmask]
        fwhmso = fwhmso[sortedmask]
        flagso = flagso[sortedmask]

        xe, ye = xe[sortedmask], ye[sortedmask]
        mage, mage_error = mage[sortedmask], mage_error[sortedmask]
        smage, smage_error = smage[sortedmask], smage_error[sortedmask]
        fwhmse = fwhmse[sortedmask]
        flagse = flagse[sortedmask]

        if use_e_beam_for_astrometry:
            xs_for_astrometry = xe
            ys_for_astrometry = ye
        else:
            xs_for_astrometry = xo
            ys_for_astrometry = yo

        pixel_coords = np.ndarray((len(xs_for_astrometry), 2))
        for j in range(len(xs_for_astrometry)) :
            pixel_coords[j] = [xs_for_astrometry[j],ys_for_astrometry[j]]
            
        if solve_astrometry:
            try:
                if p["SOLVE_ASTROMETRY_WITH_ASTROMETRY_NET"] :
                    logger.info ("Trying to solve astrometry in PHOT-MODE using astrometry.net")
                    if use_e_beam_for_astrometry:
                        fluxes_for_astrometry = 10**(-0.4*mage)
                    else:
                        fluxes_for_astrometry = 10**(-0.4*mago)
                    h, w = np.shape(data)
                    #print("INPUT parameters: ",xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, h, w, p['REF_OBJECT_HEADER'], {'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.015, 'scale-units': 'arcsecperpix', 'scale-high':p['PLATE_SCALE']+0.015, 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER']})
                    # Solve astrometry
                    solution = solve_astrometry_xy(xs_for_astrometry, ys_for_astrometry, fluxes_for_astrometry, w, h, options={'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE']-0.02, 'scale-high': p['PLATE_SCALE']+0.02, 'scale-units': 'arcsecperpix', 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER'], 'add_path': p['ASTROM_INDX_PATH']})
                    p['WCS'] = solution.wcs
                else :
            
                    logger.info("Solving astrometry in POLAR-MODE using astrometry_from_existing_wcs()")
                    pixel_coords_atm = None
                    if p['USE_DETECTED_SRC_FOR_ASTROMETRY'] :
                        pixel_coords_atm = pixel_coords
                    
                    # First pass to get initial astrometric solution
                    p['WCS'] = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords_atm, pixel_scale=p["PLATE_SCALE"], fov_search_factor=2.0, max_number_of_catalog_sources=p['MAX_NUMBER_OF_GAIA_SRCS_FOR_ASTROMETRY'], nsources_to_plot=100, sip_degree=p['SIP_DEGREE'], use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'], plot_solution=p['PLOT_ASTROMETRY_RESULTS_IN_STACK'])
                    
                    # Iterative improvement of astrometric solution
                    for iter in range(p['N_ITER_ASTROMETRY']) :
                        logger.info("Refining astrometric solution in POLAR-MODE -> iter={}".format(iter))
                        w = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords_atm, pixel_scale=p["PLATE_SCALE"], fov_search_factor=p['FOV_SERACH_FACTOR'], max_number_of_catalog_sources=p['MAX_NUMBER_OF_GAIA_SRCS_FOR_ASTROMETRY'], nsources_to_plot=100, sip_degree=p['SIP_DEGREE'], use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'], plot_solution=p['PLOT_ASTROMETRY_RESULTS_IN_STACK'])
                        p['WCS'] = w
                        
            except Exception as e:
            
                logger.warn("Could not solve astrometry in POLAR-MODE, using WCS from database: {}".format(e))
                p['WCS'] = set_wcs_from_astrom_ref_image(p["ASTROM_REF_IMG"], hdr, ra_deg=p['RA_DEG'], dec_deg=p['DEC_DEG'])
            
        if 'WCS_HEADER' in p.keys() :
            # clean wcs header keywords
            p['WCS_HEADER'] = s4utils.clean_wcs_in_header(p['WCS_HEADER'])
            # update wcs header in parameters dict
            p['WCS_HEADER'].update(p['WCS'].to_header())
        else :
            p['WCS_HEADER'] = p['WCS'].to_header()
        
        # Recover sky coordinates for the set of input sources
        catalog_sky_coords = np.array(p['WCS'].pixel_to_world_values(pixel_coords))
        p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'] = catalog_sky_coords
    
        #ras, decs = p['WCS'].all_pix2world(xs_for_astrometry, ys_for_astrometry, 0)
        ras, decs = [], []
        for i in range(len(p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'])) :
            ras.append(p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'][i][0])
            decs.append(p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'][i][1])

        nsources = len(mago)
        if maxnsources:
            nsources = maxnsources

        # save photometry data into the catalogs
        for i in range(nsources):
            catalogs[current_catalogs_len]["{}".format(i)] = (i, ras[i], decs[i], xe[i], ye[i], fwhmso[i], fwhmso[i], mago[i], mago_error[i], smago[i], smago_error[i], aperture_radius, flagso[i])
            catalogs[current_catalogs_len+1]["{}".format(i)] = (i, ras[i], decs[i], xo[i], yo[i], fwhmse[i], fwhmse[i], mage[i], mage_error[i], smage[i], smage_error[i], aperture_radius, flagse[i])
    else:
        catalogs.append({})
        
        if p['SOLAR_SYSTEM_OBJECT'] and ssobj_raw_index is not None:
            p['SOLAR_SYSTEM_OBJECT_INDEX'] = ssobj_raw_index
        
        # x, y = np.array(sources['x']), np.array(sources['y'])
        x, y, mag, mag_error, smag, smag_error, fwhms, flags = run_aperture_photometry(data, sources['x'], sources['y'], aperture_radius, r_ann, err_data=None, output_mag=True, exptime=exptime, recenter=p["RECENTER_APER_FOR_PHOTOMETRY"], readnoise=readnoise, use_astropop=p["USE_ASTROPOP_PHOTOMETRY"], fwhm_from_fit=p['FWHM_FROM_FIT_IN_STACK'], window_size=p['WINDOW_SIZE_FOR_PROFILE_FIT'], update_xycenter_from_profile_fit=update_xycenter_from_profile_fit, global_fit=fwhm_from_global_fit, calculate_fwhm=calculate_fwhm)

        if calculate_fwhm :
            # update FHWMs in the parameter dict
            p['FHWMS'] = fwhms
        else:
            # update FWHMs using values from parameter dict
            if 'FHWMS' in p.keys() and len(p['FHWMS']) == len(mag) :
                fwhms = p['FHWMS']

        if 'SORTED_MASK' in p.keys() :
            sortedmask = deepcopy(p['SORTED_MASK'])
        else :
            # create a sorted mask to sort data by magnitude if requested
            sortedmask = np.full_like(x, True, dtype=bool)
        if sortbyflux:
            sortedmask = np.argsort(mag)
            # save sortedmask into a globoal variable in the paramters file
        p['SORTED_MASK'] = sortedmask
        
        x, y = x[sortedmask], y[sortedmask]
        mag, mag_error = mag[sortedmask], mag_error[sortedmask]
        smag, smag_error = smag[sortedmask], smag_error[sortedmask]
        fwhms, flags = fwhms[sortedmask], flags[sortedmask]

        pixel_coords = np.ndarray((len(x), 2))
        for j in range(len(x)) :
            pixel_coords[j] = [x[j],y[j]]
            
        if solve_astrometry :
            try:
                if p["SOLVE_ASTROMETRY_WITH_ASTROMETRY_NET"] :
                    logger.info("Trying to solve astrometry in PHOT-MODE using astrometry.net")
                    fluxes_for_astrometry = 10**(-0.4*mag)
                    h, w = np.shape(data)
                    solution = solve_astrometry_xy(x, y, fluxes_for_astrometry, w, h, options={'ra': p['RA_DEG'], 'dec': p['DEC_DEG'], 'radius': p['SEARCH_RADIUS'], 'scale-low': p['PLATE_SCALE'] - 0.02, 'scale-high': p['PLATE_SCALE']+0.02, 'scale-units': 'arcsecperpix', 'crpix-center': 1, 'tweak-order': p['TWEAK_ORDER'], 'add_path': p['ASTROM_INDX_PATH']})
                    p['WCS'] = solution.wcs
                    
                else :
                    logger.info("Solving astrometry in PHOT-MODE using astrometry_from_existing_wcs()")
                    
                    pixel_coords_atm = None
                    if p['USE_DETECTED_SRC_FOR_ASTROMETRY'] :
                        pixel_coords_atm = pixel_coords

                    # First pass to get initial astrometric solution
                    p['WCS'] = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords_atm, pixel_scale=p["PLATE_SCALE"], fov_search_factor=2.0, max_number_of_catalog_sources=p['MAX_NUMBER_OF_GAIA_SRCS_FOR_ASTROMETRY'], nsources_to_plot=100, sip_degree=p['SIP_DEGREE'], use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'], plot_solution=p['PLOT_ASTROMETRY_RESULTS_IN_STACK'])
                    # Iterative improvement of astrometric solution
                    for iter in range(p['N_ITER_ASTROMETRY']) :
                        logger.info("Refining astrometric solution in PHOT-MODE -> iter={}".format(iter))
                        w = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords_atm, pixel_scale=p["PLATE_SCALE"], fov_search_factor=p['FOV_SERACH_FACTOR'], max_number_of_catalog_sources=p['MAX_NUMBER_OF_GAIA_SRCS_FOR_ASTROMETRY'], plot_solution=p['PLOT_ASTROMETRY_RESULTS_IN_STACK'], nsources_to_plot=100, sip_degree=p['SIP_DEGREE'], use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'])
                        p['WCS'] = w
                    
            except Exception as e:
                logger.warn("Could not solve astrometry in PHOT-MODE, using WCS from database: {}".format(e))
                p['WCS'] = set_wcs_from_astrom_ref_image(p["ASTROM_REF_IMG"], hdr, ra_deg=p['RA_DEG'], dec_deg=p['DEC_DEG'])
                
        if 'WCS_HEADER' in p.keys() :
            # clean wcs header keywords
            p['WCS_HEADER'] = s4utils.clean_wcs_in_header(p['WCS_HEADER'])
            # update wcs header in parameters dict
            p['WCS_HEADER'].update(p['WCS'].to_header())
        else :
            p['WCS_HEADER'] = p['WCS'].to_header()
            
        # Recover sky coordinates for the set of input sources
        catalog_sky_coords = np.array(p['WCS'].pixel_to_world_values(pixel_coords))
        p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'] = catalog_sky_coords
        
        #ras, decs = p['WCS'].all_pix2world(x, y, 0)
        ras, decs = [], []
        for i in range(len(p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'])) :
            ras.append(p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'][i][0])
            decs.append(p['SKYCOORDS_FROM_CATALOG_PIXCOORDS'][i][1])
            
        nsources = len(mag)
        if maxnsources:
            nsources = maxnsources

        # save photometry data into the catalog
        for i in range(nsources):
            catalogs[current_catalogs_len]["{}".format(i)] = (i, ras[i], decs[i], x[i], y[i], fwhms[i], fwhms[i], mag[i], mag_error[i], smag[i], smag_error[i], aperture_radius, flags[i])
    
    return catalogs, p


def set_sky_aperture(p, aperture_radius):
    """ Pipeline module to calculate the sky aperture radius
    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    aperture_radius : int
        photometry aperture radius in pixels
    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    """

    r_ann = deepcopy(p['PHOT_FIXED_R_ANNULUS'])

    r_in_ann = r_ann[0]

    if r_ann[0] < aperture_radius + p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS']:
        r_in_ann = aperture_radius + p['PHOT_MIN_OFFSET_FOR_SKYINNERRADIUS']

    r_out_ann = r_ann[1]
    if r_out_ann < r_in_ann + p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS']:
        r_out_ann = r_in_ann + p['PHOT_MIN_OFFSET_FOR_SKYOUTERRADIUS']

    r_ann = (r_in_ann, r_out_ann)

    return r_ann


def add_solar_system_object(p, data, sources, dist_threshold=8, ra=None, dec=None, polarimetry=False, plot=False) :
    
    """ Pipeline module to add Solar System object to targets

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data
    sources : astropy.table.Table()
        input table of detected sources
    dist_threshold : float, default=8 pixels
        maximum distance (in pixels) for matching targets against detected sources
    ra: float
        RA (deg) for Solar System object to add
    dec: float
        Dec (deg) for Solar System object to add
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources
    plot : bool, default=False
        plot data and sources

    Returns
    -------
    sources : astropy.table.Table()
        updated table of detected sources
    visible_targets_sky_coords : list of tuples
        list of RA and DEC sky coordinates of targets visible in the field
    p : dict
        dictionary to store pipeline parameters
    """

    # initialize list of target sky coordinates
    visible_targets_sky_coords, visible_targets_pixel_coords = [], []
    visible_targets_pixel_coords2 = []
        
    dx, dy = 0, 0
    if polarimetry :
        dx = p["POLAR_DUAL_BEAM_PIX_DISTANCE_X"]
        dy = p["POLAR_DUAL_BEAM_PIX_DISTANCE_Y"]

    if plot :
        src_pixel_coords = np.ndarray((len(sources['x']), 2))
        for j in range(len(sources['x'])) :
            src_pixel_coords[j] = [sources['x'][j],sources['y'][j]]

    # initialize list of target sky coordinates
    target_sky_coords = [[ra,dec]]
        
    # use current wcs to generate the set of pixel coordinates of input targets
    targets_pixel_coords = np.array(p["WCS"].world_to_pixel_values(target_sky_coords))

    # save x and y pixel coordinates into arrays
    ny, nx = np.shape(data)
    x, y, x2, y2 = [], [], [], []
    ids = []
    for i in range(len(targets_pixel_coords)) :
    
        # get x,y pix coordinates
        xcoord = targets_pixel_coords[i][0]
        ycoord = targets_pixel_coords[i][1]
        
        # get x,y pix coordinates for the second polarimetric beam
        xcoord2 = xcoord + dx
        ycoord2 = ycoord + dy
            
        # append only if star lies within the image boundaries:
        if (xcoord > 0) and (xcoord < nx) and \
           (ycoord > 0) and (ycoord < ny) and \
           (xcoord2 > 0) and (xcoord2 < nx) and \
           (ycoord2 > 0) and (ycoord2 < ny) and \
            np.isfinite(xcoord) and np.isfinite(ycoord) and \
            np.isfinite(xcoord2) and np.isfinite(ycoord2) :
           
            x.append(xcoord)
            y.append(ycoord)
            x2.append(xcoord2)
            y2.append(ycoord2)
            ids.append(p['SOLAR_SYSTEM_OBJECT_ID'])
            
            visible_targets_sky_coords.append([ra,dec])
            visible_targets_pixel_coords.append([xcoord,ycoord])
            visible_targets_pixel_coords2.append([xcoord2,ycoord2])
            
    # if number of targets in the field is 0, exit
    if len(x) == 0 :
        return sources, visible_targets_sky_coords, p
       
    # set circular apertures for photometry
    apertures = photutils.aperture.CircularAperture(visible_targets_pixel_coords, r=10)
    # calculate photometric quantities for all targets
    aper_stats = photutils.aperture.ApertureStats(data, apertures)
      
    ss_target_index = None
    # check if targets are already in the sources list
    for j in range(len(x)) :
        # calculate distace between target and all sources
        dist = np.sqrt((sources['x'] - x[j])**2 + (sources['y'] - y[j])**2)
                
        # get minimum distance
        min_dist_idx = np.nanargmin(dist)

        # Consider a match if minimum distance is within dist_threshold
        if dist[min_dist_idx] < dist_threshold:
            logger.info("Target {} already exists in catalog with index={}, skipping ...".format(ids[j], min_dist_idx))
            ss_target_index = min_dist_idx
            #row = sources[min_dist_idx]
        else :
            ss_target_index = len(sources)
            # make a hard copy of the first row in the sources table
            row = deepcopy(sources[0])
                    
            # fill in basic photometric information for new target
            row['id'] = len(sources)
            row["x"], row["y"] = x[j], y[j]
            row['flux'] = aper_stats.sum[j]
            row['xcentroid'], row['ycentroid'] = aper_stats.xcentroid[j], aper_stats.ycentroid[j]
            #row['peak'], row['fwhm'] = aper_stats.max[j], aper_stats.fwhm[j]
            row['sharpness'], row['roundness'] = np.nan, np.nan
            row['s_roundness'], row['g_roundness'] = np.nan, np.nan
            row['eccentricity'], row['elongation'] = aper_stats.eccentricity[j] , aper_stats.elongation[j]
            row['ellipticity'] = aper_stats.ellipticity[j]
            #row['cxx'], row['cyy'], row['cxy'] = aper_stats.cxx[j], aper_stats.cyy[j], aper_stats.cxy[j]
            
            # add new target to the sources table
            sources.add_row(row)
            
            if polarimetry :
                row2 = deepcopy(row)
                row2["x"], row2["y"] = x2[j], y2[j]
                sources.add_row(row2)
            
    p['SOLAR_SYSTEM_OBJECT_INDEX'] = ss_target_index
    
    if plot :
        fig = plt.figure(figsize=(10, 10))
        
        new_src_pixel_coords = np.ndarray((len(sources['x']), 2))
        for j in range(len(sources['x'])) :
            new_src_pixel_coords[j] = [sources['x'][j],sources['y'][j]]
        # plot image to check targets
        astrometry_sources_pixcoords = np.array(visible_targets_pixel_coords)
        astrometry_sources_pixcoords2 = np.array(visible_targets_pixel_coords2)
                    
        plt.imshow(data, vmin=np.percentile(data, 0.5), vmax=np.percentile(data, 99.5), origin='lower', cmap="Greys_r")
        _ = photutils.aperture.CircularAperture(src_pixel_coords, r=6.0).plot(color="g")
        _ = photutils.aperture.CircularAperture(new_src_pixel_coords, r=10.0).plot(color="m")
        
        _ = photutils.aperture.CircularAperture(astrometry_sources_pixcoords, r=14.0).plot(color="r")
        if polarimetry :
            _ = photutils.aperture.CircularAperture(astrometry_sources_pixcoords2, r=14.0).plot(color="r")
        plt.annotate(p['SOLAR_SYSTEM_OBJECT_ID'], [astrometry_sources_pixcoords[0][0]-25, astrometry_sources_pixcoords[0][1]+25], color='r')
        plt.xlabel("columns (pixel)", fontsize=16)
        plt.ylabel("rows (pixel)", fontsize=16)
        
        if p['PLOT_TO_FILE'] :
            output = p['OBJECT_STACK'].replace(".fits","_ssobj{}".format(p['PLOT_FILE_FORMAT']))
            fig.savefig(output, bbox_inches='tight')
            plt.close(fig)
        else :
            plt.show()
                    
    return sources, visible_targets_sky_coords, p


def add_targets_from_users_file(p, data, sources, dist_threshold=8, polarimetry=False, plot=False) :
    
    """ Pipeline module to add targets from the user's input list to the sources list

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data
    sources : astropy.table.Table()
        input table of detected sources
    dist_threshold : float, default=8 pixels
        maximum distance (in pixels) for matching targets against detected sources
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources
    plot : bool, default=False
        plot data and sources

    Returns
    -------
    sources : astropy.table.Table()
        updated table of detected sources
    visible_targets_sky_coords : list of tuples
        list of RA and DEC sky coordinates of targets visible in the field
    p : dict
        dictionary to store pipeline parameters
    """

    # read table of input targets
    tbl = ascii.read(p['TARGET_LIST_FILE'])
    
    # initialize list of target sky coordinates
    visible_targets_sky_coords, visible_targets_pixel_coords = [], []

    # if tbl is empty, return intact sources and p, and empty target_sky_coords
    if len(tbl) == 0 :
        return sources, visible_targets_sky_coords, p
        
    dx, dy = 0, 0
    if polarimetry :
        dx = p["POLAR_DUAL_BEAM_PIX_DISTANCE_X"]
        dy = p["POLAR_DUAL_BEAM_PIX_DISTANCE_Y"]

    if plot :
        src_pixel_coords = np.ndarray((len(sources['x']), 2))
        for j in range(len(sources['x'])) :
            src_pixel_coords[j] = [sources['x'][j],sources['y'][j]]

    # initialize list of target sky coordinates
    target_sky_coords = []

    # read targets coordinates into an array of coordinates
    for i in range(len(tbl)) :
        target_sky_coords.append([tbl["RA"][i],tbl["DEC"][i]])
        
    # use current wcs to generate the set of pixel coordinates of input targets
    targets_pixel_coords = np.array(p["WCS"].world_to_pixel_values(target_sky_coords))

    # save x and y pixel coordinates into arrays
    ny, nx = np.shape(data)
    x, y, x2, y2 = [], [], [], []
    ids = []
    for i in range(len(targets_pixel_coords)) :
    
        # get x,y pix coordinates
        xcoord = targets_pixel_coords[i][0]
        ycoord = targets_pixel_coords[i][1]
        
        # get x,y pix coordinates for the second polarimetric beam
        xcoord2 = xcoord + dx
        ycoord2 = ycoord + dy
            
        # append only if star lies within the image boundaries:
        if (xcoord > 0) and (xcoord < nx) and \
           (ycoord > 0) and (ycoord < ny) and \
           (xcoord2 > 0) and (xcoord2 < nx) and \
           (ycoord2 > 0) and (ycoord2 < ny) and \
            np.isfinite(xcoord) and np.isfinite(ycoord) and \
            np.isfinite(xcoord2) and np.isfinite(ycoord2) :
           
            x.append(xcoord)
            y.append(ycoord)
            x2.append(xcoord2)
            y2.append(ycoord2)
            ids.append(tbl['OBJECT_ID'][i])
            #print(i,tbl["OBJECT_ID"][i],targets_pixel_coords[i][0],targets_pixel_coords[i][1])
            
            visible_targets_sky_coords.append([tbl["RA"][i],tbl["DEC"][i]])
            visible_targets_pixel_coords.append([xcoord,ycoord])
            
    # if number of targets in the field is 0, exit
    if len(x) == 0 :
        return sources, visible_targets_sky_coords, p
       
    # set circular apertures for photometry
    apertures = photutils.aperture.CircularAperture(visible_targets_pixel_coords, r=10)
    # calculate photometric quantities for all targets
    aper_stats = photutils.aperture.ApertureStats(data, apertures)
            
    # check if targets are already in the sources list
    for j in range(len(x)) :
        # calculate distace between target and all sources
        dist = np.sqrt((sources['x'] - x[j])**2 + (sources['y'] - y[j])**2)
        
        # get minimum distance
        min_dist_idx = np.nanargmin(dist)
        # Consider a match if minimum distance is within dist_threshold
        if dist[min_dist_idx] <  dist_threshold:
            logger.info("Target {} already exists in catalog with index={}, skipping ...".format(ids[j], min_dist_idx))
            #row = sources[min_dist_idx]
        else :
            # make a hard copy of the first row in the sources table
            row = deepcopy(sources[0])
                    
            # fill in basic photometric information for new target
            row['id'] = len(sources)
            row["x"], row["y"] = x[j], y[j]
            row['flux'] = aper_stats.sum[j]
            row['xcentroid'], row['ycentroid'] = aper_stats.xcentroid[j], aper_stats.ycentroid[j]
            #row['peak'], row['fwhm'] = aper_stats.max[j], aper_stats.fwhm[j]
            row['sharpness'], row['roundness'] = np.nan, np.nan
            row['s_roundness'], row['g_roundness'] = np.nan, np.nan
            row['eccentricity'], row['elongation'] = aper_stats.eccentricity[j] , aper_stats.elongation[j]
            row['ellipticity'] = aper_stats.ellipticity[j]
            #row['cxx'], row['cyy'], row['cxy'] = aper_stats.cxx[j], aper_stats.cyy[j], aper_stats.cxy[j]
            
            # add new target to the sources table
            sources.add_row(row)
            
            if polarimetry :
                row2 = deepcopy(row)
                row2["x"], row2["y"] = x2[j], y2[j]
                sources.add_row(row2)
                   
    if plot :
        new_src_pixel_coords = np.ndarray((len(sources['x']), 2))
        for j in range(len(sources['x'])) :
            new_src_pixel_coords[j] = [sources['x'][j],sources['y'][j]]
        # plot image to check targets
        astrometry_sources_pixcoords = np.array(visible_targets_pixel_coords)
        plt.imshow(data, vmin=np.percentile(data, 0.5), vmax=np.percentile(data, 99.5), origin='lower', cmap="Greys_r")
        _ = photutils.aperture.CircularAperture(src_pixel_coords, r=6.0).plot(color="g")
        _ = photutils.aperture.CircularAperture(new_src_pixel_coords, r=10.0).plot(color="m")
        _ = photutils.aperture.CircularAperture(astrometry_sources_pixcoords, r=14.0).plot(color="y")
        plt.show()
                    
    return sources, visible_targets_sky_coords, p


def build_catalogs(p, data, hdr, catalogs=[], xshift=0., yshift=0., solve_astrometry=True, maxnsources=0, polarimetry=False, stackmode=False, exptime=1.0, readnoise=0., set_wcs_from_database=False):
    """ Pipeline module to generate the catalogs of sources from an image
        This module will perform the following tasks:
        1. Calculate background in the input image
        2. Detect sources using starfind
        3. Perform aperture photometry
        4. Build catalog container and stores it into the parameter dict

    Parameters
    ----------
    p : dict
        dictionary to store pipeline parameters
    data : numpy.ndarray (n x m)
        float array containing the image data
    hdr : fits.PrimaryHDU().header
        FITS image header
    catalogs : list of dicts, optional
        list of dictionaries containing the input catalogs. If not provided it
        will re-detect sources on image and build a new catalog
    xshift : float, optional
        image offset in the x direction with respect to the input catalog
    yshift : float, optional
        image offset in the y direction with respect to the input catalog
    maxnsources : int, default=0
        maximum number of sources to include in the catalog. Sources are sorted by brightness
        for maxnsources=0 it will include all detected sources
    polarimetry : bool, default=False
        whether or not input data is a dual beam polarimetric image with duplicated sources
    exptime : float, optional
        exposure time in units of seconds
    readnoise : float, optional
        CCD readout noise in units of electrons

    Returns
    -------
    p : dict
        dictionary to store pipeline parameters
    catalogs : list of dicts
        output catalogs
    """
    # read image data
    # hdul = fits.open(image_name, mode = "readonly")
    # data = np.array(hdul[0].data, dtype=float)
       
    p['RA_DEG'], p['DEC_DEG'] = None, None
    
    if p['SOLAR_SYSTEM_OBJECT'] :
        # For solar system object, query object id through JPL Horizons to get ephemerides for observation time
        obstime = Time(hdr[p['TIME_KEY']], format='isot', scale='utc')
        ssobj = Horizons(id=p['SOLAR_SYSTEM_OBJECT_ID'], location=p['JPL_HORIZONS_OBSERVATORY_CODE'], epochs=obstime.jd)
        eph = ssobj.ephemerides()
        # get ra/dec of object during observations
        p['RA_DEG'], p['DEC_DEG'] = eph['RA'][0], eph['DEC'][0]
        logger.info("Solar System object {} detected, setting coordinates at center: RA={} Dec={}".format(p['SOLAR_SYSTEM_OBJECT_ID'],p['RA_DEG'], p['DEC_DEG']))
        
    # set wcs from a reference image in the database
    p['WCS'] = set_wcs_from_astrom_ref_image(p["ASTROM_REF_IMG"], hdr, ra_deg=p['RA_DEG'], dec_deg=p['DEC_DEG'])
    
    #print("******* DEBUG ASTROMETRY **********")
    #print("WCS:\n{}".format(p['WCS']))
 
    # create a copy of input catalogs
    new_catalogs = deepcopy(catalogs)
 
    if stackmode:
        new_catalogs = []

    # When no catalog is provided, generate a new one (usually on stack image)
    if new_catalogs == []:
    
        # calculate background
        bkg, rms = background(data, global_bkg=False)
    
        try :
            # detect sources
            sources = starfind(data, threshold=p["PHOT_THRESHOLD"], background=bkg, noise=rms, sharp_limit=(0.2, 3.0))
        except Exception as e:
            logger.warn("Could not build catalog of sources -- ERROR on astropop.starfind() -> {}".format(e))
            return p, new_catalogs

        # get fwhm
        fwhm = sources.meta['astropop fwhm']

        # set aperture radius from
        r_ann = set_sky_aperture(p, p['PHOT_FIXED_APERTURE'])
    
        logger.info("Creating new catalog of detected sources:")
        
        # The step below is important to solve astrometry before add targets from user or object header key
        # print("Running aperture photometry with aperture_radius={} r_ann={}".format(aperture_radius,r_ann))
        new_catalogs, p = generate_catalogs(p, data, hdr, sources, fwhm, err_data=rms, catalogs=[], aperture_radius=p['PHOT_FIXED_APERTURE'], r_ann=r_ann, sortbyflux=True, polarimetry=polarimetry, solve_astrometry=p["SOLVE_ASTROMETRY_IN_STACK"], exptime=exptime, readnoise=readnoise, update_xycenter_from_profile_fit=p["UPDATE_XY_SRC_COORDINATES_FROM_PROFILE_FIT"], fwhm_from_global_fit=p["FWHM_FROM_GLOBAL_FIT_IN_STACK"], calculate_fwhm=True)
        
        p['SOLAR_SYSTEM_OBJECT_INDEX'] = None
        
        if p['SOLAR_SYSTEM_OBJECT'] :
    
            logger.info("Adding solar system object manually to the catalog of sources")
            # run routine to add solar system object
            sources, target_sky_coords, p = add_solar_system_object(p, data, sources, ra=p['RA_DEG'], dec=p['DEC_DEG'], dist_threshold=2*fwhm, polarimetry=polarimetry, plot=p['PLOT_STACK_WITH_SS_OBJ_IDENTIFIED'])
            logger.info("The raw index of the Solar System object is {}".format(p['SOLAR_SYSTEM_OBJECT_INDEX']))
            
            if len(target_sky_coords) :
                # delete preview sorted mask, which has a size inconsistent with the new one
                del p['SORTED_MASK']
                # generate catalogs again, now with the SS object added
                new_catalogs, p = generate_catalogs(p, data, hdr, sources, fwhm, err_data=rms, catalogs=[], aperture_radius=p['PHOT_FIXED_APERTURE'], r_ann=r_ann, polarimetry=polarimetry, solve_astrometry=p["SOLVE_ASTROMETRY_IN_STACK"], exptime=exptime, readnoise=readnoise, sortbyflux=True, ssobj_raw_index=p['SOLAR_SYSTEM_OBJECT_INDEX'], update_xycenter_from_profile_fit=p["UPDATE_XY_SRC_COORDINATES_FROM_PROFILE_FIT_SS_OBJECT"], fwhm_from_global_fit=p["FWHM_FROM_GLOBAL_FIT_IN_STACK"], calculate_fwhm=True)
            # update main target index to the Solar System object index
            if p['UPDATE_TARGET_INDEX_TO_SS_OBJECT'] :
                p['TARGET_INDEX'] = p['SOLAR_SYSTEM_OBJECT_INDEX']

        if p['TARGET_LIST_FILE'] != "" :
            logger.info("Adding sources manually from input target list file: {}".format(p['TARGET_LIST_FILE']))
                
            # run routine to add targets from user's list to the sources
            sources, target_sky_coords, p = add_targets_from_users_file(p, data, sources, dist_threshold=2*fwhm, polarimetry=polarimetry, plot=False)
            
            if len(target_sky_coords) :
                # delete preview sorted mask, which has a size inconsistent with the new one
                del p['SORTED_MASK']
                # generate catalogs again, now with the new targets added
                new_catalogs, p = generate_catalogs(p, data, hdr, sources, fwhm, err_data=rms, catalogs=[], aperture_radius=p['PHOT_FIXED_APERTURE'], r_ann=r_ann, polarimetry=polarimetry, solve_astrometry=p["SOLVE_ASTROMETRY_IN_STACK"], exptime=exptime, readnoise=readnoise, sortbyflux=True, update_xycenter_from_profile_fit=p["UPDATE_XY_SRC_COORDINATES_FROM_PROFILE_FIT"], fwhm_from_global_fit=p["FWHM_FROM_GLOBAL_FIT_IN_STACK"], calculate_fwhm=True)
               
                if p['UPDATE_TARGET_INDEX_FROM_INPUT'] and not p['UPDATE_TARGET_INDEX_TO_SS_OBJECT']:
                    # use current wcs to generate the set of pixel coordinates of input targets
                    targets_pixel_coords = np.array(p["WCS"].world_to_pixel_values(target_sky_coords))
                    # read catalog x,y pixel coordinates
                    catalog = new_catalogs[0]
                    src_x, src_y = np.array([]), np.array([])
                    for src in catalog.keys() :
                        src_x = np.append(src_x,catalog[src][3])
                        src_y = np.append(src_y,catalog[src][4])
                    # find out index of main user target
                    dist = np.sqrt((src_x - targets_pixel_coords[p['TARGET_INDEX_FROM_INPUT']][0])**2 + (src_y - targets_pixel_coords[p['TARGET_INDEX_FROM_INPUT']][1])**2)
                    # update target index by an input target
                    p['TARGET_INDEX'] = np.nanargmin(dist)
                    

        if p['MULTI_APERTURES']:
            logger.info("Running photometry for multiple apertures:")

            new_catalogs = []
            
            for i in range(len(p['PHOT_APERTURES'])):
                aperture_radius = p['PHOT_APERTURES'][i]
                logger.info("Aperture radius of {} pixels: {} of {} :".format(aperture_radius,i+1,len(p['PHOT_APERTURES'])))

                r_ann = set_sky_aperture(p, aperture_radius)

                # print("Running aperture photometry with aperture_radius={} r_ann={}".format(aperture_radius,r_ann))
                new_catalogs, p = generate_catalogs(p, data, hdr, sources, fwhm, err_data=rms, catalogs=new_catalogs, aperture_radius=aperture_radius, r_ann=r_ann, sortbyflux=False, polarimetry=polarimetry, exptime=exptime, readnoise=readnoise, update_xycenter_from_profile_fit=p["UPDATE_XY_SRC_COORDINATES_FROM_PROFILE_FIT"], fwhm_from_global_fit=p["FWHM_FROM_GLOBAL_FIT_IN_STACK"], calculate_fwhm=False)
                
    else:
        # Here's when a catalog is provided:
        logger.info("Running aperture photometry for catalogs with an offset of dx={} dy={}".format(xshift, yshift))
        
        ras, decs, x, y = read_catalog_coords(deepcopy(catalogs[0]))
        
        pixel_coords = np.ndarray((len(x), 2))
        sky_coords = np.ndarray((len(ras), 2))
        for i in range(len(x)) :
            pixel_coords[i] = [x[i]+xshift,y[i]+yshift]
            sky_coords[i] = [ras[i],decs[i]]
                    
        if p['SOLAR_SYSTEM_OBJECT'] :
            # Remove Solar System object before solving astrometry
            idx = p['SOLAR_SYSTEM_OBJECT_INDEX']
            pixel_coords = np.delete(pixel_coords,idx,0)
            sky_coords = np.delete(sky_coords,idx,0)

        if p['SOLVE_ASTROMETRY_IN_INDIVIDUAL_FRAMES'] :
            pixel_coords_atm = None
            if p['USE_DETECTED_SRC_FOR_ASTROMETRY'] :
                pixel_coords_atm = pixel_coords
            
            # First pass to get initial astrometric solution
            p['WCS'] = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords_atm, pixel_scale=p["PLATE_SCALE"], fov_search_factor=2.0, max_number_of_catalog_sources=p['MAX_NUMBER_OF_GAIA_SRCS_FOR_ASTROMETRY'], nsources_to_plot=100, sip_degree=None, use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'], plot_solution=False)
            # Iterative improvement of astrometric solution
            for iter in range(p['N_ITER_ASTROMETRY']) :
                # run full astrometry for individual frames using the existing WCS as reference
                w = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords_atm, pixel_scale=p["PLATE_SCALE"], fov_search_factor=p['FOV_SERACH_FACTOR'], max_number_of_catalog_sources=p['MAX_NUMBER_OF_GAIA_SRCS_FOR_ASTROMETRY'], plot_solution=False, nsources_to_plot=100, sip_degree=p['SIP_DEGREE'], use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'])
                p['WCS'] = w

        else :
            # update WCS using the set of x+offset,y+offset and ra,dec arrays in the catalog.
            w = astrometry_from_existing_wcs(deepcopy(p['WCS']), data, pixel_coords=pixel_coords, sky_coords=sky_coords, pixel_scale=p["PLATE_SCALE"], plot_solution=False, nsources_to_plot=100, sip_degree=p['SIP_DEGREE'], use_vizier=p['USE_VIZIER'], vizier_catalogs=p['VIZIER_CATALOGS'], vizier_catalog_idx=p['VIZIER_CATALOG_IDX'])
            p['WCS'] = w
        
        if 'WCS_HEADER' in p.keys() :
            # clean wcs header keywords
            p['WCS_HEADER'] = s4utils.clean_wcs_in_header(p['WCS_HEADER'])
            # update wcs header in parameters dict
            p['WCS_HEADER'].update(p['WCS'].to_header())
        else :
            p['WCS_HEADER'] = p['WCS'].to_header()
        
        # initialize switch to calculate fwhm
        calculate_fwhm = True
        lim_index = 0
        if polarimetry :
            lim_index = 1
            
        for j in range(len(catalogs)):
            
            # load coordinates from an input catalog
            ras, decs, x, y = read_catalog_coords(deepcopy(catalogs[j]))
                
            # apply shifts
            if np.isfinite(xshift):
                x += xshift
            if np.isfinite(yshift):
                y += yshift

            if p['SOLAR_SYSTEM_OBJECT'] :
                idx = p['SOLAR_SYSTEM_OBJECT_INDEX']
                
                #ssobj = Horizons(id=p['SOLAR_SYSTEM_OBJECT_ID'], location=p['JPL_HORIZONS_OBSERVATORY_CODE'], epochs=obstime.jd)
                #eph = ssobj.ephemerides()
                ss_pix_coords = np.array(p["WCS"].world_to_pixel_values([[eph['RA'][0], eph['DEC'][0]]]))
                
                if polarimetry :
                    if (j % 2) == 0 :
                        x[idx], y[idx] = ss_pix_coords[0][0], ss_pix_coords[0][1]
                    else :
                        x[idx] = ss_pix_coords[0][0]+p["POLAR_DUAL_BEAM_PIX_DISTANCE_X"]
                        y[idx] = ss_pix_coords[0][1]+p["POLAR_DUAL_BEAM_PIX_DISTANCE_Y"]
                else :
                    x[idx], y[idx] = ss_pix_coords[0][0], ss_pix_coords[0][1]
                    
            aperture_radius = catalogs[j]['0'][11]
            r_ann = set_sky_aperture(p, aperture_radius)

            # reset copy of catalogs data
            for i in range(len(catalogs[j])):
                new_catalogs[j]["{}".format(i)] = (i, ras[i], decs[i], x[i], y[i], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, aperture_radius, 0)
                
            #logger.info("Running aperture photometry for catalog={}/{} xshift={} yshift={} with aperture_radius={} r_ann={}".format(j+1,len(catalogs),xshift,yshift,aperture_radius,r_ann))
            
            # run aperture photometry
            x, y, mag, mag_error, smag, smag_error, fwhms, flags = run_aperture_photometry(data, x, y, aperture_radius, r_ann, output_mag=True, exptime=exptime, recenter=p["RECENTER_APER_FOR_PHOTOMETRY"], readnoise=readnoise, use_astropop=p["USE_ASTROPOP_PHOTOMETRY"], fwhm_from_fit=p['FWHM_FROM_FIT_IN_INDIVIDUAL_FRAMES'], window_size=p['WINDOW_SIZE_FOR_PROFILE_FIT'], update_xycenter_from_profile_fit=False, global_fit=p['FWHM_FROM_GLOBAL_FIT_IN_INDIVIDUAL_FRAMES'], calculate_fwhm=calculate_fwhm)
    
            # update values of FHWMs if they have been calculated, and get from p when they haven't.
            if polarimetry :
                if calculate_fwhm :
                    if (j % 2) == 0 :
                        p['FHWMS_O'] = fwhms
                    else :
                        p['FHWMS_E'] = fwhms
                else:
                    if (j % 2) == 0 :
                        # update FWHMs using values from parameter dict
                        if 'FHWMS_O' in p.keys() and len(p['FHWMS_O']) == len(mag) :
                            fwhms = p['FHWMS_O']
                    else :
                        if 'FHWMS_E' in p.keys() and len(p['FHWMS_E']) == len(mag) :
                            fwhms = p['FHWMS_E']
            else :
                if calculate_fwhm :
                    p['FWHM'] = fwhms
                else :
                    if 'FHWMS' in p.keys() and len(p['FHWMS']) == len(mag) :
                        fwhms = p['FWHM']
    
            #logger.info("Photometry of source=0: x={:.2f} y={:.2f} mag={:.5f} emag={:.5f} smag={:.5f} esmag={:.5f} fwhm={:.2f} flag={}".format(x[0], y[0], mag[0], mag_error[0], smag[0], smag_error[0], fwhms[0], flags[0]))
            
            # save data back into the catalog
            for i in range(len(mag)):
                new_catalogs[j]["{}".format(i)] = (i, ras[i], decs[i], x[i], y[i], fwhms[i], fwhms[i], mag[i], mag_error[i], smag[i], smag_error[i], aperture_radius, flags[i])

            # turn off calculating fwhm for improved performance.
            if j == lim_index :
                calculate_fwhm = False
                
    return p, new_catalogs


def phot_time_series(sci_list,
                     reduce_dir="./",
                     ts_suffix="",
                     time_key='DATE-OBS',
                     time_format='isot',
                     time_scale='utc',
                     longitude=-45.5825,
                     latitude=-22.5344,
                     altitude=1864,
                     catalog_names=[],
                     time_span_for_rms=5,
                     keys_to_add_header_data=[],
                     best_apertures=False,
                     force=True):
    """ Pipeline module to calculate photometry differential time series for a given list of sparc4 sci image products

    Parameters
    ----------
    sci_list : list
        list of paths to science image products
    ts_suffix : str (optional)
        time series suffix to add into the output file name
    reduce_dir : str (optional)
        path to the reduce directory
    time_keyword : str (optional)
        Time keyword in fits header. Default is 'DATE-OBS'
    time_format : str (optional)
        Time format in fits header. Default is 'isot'
    time_scale : str (optional)
        Time scale in fits header. Default is 'utc'
    longitude : float (optional)
        East geographic longitude [deg] of observatory; default is OPD longitude of -45.5825 degrees
    latitude : float (optional)
        North geographic latitude [deg] of observatory; default is OPD latitude of -22.5344 degrees
    altitude : float (optional)
        Observatory elevation [m] above sea level. Default is OPD altitude of 1864 m
    catalog_names : list (optional)
        list of catalog names to be included in the final data product. The less the faster. Default is [], which means all catalogs.
    time_span_for_rms : float, optional
        Time span (in minutes) around a given observation to include other observations to
        calculate running rms.
    keys_to_add_header_data : list of str, optional
        list of header keywords to get data from and include in time series
    best_apertures : bool, optional
        Boolean to include extension with best apertures
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists

    Returns
    -------
    output : str
        path to the output time series product file
    """

    # set output light curve product file name
    output = os.path.join(reduce_dir, "{}_lc.fits".format(ts_suffix))
    if os.path.exists(output) and not force:
        return output

    # initialize data container as dict
    tsdata = {}

    # get information from the first image in the time series
    hdul = fits.open(sci_list[0])
    hdr = hdul[0].header

    if catalog_names == []:
        for hdu in hdul:
            if hdu.name != "PRIMARY":
                catalog_names.append(hdu.name)

    # get number of exposures in time series
    nexps = len(sci_list)

    # bool to get time info only once:
    has_time_info = False

    # apertures array
    apertures = {}

    # loop below to get photometry data for all catalogs
    for key in catalog_names:

        logger.info("Packing time series data for catalog: {}".format(key))

        catdata = s4p.readPhotTimeSeriesData(sci_list,
                                             catalog_key=key,
                                             longitude=longitude,
                                             latitude=latitude,
                                             altitude=altitude,
                                             time_keyword=time_key,
                                             time_format=time_format,
                                             time_scale=time_scale,
                                             time_span_for_rms=5,
                                             keys_to_add_header_data=keys_to_add_header_data)

        apertures[key] = fits.getheader(sci_list[0], key)['APRADIUS']

        if not has_time_info:
            # get time array
            times = catdata[catdata["SRCINDEX"] == 0]["TIME"]

            # get first and last times
            tstart = Time(times[0], format='jd', scale='utc')
            tstop = Time(times[-1], format='jd', scale='utc')

            # get number of sources
            nsources = fits.getheader(sci_list[0], key)['NOBJCAT']

            has_time_info = True

        tsdata[key] = catdata

        del catdata

    # Construct information dictionary to add to the header of FITS product
    info = {}

    info['OBSERV'] = ('OPD', 'observatory')
    info['OBSLAT'] = (latitude, '[DEG] observatory latitude (N)')
    info['OBSLONG'] = (longitude, '[DEG] observatory longitude (E)')
    info['OBSALT'] = (altitude, '[m] observatory altitude')
    info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
    info['INSTRUME'] = ('SPARC4', 'instrument')
    info['CHANNEL'] = (hdr["CHANNEL"], 'Instrument channel')
    info['OBJECT'] = (hdr["OBJECT"], 'ID of object of interest')
    equinox = 'J2000.0'
    source = SkyCoord(hdr["RA"], hdr["DEC"], unit=(
        u.hourangle, u.deg), frame='icrs', equinox=equinox)
    info['RA'] = (source.ra.value, '[DEG] RA of object of interest')
    info['DEC'] = (source.dec.value, '[DEG] DEC of object of interest')
    info['RADESYS'] = ('ICRS    ', 'reference frame of celestial coordinates')
    info['EQUINOX'] = (2000.0, 'equinox of celestial coordinate system')
    info['PHZEROP'] = (0., '[mag] photometric zero point')
    info['PHOTSYS'] = ("SPARC4", 'photometric system')
    info['TIMEKEY'] = (time_key, 'keyword used to extract times')
    info['TIMEFMT'] = (time_format, 'time format in img files')
    info['TIMESCL'] = (time_scale, 'time scale')
    if "INSTMODE" in hdr.keys() :
        info['INSTMODE'] = (hdr["INSTMODE"], 'Instrument mode')
    info['TSTART'] = (tstart.jd, 'observation start time in BJD')
    info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
    info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
    info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')
    info['NEXPS'] = (nexps, 'number of exposures')
    info['NSOURCES'] = (nsources, 'number of sources')

    if best_apertures:
        minrms = np.arange(nsources)*0 + 1e20
        selected_apertures = np.zeros_like(minrms)
        selected_data = []
        for i in range(nsources):
            selected_data.append(None)

        # loop over each aperture container
        for key in tsdata.keys():

            # get aperture table data
            tbl = tsdata[key]

            # save minimum rms for each source
            for i in range(nsources):
                loc_tbl = tbl[tbl["SRCINDEX"] == i]
                m_rms = np.nanmedian(loc_tbl['RMS'])
                if m_rms < minrms[i]:
                    minrms[i] = m_rms
                    selected_apertures[i] = apertures[key]
                    loc_tbl["APRADIUS"] = np.full_like(loc_tbl["MAG"], apertures[key])
                    selected_data[i] = loc_tbl

        # add selected data into data container
        tsdata["BEST_APERTURES"] = vstack(selected_data)
        apertures["BEST_APERTURES"] = np.nanmedian(selected_apertures)

    # generate the photometric time series product
    s4p.photTimeSeriesProduct(tsdata, apertures, info=info, filename=output)

    return output


def get_waveplate_angles(wppos):
    """ Pipeline module to get position angles of the wave plate
    given the position index(es)

    Parameters
    ----------
    wppos :
        index for the position angle of the wave plate

    Returns
    -------
    angles : float
        return the angle(s) corresponding to the input position index(es)
    """

    angles = [0., 22.5, 45., 67.5, 90.,
              112.5, 135., 157.5, 180.,
              202.5, 225., 247.5, 270.,
              292.5, 315., 337.5]

    return angles[wppos]


def load_list_of_sci_image_catalogs(sci_list, wppos_key="WPPOS", polarimetry=False):
    """ Pipeline module to load information in the catalog extensions of a list of
    science image products

    Parameters
    ----------
    sci_list : list
        input list of file paths
    wppos_key : str
        image header keyword to get the wave plate position information
    polarimetry : bool
        whether or not in polarimetry mode

    Returns
    -------
    beam or (beam1 and beam2): dict
        dict container to store catalog data
    waveplate_angles : list, only if polarimetry==True
        position angles of the wave plate
    apertures: list
        aperture radii using in the photometry
    nsources: int
        number of sources in the catalog
    """

    nsources = 0

    if polarimetry:
        beam1, beam2 = {}, {}

        waveplate_angles = np.array([])
        apertures = np.array([])

        for i in range(len(sci_list)):

            hdulist = fits.open(sci_list[i])

            wppos = int(hdulist[0].header[wppos_key])
            waveplate_angles = np.append(waveplate_angles, get_waveplate_angles(wppos-1))

            phot1data, phot2data = [], []

            for ext in range(1, len(hdulist)):
                if (ext % 2) != 0:
                    if i == 0 :
                        apertures = np.append(apertures, hdulist[ext].data[0][11])
                        nsources = len(hdulist[ext].data)
                
                    phot1data.append(Table(hdulist[ext].data))
                    phot2data.append(Table(hdulist[ext+1].data))
                else:
                    continue
            beam1[sci_list[i]] = phot1data
            beam2[sci_list[i]] = phot2data

        # return beam1, beam2, waveplate_angles*u.degree, apertures, nsources
        return beam1, beam2, waveplate_angles, apertures, nsources

    else:
        beam = {}

        apertures = np.array([])

        for i in range(len(sci_list)):

            hdulist = fits.open(sci_list[i])

            photdata = []

            for ext in range(1, len(hdulist)):
                if i == 0:
                    apertures = np.append(apertures, hdulist[ext].data[0][11])
                    nsources = len(hdulist[ext].data)

                photdata.append(Table(hdulist[ext].data))

            beam[sci_list[i]] = photdata

        return beam, apertures, nsources


def combine_ts_products_of_polar_beams(ts_product_S, ts_product_N, keys_to_add_header_data=[]) :
    """ Create a combined timeseries FITS product

    Parameters
    ----------
    ts_product_S : astropy.io.fits.HDUList
        photometric time series product for the Southern polar beam
    ts_product_N : astropy.io.fits.HDUList
        photometric time series product for the Northern polar beam
    keys_to_add_header_data : list
        list of keywords to add data from iamge header
    Returns
    -------
    output : str
        output photometric time series product for the sum of S and N beam
    """
    
    # open ts products
    slc = fits.open(ts_product_S)
    nlc = fits.open(ts_product_N)

    # create empty header
    header = slc[0].header
    # create primary hdu for output file
    primary_hdu = fits.PrimaryHDU(header=header)
    # initialize list of hdus with the primary hdu
    hdu_array = [primary_hdu]
    
    # iterate over each aperture FITS extension
    for ext in range(1,len(slc)) :
    
        # get input Table with data
        s_tbl = slc[ext].data
        n_tbl = nlc[ext].data

        # intialize array for output magnitudes
        mags, emags = np.array([]), np.array([])

        # Load magnitudes into QFloat and convert them to flux Qfloats
        s_mag, s_emag = s_tbl["MAG"], s_tbl["EMAG"]
        n_mag, n_emag = n_tbl["MAG"], n_tbl["EMAG"]
        
        for i in range(len(s_mag)) :
        
            s_umag = ufloat(s_mag[i],s_emag[i])
            n_umag = ufloat(n_mag[i],n_emag[i])

            s_uflux = 10**(-0.4*s_umag)
            n_uflux = 10**(-0.4*n_umag)

            uflux = s_uflux + n_uflux
            umag = uflux_to_magnitude(uflux)
            
            mags = np.append(mags, umag.nominal_value)
            emags = np.append(emags, umag.std_dev)
            
        # initialize time series data container
        tsdata = {}
        
        # set unchanged quantities
        tsdata["TIME"] = s_tbl["TIME"]
        tsdata["SRCINDEX"] = s_tbl["SRCINDEX"]
        tsdata["RA"] = s_tbl["RA"]
        tsdata["DEC"] = s_tbl["DEC"]
        
        # Calculate mean detector coordinates (photocenter would be better!)
        tsdata["X"] = (s_tbl["X"] + n_tbl["X"]) / 2
        tsdata["Y"] = (s_tbl["Y"] + n_tbl["Y"]) / 2
        
        # Calculate mean FWHM
        tsdata["FWHM"] = (s_tbl["FWHM"] + n_tbl["FWHM"]) / 2
        
        # set output MAG and EMAG with combined values
        tsdata["MAG"] = mags
        tsdata["EMAG"] = emags
        
        # set sky mags as those from the south beam
        tsdata["SKYMAG"] = s_tbl["SKYMAG"]
        tsdata["ESKYMAG"] = s_tbl["SKYMAG"]
        
        tsdata["FLAG"] = s_tbl["FLAG"] + n_tbl["FLAG"]
        # Sum RMS in quadrature
        tsdata["RMS"] = np.sqrt(s_tbl["RMS"]*s_tbl["RMS"] + n_tbl["RMS"]*n_tbl["RMS"])

        for i in range(len(keys_to_add_header_data)) :
            tsdata[keys_to_add_header_data[i]] = s_tbl[keys_to_add_header_data[i]]

        # set catalog key
        catalog_key = slc[ext].name.replace("POL_S","PHOT")
        
        # create catalog photometry hdu
        catalog_phot_hdu = fits.BinTableHDU(data=Table(tsdata), header=slc[ext].header, name=catalog_key)

        # append hdu
        hdu_array.append(catalog_phot_hdu)
            
    # set output file name
    output = ts_product_S.replace("_S_lc.fits","_S+N_lc.fits")
    
    # create hdu list
    hdu_list = s4p.create_hdu_list(hdu_array)
    
    # write output fits ts product
    hdu_list.writeto(output, overwrite=True, output_verify="fix+warn")
    
    return output
    
                
def nan_proof_keyword(value):
    if np.isnan(value):
        return "NaN"
    elif np.isinf(value):
        return "inf"
    else:
        return value



def get_qflux(beam, filename, aperture_index, source_index, magkey="MAG", emagkey="EMAG"):
    """ Pipeline module to get catalog fluxes into Qfloat

    Parameters
    ----------
    beam : dict
        dict container to store catalog data
    filename : str
        file name  to select only data for a given exposure
    aperture_index : int
        aperture index to select only data for a given aperture
    source_index : int
        source index to select only data for a given source
    magkey : str
        magnitude keyword
    emagkey : str
        magnitude error keyword

    Returns
    -------
    qflux : Qfloat
        return flux+/-flux_err information from input catalog
    """

    qflux = QFloat(np.nan, np.nan)
    
    try :
        umag = ufloat(beam[filename][aperture_index][magkey][source_index],beam[filename][aperture_index][emagkey][source_index])

        uflux = 10**(-0.4*umag)

        qflux = QFloat(uflux.nominal_value, uflux.std_dev)

    except Exception as e:
        logger.warn("Could not retrieve flux data for file={} aperture_index={} source_index={} : {}".format(filename,aperture_index,source_index,e))

    return qflux


def get_photometric_data_for_polar_catalog(beam1, beam2, sci_list, aperture_index=8, source_index=0):
    """ Pipeline module to get photometric data from the catalogs of a
    list of polarimetry exposures

    Parameters
    ----------
    beam1, beam2 : dict
        dict containers to store photometry catalog data
    sci_list : list
        input list of file paths
    aperture_index : int
        aperture index to select only data for a given aperture
    source_index : int
        source index to select only data for a given source

    Returns
    -------
    ra : numpy.ndarray (N sources)
        float array containing the right ascension [DEG] data for all sources
    dec : numpy.ndarray (N sources)
        float array containing the declination [DEG] data for all sources
    x1, y1, x2, y2: numpy.ndarray (N sources)
        float arrays containing the x- and y-positions [pix] data for all sources
    mag.nominal : numpy.ndarray (N sources)
        float array containing the instrumental magnitude data for all sources
    mag.std_dev : numpy.ndarray (N sources)
        float array containing the instrumental magnitude error data for all sources
    fwhm : float
        median full-width at half-maximum (pix)
    skymag.nominal : numpy.ndarray (N sources)
        float array containing the mean instrumental sky magnitude data for all sources
    skymag.std_dev : numpy.ndarray (N sources)
        float array containing the mean instrumental sky magnitude error data for all sources
    flag : int
        sum of photometric flags

    """

    i, j = int(aperture_index), int(source_index)

    ra, dec = np.nan, np.nan
    x1, y1 = np.nan, np.nan
    x2, y2 = np.nan, np.nan

    flux1, flux2 = QFloat(0, 0), QFloat(0, 0)
    skyflux1, skyflux2 = QFloat(0, 0), QFloat(0, 0)

    fwhms = np.array([])

    flag1, flag2 = 0, 0

    for k in range(len(sci_list)):

        if k == 0:
            ra, dec = beam1[sci_list[0]][i]["RA"][j], beam1[sci_list[0]][i]["DEC"][j]
            x1, y1 = beam1[sci_list[0]][i]["X"][j], beam1[sci_list[0]][i]["Y"][j]
            x2, y2 = beam2[sci_list[0]][i]["X"][j], beam2[sci_list[0]][i]["Y"][j]

        flux1 += get_qflux(beam1, sci_list[k], i, j)
        flux2 += get_qflux(beam2, sci_list[k], i, j)

        skyflux1 += get_qflux(beam1, sci_list[k],i, j, magkey="SKYMAG", emagkey="ESKYMAG")
        skyflux2 += get_qflux(beam2, sci_list[k],i, j, magkey="SKYMAG", emagkey="ESKYMAG")

        flag1 += beam1[sci_list[k]][i]["FLAG"][j]
        flag2 += beam2[sci_list[k]][i]["FLAG"][j]
            
        fwhms = np.append(fwhms, beam1[sci_list[k]][i]["FWHMX"][j])
        fwhms = np.append(fwhms, beam1[sci_list[k]][i]["FWHMY"][j])
        fwhms = np.append(fwhms, beam2[sci_list[k]][i]["FWHMX"][j])
        fwhms = np.append(fwhms, beam2[sci_list[k]][i]["FWHMY"][j])

    fwhm = np.nanmedian(fwhms)

    flux = flux1 + flux2
    mag = -2.5 * np.log10(flux)

    skyflux = (skyflux1 + skyflux2) / (2 * len(sci_list))
    skymag = -2.5 * np.log10(skyflux)

    flag = flag1 + flag2

    return ra, dec, x1, y1, x2, y2, mag.nominal, mag.std_dev, fwhm, skymag.nominal, skymag.std_dev, flag


def compute_polarimetry(sci_list, output_filename="", wppos_key='WPPOS', save_output=True, wave_plate='halfwave', compute_k=True, fit_zero=False, zero=0, base_aperture=8, exptimekey="EXPTIME", min_n_wppos=4, force=False):

    """ Pipeline module to compute polarimetry for given polarimetric sequence and
        saves the polarimetry data into a FITS SPARC4 product

    Parameters
    ----------
    sci_list : list
        input list of file paths
    output_filename : str (optional)
        file path to save output product, if not given,
         an automatic path convention will be used
    wppos_key : str
        image header keyword to get the wave plate position information
    wave_plate : str
        type of wave plate used in the sequence.
        Accepted values are: "halfwave" or "quarterwave"
    compute_k : bool
        whether or not to include a constant "k" in the polarization model
    zero : float
        zero of polarization in degrees
    fit_zero : bool
        whether or not to fit zero of polarization
    base_aperture : int
        base aperture index
    exptimekey : str
        header keyword for exposure time
    min_n_wppos : int (Deafult: min_n_wppos=4)
        minimum number of waveplate positions acceptable for polimetry
    force : bool, optional
        Boolean to decide whether or not to force reduction if a product already exists

    Returns
    -------
    output_filename : str
        file path for the output product
    """

    # define output polarimetry product file name
    if output_filename == "":
        if wave_plate == 'halfwave':
            output_filename = sci_list[0].replace("_proc.fits", "_l2_polar.fits")
        elif wave_plate == 'quarterwave':
            output_filename = sci_list[0].replace("_proc.fits", "_l4_polar.fits")
        else:
            logger.error("Wave plate mode not supported, exiting ...")
            exit()

    if os.path.exists(output_filename) and not force:
        logger.info("There is already a polarimetry product :".format(output_filename))
        return output_filename

    # get data from a list of science image products
    beam1, beam2, waveplate_angles, apertures, nsources = load_list_of_sci_image_catalogs(sci_list, wppos_key=wppos_key, polarimetry=True)

    logger.info("Number of sources in catalog: {}".format(nsources))
    logger.info("Number of apertures: {}  varying from {} to {} in steps of {} pix".format(len(apertures), apertures[0], apertures[-1], np.abs(np.nanmedian(apertures[1:]-apertures[:-1]))))

    # set number of free parameters equals 2: u and q
    number_of_free_params = 2

    if wave_plate == 'halfwave':
        zero = 0

    # add one parameter: v
    if wave_plate == 'quarterwave':
        number_of_free_params += 1

    # if zero is free, then add another parameter
    if fit_zero:
        zero = None
        # we do not fit zero and k simultaneously
        compute_k = False
        number_of_free_params += 1

    # initialize astropop SLSDualBeamPolarimetry object
    pol = SLSDualBeamPolarimetry(wave_plate, compute_k=compute_k, zero=zero, iter_tolerance=1e-6)

    # create an empty dict to store polar catalogs
    polar_catalogs = {}

    # set names and data format for each column in the catalog table
    variables = ['APERINDEX', 'APER', 'SRCINDEX',
                 'RA', 'DEC', 'X1', 'Y1', 'X2', 'Y2',
                 'FWHM', 'MAG', 'EMAG',
                 'SKYMAG', 'ESKYMAG', 'PHOTFLAG',
                 'Q', 'EQ', 'U', 'EU', 'V', 'EV',
                 'P', 'EP', 'THETA', 'ETHETA',
                 'K', 'EK', 'ZERO', 'EZERO', 'NOBS', 'NPAR',
                 'CHI2', 'RMS', 'TSIGMA', 'POLARFLAG']
                 
    for i in range(len(sci_list)):
        variables.append('FO{:04d}'.format(i))
        variables.append('EFO{:04d}'.format(i))
        variables.append('FE{:04d}'.format(i))
        variables.append('EFE{:04d}'.format(i))

    # Initialize container to store polarimetric data
    aperture_keys = []
    for i in range(len(apertures)):
        key = "POLARIMETRY_AP{:03d}".format(int(apertures[i]))
        aperture_keys.append(key)
        polar_catalogs[key] = {}
        for var in variables:
            polar_catalogs[key][var] = np.array([])

    # loop over each aperture
    for i in range(len(apertures)):

        logger.info("Calculating {} polarimetry for aperture {} of {}".format(wave_plate, i+1, len(apertures)))

        # loop over each source in the catalog
        for j in range(nsources):

            # retrieve photometric information in a pair of polar catalog
            ra, dec, x1, y1, x2, y2, mag, mag_err, fwhm, skymag, skymag_err, photflag = get_photometric_data_for_polar_catalog(beam1, beam2, sci_list, aperture_index=i, source_index=j)

            n_fo, n_fe = [], []
            en_fo, en_fe = [], []

            for filename in sci_list:
                # get fluxes as qfloats
                fo = get_qflux(beam1, filename, i, j)
                fe = get_qflux(beam2, filename, i, j)
                # print(filename, i, j, fo, fe)
                n_fo.append(fo.nominal)
                n_fe.append(fe.nominal)
                en_fo.append(fo.std_dev)
                en_fe.append(fe.std_dev)

            n_fo, n_fe = np.array(n_fo), np.array(n_fe)
            en_fo, en_fe = np.array(en_fo), np.array(en_fe)

            zi, zi_err = np.full_like(n_fo, np.nan), np.full_like(n_fo, np.nan)

            keep = np.isfinite(n_fo)
            keep &= np.isfinite(n_fe)
            keep &= (n_fo > np.nanmedian(en_fo)) & (n_fe > np.nanmedian(en_fe))
            keep &= (en_fo > 0) & (en_fe > 0)

            valid_sci_list = []
            for k in range(len(sci_list)):
                if keep[k]:
                    valid_sci_list.append(sci_list[k])

            number_of_observations = len(valid_sci_list)

            polar_flag = 1
            chi2 = np.nan
            qpol, q_err = np.nan, np.nan
            upol, u_err = np.nan, np.nan
            vpol, v_err = np.nan, np.nan
            ptot, ptot_err = np.nan, np.nan
            theta, theta_err = np.nan, np.nan
            k_factor, k_factor_err = np.nan, np.nan
            zero, zero_err = np.nan, np.nan
            z_rms, theor_sigma = np.nan, np.nan
            
            observed_model = np.full_like(waveplate_angles[keep], np.nan)

            try:
                #logger.info("Computing polarimetry for the following flux array sizes: {} ".format(len(waveplate_angles[keep])))
                
                # compute polarimetry
                if len(waveplate_angles[keep]) >= min_n_wppos :
                    norm = pol.compute(waveplate_angles[keep], n_fo[keep], n_fe[keep], f_ord_error=en_fo[keep], f_ext_error=en_fe[keep])

                    if wave_plate == 'halfwave':
                        #logger.info("Computing halfwave (L2) observed polarization model for q={} u={} ".format(norm.q.nominal, norm.u.nominal))
                    
                        observed_model = halfwave_model(waveplate_angles[keep], norm.q.nominal, norm.u.nominal)

                    elif wave_plate == 'quarterwave':
                        #logger.info("Computing quarterwave (L4) observed polarization model for q={} u={} v={} zero={}".format(norm.q.nominal, norm.u.nominal, norm.v.nominal, norm.zero.nominal))
                    
                        observed_model = quarterwave_model(waveplate_angles[keep], norm.q.nominal, norm.u.nominal, norm.v.nominal, zero=norm.zero.nominal)

                    zi[keep] = norm.zi.nominal
                    zi_err[keep] = norm.zi.std_dev

                    #logger.info("Computing chi-square for number_of_observations={} and number_of_free_params={}".format(number_of_observations,number_of_free_params))
                    rms = np.sqrt(np.nanmean((norm.zi.nominal - observed_model)**2))
                    chi2 = np.nansum(((norm.zi.nominal - observed_model)/norm.zi.std_dev)** 2) / (number_of_observations - number_of_free_params)

                    polar_flag = 0

                    if type(norm.q.nominal) is float :
                        qpol, q_err = norm.q.nominal, norm.q.std_dev
                    
                    if type(norm.u.nominal) is float :
                        upol, u_err = norm.u.nominal, norm.u.std_dev
                    
                    if wave_plate == 'quarterwave' and type(norm.v.nominal) is float:
                        vpol, v_err = norm.v.nominal, norm.v.std_dev
                    
                    if type(norm.p.nominal) is float :
                        ptot, ptot_err = norm.p.nominal, norm.p.std_dev
                    
                    if type(norm.theta.nominal) is float :
                        theta, theta_err = norm.theta.nominal, norm.theta.std_dev
    
                    if type(norm.k) is float :
                        k_factor = norm.k
                        
                    if type(norm.zero.nominal) is float :
                        zero, zero_err = norm.zero.nominal, norm.zero.std_dev
       
                    #if type(norm.rms) is float :
                    #    z_rms = norm.rms
                    if type(rms) is float :
                        z_rms = rms
                    
                    if type(norm.theor_sigma['p']) is float :
                        theor_sigma = norm.theor_sigma['p']

            except Exception as e:
                logger.warn("Could not compute polarimetry for source_index={} and aperture={} pixels: {}".format(j, apertures[i], e))

            var_values = [i, apertures[i], j,
                          ra, dec, x1, y1, x2, y2,
                          fwhm, mag, mag_err,
                          skymag, skymag_err, photflag,
                          qpol, q_err,
                          upol, u_err,
                          vpol, v_err,
                          ptot, ptot_err,
                          theta, theta_err,
                          k_factor, k_factor_err,
                          zero, zero_err,
                          number_of_observations, number_of_free_params,
                          chi2, z_rms, theor_sigma, polar_flag]
            
            for ii in range(len(n_fo)):
                var_values.append(n_fo[ii])
                var_values.append(en_fo[ii])
                var_values.append(n_fe[ii])
                var_values.append(en_fe[ii])

            for ii in range(len(variables)):
                polar_catalogs[aperture_keys[i]][variables[ii]] = np.append(polar_catalogs[aperture_keys[i]][variables[ii]], var_values[ii])

    if save_output:
        info = {}
        hdr_start = fits.getheader(sorted(sci_list)[0])
        hdr_end = fits.getheader(sorted(sci_list)[-1])
        if "OBJECT" in hdr_start.keys():
            info['OBJECT'] = (hdr_start["OBJECT"], hdr_start.comments["OBJECT"])
        if "OBSLAT" in hdr_start.keys():
            info['OBSLAT'] = (hdr_start["OBSLAT"], hdr_start.comments["OBSLAT"])
        if "OBSLONG" in hdr_start.keys():
            info['OBSLONG'] = (hdr_start["OBSLONG"], hdr_start.comments["OBSLONG"])
        if "OBSALT" in hdr_start.keys():
            info['OBSALT'] = (hdr_start["OBSALT"], hdr_start.comments["OBSALT"])
        info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
        if "INSTRUME" in hdr_start.keys():
            info['INSTRUME'] = (hdr_start["INSTRUME"], hdr_start.comments["INSTRUME"])
        if "EQUINOX" in hdr_start.keys():
            info['EQUINOX'] = (hdr_start["EQUINOX"], hdr_start.comments["EQUINOX"])
        info['PHZEROP'] = (0., '[mag] photometric zero point')
        info['PHOTSYS'] = ("SPARC4", 'photometric system')
        if "CHANNEL" in hdr_start.keys():
            info['CHANNEL'] = (hdr_start["CHANNEL"], hdr_start.comments["CHANNEL"])
        info['POLTYPE'] = (wave_plate, 'polarimetry type l/2 or l/4')

        tstart = Time(hdr_start["BJD"], format='jd', scale='utc')

        exptime = s4utils.get_exptime(hdr_end,exptimekey=exptimekey)

        tstop = Time(hdr_end["BJD"]+exptime/(24*60*60),format='jd', scale='utc')

        info['TSTART'] = (tstart.jd, 'obs time of first exposure in BJD')
        info['TSTOP'] = (tstop.jd, 'obs time of last exposure in BJD')
        info['MEANBJD'] = ((tstop.jd + tstart.jd)/2, 'observation mean time in BJD')
        info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
        info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')
        info['NSOURCES'] = (nsources, 'number of sources')
        info['NEXPS'] = (len(sci_list), 'number of exposures in sequence')

        for k in range(len(sci_list)):
            hdr = fits.getheader(sci_list[k])
            info["FILE{:04d}".format(k)] = (os.path.basename(sci_list[k]), 'file name of exposure')
            info["EXPT{:04d}".format(k)] = (exptime, 'exposure time (s)')
            info["BJD{:04d}".format(k)] = (hdr["BJD"], hdr.comments["BJD"])
            info["WPPO{:04d}".format(k)] = (hdr[wppos_key],  hdr.comments[wppos_key])
            info["WANG{:04d}".format(k)] = (waveplate_angles[k], 'WP angle of exposure (deg)')
            if "CALW" in hdr :
                info["CALW{:04d}".format(k)] = (hdr["CALW"], 'Selected element in calibration wheel ')

        logger.info("Saving output {} polarimetry product: {}".format(wave_plate, output_filename))
        output_hdul = s4p.polarProduct(polar_catalogs, info=info, filename=output_filename)

    return output_filename


def get_polarimetry_results(filename, source_index=0, aperture_radius=None, min_aperture=0, max_aperture=1024, compute_k=False, k=None, k_err=None, zero=None, zero_err=None, min_n_wppos=4, plot=False, verbose=False, plot_filename='', figsize=(12, 6)):

    """ Pipeline module to compute polarimetry for given polarimetric product.
        It returns the polarimetry results obtained using the input parameters as a python dict.

    Parameters
    ----------
    filename : str
        file path for a polarimetry product
    source_index : int
        source index to select only data for a given source
    aperture_radius : float
        to select a given aperture radius, if not provided
        then the minimum chi-square solution will be adopted
    min_aperture : float
        minimum aperture radius (pix)
    max_aperture : float
        minimum aperture radius (pix)
    compute_k: bool
        whether or not to compute k
    k: float
        normalization factor "k"
    k_err: float
        uncertainty of the normalization factor "k"
    zero: float
        zero of waveplate
    zero_err: float
        uncertainty of the zero of waveplate
    min_n_wppos : int (Deafult: min_n_wppos=4)
        minimum number of waveplate positions acceptable for polimetry
    plot: bool
        whether or not to plot results
    plot_filename : str, optional
        The output plot file name to save graphic to file. If empty, it won't be saved.
    figsize : (int,int)
        Horizontal and vertical figure size
    Returns
    -------
    loc : dict
        container to store polarimetry results for given target and aperture
    """

    loc = {}
    loc["POLAR_PRODUCT"] = filename
    loc["SOURCE_INDEX"] = source_index

    loc["POLARIMETRY_SUCCESS"] = False
    
    # open polarimetry product FITS file
    hdul = fits.open(filename)
    wave_plate = hdul[0].header['POLTYPE']

    # initialize aperture index
    aperture_index = 1

    # if an aperture index is not given, then consider the one with minimum chi2
    if aperture_radius is None:
        minchi2 = 1e30
        for i in range(len(hdul)):
            if hdul[i].name == 'PRIMARY' or hdul[i].header['APRADIUS'] < min_aperture or hdul[i].header['APRADIUS'] > max_aperture:
                continue
            curr_chi2 = hdul[i].data[hdul[i].data['SRCINDEX'] == source_index]['CHI2'][0]
            if curr_chi2 < minchi2:
                minchi2 = curr_chi2
                aperture_index = i
    else:
        minapdiff = 1e30
        for i in range(len(hdul)):
            if hdul[i].name == 'PRIMARY':
                continue
            curr_apdiff = np.abs(hdul[i].header['APRADIUS'] - aperture_radius)
            if curr_apdiff < minapdiff:
                minapdiff = curr_apdiff
                aperture_index = i

    # get selected aperture radius
    aperture_radius = hdul[aperture_index].header['APRADIUS']
    loc["APERTURE_INDEX"] = aperture_index
    loc["APERTURE_RADIUS"] = aperture_radius
    # isolate data for selected aperture and target
    tbl = hdul[aperture_index].data[hdul[aperture_index].data['SRCINDEX'] == source_index]

    # get number of exposure in the polarimetric sequence
    nexps = hdul[0].header['NEXPS']
    loc["NEXPS"] = nexps

    # get source magnitude
    mag = QFloat(float(tbl["MAG"][0]), float(tbl["EMAG"][0]))
    loc["MAG"] = mag

    # get source coordinates
    ra, dec = tbl["RA"][0], tbl["DEC"][0]

    loc["RA"] = ra
    loc["DEC"] = dec
    loc["FWHM"] = tbl["FWHM"][0]
    loc["X1"] = tbl["X1"][0]
    loc["Y1"] = tbl["Y1"][0]
    loc["X2"] = tbl["X2"][0]
    loc["Y2"] = tbl["Y2"][0]

    # get polarization data and the WP position angles
    fos, efos = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    fes, efes = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    zis, zierrs = np.arange(nexps)*np.nan, np.arange(nexps)*np.nan
    waveplate_angles = np.arange(nexps)*np.nan

    for ii in range(nexps):
        if np.isfinite(tbl["FO{:04d}".format(ii)]) :
            fos[ii] = tbl["FO{:04d}".format(ii)]
        if np.isfinite(tbl["EFO{:04d}".format(ii)]) :
            efos[ii] = tbl["EFO{:04d}".format(ii)]
        if np.isfinite(tbl["FE{:04d}".format(ii)]) :
            fes[ii] = tbl["FE{:04d}".format(ii)]
        if np.isfinite(tbl["EFE{:04d}".format(ii)]) :
            efes[ii] = tbl["EFE{:04d}".format(ii)]
        waveplate_angles[ii] = hdul[0].header["WANG{:04d}".format(ii)]

    # filter out nan data
    keep = np.isfinite(waveplate_angles)
    keep &= (np.isfinite(fos)) & (np.isfinite(fes))
    keep &= (np.isfinite(efos)) & (np.isfinite(efes))

    # get polarimetry results
    qpol = QFloat(np.nan, np.nan)
    upol = QFloat(np.nan, np.nan)
    vpol = QFloat(np.nan, np.nan)
    ppol = QFloat(np.nan, np.nan)
    theta = QFloat(np.nan, np.nan)
    kcte = QFloat(np.nan, np.nan)
    qzero = QFloat(np.nan, np.nan)
    
    if k is not None :
        kcte.nominal = k
    if k_err is not None :
        kcte.std_dev = k_err

    if zero is not None :
        qzero.nominal = zero
    if zero_err is not None :
        qzero.std_dev = zero_err
        
    # cast zi data into QFloat
    fo_out = QFloat(fos, efos)
    fe_out = QFloat(fes, efes)
    zi = QFloat(zis, zierrs)
    n, m = 0, 0
    chi2 = np.nan
    rms = np.nan
    theor_sigma = np.nan
    observed_model = QFloat(np.arange(nexps)*np.nan, np.arange(nexps)*np.nan)
        
    if len(fos[keep]) != 0 :
    
        # get polarimetry results
        qpol = QFloat(tbl['Q'][0], tbl['EQ'][0])
        upol = QFloat(tbl['U'][0], tbl['EU'][0])
        vpol = QFloat(tbl['V'][0], tbl['EV'][0])
        ppol = QFloat(tbl['P'][0], tbl['EP'][0])
        theta = QFloat(tbl['THETA'][0], tbl['ETHETA'][0])
        
        if not np.isfinite(kcte.nominal) :
            kcte.nominal = tbl['K'][0]
        if not np.isfinite(kcte.std_dev) :
            kcte.std_dev = tbl['EK'][0]
        
        if not np.isfinite(qzero.nominal) :
            qzero.nominal = tbl['ZERO'][0]
        if not np.isfinite(qzero.std_dev) :
            qzero.std_dev = tbl['EZERO'][0]
            
        rms, theor_sigma = tbl['RMS'][0], tbl['TSIGMA'][0]
        n, m = tbl['NOBS'][0], tbl['NPAR'][0]

        # cast flux data into QFloat
        fo = QFloat(fos[keep], efos[keep])
        fe = QFloat(fes[keep], efes[keep])
        
        # cast flux data into QFloat keeping nans
        fo_out = QFloat(fos, efos)
        fe_out = QFloat(fes, efes)
    
        k_value = kcte.nominal
        if compute_k :
            k_value = None
            
        # calculate polarimetry model and get statistical quantities
        observed_model = np.full_like(waveplate_angles, np.nan)
        
        if wave_plate == "halfwave":
            # initialize astropop SLSDualBeamPolarimetry object
            pol = SLSDualBeamPolarimetry(wave_plate, compute_k=compute_k, k=k_value, zero=0)
            
            try :
                if len(waveplate_angles[keep]) >= min_n_wppos and np.isfinite(qpol.nominal) and np.isfinite(upol.nominal):
                    norm = pol.compute(waveplate_angles[keep], fos[keep], fes[keep], f_ord_error=efos[keep], f_ext_error=efes[keep])
                
                    # update polarimetric results
                    qpol = norm.q
                    upol = norm.u
                    vpol = norm.v
                    ppol = norm.p
                    theta = norm.theta
                    if compute_k :
                        kcte.nominal = norm.k
                    
                    observed_model[keep] = halfwave_model(waveplate_angles[keep], qpol.nominal, upol.nominal)
                    
                    # get zis
                    zis[keep] = norm.zi.nominal
                    zierrs[keep] = norm.zi.std_dev
                    theor_sigma = norm.theor_sigma['p']
                    
            except Exception as e :
                logger.warn("Could not compute polarimetry: {}".format(e))
                pass
                
        elif wave_plate == "quarterwave":
            # initialize astropop SLSDualBeamPolarimetry object
            pol = SLSDualBeamPolarimetry(wave_plate, compute_k=compute_k, k=k_value, zero=qzero.nominal)
            
            try :
                if len(waveplate_angles[keep]) >= min_n_wppos and np.isfinite(qpol.nominal) and np.isfinite(upol.nominal) and np.isfinite(vpol.nominal):
                    norm = pol.compute(waveplate_angles[keep], fos[keep], fes[keep], f_ord_error=efos[keep], f_ext_error=efes[keep])
            
                    # update polarimetric results
                    qpol = norm.q
                    upol = norm.u
                    vpol = norm.v
                    ppol = norm.p
                    theta = norm.theta
                    if compute_k :
                        kcte.nominal = norm.k
                
                    #observed_model[keep] = quarterwave_model(waveplate_angles[keep], qpol.nominal, upol.nominal, vpol.nominal, zero=qzero.nominal)
                    observed_model[keep] = norm.model(norm.psi)
                    
                    # get zis
                    zis[keep] = norm.zi.nominal
                    zierrs[keep] = norm.zi.std_dev
                    theor_sigma = norm.theor_sigma['p']
                    
            except Exception as e :
                logger.warn("Could not compute polarimetry: {}".format(e))
                pass

        # cast zi data into QFloat
        zi = QFloat(zis, zierrs)
        # get statistics
        rms = np.sqrt(np.nanmean((zis[keep] - observed_model[keep])**2))
        chi2 = np.nansum(((zis[keep] - observed_model[keep])/zierrs[keep])**2) / (n - m)
        
    else :
        logger.warn("No useful polarization data for Source index: {}  and aperture: {} pix ".format(source_index, aperture_radius))


    # print results
    if verbose:
        logger.info("Source index: i={} ".format(source_index))
        logger.info("Source RA={} Dec={} mag={}".format(ra, dec, mag))
        logger.info("Best aperture radius: {} pixels".format(aperture_radius))
        logger.info("Polarization in Q: {}".format(qpol))
        logger.info("Polarization in U: {}".format(upol))
        logger.info("Polarization in V: {}".format(vpol))
        logger.info("Total linear polarization p: {}".format(ppol))
        logger.info("Angle of polarization theta: {}".format(theta))
        logger.info("Free constant k: {}".format(kcte))
        logger.info("Zero of polarization: {}".format(qzero))
        logger.info("RMS of zi residuals: {}".format(rms))
        logger.info("Reduced chi-square (n={}, DOF={}): {:.2f}".format(n, n-m, chi2))

    loc["WAVEPLATE_ANGLES"] = waveplate_angles
    loc["ZI"] = zi
    loc["FO"] = fo_out
    loc["FE"] = fe_out
    loc["OBSERVED_MODEL"] = observed_model
    loc["Q"] = qpol
    loc["U"] = upol
    loc["V"] = vpol
    loc["P"] = ppol
    loc["THETA"] = theta
    loc["K"] = kcte
    loc["ZERO"] = qzero
    loc["CHI2"] = chi2
    loc["RMS"] = rms
    loc["TSIGMA"] = theor_sigma
    loc["NOBS"] = n
    loc["NPAR"] = m

    if wave_plate == "halfwave" :
        if len(waveplate_angles[keep]) and np.isfinite(loc["P"].nominal) :
            loc["POLARIMETRY_SUCCESS"] = True
    elif wave_plate == "quarterwave" :
        if len(waveplate_angles[keep]) and np.isfinite(loc["P"].nominal) and np.isfinite(loc["V"].nominal)  :
            loc["POLARIMETRY_SUCCESS"] = True
            
    polar_mode_label = "L2"
    if wave_plate == "quarterwave" :
        polar_mode_label = "L4"

    channel_index = int(hdul[0].header['CHANNEL'])
    bands = ["g","r","i","z"]

    title_label = r"Object: {}  date: {}  mode: {} S4C{} ({}-band)".format(hdul[0].header['OBJECT'],hdul[0].header['DATE-OBS'][:10],polar_mode_label,channel_index,bands[channel_index-1])
    title_label += "\n"
    # set title to appear in the plot header
    title_label += r"Source index: {}    aperture: {} pix    $\chi^2$: {:.2f}    RMS: {:.6f}".format(source_index, aperture_radius, chi2, rms)

    loc["TITLE_LABEL"] = title_label
    loc["WAVE_PLATE"] = wave_plate
    
    # plot polarization data and best-fit model
    if plot and loc["POLARIMETRY_SUCCESS"] :
        try :
            s4plt.plot_polarimetry_results(loc, title_label=title_label, wave_plate=wave_plate, output=plot_filename, figsize=figsize)
        except Exception as e:
            logger.warn("Could not generate polarimetry plot : {}".format(e))
            
    hdul.close()

    return loc


def psf_analysis(filename, aperture=10, half_windowsize=15, nsources=0, percentile=99.5, polarimetry=False, plot=False, verbose=False):
    """ Pipeline module to compute point spread function analysis and
        save the PSF data into a FITS SPARC4 product

    Parameters
    ----------
    filename : str
        file path for a science image product
    plot: bool
        whether or not to plot results

    Returns
    -------
    loc : dict
        container to store polarimetry results for given target and aperture
    """

    loc = {}
    loc["PSF_PRODUCT"] = filename

    # open science image product FITS file
    hdulist = fits.open(filename)
    img_data = hdulist[0].data[0]
    indices = np.indices(img_data.shape)

    # get pixel scale from the WCS and convert it to arcsec
    wcs_obj = WCS(hdulist[0].header, naxis=2)
    pixel_scale = proj_plane_pixel_scales(wcs_obj)
    pixel_scale *= 3600
    logger.info("Pixel scale: x: {:.3f} arcsec/pix y: {:.3f} arcsec/pix".format(pixel_scale[0], pixel_scale[1]))

    if polarimetry:

        catN_label = "CATALOG_POL_N_AP{:03d}".format(aperture)
        catS_label = "CATALOG_POL_S_AP{:03d}".format(aperture)
        photN_data = Table(hdulist[catN_label].data)
        photS_data = Table(hdulist[catS_label].data)

        nsources = len(hdulist[catN_label].data)

        fwhms = np.array([])

        for j in range(nsources):
            xN, yN = photN_data["X"][j], photN_data["Y"][j]
            xS, yS = photS_data["X"][j], photS_data["Y"][j]

            fwhms = np.append(fwhms, photN_data["FWHMX"][j])
            fwhms = np.append(fwhms, photN_data["FWHMY"][j])
            fwhms = np.append(fwhms, photS_data["FWHMX"][j])
            fwhms = np.append(fwhms, photS_data["FWHMY"][j])

    else:

        cat_label = "CATALOG_PHOT_AP{:03d}".format(aperture)
        phot_data = Table(hdulist[cat_label].data)

        if nsources == 0 or nsources > len(hdulist[cat_label].data):
            nsources = len(hdulist[cat_label].data)

        fwhms = np.array([])

        boxes = np.zeros([nsources, 2*half_windowsize+1,
                         2*half_windowsize+1]) * np.nan
        xbox, ybox = None, None
        for j in range(nsources):
            x, y = phot_data["X"][j], phot_data["Y"][j]
            fwhms = np.append(fwhms, phot_data["FWHMX"][j])
            fwhms = np.append(fwhms, phot_data["FWHMY"][j])

            # get  box around source
            box = trim_array(img_data, 2*half_windowsize,
                             (x, y), indices=indices)
            zbox = box[0]

            nx, ny = np.shape(zbox)

            if nx == 2*half_windowsize+1 and ny == 2*half_windowsize+1:
                maxvalue = np.nanmax(zbox)
                nzbox = zbox/maxvalue
                # print(j, maxvalue, np.nanmedian(zbox))
                boxes[j, :, :] = nzbox
                xbox, ybox = box[1]-x, box[2]-y

                # vmin, vmax = np.percentile(nzbox, 1.), np.percentile(nzbox, 99.)
                # s4plt.plot_2d(xbox, ybox, nzbox, LIM=None, LAB=["x (pix)", "y (pix)","flux fraction"], z_lim=[vmin,vmax], title="source: {}".format(j), pfilename="", cmap="gist_heat")

        master_box = np.nanmedian(boxes, axis=0)
        master_box_err = np.nanmedian(np.abs(boxes - master_box), axis=0)

        min = np.percentile(master_box, 0.5)
        master_box = master_box - min

        max = np.nanmax(master_box)
        master_box /= max
        master_box_err /= max

        # vmin, vmax = np.percentile(master_box, 3.), np.percentile(master_box, 97.)
        # s4plt.plot_2d(xbox*0.33, ybox*0.33, master_box, LIM=None, LAB=["x (arcsec)", "y (arcsec)","flux fraction"], z_lim=[vmin,vmax], title="PSF", pfilename="", cmap="gist_heat")

        master_fwhm = _fwhm_loop('gaussian', master_box, xbox, ybox, 0, 0)

        logger.info("Median FWHM: {:.3f} pix   Master PSF FWHM: {:.3f} pix".format(np.median(fwhms), master_fwhm))

        # multiply x and y by the pixel scale
        xbox *= pixel_scale[0]
        ybox *= pixel_scale[1]

        # plot PSF results
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False, gridspec_kw={
                                 'hspace': 0.5, 'height_ratios': [1, 1]})

        # plot PSF data
        vmin, vmax = np.percentile(
            master_box, 10), np.percentile(master_box, 99)
        axes[0, 0].pcolor(xbox, ybox, master_box, vmin=vmin,
                          vmax=vmax, shading='auto', cmap="cividis")
        axes[0, 0].plot(xbox[half_windowsize], np.zeros_like(
            xbox[half_windowsize]), '--', lw=1., color='white', zorder=3)
        axes[0, 0].plot(np.zeros_like(ybox[:, half_windowsize]),
                        ybox[:, half_windowsize], '--', lw=1., color='white', zorder=3)
        axes[0, 0].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[0, 0].set_ylabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[0, 0].set_title("PSF data", pad=10, fontsize=20)

        # contour plot
        axes[0, 1].contour(xbox, ybox, master_box,
                           vmin=vmin, vmax=vmax, colors='k')
        axes[0, 1].plot(xbox[half_windowsize], np.zeros_like(
            xbox[half_windowsize]), '--', lw=1., color='k', zorder=3)
        axes[0, 1].plot(np.zeros_like(ybox[:, half_windowsize]),
                        ybox[:, half_windowsize], '--', lw=1., color='k', zorder=3)
        axes[0, 1].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[0, 1].set_ylabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[0, 1].set_title("PSF contour", pad=10, fontsize=20)

        # Fit the data using a Gaussian
        model = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
        fitter = fitting.LevMarLSQFitter()
        best_fit = fitter(model, xbox[half_windowsize], master_box[half_windowsize],
                          weights=1.0/master_box_err[half_windowsize]**2)
        # print(best_fit)

        # plot x profile
        axes[1, 0].set_title("E-W profile \nFWHM: {:.2f} arcsec".format(
            2.355*best_fit.stddev.value), pad=10, fontsize=20)
        axes[1, 0].errorbar(xbox[half_windowsize], master_box[half_windowsize],
                            yerr=master_box_err[half_windowsize], lw=0.5, fmt='o', color='k')
        axes[1, 0].plot(xbox[half_windowsize], best_fit(
            xbox[half_windowsize]), '-', lw=2, color='brown')
        axes[1, 0].set_xlabel(r"$\Delta\,\alpha$ (arcsec)", fontsize=16)
        axes[1, 0].set_ylabel("flux", fontsize=16)
        # axes[1,0].legend(fontsize=16)

        best_fit = fitter(model, ybox[:, half_windowsize], master_box[:,
                          half_windowsize], weights=1.0/master_box_err[:, half_windowsize]**2)
        # print(best_fit)
        # plot y profile
        axes[1, 1].set_title("N-S profile \nFWHM: {:.2f} arcsec".format(
            2.355*best_fit.stddev.value), pad=10, fontsize=20)
        axes[1, 1].errorbar(ybox[:, half_windowsize], master_box[:, half_windowsize],
                            master_box_err[:, half_windowsize], lw=0.5, fmt='o', color='k')
        axes[1, 1].plot(ybox[:, half_windowsize], best_fit(
            ybox[:, half_windowsize]), '-', lw=2, color='darkblue')
        axes[1, 1].set_xlabel(r"$\Delta\,\delta$ (arcsec)", fontsize=16)
        axes[1, 1].set_ylabel("flux", fontsize=16)
        # axes[1,1].legend(fontsize=16)

        plt.show()

        # vmin, vmax = np.percentile(img, 3.), np.percentile(img, 97)
        # s4plt.plot_2d(xw, yw, img, LIM=None, LAB=["x (pix)", "y (pix)","flux fraction"], z_lim=[vmin,vmax], title="PSF plot", pfilename="", cmap="gist_heat")

    return loc


def stack_and_reduce_sci_images(p, sci_list, reduce_dir, ref_img="", stack_suffix="", force=True, match_frames=True, polarimetry=False, plot=False, plot_proc_frames=False):
    """ Pipeline module to run stack and reduction of science images

    Parameters
    ----------
    p : dict
        params
    sci_list : list
        list of science file paths
    reduce_dir : str
        path to reduce directory
    ref_img : str
        path to a reference image. If empty will take it from "p"
    stack_suffix : str
        suffix to be appended to the output stack file name
    force : bool
        force reduction, even if products already exist
    match_frames : bool
        match frames in sci list
    polarimetry : bool
        is it polarimetry data?
    plot : bool
        do plots
    plot_proc_frames : bool
        plot processed frames

    Returns
    -------
    p : dict
        params
    """

    # clean list of reduced images from previous channel/object
    if 'OBJECT_REDUCED_IMAGES' in p.keys():
        del p['OBJECT_REDUCED_IMAGES']

    # calculate stack
    p = stack_science_images(p,
                             sci_list,
                             reduce_dir=reduce_dir,
                             force=force,
                             stack_suffix=stack_suffix,
                             polarimetry=polarimetry)

    # plot stack frame
    if match_frames and plot:
        stack_plot_file = ""
        if p['PLOT_TO_FILE'] :
            stack_plot_file = p['OBJECT_STACK'].replace(".fits",p['PLOT_FILE_FORMAT'])
        try :
            if polarimetry :
                s4plt.plot_sci_polar_frame(p['OBJECT_STACK'], percentile=99.5, output=stack_plot_file)
            else :
                s4plt.plot_sci_frame(p['OBJECT_STACK'], nstars=20, use_sky_coords=True, output=stack_plot_file)
        except Exception as e:
            logger.warn("Could not generate plot for product {} : {}".format(p['OBJECT_STACK'], e))
    
    # set numbe of science reduction loops to avoid memory issues.
    nloops = int(np.ceil(len(sci_list) / p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP']))

    logger.info("The {} images will be reduced in {} loops of {} images each time".format(len(sci_list), nloops, p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP']) )
        
    # set reference image
    if ref_img == "":
        ref_img = p['REFERENCE_IMAGE']

    for loop in range(nloops):
        first = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * loop
        last = p['MAX_NUMBER_OF_SCI_FRAMES_PER_LOOP'] * (loop+1)
        if last > len(sci_list):
            last = len(sci_list)

        logger.info("Running loop {} of {} -> images in loop: {} to {} ... ".format(loop, nloops, first, last))
    
        animated_gif = ""
        if p['PLOT_PROC_FRAMES'] and p['CREATE_ANIMATED_GIF'] and p['PLOT_TO_FILE']:
            animated_gif = "{}_{:04d}.gif".format(p['OBJECT_STACK'].replace(".fits",""),loop)

        # reduce science data and calculate stack
        p = reduce_science_images(p,
                                  sci_list[first:last],
                                  reduce_dir=reduce_dir,
                                  ref_img=ref_img,
                                  force=force,
                                  match_frames=match_frames,
                                  polarimetry=polarimetry,
                                  plot=plot_proc_frames,
                                  animated_gif=animated_gif)
            
    # clean up catalogs
    if 'CATALOGS' in p.keys() :
        del p['CATALOGS']
        
    return p


def polar_time_series(sci_pol_list,
                      reduce_dir="./",
                      ts_suffix="",
                      aperture_radius=None,
                      min_aperture=0,
                      max_aperture=1024,
                      compute_k=True,
                      force=True):
    """ Pipeline module to calculate polarimetric time series for a given list of sparc4 sci image products

    Parameters
    ----------
    sci_pol_list : list
        list of paths to science polar products
    ts_suffix : str (optional)
        time series suffix to add into the output file name
    reduce_dir : str (optional)
        path to the reduce directory
    aperture_radius : float (optional)
        select aperture radius. Default is None and it will calculate best aperture
    min_aperture : float
        minimum aperture radius (pix)
    max_aperture : float
        minimum aperture radius (pix)
    compute_k : bool
        whether or not to compute k
    force : bool
        force reduction even if product already exists

    Returns
    -------
    output : str
        path to the output time series product file
    """

    # set output light curve product file name
    output = os.path.join(reduce_dir, "{}_ts.fits".format(ts_suffix))
    if os.path.exists(output) and not force:
        return output

    # get information from the first image in the time series
    basehdul = fits.open(sci_pol_list[0])
    basehdr = basehdul[0].header

    # set number of input polar files (size of time series)
    npolfiles = len(sci_pol_list)

    # get number of sources in polar catalog of first sequence:
    nsources = basehdr['NSOURCES']

    # initialize data container as dict
    tsdata = {}

    tsdata['TIME'] = np.array([])
    tsdata['SRCINDEX'] = np.array([])
    tsdata['RA'] = np.array([])
    tsdata['DEC'] = np.array([])
    tsdata['X1'] = np.array([])
    tsdata['Y1'] = np.array([])
    tsdata['X2'] = np.array([])
    tsdata['Y2'] = np.array([])
    tsdata['FWHM'] = np.array([])
    tsdata['MAG'] = np.array([])
    tsdata['EMAG'] = np.array([])

    tsdata['Q'] = np.array([])
    tsdata['EQ'] = np.array([])
    tsdata['U'] = np.array([])
    tsdata['EU'] = np.array([])
    tsdata['V'] = np.array([])
    tsdata['EV'] = np.array([])
    tsdata['P'] = np.array([])
    tsdata['EP'] = np.array([])
    tsdata['THETA'] = np.array([])
    tsdata['ETHETA'] = np.array([])
    tsdata['K'] = np.array([])
    tsdata['EK'] = np.array([])
    tsdata['ZERO'] = np.array([])
    tsdata['EZERO'] = np.array([])
    tsdata['NOBS'] = np.array([])
    tsdata['NPAR'] = np.array([])
    tsdata['CHI2'] = np.array([])
    tsdata['RMS'] = np.array([])
    tsdata['TSIGMA'] = np.array([])

    ti, tf = 0, 0

    for i in range(len(sci_pol_list)):

        logger.info("Packing time series data for polar file {} of {}".format(i+1, len(sci_pol_list)))

        hdul = fits.open(sci_pol_list[i])
        header = hdul[0].header

        mid_bjd = (header["TSTART"] + header["TSTOP"]) / 2

        if i == 0:
            ti = header["TSTART"]
        if i == len(sci_pol_list) - 1:
            tf = header["TSTOP"]

        for j in range(nsources):

            # read polarimetry results for the base sequence in the time series
            polar = get_polarimetry_results(sci_pol_list[i],
                                            source_index=j,
                                            aperture_radius=aperture_radius,
                                            min_aperture=min_aperture,
                                            max_aperture=max_aperture,
                                            compute_k=compute_k)

            tsdata['TIME'] = np.append(tsdata['TIME'], mid_bjd)
            tsdata['SRCINDEX'] = np.append(tsdata['SRCINDEX'], j)

            tsdata['RA'] = np.append(tsdata['RA'], polar['RA'])
            tsdata['DEC'] = np.append(tsdata['DEC'], polar['DEC'])
            tsdata['MAG'] = np.append(tsdata['MAG'], polar['MAG'].nominal)
            tsdata['EMAG'] = np.append(tsdata['EMAG'], polar['MAG'].std_dev)
            tsdata['FWHM'] = np.append(tsdata['FWHM'], polar['FWHM'])
            tsdata['X1'] = np.append(tsdata['X1'], polar['X1'])
            tsdata['Y1'] = np.append(tsdata['Y1'], polar['Y1'])
            tsdata['X2'] = np.append(tsdata['X2'], polar['X2'])
            tsdata['Y2'] = np.append(tsdata['Y2'], polar['Y2'])

            tsdata['Q'] = np.append(tsdata['Q'], polar['Q'].nominal)
            tsdata['EQ'] = np.append(tsdata['EQ'], polar['Q'].std_dev)
            tsdata['U'] = np.append(tsdata['U'], polar['U'].nominal)
            tsdata['EU'] = np.append(tsdata['EU'], polar['U'].std_dev)
            if polar['V'] is not None :
                tsdata['V'] = np.append(tsdata['V'], polar['V'].nominal)
                tsdata['EV'] = np.append(tsdata['EV'], polar['V'].std_dev)
            else :
                tsdata['V'] = np.append(tsdata['V'], np.nan)
                tsdata['EV'] = np.append(tsdata['EV'], np.nan)
                
            tsdata['P'] = np.append(tsdata['P'], polar['P'].nominal)
            tsdata['EP'] = np.append(tsdata['EP'], polar['P'].std_dev)
            tsdata['THETA'] = np.append(tsdata['THETA'], polar['THETA'].nominal)
            tsdata['ETHETA'] = np.append(tsdata['ETHETA'], polar['THETA'].std_dev)
            tsdata['K'] = np.append(tsdata['K'], polar['K'].nominal)
            tsdata['EK'] = np.append(tsdata['EK'], polar['K'].std_dev)
            tsdata['ZERO'] = np.append(tsdata['ZERO'], polar['ZERO'].nominal)
            tsdata['EZERO'] = np.append(tsdata['EZERO'], polar['ZERO'].std_dev)
            tsdata['NOBS'] = np.append(tsdata['NOBS'], polar['NOBS'])
            tsdata['NPAR'] = np.append(tsdata['NPAR'], polar['NPAR'])
            tsdata['CHI2'] = np.append(tsdata['CHI2'], polar['CHI2'])
            tsdata['RMS'] = np.append(tsdata['RMS'], polar['RMS'])
            
            if type(polar['TSIGMA']) is not float :
                polar['TSIGMA'] = 0.
            tsdata['TSIGMA'] = np.append(tsdata['TSIGMA'], polar['TSIGMA'])

        hdul.close()
        del hdul

    # Construct information dictionary to add to the header of FITS product
    info = {}

    info['OBSERV'] = ('OPD', 'observatory')
    info['OBSLAT'] = (basehdr["OBSLAT"], '[DEG] observatory latitude (N)')
    info['OBSLONG'] = (basehdr["OBSLONG"], '[DEG] observatory longitude (E)')
    info['OBSALT'] = (basehdr["OBSALT"], '[m] observatory altitude')
    info['TELESCOP'] = ('OPD-PE 1.6m', 'telescope')
    info['INSTRUME'] = ('SPARC4', 'instrument')
    info['OBJECT'] = (basehdr["OBJECT"], 'ID of object of interest')
    info['CHANNEL'] = (basehdr["CHANNEL"], 'Instrument channel')
    info['PHZEROP'] = (0., '[mag] photometric zero point')
    info['PHOTSYS'] = ("SPARC4", 'photometric system')
    info['POLTYPE'] = (basehdr["POLTYPE"], 'polarimetry type l/2 or l/4')
    info['NSOURCES'] = (basehdr["NSOURCES"], 'number of sources')

    # get first and last times
    tstart = Time(ti, format='jd', scale='utc')
    tstop = Time(tf, format='jd', scale='utc')
    info['TSTART'] = (tstart.jd, 'observation start time in BJD')
    info['TSTOP'] = (tstop.jd, 'observation stop time in BJD')
    info['DATE-OBS'] = (tstart.isot, 'TSTART as UTC calendar date')
    info['DATE-END'] = (tstop.isot, 'TSTOP as UTC calendar date')

    # generate the photometric time series product
    s4p.polarTimeSeriesProduct(tsdata, info=info, filename=output)

    return output


def select_best_phot_aperture_per_target(filename, plot=False):
    """ Pipeline module to get the best apertures per target

    Parameters
    ----------
    filename : str
        input light curves product (fits)
    plot : bool
        plots

    Returns
    -------
    output : numpy array of floats
        array of best aperture radii for all sources
    """

    # open time series fits file
    hdul = fits.open(filename)

    nsources = hdul[0].header['NSOURCES']

    minrms = np.arange(nsources)*0 + 1e20
    selected_apertures = np.zeros_like(minrms)
    mags, emags = np.full_like(minrms, np.nan),  np.full_like(minrms, np.nan)

    # loop over each hdu
    for hdu in hdul:
        # skip primary hdu
        if hdu.name == "PRIMARY":
            continue

        # get aperture table  data
        tbl = hdul[hdu.name].data

        # get aperture radius value
        aperture_radius = hdu.header["APRADIUS"]

        for i in range(nsources):
            loc_tbl = tbl[tbl["SRCINDEX"] == i]
            m_rms = np.nanmedian(loc_tbl['RMS'])
            if m_rms < minrms[i]:
                minrms[i] = m_rms
                selected_apertures[i] = aperture_radius
                mags[i] = np.nanmedian(loc_tbl['MAG'])
                emags[i] = np.nanmedian(loc_tbl['EMAG'])
            if plot:
                plt.plot([aperture_radius], [m_rms], '.')

    if plot:
        for i in range(nsources):
            plt.plot([selected_apertures[i]], [minrms[i]], 'rx')

        plt.ylabel("RMS [mag]")
        plt.xlabel("Aperture radius [pix]")
        plt.show()

        plt.errorbar(selected_apertures, mags, yerr=emags, fmt='o')
        plt.ylabel("Instrumental magnitude")
        plt.xlabel("Aperture radius [pix]")
        plt.show()

    return selected_apertures

