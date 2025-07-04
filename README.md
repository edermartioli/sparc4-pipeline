![Alt text](Figures/sparc4-pipeline-logo.png?raw=true "Title")
# sparc4-pipeline

The `sparc4-pipeline` is a set of routines that make use of the [`AstroPoP`](https://github.com/juliotux/astropop) package and other astronomical packages to reduce the data of photometric and polarimetric observations obtained with the [SPARC4](https://ui.adsabs.harvard.edu/abs/2012AIPC.1429..252R/abstract) instrument installed at the [Pico dos Dias Observatory (OPD/LNA)](https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd). The pipeline has a main module called `sparc4_mini_pipeline.py` to run the pipeline from the command line and allow the reduction of data from the four SPARC4 channels automatically. The pipeline also has a file called `sparc4_params.yaml` with the pipeline execution parameters, where one can configure the reduction parameters according to the science needs. 

<code style="color : red"> **WARNING**: </code> The current version of the `sparc4-pipeline` requires a pre-release version of `astropop`. To install this version of `astropop`, use the following command:
```
pip install -U --pre astropop
```

# Installation

```
git clone https://github.com/edermartioli/sparc4-pipeline.git

cd sparc4-pipeline

pip install -U .
```


# Execution

Below is an example to run the SPARC4 pipeline :

```
python -W ignore ~/sparc4-pipeline/scripts/sparc4_mini_pipeline.py --nightdir=20230605 -vp
```

The pipeline routines are organized in the following 5 main libraries:

* `pipeline_lib.py`: pipeline execution routines and functions
* `db.py`: a simple interface to create and manage a database of input raw data 
* `utils.py`: utility routines for reduction
* `products.py`: I/O routines containing the definition of SPARC4 reduction products
* `product_plots.py`: routines to get diagnostic plots of reduction products

These libraries can also be used independently to reduce data step by step, as in the examples provided in the Jupyter [notebooks](https://github.com/edermartioli/sparc4-pipeline/tree/main/notebooks).

Download the [minidata package](https://drive.google.com/file/d/1tAVjyhYGMDcrU5sDdGCmd_f5HoazZ294/view?usp=drive_link) containing SPARC4 data to test the pipeline. You may also want to downaload the [minilcdata package](https://drive.google.com/file/d/1GJA7HB-j2YhbmLO82T1g-LNrbpYFn6OR/view?usp=drive_link) containing time series data in photometric mode. 

See the [SPARC4 Quick Tutorial](https://github.com/edermartioli/sparc4-pipeline/blob/257cde7c85666b2cd83a76834a9f0023365393fa/docs/Manual%20da%20SPARC4%20Pipeline.pdf) (in Portuguese) to start using the pipeline quickly.

The [SPARC4 Pipeline Workshop Guidelines](https://docs.google.com/document/d/139lela_5Od0tttfZycWEukB7HSjlJ4hL4iNhtqP97mQ/edit?usp=sharing) ([pdf version](https://github.com/edermartioli/sparc4-pipeline/blob/main/docs/SPARC4%20Pipeline%20Workshop%20Guidelines.pdf)) are now available, providing step-by-step instructions to reduce the minidata. Make sure you also check the [slides presented in the workshop at the XLVII RASAB 2024](https://github.com/edermartioli/sparc4-pipeline/blob/main/docs/sparc4-pipeline_sab2024_hands-on.pdf) as well as the [hands-on SPARC4 Pipeline Workshop Jupyter notebooks](https://drive.google.com/file/d/1yJl6maK2WXIWPPt7f8XB0CQFZwhljsCZ/view?usp=sharing) with examples for accessing the pipeline products obtained from the reduction of the minidata.

# Warnings
Date: 2025-Jun-26

* The sources and sky magnitude values provided in the catalogs are instrumental, meaning they are not calibrated for any photometric system.
* The position angle of the linear polarization is not calibrated to the equatorial system. This applies to both half-wave and quarter-wave retarders.
* Polarimetry using a quarter-wave plate requires prior determination of the fast-axis direction. Simultaneously fitting the fast-axis direction and the normalization constant is unreliable and not permitted in the pipeline.
* If a new reduction is to be rerun with updated parameters, it is mandatory to delete any products affected by this new reduction; otherwise, it will have no effect. The current version of the pipeline only checks for the existence of a product by its file path, not whether its contents are consistent with the parameters used for the reduction.
-----

The SPARC4 pipeline is under development in collaboration with the scientific community. If you are interested in collaborating, send an email to the pipeline team at the following address: `sparc4-pipeline@googlegroups.com`.


