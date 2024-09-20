![Alt text](Figures/sparc4-pipeline-logo.png?raw=true "Title")
# sparc4-pipeline

The `sparc4-pipeline` is a set of routines that make use of the [`AstroPoP`](https://github.com/juliotux/astropop) package and other astronomical packages to reduce the data of photometric and polarimetric observations obtained with the [SPARC4](https://ui.adsabs.harvard.edu/abs/2012AIPC.1429..252R/abstract) instrument installed at the [Pico dos Dias Observatory (OPD/LNA)](https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd). The pipeline has a main module called `sparc4_mini_pipeline.py` to run the pipeline from the command line and allow the reduction of data from the four SPARC4 channels automatically. The pipeline also has a file called `sparc4_params.yaml` with the pipeline execution parameters, where one can configure the reduction parameters according to the science needs. 

<span style="color: red"> WARNING: The current version was tested with `AstroPoP Version: 0.9.3` </span>

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

The SPARC4 pipeline is under development in collaboration with the scientific community. If you are interested in collaborating, send an email to the pipeline team at the following address: `sparc4-pipeline@googlegroups.com`.
