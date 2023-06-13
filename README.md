# sparc4-pipeline

The `sparc4-pipeline` is a set of routines that make use of the [`AstroPoP`](https://github.com/juliotux/astropop) package to reduce the data of photometric and polarimetric observations obtained with the [SPARC4](https://ui.adsabs.harvard.edu/abs/2012AIPC.1429..252R/abstract) instrument installed at the [Pico dos Dias Observatory (OPD/LNA)](https://www.gov.br/lna/pt-br/composicao-1/coast/obs/opd). The pipeline has a main module called `sparc4_mini_pipeline.py` to run the pipeline from the command line and allow the reduction of data from the four SPARC4 channels automatically. The pipeline also has a file called `sparc4_params.py` with the pipeline execution parameters, where one can configure the reduction parameters according to the science needs. 

Below is an example to run the SPARC4 pipeline :

```
python ~/sparc4-pipeline/sparc4_mini_pipeline.py --datadir=~/minidata/ --reducedir=~/minidata/reduced --nightdir=20230503
```

The pipeline routines are organized in the following 4 main libraries:

* `sparc4_pipeline_lib.py`: pipeline execution routines and functions
* `sparc4_utils.py`: utility routines for reduction
* `sparc4_products.py`: I/O routines containing the definition of SPARC4 reduction products
* `sparc4_product_plots.py`: routines to get diagnostic plots of reduction products

These libraries can be used independently to reduce data step by step, as in the examples provided in the Jupyter notebooks.

Download the [minidata package](https://drive.google.com/file/d/1wixhMG6X9_kpQ95CSTi2H2wGlbZrklzM/view?usp=drive_link) containing SPARC4 data to test the pipeline.

See the [SPARC4 Quick Tutorial](https://docs.google.com/document/d/1gHF8UVxStBSGTgpc1EWEM6cGChxysxAqrQvta2II9Jg/edit?usp=drive_link) (in Portuguese) to start using the pipeline quickly.

The SPARC4 pipeline is under development in collaboration with the scientific community. If you are interested in collaborating with the development of the SPARC4 pipeline, please send an email to the pipeline team at the following address: `sparc4-pipeline@googlegroups.com`.
