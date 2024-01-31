#FROM jupyter/base-notebook
#FROM cschranz/gpu-jupyter

FROM jupyter/scipy-notebook:python-3.9.13

USER root

RUN apt-get update  && \
    apt-get install -y unzip && \
    apt-get clean && rm -rf var/lib/apt/lists/*

USER $NB_UID
RUN pip install \
    "cvxpy==1.4.2" \
    "matplotlib==3.6.0" \
    "numpy==1.23.3" \
    "pandas==1.5" \
    "scikit-learn==1.1.2" \
    "scipy==1.9.1" \
    "yfinance==0.2.36" \
    "PyPortfolioOpt==1.5.5"


#RUN conda install --quiet --yes \
#    'pandas' \
#    'tqdm' && \
#    conda clean --all -f -y #&& \
#    fix-permissions "/home/${NB_USER}"


