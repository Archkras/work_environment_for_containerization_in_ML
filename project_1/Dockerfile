#FROM jupyter/base-notebook
#FROM cschranz/gpu-jupyter

FROM jupyter/scipy-notebook

USER root

RUN apt-get update  && \
    apt-get install -y unzip && \
    apt-get clean && rm -rf var/lib/apt/lists/*

USER $NB_UID
RUN pip install \
    "cvxpy>=1.1.19" \
    "matplotlib>=3.2.0" \
    "numpy>=1.0.0" \
    "pandas>=0.19" \
    "scikit-learn>=0.24.1" \
    "scipy>=1.3.0" \
    "yfinance" \
    "PyPortfolioOpt"


#RUN conda install --quiet --yes \
#    'pandas' \
#    'tqdm' && \
#    conda clean --all -f -y #&& \
#    fix-permissions "/home/${NB_USER}"


