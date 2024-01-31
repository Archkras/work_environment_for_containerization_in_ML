#FROM jupyter/base-notebook
#FROM cschranz/gpu-jupyter

FROM jupyter/scipy-notebook:python-3.9.13

USER root

RUN apt-get update  && \
    apt-get install -y unzip && \
    apt-get clean && rm -rf var/lib/apt/lists/*

USER $NB_UID
RUN pip install kaggle && \
    pip install dvc && \
    pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip



