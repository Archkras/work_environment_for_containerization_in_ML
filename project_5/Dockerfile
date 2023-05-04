FROM python:3.10

ARG PROJECT_NAME=project_5
ARG GROUP_ID=1000
ARG USER_ID=1000

RUN groupadd --gid ${GROUP_ID} ${PROJECT_NAME} && \
    useradd --home-dir /home/${PROJECT_NAME} --create-home --uid ${USER_ID} \
    --gid ${GROUP_ID} --shell /bin/sh --skel /dev/null ${PROJECT_NAME} && \
    mkdir /app && \
    chown -R ${PROJECT_NAME}:${PROJECT_NAME} /app

WORKDIR /app

RUN apt-get update  && \
    apt-get install -y unzip && \
    apt-get clean && rm -rf var/lib/apt/lists/*

USER ${USER_ID}

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY script.py /app/script.py

