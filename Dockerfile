# syntax=docker/dockerfile:1

FROM docker-eu-public.artifactory.swg-devops.com/zrl-sa-docker-virtual/python:3.11-slim-bullseye AS base


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1

ENV PATH="$PATH:/root/.local/bin" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    # Disable git host key verification.
    GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

# Install poetry
RUN --mount=type=cache,target=/var/cache/apt,id=apt,sharing=locked \
    apt-get update \
    && apt-get -y install curl build-essential git libssl-dev wget \
    #&& apt-get -y install libblas-dev liblapack-dev libatlas-base-dev \
    #&& apt-get -y install poppler-utils \
    # intall top
    #&& apt-get -y install procps \
    && apt-get -y install libgl1-mesa-glx libglib2.0-0 \
    # fonts needed for pdf -> png convertion
    #&& apt-get -y install fonts-wqy-zenhei \
    # library and include for pcre
    #&& apt-get -y install libpcre3-dev \
    && curl -sSL 'https://install.python-poetry.org' > install-poetry.py \
    && python install-poetry.py \
    && poetry --version \
    && rm install-poetry.py


# Create user
RUN useradd -ms /bin/bash app \
    && mkdir /app \
    && chown -R app:0 /app \
    && chmod g=u -R /app

WORKDIR /app

COPY --chown=app:0 . /app

ARG DEPENDENCY_VERSION=0

RUN --mount=type=cache,id=cache,sharing=locked,target=/root/.cache \
    # Poetry configuration files for authentication
    --mount=type=secret,id=poetry-config-toml,dst=/root/.config/pypoetry/config.toml \
    --mount=type=secret,id=poetry-auth-toml,dst=/root/.config/pypoetry/auth.toml \
    # SSH for git dependencies
    --mount=type=ssh \
    # Optionally update dependencies
    bash -c "[ ${DEPENDENCY_VERSION} != '0' ] && poetry update --no-dev || true" \
    && poetry install
    # Skip dev dependencies
    #--without dev


#RUN --mount=type=cache,id=cache,sharing=locked,target=/root/.cache \
#    --mount=type=secret,id=poetry-config-toml,dst=/root/.config/pypoetry/config.toml \
#    --mount=type=secret,id=poetry-auth-toml,dst=/root/.config/pypoetry/auth.toml \
#    --mount=type=ssh \
#    poetry run pip install git+ssh://git@github.ibm.com/VWE/image-segmentation@vwe/for-external-git
