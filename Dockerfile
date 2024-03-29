# syntax=docker/dockerfile:1

FROM python:3.11-slim-bullseye AS base


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
    && apt-get -y install libgl1-mesa-glx libglib2.0-0 \
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
