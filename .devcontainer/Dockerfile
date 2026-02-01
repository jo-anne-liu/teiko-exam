# Use the official Python image (choose the version you prefer)
FROM python:3.11-slim

# Install OS‑level dependencies needed by scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libatlas-base-dev \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libffi-dev \
        && rm -rf /var/lib/apt/lists/*

# Create a non‑root user (optional but nice)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /workspaces/${PWD##*/}

# Switch to the user for the rest of the steps
USER $USERNAME
