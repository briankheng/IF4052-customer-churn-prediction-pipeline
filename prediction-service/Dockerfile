# Use an official Python image as a base
FROM python:3.12.8-bullseye

# Install dependencies for pyenv
RUN apt-get update && apt-get install -y \
    curl \
    git \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Set environment variables for pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install desired Python version via pyenv
RUN pyenv install 3.12.8 && \
    pyenv global 3.12.8

# Reinstall pip for the pyenv Python
RUN curl https://bootstrap.pypa.io/get-pip.py | python

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8000

# Run bash
CMD ["mlflow", "models", "serve", "-m", "models:/customer_churn_prediction_model/latest", "-p", "8000", "--no-conda", "--host", "0.0.0.0"]