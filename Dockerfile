# Base image with CUDA 12.6
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.11 \
    PATH="/root/.pyenv/versions/${PYTHON_VERSION}/bin:${PATH}" \
    PYENV_ROOT="/root/.pyenv"

# Install system dependencies for Python build (pyenv) and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install bash first (though usually present)
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Add pyenv to PATH and set PYENV_SHELL
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:${PATH}" \
    PYENV_SHELL=bash

# Install Python using pyenv, set global version, and upgrade pip
# This RUN command will use the updated PATH
RUN pyenv install ${PYTHON_VERSION} \
    && pyenv global ${PYTHON_VERSION} \
    && pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Run commands from cog.yaml (pget and patch)
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget && \
    wget -O `${PYENV_ROOT}/versions/${PYTHON_VERSION}/lib/python${PYTHON_VERSION}/site-packages/torch/_inductor/fx_passes/post_grad.py` https://gist.githubusercontent.com/alexarmbr/d3f11394d2cb79300d7cf2a0399c2605/raw/378fe432502da29f0f35204b8cd541d854153d23/patched_torch_post_grad.py

# Expose port if necessary (RunPod serverless usually handles this)
# EXPOSE 8000

# Command to run the application
CMD ["python3", "-u", "rp_handler.py"]
