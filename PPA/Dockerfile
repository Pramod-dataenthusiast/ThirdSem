# Use Miniconda with Python 3.7 as the base image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /workspace

# Add the conda-forge channel and create the Conda environment
RUN conda create -y -n tgan_env -c conda-forge python=3.7 \
    numpy=1.16.4 pandas=0.24.2 scipy=1.2.1 scikit-learn=0.20.2 tensorflow=1.14 \
    && conda run -n tgan_env pip install tgan==0.1.0 ctgan

# Set the default command to use the Conda environment without needing to activate it explicitly
SHELL ["conda", "run", "-n", "tgan_env", "/bin/bash", "-c"]

# Copy any necessary files into the working directory
COPY . /workspace
