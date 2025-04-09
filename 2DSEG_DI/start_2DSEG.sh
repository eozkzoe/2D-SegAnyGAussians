#!/bin/bash

echo "+---------------------------------+"
echo "| Running 2DSEG container setup... |"
echo "+---------------------------------+"

# Extract the environment name from environment.yml
ENV_NAME=2dseg

# Activate the Conda environment
source /opt/conda/bin/activate "$ENV_NAME"

./install_2dseg.sh

# Keep container running indefinitely after installation
tail -f /dev/null
