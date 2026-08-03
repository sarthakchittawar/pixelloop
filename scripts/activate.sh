#!/bin/bash
# Ensure pixi's C++ libraries are found first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
