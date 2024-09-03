Installation
============

Recommended system resources
-----------------------------
TODO

* multi-core CPU
* > 128 GB system RAM
* 1 NVIDIA GPU ( > 12 GB VRAM)
* fast local scratch disk if insufficient RAM to hold all particles + 25% margin at float32 precision

TODO add minimum system requirements for running commandtest with toy models (e.g. macbook)


Setting up a new tomoDRGN environment
--------------------------------------

.. code-block:: bash

    # Create conda environment
    conda create --name tomodrgn "python>=3.10"
    conda activate tomodrgn

    # Clone source code and install
    git clone https://github.com/bpowell122/tomodrgn.git
    cd tomodrgn
    python -m pip install .

    # Optionally, install development dependencies
    python -m pip install .[docs]


Potential errors during installation
-------------------------------------

.. code-block:: bash

    # on my Ubuntu 24.04 machine, I had to install the following in order to build fastcluster dependency during install
    sudo apt install make
    sudo apt install build-essential
    sudo apt install cmake


Optional: verify code+dependency functionality on your system
---------------------------------------------------------------

.. code-block:: bash

    cd tomodrgn/testing

    # ~1 minute
    # tests train_vae and analyze
    python ./quicktest.py

    # ~50 minutes on Macbook Pro, ~10 minutes on Ubuntu workstation with 4060Ti
    # tests all commands with multiple options (except jupyter notebooks)
    # a useful reference for commonly used command syntax
    python ./commandtest.py
