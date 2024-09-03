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


Potential errors during installation
-------------------------------------

On an Ubuntu 24.04 machine, I had to install the following in order to build ``fastcluster`` dependency during install:

.. code-block:: bash

    sudo apt install make
    sudo apt install build-essential
    sudo apt install cmake

TomoDRGN requires ``pytorch>=2.3``, but `pytorch does not distribute prebuilt pip packages for x86 Macs starting with pytorch 2.3 <https://github.com/pytorch/pytorch/issues/114602>`_.
Therefore pytorch must be `built from source for x86 Macs <https://github.com/pytorch/pytorch#from-source>`_:

.. code-block:: bash

    pip install mkl-static mkl-include
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    conda install cmake ninja
    pip install -r requirements.txt
    python3 setup.py develop


Optional: verify code+dependency functionality on your system
---------------------------------------------------------------

Run a quick test of the most essential and frequently used commands, ``tomodrgn train_vae`` and ``tomodrgn analyze``.
Takes about 1 minute.

.. code-block:: bash

    cd testing
    python ./quicktest.py

Run a comprehensive end-to-end test of all commands with multiple options (except Jupyter notebooks).
Takes about 50 minutes on a MacBook, about 10 minutes on an Ubuntu workstation with a 4060Ti.
Produces about 1 GB of outputs in ``testing/outputs``.

.. code-block:: bash

    cd testing
    python ./quicktest.py

    # a useful reference for commonly used command syntax
    python ./commandtest.py


Optional: build documentation
-----------------------------

Documentation is built with sphinx in the ``tomodrgn`` environment:

.. code-block:: bash

    python -m pip install .[docs]
    cd docs
    make clean && make html
    # documentation is accessible at ./docs/_build/html/index.html and can be viewed in a web browser
