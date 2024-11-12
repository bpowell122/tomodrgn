Installation
============

System requirements
--------------------

.. tab-set::

   .. tab-item:: Recommended system configuration

      Recommended system resources to run tomoDRGN on real-world datasets:

      * CPU: 8+ cores
      * RAM: 128+ GB (enough to hold all particles + 25% margin at float32 precision)
      * GPU: 1x Nvidia GPU with > 12 GB VRAM
      * Disk: 10-100 GB disk space for tomoDRGN outputs + fast local scratch disk if insufficient RAM to hold all particles

   .. tab-item:: Minimum system configuration

      Minimum system resources to run tomoDRGN on toy datasets, develop source code, build documentation, etc.:

      * CPU: 4+ cores
      * RAM: 8+ GB
      * Disk: 2 GB disk space for toy dataset outputs


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

TomoDRGN requires ``torch>=2.3``, but `pytorch does not distribute prebuilt pip packages for x86 Macs starting with pytorch 2.3 <https://github.com/pytorch/pytorch/issues/114602>`_.
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

Running tests requires installation of testing dependencies.

.. code-block:: bash

    cd $TOMODRGN_SOURCE_DIR
    python -m pip install .[tests]

Run a quick test of the most essential and frequently used commands, ``tomodrgn train_vae`` and ``tomodrgn analyze``.
Takes about 1 minute.

.. code-block:: bash

    cd $TOMODRGN_SOURCE_DIR/testing
    pytest --script-launch-mode=subprocess quicktest.py

Run a comprehensive end-to-end test of all commands with multiple options (except Jupyter notebooks).
Takes about 50 minutes on a MacBook, about 10 minutes on an Ubuntu workstation with a 4060Ti.
Produces about 1 GB of outputs in ``testing/outputs``.
Also serves as a useful reference for commonly used command syntax.

.. code-block:: bash

    cd $TOMODRGN_SOURCE_DIR/testing
    pytest --script-launch-mode=subprocess ./commandtest.py
    pytest --script-launch-mode=subprocess ./commandtest_warptools.py

Some useful arguments that can be supplied to the pytest commands above:

* ``--capture=tee-sys``: log pytest and tomodrgn output to STDOUT (a useful way to see any warnings as opposed to always-visible errors)
* ``--basetemp=/custom/output/directory``: change the directory to save all outputs from the test (perhaps to see sample outputs)


Optional: build documentation
-----------------------------

Documentation is built with sphinx in the ``tomodrgn`` environment:

.. code-block:: bash

    CD $TOMODRGN_SOURCE_DIR
    python -m pip install .[docs]
    cd docs
    make clean
    rm -rfv docs/_source/api/_autosummary  # this ensures all files from previous builds are removed, including autosummary API files missed by make clean
    make html  # note that a large number of warnings about `torch.nn.modules.Module` are expected and can be ignored
    # documentation is accessible at ./docs/_build/html/index.html and can be viewed in a web browser
