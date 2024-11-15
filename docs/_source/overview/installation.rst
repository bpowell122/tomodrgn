Installation
============

System requirements
--------------------

.. tab-set::

   .. tab-item:: Recommended system configuration

      Recommended system resources to run tomoDRGN on real-world datasets:

      * CPU: 8+ cores
      * RAM: 128+ GB (enough to hold all particles + 25% margin if not using ``--lazy``)
      * GPU: 1x Nvidia GPU with > 12 GB VRAM
      * Disk: 10-100 GB disk space for tomoDRGN outputs + fast local scratch disk if insufficient RAM to hold all particles

   .. tab-item:: Minimum system configuration

      Minimum system resources to run tomoDRGN on toy datasets, develop source code, build documentation, etc.:

      * CPU: 4+ cores
      * RAM: 8+ GB
      * Disk: 2 GB disk space for toy dataset outputs


Creating a tomoDRGN environment
---------------------------------

We recommend creating separate virtual environments for separate software.
Conda is a package manager that can create and manage such virtual environments, and can install the requisite Python and non-Python dependencies.
Conda can be installed following the latest instructions `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation>`_.

Once you have installed conda, run the commands below to create a tomoDRGN environment and install tomoDRGN.

.. code-block:: bash

    # Create conda environment
    conda create --name tomodrgn "python>=3.10, <3.13"
    conda activate tomodrgn

    # Clone source code and install tomoDRGN + dependencies
    git clone https://github.com/bpowell122/tomodrgn.git
    cd tomodrgn  # note: this directory is referred to as $TOMODRGN_SOURCE_DIR below
    git checkout v1.0.0
    python -m pip install .

If you did not see any errors, great!
Your installation of tomoDRGN is ready to use, and you may confirm this by running tomoDRGN's suite of CLI tests as described :doc:`here <run_tests>`.

Resolving potential installation errors
----------------------------------------

If you did see errors during the installation process, here we describe some known issues and their solutions.

* On an Ubuntu 24.04 system, tomoDRGN installation failed while installing the ``fastcluster`` dependency. The root cause appeared to be missing ``cmake`` installation.

  .. code-block:: bash

      sudo apt install make
      sudo apt install build-essential
      sudo apt install cmake
      # resume installation from ``python -m pip install .`` above

* On an x86 Mac system, tomoDRGN installation failed while installing the ``torch`` dependency. The root cause appeared to be tomoDRGN's requirement of ``torch>=2.3`` which `cannot be satisfied from PyPI / conda for x86 Macs <https://github.com/pytorch/pytorch/issues/114602>`_. Therefore pytorch must be `built from source for x86 Macs <https://github.com/pytorch/pytorch#from-source>`_:

  .. code-block:: bash

      pip install mkl-static mkl-include
      git clone --recursive https://github.com/pytorch/pytorch
      cd pytorch
      conda install cmake ninja
      pip install -r requirements.txt
      python3 setup.py develop
      # resume installation from ``python -m pip install .`` above






