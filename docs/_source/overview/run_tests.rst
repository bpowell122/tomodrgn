Optional: run tests
====================

If you wish to explicitly test that tomoDRGN has been installed and is functioning correctly on your system, you can run the following tests.
Each test uses a small toy dataset included in the tomoDRGN source code.

If these tests fail, please report the issue on `tomoDRGN's Github page <https://github.com/bpowell122/tomodrgn>`_.
The most likely cause for failure is conda installing a more updated dependency version that is not compatible with tomoDRGN.

.. note::

    Running any of the following tests requires a further installation of tomoDRGN's testing dependencies.

    .. code-block:: bash

        cd $TOMODRGN_SOURCE_DIR
        python -m pip install ".[tests]"

.. note::

    Some useful arguments that can be supplied to the pytest commands below:

    * ``--capture=tee-sys``: log pytest and tomodrgn output to STDOUT
    * ``--basetemp=/custom/output/directory``: specify a directory to save all outputs from the test (perhaps to see sample tomoDRGN outputs)

Optional test 1
----------------

Run a quick end-to-end test of the most essential and frequently used CLI commands, ``tomodrgn train_vae`` and ``tomodrgn analyze``.
Takes about 1 minute.
Produces about 20 MB of outputs.

.. code-block:: bash

    cd $TOMODRGN_SOURCE_DIR/testing
    pytest --script-launch-mode=subprocess quicktest.py

Optional test 2
----------------

Run a comprehensive end-to-end test of all CLI commands with multiple combinations of CLI options (except Jupyter notebooks).
Takes about about 10 minutes (for each pytest command) on an Ubuntu workstation with a 4060Ti, or about 50 minutes on a MacBook.
Produces about 1 GB of outputs.

.. code-block:: bash

    cd $TOMODRGN_SOURCE_DIR/testing
    pytest --script-launch-mode=subprocess ./commandtest.py
    pytest --script-launch-mode=subprocess ./commandtest_warptools.py