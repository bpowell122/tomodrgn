Optional: build documentation
==============================

If you edit your local tomoDRGN source code or documentation, you can re-build documentation pages to reflect your changes.
The source pages for documentation are available at ``$TOMODRGN_SOURCE_DIR/docs/_source/`` and are written in reStructuredText (``.rst``).
Documentation is then built with ``sphinx`` in the ``tomodrgn`` environment as described below.

.. note::

    Building documentation requires a further installation of tomoDRGN's documentation-building dependencies.

    .. code-block:: bash

        cd $TOMODRGN_SOURCE_DIR
        python -m pip install ".[docs]"

Run the following commands to build local documentation.

.. code-block:: bash

    cd $TOMODRGN_SOURCE_DIR/docs
    make clean
    rm -rfv docs/_source/api/_autosummary  # this ensures all files from previous builds are removed, including autosummary API files missed by make clean
    make html  # note that a large number of warnings about `torch.nn.modules.Module` are expected and can be ignored

Built documentation can be viewed by opening ``./docs/_build/html/index.html`` in a web browser.
