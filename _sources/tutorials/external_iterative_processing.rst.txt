External iterative processing
==============================

Once star files describing homogeneous particle subsets resolved by tomoDRGN have been generated, these subsets can be refined in external STA software to potentially improve both global and local resolution.
The distinct structural states can also be treated as distinct molecular species in, for example, `the software M <https://warpem.github.io/warp/user_guide/m/quick_start_m/>`_ or `MCore <https://warpem.github.io/warp/user_guide/warptools/quick_start_warptools_tilt_series/#high-resolution-refinements-in-m>`_, allowing multi-species spatially-coherent STA refinement.
TomoDRGN can form a virtuous cycle in this way:

#. perform initial coarse round of STA
#. export particles, train a tomoDRGN model, and isolate less heterogeneous subsets of particles (e.g. junk filtration, structural state separation)
#. perform another round of STA separately for each structural state (to get optimal poses)
#. perform multi-species refinements in M / MCore (or analogous software)
#. re-export particle subtomograms to tomoDRGN to identify residual heterogeneity
#. rinse and repeat

Exporting to external STA software and performing validating refinements with these tools also generates the appropriate files `required <https://www.ebi.ac.uk/emdb/documentation/policies#1>`_ to deposit maps of your dataset's unique structural states to the EMDB.
For multiple reasons, including the fact that tomoDRGN does not generate half maps, tomoDRGN-generated volumes cannot be uploaded to the EMDB.
Instead, the trained tomoDRGN model (``weights.EPOCH.pkl``), the config file (``config.pkl``), the latent embeddings (``z.EPOCH.train.pkl``), selected indices and/or class labels, and selected tomoDRGN volumes should be uploaded to a more general data sharing site (such as `Zenodo <https://zenodo.org/>`_).
