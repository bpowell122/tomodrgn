External iterative processing
==============================

Once star files describing homogeneous particle subsets have been generated, these subsets can be refined in external STA software to potentially improve both global and local resolution.
The distinct structural states can also be treated as distinct molecular species in `the software M <https://warpem.github.io/warp/home/m/>`_, allowing multi-species spatially-coherent STA refinement.
TomoDRGN can form a virtous cycle in this way:

#. perform initial coarse round of STA
#. export particles, train a tomoDRGN model, and isolate less heterogeneous subsets of particles (e.g. junk filtration, structural state separation)
#. perform another round of STA separately for each structural state (to get optimal poses)
#. perform multi-species refinements in M
#. export particle subtomograms from M to tomoDRGN to identify residual heterogeneity
#. rinse and repeat

Exporting to external STA software and performing validating refinements with these tools also generates the appropriate files `required <https://www.ebi.ac.uk/emdb/documentation/policies#1>`_ to deposit maps of your dataset's unique structural states to the EMDB.
