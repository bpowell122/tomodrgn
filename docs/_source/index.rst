tomoDRGN documentation
========================

.. figure:: assets/empiar10499_00255_ribosomes.png
    :alt: example heterogeneity learned with tomoDRGN on dataset EMPIAR-10499
    :scale: 50%

    Example ribosomal heterogeneity learned with tomoDRGN on dataset EMPIAR-10499


[CryoDRGN](https://github.com/zhonge/cryodrgn) has proven a powerful deep learning method for heterogeneity analysis in single particle cryo-EM. In particular, the method models a continuous distribution over 3D structures by using a Variational Auto-Encoder (VAE) based architecture to generate a reconstruction voxel-by-voxel once given a fixed coordinate from a continuous learned  latent space.

TomoDRGN extends the cryoDRGN framework to cryo-ET by learning heterogeneity from datasets in which each particle is sampled by multiple projection images at different stage tilt angles. For cryo-ET samples imaging particles *in situ*, tomoDRGN therefore enables continuous heterogeneity analysis at a single particle level within the native cellular environment. This new type of input necessitates modification of the cryoDRGN architecture, enables tomography-specific processing opportunities (e.g. dose weighting for loss weighting and efficient voxel subset evaluation during training), and benefits from tomography-specific interactive visualizations.

References
-----------
| [1] GitHub: https://github.com/bpowell122/tomodrgn
| [2] BioRxiv: https://www.biorxiv.org/content/10.1101/2023.05.31.542975v1
| [3] Nature Methods:


.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   quickstart <quickstart/index>
   command usage <command_usage/index>
   API <api/index>
   tutorials <tutorial_empiar10499/index>

