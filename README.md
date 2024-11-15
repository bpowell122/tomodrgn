# TomoDRGN: analyzing structural heterogeneity in cryo-electron sub-tomograms

<p style="text-align: center;">
    <img src="docs/_source/assets/empiar10499_00255_ribosomes.png" alt="Unique per-particle ribosome volumes, calculated from a tomoDRGN variational autoencoder trained on EMPIAR-10499 _Mycoplasma pneumonaie_ ribosomes, mapped back to the tomographic cellular environment"/> 
</p>

[CryoDRGN](https://github.com/zhonge/cryodrgn) has proven a powerful deep learning method for heterogeneity analysis in single particle cryo-EM. In particular, the method models a continuous distribution over 3D structures by using a Variational Auto-Encoder (VAE) based architecture to generate a reconstruction voxel-by-voxel once given a fixed coordinate from a continuous learned  latent space.

TomoDRGN extends the cryoDRGN framework to cryo-ET by learning heterogeneity from datasets in which each particle is sampled by multiple projection images at different stage tilt angles. For cryo-ET samples imaging particles _in situ_, tomoDRGN therefore enables continuous heterogeneity analysis at a single particle level within the native cellular environment. This new type of input necessitates modification of the cryoDRGN architecture, enables tomography-specific processing opportunities (e.g. dose weighting for loss weighting and efficient voxel subset evaluation during training), and benefits from tomography-specific interactive visualizations.

## Documentation
TomoDRGN documentation, including installation instructions, tutorials, CLI and API references, and more, are now hosted on [GitHub Pages](https://bpowell122.github.io/tomodrgn/index.html).


## Relevant literature
1. Powell, B.M., Davis, J.H. Learning structural heterogeneity from cryo-electron sub-tomograms with tomoDRGN. bioRxiv
2. Zhong, E.D., Bepler, T., Berger, B. & Davis, J.H. CryoDRGN: Reconstruction of Heterogeneous cryo-EM Structures Using Neural Networks. Nature Methods, [doi:10.1038/s41592-020-01049-4](https://doi.org/10.1038/s41592-020-01049-4) (2021)
3. Kinman, L.F., Powell, B.M., Zhong, E.D. et al. Uncovering structural ensembles from single-particle cryo-EM data using cryoDRGN. Nat Protoc 18, 319–339 (2023). [https://doi.org/10.1038/s41596-022-00763-x](https://doi.org/10.1038/s41596-022-00763-x)
4. Sun, J., Kinman, L., Jahagirdar, D., Ortega, J., Davis. J. KsgA facilitates ribosomal small subunit maturation by proofreading a key structural lesion. bioRxiv, [doi:10.1101/2022.07.13.499473](https://doi.org/10.1101/2022.07.13.499473) (2022)


## Contact
Please file bug reports, feature requests, etc on this GitHub repository's Issues page.