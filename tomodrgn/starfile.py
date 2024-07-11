"""
Lightweight parsers for starfiles
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
import matplotlib.pyplot as plt
from typing import TextIO

from tomodrgn import mrc, utils
from tomodrgn.mrc import LazyImage

log = utils.log
vlog = utils.vlog


def prefix_paths(mrcs: list[str],
                 datadir: str) -> list[str]:
    """
    Test which of various modifications to the image .mrcs files correctly locates the files on disk.
    Tries no modification; prepending `datadir` to the basename of each image; prepending `datadir` to the full path of each image.
    :param mrcs: list of strings corresponding to the path to each image file specified in the star file (expected format: the `path_to_mrc` part of `index@path_to_mrc`)
    :param datadir: str corresponding to absolute or relative path to prepend to `mrcs`
    :return: list of strings corresponding to the confirmed path to each image file
    """

    filename_patterns = [
        mrcs,
        [f'{datadir}/{os.path.basename(x)}' for x in mrcs],
        [f'{datadir}/{x}' for x in mrcs],
    ]

    for filename_pattern in filename_patterns:
        if all([os.path.isfile(file) for file in set(filename_pattern)]):
            return filename_pattern

    raise FileNotFoundError(f'Not all files (or possibly no files) could be found using any of the filename patterns: {[filename_pattern[0] for filename_pattern in filename_patterns]}')


class GenericStarfile:
    """
    Class to parse any star file with any number of blocks on disk, or a pre-existing pandas dataframe, to a (dictionary of) pandas dataframes.
    Input
        starfile     : path to star file on disk, mutually exclusive with setting `dataframe`
        dataframe    : pre-existing pandas dataframe, mutually exclusive with setting `starfile`
    Attributes:
        sourcefile   : absolute path to source data star file
        preambles    : list of lists containing text preceeding each block in starfile
        block_names  : list of names for each data block
        blocks       : dictionary of {block_name: data_block_as_pandas_df}
    Methods:
        skeletonize  : automatically called by __init__ to identify data blocks, column headers, and line numbers to load later
        load         : automatically called by __init__ to read all data from .star into pandas dataframes
        write        : writes all object data to `outstar` optionally with timestamp
        get_particles_stack : loads all particle images specified by the star file into a numpy array
    Notes:
        Stores data blocks not initiated with `loop_` keyword as a list in the `preambles` attribute
        Will ignore comments between `loop_` and beginning of data block; will not be preserved if using .write()
        Will raise a RuntimeError if a comment is found within a data block initiated with `loop`
    """

    def __init__(self,
                 starfile: str = None,
                 *,
                 dataframe: pd.DataFrame = None):
        """
        Create the GenericStarfile object either by reading a star file on disk, or by passing in a pre-existing pandas dataframe.
        :param starfile: path to star file on disk, mutually exclusive with setting `dataframe`
        :param dataframe: pre-existing pandas dataframe, mutually exclusive with setting `starfile`
        """
        if starfile is not None:
            assert not dataframe, 'Creating a GenericStarfile from a star file is mutually exclusive with creating a GenericStarfile from a dataframe.'
            self.sourcefile = os.path.abspath(starfile)
            preambles, blocks = self.skeletonize()
            self.preambles = preambles
            if len(blocks) > 0:
                blocks = self.load(blocks)
                self.block_names = list(blocks.keys())
            self.blocks = blocks
        elif dataframe is not None:
            self.sourcefile = None
            self.preambles = [['', 'data_', '', 'loop_']]
            self.block_names = ['data_']
            self.blocks = {'data_': dataframe}

    def __len__(self):
        return len(self.block_names)

    def skeletonize(self) -> tuple[list[list[str]], dict[str, [list[str], int, int]]]:
        """
        Parse star file for key data including preamble lines, header lines, and first and last row numbers associated with each data block. Does not load the entire file.
        :return: preambles: list (for each data block) of lists (each line preceeding column header lines and following data rows, as relevant)
        :return: blocks: dict mapping block names (e.g. `data_particles`) to a list of constituent column headers (e.g. `_rlnImageName),
            the first file line containing the data values of that block, and the last file line containing data values of that block
        """

        def parse_preamble(filehandle: TextIO,
                           _line_count: int) -> tuple[list[str], str | None, int]:
            """
            Parse a star file preamble (the lines preceeding column header lines and following data rows, as relevant)
            :param filehandle: pre-existing file handle from which to read the star file
            :param _line_count: the currently active line number in the star file
            :return: _preamble: list of lines comprising the preamble section
            :return: _block_name: the name of the data block following the preamble section, or None if no data block follows
            :return: _line_count: the currently active line number in the star file after parsing the preamble
            """
            # parse all lines preceeding column headers (including 'loop_')
            _preamble = []
            while True:
                line = filehandle.readline()
                _line_count += 1
                if not line:
                    # end of file detected
                    return _preamble, None, _line_count
                _preamble.append(line.strip())
                if line.startswith('loop_'):
                    # entering loop
                    _block_name = [line for line in _preamble if line != ''][-2]
                    return _preamble, _block_name, _line_count

        def parse_single_block(_f: TextIO,
                               _line_count: int) -> tuple[list[str], int, int, bool]:
            """
            Parse a single data block of a star file
            :param _f: pre-existing file handle from which to read the star file
            :param _line_count: the currently active line number in the star file
            :return: _header: list of lines comprising the column headers of the data block
            :return: _block_start_line: the first file line containing the data values of the data block
            :return: _line_count: the currently active line number in the star file after parsing the data block
            :return: end_of_file: boolean indicating whether the entire file ends immediately following the data block
            """
            _header = []
            _block_start_line = _line_count
            while True:
                # populate header
                line = _f.readline()
                _line_count += 1
                if not line.strip():
                    # blank line between `loop_` and first header row
                    continue
                elif line.startswith('_'):
                    # column header
                    _header.append(line)
                    continue
                elif line.startswith('#'):
                    # line is a comment, discarding for now
                    print(f'Found comment at STAR file line {_line_count}, will not be preserved if writing star file later')
                    continue
                elif len(line.split()) == len([column for column in _header if column.startswith('_')]):
                    # first data line
                    _block_start_line = _line_count
                    break
                else:
                    # unrecognized data block format
                    raise RuntimeError
            while True:
                # get length of data block
                line = _f.readline()
                _line_count += 1
                if not line:
                    # end of file, therefore end of data block
                    return _header, _block_start_line, _line_count, True
                if line.strip() == '':
                    # end of data block
                    return _header, _block_start_line, _line_count, False

        preambles = []
        blocks = {}
        line_count = 0
        with open(self.sourcefile, 'r') as f:
            while True:
                # iterates once per preamble/header/block combination, ends when parse_preamble detects EOF
                preamble, block_name, line_count = parse_preamble(f, line_count)
                if preamble:
                    preambles.append(preamble)
                if block_name is None:
                    return preambles, blocks

                header, block_start_line, line_count, end_of_file = parse_single_block(f, line_count)
                blocks[block_name] = [header, block_start_line, line_count]

                if end_of_file:
                    return preambles, blocks

    def load(self,
             blocks: dict[str, [list[str], int, int]]) -> dict[str, pd.DataFrame]:
        """
        Load each data block of a pre-skeletonized star file into a pandas dataframe
        :param blocks: dict mapping block names (e.g. `data_particles`) to a list of constituent column headers (e.g. `_rlnImageName),
            the first file line containing the data values of that block, and the last file line containing data values of that block
        :return: dict mapping block names (e.g. `data_particles`) to the corresponding data as a pandas dataframe
        """

        def load_single_block(_header: list[str],
                              _block_start_line: int,
                              _block_end_line: int) -> pd.DataFrame:
            """
            Load a single data block of a pre-skeletonized star file into a pandas dataframe
            :param _header: list of column headers (e.g. `_rlnImageName) of the data block
            :param _block_start_line: the first file line containing the data values of the data block
            :param _block_end_line: the last file line containing data values of the data block
            :return: pandas dataframe of the data block values
            """
            columns = [line.split(' ')[0] for line in _header if line.startswith('_')]

            # load the first 1 row to get dtypes of columns
            df = pd.read_csv(self.sourcefile,
                             sep='\s+',
                             header=None,
                             names=columns,
                             index_col=None,
                             skiprows=_block_start_line - 1,
                             nrows=1,
                             low_memory=True,
                             engine='c',
                             )
            df_dtypes = {column: dtype for column, dtype in zip(df.columns.values.tolist(), df.dtypes.values.tolist())}

            # convert object dtype columns to string
            for column, dtype in df_dtypes.items():
                if dtype == 'object':
                    df_dtypes[column] = pd.StringDtype()

            # load the full dataframe with dtypes specified
            df = pd.read_csv(self.sourcefile,
                             sep='\s+',
                             header=None,
                             names=columns,
                             index_col=None,
                             skiprows=_block_start_line - 1,
                             nrows=_block_end_line - _block_start_line,
                             low_memory=True,
                             engine='c',
                             dtype=df_dtypes,
                             )
            return df

        for block_name in blocks.keys():
            header, block_start_line, block_end_line = blocks[block_name]
            blocks[block_name] = load_single_block(header, block_start_line, block_end_line)
        return blocks

    def write(self,
              outstar: str,
              timestamp: bool = False) -> None:
        """
        Write out the starfile dataframe(s) as a new file
        :param outstar: name of the output star file, optionally as absolute or relative path
        :param timestamp: whether to include the timestamp of file creation as a comment in the first line of the file
        :return: None
        """

        def write_single_block(_f: TextIO,
                               _block_name: str) -> None:
            """
            Write a single star file block to a pre-existing file handle
            :param _f: pre-existing file handle to which to write this block's contents
            :param _block_name: name of star file block to write (e.g. `data_`, `data_particles`)
            :return: None
            """
            df = self.blocks[_block_name]
            headers = [f'{header} #{i + 1}' for i, header in enumerate(df.columns.values.tolist())]
            _f.write('\n'.join(headers))
            _f.write('\n')
            df.to_csv(_f, index=False, header=False, mode='a', sep='\t')

        with open(outstar, 'w') as f:
            if timestamp:
                f.write('# Created {}\n'.format(dt.now()))

            for preamble, block_name in zip(self.preambles, self.block_names):
                for row in preamble:
                    f.write(row)
                    f.write('\n')
                write_single_block(f, block_name)
                f.write('\n')

        print(f'Wrote {os.path.abspath(outstar)}')

    def get_particles_stack(self,
                            particles_block_name: str = None,
                            particles_path_column: str = None,
                            datadir: str = None,
                            lazy: bool = False) -> np.ndarray | list[LazyImage]:
        """
        Load particle images referenced by starfile
        :param particles_block_name: name of star file block containing particle path column (e.g. `data_`, `data_particles`)
        :param particles_path_column: name of star file column containing path to particle images .mrcs (e.g. `_rlnImageName`)
        :param datadir: absolute path to particle images .mrcs to override particles_path_column
        :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True)
        :return: np.ndarray of shape (n_ptcls * n_tilts, D, D) or list of LazyImage objects of length (n_ptcls * n_tilts)
        """

        # validate inputs
        assert particles_block_name is not None
        assert particles_path_column is not None

        images = self.blocks[particles_block_name][particles_path_column]
        images = [x.split('@') for x in images]  # assumed format is index@path_to_mrc
        self.blocks[particles_block_name]['_rlnImageNameInd'] = [int(x[0]) - 1 for x in images]  # convert to 0-based indexing of full dataset
        self.blocks[particles_block_name]['_rlnImageNameBase'] = [x[1] for x in images]

        mrcs = []
        ind = []
        # handle starfiles where .mrcs stacks are referenced non-contiguously
        for i, group in self.blocks[particles_block_name].groupby(
                (self.blocks[particles_block_name]['_rlnImageNameBase'].shift() != self.blocks[particles_block_name]['_rlnImageNameBase']).cumsum(), sort=False):
            # mrcs = [path1, path2, ...]
            mrcs.append(group['_rlnImageNameBase'].iloc[0])
            # ind = [ [0, 1, 2, ..., N], [0, 3, 4, ..., M], ..., ]
            ind.append(group['_rlnImageNameInd'].to_numpy())

        if datadir is not None:
            mrcs = prefix_paths(mrcs, datadir)
        for path in set(mrcs):
            assert os.path.exists(path), f'{path} not found'

        header = mrc.parse_header(mrcs[0])
        boxsize = header.D  # image size along one dimension in pixels
        dtype = header.dtype
        stride = dtype().itemsize * boxsize * boxsize
        if lazy:
            lazyparticles = [LazyImage(file, (boxsize, boxsize), dtype, 1024 + ind_img * stride)
                             for ind_stack, file in zip(ind, mrcs)
                             for ind_img in ind_stack]
            return lazyparticles
        else:
            # preallocating numpy array for in-place loading, fourier transform, fourier transform centering, etc
            particles = np.zeros((len(self.blocks[particles_block_name]), boxsize + 1, boxsize + 1), dtype=np.float32)
            offset = 0
            for ind_stack, file in zip(ind, mrcs):
                particles[offset:offset + len(ind_stack), :-1, :-1] = mrc.LazyImageStack(file, dtype, (boxsize, boxsize), ind_stack).get()
                offset += len(ind_stack)
            return particles


class TiltSeriesStarfile(GenericStarfile):
    """
    Class to parse a particle image-series star file from multiple upstream STA software into a consistent format
    Input            : path to star file
    Attributes:
        df                     : alias to the particles dataframe
        header_pose_phi        : particles dataframe column header for pose angle phi in degrees following RELION conventions
        header_pose_theta      : particles dataframe column header for pose angle theta in degrees following RELION conventions
        header_pose_psi        : particles dataframe column header for pose angle psi in degrees following RELION conventions
        header_pose_tx         : particles dataframe column header for pose shift in x in pixels
        header_pose_tx_angst   : particles dataframe column header for pose shift in x in Ångstroms
        header_pose_ty         : particles dataframe column header for pose shift in y in pixels
        header_pose_ty_angst   : particles dataframe column header for pose shift in y in Ångstroms
        header_ctf_angpix      : particles dataframe column header for extracted particle pixel size in Ångstroms per pixel
        header_ctf_defocus_u   : particles dataframe column header for particle defocus U in Ångstroms
        header_ctf_defocus_v   : particles dataframe column header for particle defocus V in Ångstroms
        header_ctf_defocus_ang : particles dataframe column header for particle defocus angle in degrees
        header_ctf_voltage     : particles dataframe column header for microscope voltage in kV
        header_ctf_cs          : particles dataframe column header for microscope spherical aberration in millimeters
        header_ctf_w           : particles dataframe column header for microscope amplitude contrast
        header_ctf_ps          : particles dataframe column header for particle phase shift in degrees
        header_ptcl_uid        : particles dataframe column header for unique identifier shared among all images of a particle
        header_ptcl_dose       : particles dataframe column header for cumulative electron dose applied to each image in electrons per square Ångstrom
        header_ptcl_tilt       : particles dataframe column header for stage tilt relative to electron beam in degrees
        header_ptcl_image      : particles dataframe column header for index and path of extracted particle image
        header_ptcl_micrograph : particles dataframe column header for source particle image micrograph or tomogram
        image_ctf_corrected    : whether extracted particle images are CTF corrected
        image_dose_weighted    : whether extracted particle images are dose weighted
        image_tilt_weighted    : whether extracted particle iamges are tilt weighted
    Methods:
        infer_metadata_mapping  : automatically called by __init__ to infer key column names and upstream particle image preprocessing
        get_tiltseries_pixelsize : returns the extracted particle pixel size in Ångstroms per pixel
        get_tiltseries_voltage   : returns the microscope acceleration voltage in kV
        get_ptcl_img_indices     : returns the indices of each tilt image in the particles dataframe grouped by particle ID.
        get_image_size : returns the image size in pixels by loading the first image's header.
        make_test_train_split  : creates indices for tilt images assigned to train vs test split
        plot_particle_uid_ntilt_distribution: plots the distribution of the number of tilt images per particle as a line plot (against star file particle index) and as a histogram.
    """

    def __init__(self, starfile: str):
        # initialize object from parent class with parent attributes assigned at parent __init__
        super().__init__(starfile)

        # pre-initialize header aliases as None, to be set as appropriate by guess_metadata_interpretation()
        self.block_particles = None
        self.df = None
        self.header_pose_phi = None
        self.header_pose_theta = None
        self.header_pose_psi = None
        self.header_pose_tx = None
        self.header_pose_tx_angst = None
        self.header_pose_ty = None
        self.header_pose_ty_angst = None
        self.header_ctf_angpix = None
        self.header_ctf_defocus_u = None
        self.header_ctf_defocus_v = None
        self.header_ctf_defocus_ang = None
        self.header_ctf_voltage = None
        self.header_ctf_cs = None
        self.header_ctf_w = None
        self.header_ctf_ps = None
        self.header_ptcl_uid = None
        self.header_ptcl_dose = None
        self.header_ptcl_tilt = None
        self.header_ptcl_image = None
        self.header_ptcl_micrograph = None
        self.image_ctf_corrected = None
        self.image_dose_weighted = None
        self.image_tilt_weighted = None

        # infer the upstream metadata format
        self.infer_metadata_mapping()

    def __init__(self, headers, df, mrc_path):
        assert headers == list(df.columns), f'{headers} != {df.columns}'
        self.headers = headers
        self.df = df

    def infer_metadata_mapping(self) -> None:
        """
        Infer particle source software and version for key metadata and extraction-time processing corrections
        :return: None
        """
        known_star_headers = {
            'warpm_v1': {
                'data_': [
                    '_rlnMagnification',
                    '_rlnDetectorPixelSize',
                    '_rlnVoltage',
                    '_rlnSphericalAberration',
                    '_rlnAmplitudeContrast',
                    '_rlnPhaseShift',
                    '_rlnDefocusU',
                    '_rlnDefocusV',
                    '_rlnDefocusAngle',
                    '_rlnImageName',
                    '_rlnMicrographName',
                    '_rlnCoordinateX',
                    '_rlnCoordinateY',
                    '_rlnAngleRot',
                    '_rlnAngleTilt',
                    '_rlnAnglePsi',
                    '_rlnCtfBfactor',
                    '_rlnCtfScalefactor',
                    '_rlnRandomSubset',
                    '_rlnGroupName'
                ],
            },
            'cryosrpnt_v0.1': {
                'data_': [
                    '_rlnImageName',
                    '_rlnDetectorPixelSize',
                    '_rlnDefocusU',
                    '_rlnDefocusV',
                    '_rlnDefocusAngle',
                    '_rlnVoltage',
                    '_rlnSphericalAberration',
                    '_rlnAmplitudeContrast',
                    '_rlnPhaseShift',
                    '_rlnAngleRot',
                    '_rlnAngleTilt',
                    '_rlnAnglePsi',
                    '_rlnCtfBfactor',
                    '_rlnCtfScalefactor',
                    '_rlnGroupName',
                ],
            },
            'nextpyp': {
                'data_optics': [
                    '_rlnOpticsGroupName',
                    '_rlnOpticsGroup',
                    '_rlnMicrographOriginalPixelSize',
                    '_rlnVoltage',
                    '_rlnSphericalAberration',
                    '_rlnAmplitudeContrast',
                    '_rlnImagePixelSize',
                    '_rlnImageSize',
                    '_rlnImageDimensionality',
                ],
                'data_particles': [
                    '_rlnImageName',
                    '_rlnMicrographName',
                    '_rlnCoordinateX',
                    '_rlnCoordinateY',
                    '_rlnAnglePsi',
                    '_rlnAngleTilt',
                    '_rlnAngleRot',
                    '_rlnOriginXAngst',
                    '_rlnOriginYAngst',
                    '_rlnDefocusU',
                    '_rlnDefocusV',
                    '_rlnDefocusAngle',
                    '_rlnPhaseShift',
                    '_rlnOpticsGroup',
                    '_rlnGroupNumber',
                    '_rlnCtfBfactor',
                    '_rlnCtfScalefactor',
                    '_rlnLogLikeliContribution',
                    '_rlnRandomSubset',
                    '_rlnTiltIndex',
                ]
            },
            'cistem': {
                'data_': [
                    '_cisTEMPositionInStack',
                    '_cisTEMAnglePsi',
                    '_cisTEMAngleTheta',
                    '_cisTEMAnglePhi',
                    '_cisTEMXShift',
                    '_cisTEMYShift',
                    '_cisTEMDefocus1',
                    '_cisTEMDefocus2',
                    '_cisTEMDefocusAngle',
                    '_cisTEMPhaseShift',
                    '_cisTEMOccupancy',
                    '_cisTEMLogP',
                    '_cisTEMSigma',
                    '_cisTEMScore',
                    '_cisTEMScoreChange',
                    '_cisTEMPixelSize',
                    '_cisTEMMicroscopeVoltagekV',
                    '_cisTEMMicroscopeCsMM',
                    '_cisTEMAmplitudeContrast',
                    '_cisTEMBeamTiltX',
                    '_cisTEMBeamTiltY',
                    '_cisTEMImageShiftX',
                    '_cisTEMImageShiftY',
                    '_cisTEMBest2DClass',
                    '_cisTEMBeamTiltGroup',
                    '_cisTEMParticleGroup',
                    '_cisTEMPreExposure',
                    '_cisTEMTotalExposure',
                ]
            },
        }

        headers = {block: self.blocks[block].columns.values.tolist() for block in self.block_names}
        match headers:

            case {'data_': headers1} if set(known_star_headers['warpm_v1']['data_']).issubset(headers1):
                print('Detected STAR source software: Warp_v1 | M_V1')

                # easy reference to particles data block
                self.block_particles = 'data_'
                self.df = self.blocks[self.block_particles]

                # set header aliases used by tomodrgn
                self.header_pose_phi = '_rlnAngleRot'
                self.header_pose_theta = '_rlnAngleTilt'
                self.header_pose_psi = '_rlnAnglePsi'
                self.header_pose_tx = '_rlnOriginX'
                self.header_pose_ty = '_rlnOriginY'
                self.header_ctf_angpix = '_rlnDetectorPixelSize'
                self.header_ctf_defocus_u = '_rlnDefocusU'
                self.header_ctf_defocus_v = '_rlnDefocusV'
                self.header_ctf_defocus_ang = '_rlnDefocusAngle'
                self.header_ctf_voltage = '_rlnVoltage'
                self.header_ctf_cs = '_rlnSphericalAberration'
                self.header_ctf_w = '_rlnAmplitudeContrast'
                self.header_ctf_ps = '_rlnPhaseShift'
                self.header_ptcl_uid = '_rlnGroupName'
                self.header_ptcl_dose = '_tomodrgnTotalDose'
                self.header_ptcl_tilt = '_tomodrgnPseudoStageTilt'  # pseudo because arccos returns values in [0,pi] so lose +/- tilt information
                self.header_ptcl_image = '_rlnImageName'
                self.header_ptcl_micrograph = '_rlnMicrographName'

                # set additional headers needed by tomodrgn
                self.df[self.header_ptcl_dose] = self.df['_rlnCtfBfactor'] / -4
                self.df[self.header_ptcl_tilt] = np.arccos(self.df['_rlnCtfScalefactor'])

                # image processing applied during particle extraction
                self.image_ctf_corrected = False
                self.image_dose_weighted = False
                self.image_tilt_weighted = False

            case {'data_': headers1} if set(known_star_headers['cryosrpnt_v0.1']['data_']).issubset(headers1):
                print('Detected STAR source software: cryoSRPNT_v0.1')

                # easy reference to particles data block
                self.block_particles = 'data_'
                self.df = self.blocks[self.block_particles]

                # set header aliases used by tomodrgn
                self.header_pose_phi = '_rlnAngleRot'
                self.header_pose_theta = '_rlnAngleTilt'
                self.header_pose_psi = '_rlnAnglePsi'
                self.header_pose_tx = '_rlnOriginX'
                self.header_pose_ty = '_rlnOriginY'
                self.header_ctf_angpix = '_rlnDetectorPixelSize'
                self.header_ctf_defocus_u = '_rlnDefocusU'
                self.header_ctf_defocus_v = '_rlnDefocusV'
                self.header_ctf_defocus_ang = '_rlnDefocusAngle'
                self.header_ctf_voltage = '_rlnVoltage'
                self.header_ctf_cs = '_rlnSphericalAberration'
                self.header_ctf_w = '_rlnAmplitudeContrast'
                self.header_ctf_ps = '_rlnPhaseShift'
                self.header_ptcl_uid = '_rlnGroupName'
                self.header_ptcl_dose = '_tomodrgnTotalDose'
                self.header_ptcl_tilt = '_tomodrgnPseudoStageTilt'  # pseudo because arccos returns values in [0,pi] so lose +/- tilt information
                self.header_ptcl_image = '_rlnImageName'
                self.header_ptcl_micrograph = '_rlnMicrographName'

                # set additional headers needed by tomodrgn
                self.df[self.header_ptcl_dose] = self.df['_rlnCtfBfactor'] / -4
                self.df[self.header_ptcl_tilt] = np.arccos(self.df['_rlnCtfScalefactor'])

                # image processing applied during particle extraction
                self.image_ctf_corrected = False
                self.image_dose_weighted = False
                self.image_tilt_weighted = False

            case {'data_optics': headers1, 'data_particles': headers2} if (
                    set(known_star_headers['nextpyp']['data_optics']).issubset(headers1) and set(known_star_headers['nextpyp']['data_particles']).issubset(headers2)):
                print('Detected STAR source software: nextPYP')

                # easy reference to particles data block
                self.block_particles = 'data_particles'
                self.df = self.blocks[self.block_particles]

                # set header aliases used by tomodrgn
                self.header_pose_phi = '_rlnAngleRot'
                self.header_pose_theta = '_rlnAngleTilt'
                self.header_pose_psi = '_rlnAnglePsi'

                self.header_pose_tx = '_rlnOriginX'  # note: may not yet exist
                self.header_pose_tx_angst = '_rlnOriginXAngst'
                self.header_pose_ty = '_rlnOriginY'  # note: may not yet exist
                self.header_pose_ty_angst = '_rlnOriginYAngst'

                self.header_ctf_angpix = '_rlnImagePixelSize'
                self.header_ctf_defocus_u = '_rlnDefocusU'
                self.header_ctf_defocus_v = '_rlnDefocusV'
                self.header_ctf_defocus_ang = '_rlnDefocusAngle'
                self.header_ctf_voltage = '_rlnVoltage'
                self.header_ctf_cs = '_rlnSphericalAberration'
                self.header_ctf_w = '_rlnAmplitudeContrast'
                self.header_ctf_ps = '_rlnPhaseShift'

                self.header_ptcl_uid = '_rlnGroupNumber'
                self.header_ptcl_dose = '_tomodrgnTotalDose'
                self.header_ptcl_tilt = '_tomodrgnPseudoStageTilt'  # pseudo because arccos returns values in [0,pi] so lose +/- tilt information
                self.header_ptcl_image = '_rlnImageName'
                self.header_ptcl_micrograph = '_rlnMicrographName'

                # merge optics groups block with particle data block
                self.df = self.df.merge(self.blocks['data_optics'], on='_rlnOpticsGroup', how='left', validate='many_to_one')

                # set additional headers needed by tomodrgn
                self.df[self.header_ptcl_dose] = self.df['_rlnCtfBfactor'] / -4
                self.df[self.header_ptcl_tilt] = np.arccos(self.df['_rlnCtfScalefactor'])
                self.df[self.header_pose_tx] = self.df[self.header_pose_tx_angst] / self.df[self.header_ctf_angpix].iloc[0]
                self.df[self.header_pose_ty] = self.df[self.header_pose_ty_angst] / self.df[self.header_ctf_angpix].iloc[0]

                # image processing applied during particle extraction
                self.image_ctf_corrected = False
                self.image_dose_weighted = False
                self.image_tilt_weighted = False

            case {'data_': headers1} if set(known_star_headers['cistem']['data_']).issubset(headers1):
                print('Detected STAR source software: cisTEM')
                raise NotImplementedError

            case _:
                raise NotImplementedError(f'STAR file headers do not match any pattern known to tomoDRGN: {headers}')

    def __len__(self) -> int:
        return len(self.df)

    def get_tiltseries_pixelsize(self) -> float | int:
        """
        Returns the pixel size of the extracted particles in Ångstroms.
        Assumes all particles have the same pixel size.
        :return: pixel size in Ångstroms/pixel
        """
        return self.df[self.header_ctf_angpix].iloc[0]

    def get_tiltseries_voltage(self) -> float | int:
        """
        Returns the voltage of the microscope used to image the particles in kV.
        :return: voltage in kV
        """
        return self.df[self.header_ctf_voltage].iloc[0]

    def get_ptcl_img_indices(self) -> list[np.ndarray[int]]:
        """
        Returns the indices of each tilt image in the particles dataframe grouped by particle ID.
        The number of tilt images per particle may vary across the STAR file, so a list (or object-type numpy array or ragged torch tensor) is required
        :return: indices of each tilt image in the particles dataframe grouped by particle ID
        """
        df_grouped = self.df.groupby(self.header_ptcl_uid, sort=False)
        return [df_grouped.get_group(ptcl).index.to_numpy() for ptcl in df_grouped.groups]

    def get_image_size(self,
                       datadir: str = None) -> int:
        """
        Returns the image size in pixels by loading the first image's header.
        Assumes images are square.
        :param datadir: Relative or absolute path to overwrite path to particle image .mrcs specified in the STAR file
        :return: image size in pixels
        """
        # expected format of path to images .mrcs file is index@path_to_mrc, 1-indexed
        first_image = self.df[self.header_ptcl_image].iloc[0]
        stack_index, stack_path = first_image.split('@')
        if datadir is not None:
            stack_path = prefix_paths([stack_path], datadir)[0]
        assert os.path.exists(stack_path), f'{stack_path} not found'
        header = mrc.parse_header(stack_path)
        return header.D

    def make_test_train_split(self,
                              fraction_train: float = 0.5,
                              first_ntilts: int = None,
                              show_summary_stats: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Create indices for tilt images assigned to train vs test split
        :param fraction_train: fraction of each particle's tilt images to put in train dataset
        :param first_ntilts: if not None, only use first ntilts images of each particle in star file
        :param show_summary_stats: print summary statistics of particle sampling for test/train
        :return: inds_train: array of indices of tilt images assigned to train split from particles dataframe
        :return: inds_test: array of indices of tilt images assigned to test split from particles dataframe
        """

        # check required inputs are present
        df_grouped = self.df.groupby(self.header_ptcl_uid, sort=False)
        assert 0 < fraction_train <= 1.0

        # find minimum number of tilts present for any particle
        mintilt_df = np.nan
        for _, group in df_grouped:
            mintilt_df = min(len(group), mintilt_df)

        # get indices associated with train and test
        inds_train = []
        inds_test = []

        for particle_id, group in df_grouped:
            # get all image indices of this particle
            inds_img = group.index.to_numpy(dtype=int)

            if first_ntilts is not None:
                assert len(inds_img) >= first_ntilts, f'Requested use_first_ntilts: {first_ntilts} larger than number of tilts: {len(inds_img)} for particle: {particle_id}'
                inds_img = inds_img[:first_ntilts]

            # calculate the number of images to use in train split for this particle
            n_inds_train = np.rint(len(inds_img) * fraction_train).astype(int)

            # generate sorted indices for images in train split by sampling without replacement
            inds_img_train = np.random.choice(inds_img, size=n_inds_train, replace=False)
            inds_img_train = np.sort(inds_img_train)
            inds_train.append(inds_img_train)

            # assign all other images of this particle to the test split
            inds_img_test = np.array(list(set(inds_img) - set(inds_img_train)))
            inds_test.append(inds_img_test)

        # provide summary statistics
        if show_summary_stats:
            print(f'    Number of tilts sampled by inds_train: {set([len(inds_img_train) for inds_img_train in inds_train])}')
            print(f'    Number of tilts sampled by inds_test: {set([len(inds_img_test) for inds_img_test in inds_test])}')

        # flatten indices
        inds_train = np.asarray([ind_img for inds_img_train in inds_train for ind_img in inds_img_train])
        inds_test = np.asarray([ind_img for inds_img_test in inds_test for ind_img in inds_img_test])

        # sanity check: the intersection of inds_train and inds_test should be empty
        assert len(set(inds_train) & set(inds_test)) == 0, len(set(inds_train) & set(inds_test))
        # sanity check: the union of inds_train and inds_test should be the total number of images in the particles dataframe
        if first_ntilts is None:
            assert len(set(inds_train) | set(inds_test)) == len(self.df), len(set(inds_train) | set(inds_test))

        return inds_train, inds_test

    def plot_particle_uid_ntilt_distribution(self,
                                             outdir: str = None) -> None:
        """
        Plot the distribution of the number of tilt images per particle as a line plot (against star file particle index) and as a histogram.
        Defaults to saving plot in the same location as the input star file.
        :param outdir: directory in which to save the plot
        :return: None
        """
        ptcls_to_imgs_ind = self.get_ptcl_img_indices()
        n_tilts = np.asarray([len(ptcl_to_imgs_ind) for ptcl_to_imgs_ind in ptcls_to_imgs_ind])

        fig, axes = plt.subplots(2, 1)
        axes[0].plot(n_tilts, linewidth=0.5)
        axes[0].set_xlabel('star file particle index')
        axes[0].set_ylabel('ntilts per particle')

        axes[1].hist(n_tilts, bins=np.arange(np.min(n_tilts), np.max(n_tilts) + 2, 1))
        axes[1].set_xlabel('ntilts per particle')
        axes[1].set_ylabel('count')

        plt.tight_layout()
        if outdir is None:
            outdir = os.path.dirname(self.sourcefile)
        basename = os.path.splitext(os.path.basename(self.sourcefile))[0]
        plt.savefig(f'{outdir}/{basename}_particle_uid_ntilt_distribution.png', dpi=200)
        plt.close()

    def get_particles_stack(self,
                            *,
                            datadir: str = None,
                            lazy: bool = False,
                            **kwargs) -> np.ndarray | list[LazyImage]:
        """
        Calls parent GenericStarfile get_particles_stack.
        Parent method parameters `particles_block_name` and `particles_path_column` are presupplied due to identification of these values during TiltSeriesStarfile instance creation.
        :param datadir: absolute path to particle images .mrcs to override particles_path_column
        :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True)
        :return: np.ndarray of shape (n_ptcls * n_tilts, D, D) or list of LazyImage objects of length (n_ptcls * n_tilts)
        """
        return super().get_particles_stack(particles_block_name=self.block_particles,
                                           particles_path_column=self.header_ptcl_image,
                                           datadir=datadir,
                                           lazy=lazy)
