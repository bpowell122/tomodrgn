"""
Lightweight parsers for starfiles
"""
import os
import re
import shutil
from datetime import datetime as dt
from enum import Enum
from itertools import pairwise
from typing import TextIO, Literal, get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tomodrgn import mrc, utils


class TiltSeriesStarfileStarHeaders(Enum):
    """
    Enumeration of known source software with constituent data block names and headers which are compatible with the ``TiltSeriesStarfile`` class.
    """
    warp = {
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
    }
    cryosrpnt = {
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
    }
    nextpyp = {
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
    }
    cistem = {
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
    }


class TomoParticlesStarfileStarHeaders(Enum):
    """
    Enumeration of known source software with constituent data block names and headers which are compatible with the ``TomoParticlesStarfile`` class.
    """
    warptools = {
        'data_general': [
            '_rlnTomoSubTomosAre2DStacks'
        ],
        'data_optics': [
            '_rlnOpticsGroup',
            '_rlnOpticsGroupName',
            '_rlnSphericalAberration',
            '_rlnVoltage',
            '_rlnTomoTiltSeriesPixelSize',
            '_rlnCtfDataAreCtfPremultiplied',
            '_rlnImageDimensionality',
            '_rlnTomoSubtomogramBinning',
            '_rlnImagePixelSize',
            '_rlnImageSize',
            '_rlnAmplitudeContrast',
        ],
        'data_particles': [
            '_rlnTomoName',
            '_rlnTomoParticleId',
            '_rlnCoordinateX',
            '_rlnCoordinateY',
            '_rlnCoordinateZ',
            '_rlnAngleRot',
            '_rlnAngleTilt',
            '_rlnAnglePsi',
            '_rlnTomoParticleName',
            '_rlnOpticsGroup',
            '_rlnImageName',
            '_rlnOriginXAngst',
            '_rlnOriginYAngst',
            '_rlnOriginZAngst',
            '_rlnTomoVisibleFrames',
        ]
    }
    relion = {
        'data_general': [
            '_rlnTomoSubTomosAre2DStacks'
        ],
        'data_optics': [
            '_rlnOpticsGroup',
            '_rlnOpticsGroupName',
            '_rlnSphericalAberration',
            '_rlnVoltage',
            '_rlnTomoTiltSeriesPixelSize',
            '_rlnCtfDataAreCtfPremultiplied',
            '_rlnImageDimensionality',
            '_rlnTomoSubtomogramBinning',
            '_rlnImagePixelSize',
            '_rlnImageSize',
            '_rlnAmplitudeContrast',
        ],
        'data_particles': [
            '_rlnTomoName',
            '_rlnTomoParticleId',
            '_rlnCoordinateX',
            '_rlnCoordinateY',
            '_rlnCoordinateZ',
            '_rlnAngleRot',
            '_rlnAngleTilt',
            '_rlnAnglePsi',
            '_rlnTomoParticleName',
            '_rlnOpticsGroup',
            '_rlnImageName',
            '_rlnOriginXAngst',
            '_rlnOriginYAngst',
            '_rlnOriginZAngst',
            '_rlnTomoVisibleFrames',
            '_rlnGroupNumber',
            '_rlnClassNumber',
            '_rlnNormCorrection',
            '_rlnRandomSubset',
            '_rlnLogLikeliContribution',
            '_rlnMaxValueProbDistribution',
            '_rlnNrOfSignificantSamples',
        ]
    }


# to avoid potential mistakes while repeating names of supported star file source software, dynamically define the Literal of allowable source software
# note that this does not work for static typing, but does work correctly at runtime (e.g. for building documentation)
TILTSERIESSTARFILE_STAR_SOURCES = Literal['auto', 'warp', 'cryosrpnt', 'nextpyp', 'cistem']

TOMOPARTICLESSTARFILE_STAR_SOURCES = Literal['auto', 'warptools', 'relion']

KNOWN_STAR_SOURCES = Literal[TILTSERIESSTARFILE_STAR_SOURCES, TOMOPARTICLESSTARFILE_STAR_SOURCES]


class GenericStarfile:
    """
    Class to parse a STAR file, a pre-existing pandas dataframe, or a pre-existing dictionary, to a dictionary of dictionaries or pandas dataframes.
    Simple two-column STAR blocks are parsed as dictionaries, while complex table-style blocks are parsed as pandas dataframes.

    Notes:

    * Will ignore comments between `loop_` and beginning of data block; will not be preserved if using .write()
    * Will raise a RuntimeError if a comment is found within a data block initiated with `loop`

    """

    def __init__(self,
                 starfile: str = None,
                 *,
                 dictionary: dict = None,
                 dataframe: pd.DataFrame = None):
        """
        Create the GenericStarfile object by reading a star file on disk, by passing in a pre-existing dictionary, or by passing in a pre-existing pandas dataframe.

        :param starfile: path to star file on disk, mutually exclusive with setting `dictionary` or `dataframe`
        :param dictionary: pre-existing python dictionary, mutually exclusive with setting `starfile` or `dataframe
        :param dataframe: pre-existing pandas dataframe, mutually exclusive with setting `starfile` or `dictionary`
        """
        if starfile is not None:
            assert not dataframe, 'Creating a GenericStarfile from a star file is mutually exclusive with creating a GenericStarfile from a dataframe.'
            assert not dictionary, 'Creating a GenericStarfile from a star file is mutually exclusive with creating a GenericStarfile from a dictionary.'
            self.sourcefile = os.path.abspath(starfile)
            preambles, blocks = self._skeletonize(sourcefile=self.sourcefile)
            self.preambles = preambles
            if len(blocks) > 0:
                blocks = self._load(blocks)
                self.block_names = list(blocks.keys())
            else:
                self.block_names = []
            self.blocks = blocks
        elif dictionary is not None:
            assert not starfile, 'Creating a GenericStarfile from a dictionary is mutually exclusive with creating a GenericStarfile from a star file.'
            assert not dataframe, 'Creating a GenericStarfile from a dictionary is mutually exclusive with creating a GenericStarfile from a dataframe.'
            self.sourcefile = None
            self.preambles = [['', 'data_', '']]
            self.block_names = ['data_']
            self.blocks = dictionary
        elif dataframe is not None:
            assert not starfile, 'Creating a GenericStarfile from a dataframe is mutually exclusive with creating a GenericStarfile from a star file.'
            assert not dictionary, 'Creating a GenericStarfile from a dataframe is mutually exclusive with creating a GenericStarfile from a dictionary.'
            self.sourcefile = None
            self.preambles = [['', 'data_', '', 'loop_']]
            self.block_names = ['data_']
            self.blocks = {'data_': dataframe}

    def __len__(self):
        return len(self.block_names)

    @staticmethod
    def _skeletonize(sourcefile) -> tuple[list[list[str]], dict[str, [list[str], int, int]]]:
        """
        Parse star file for key data including:

        * preamble lines,
        * simple two-column blocks parsed as a dictionary, and
        * header lines, first, and last row numbers associated with each table-style data block.

        Does not load the entire file.

        :param sourcefile: path to star file on disk
        :return: preambles: list (for each data block) of lists (each line preceeding data block header lines and following data rows, as relevant)
        :return: blocks: dict mapping block names (e.g. `data_particles`) to either the block contents as a dictionary (for simple two-column data blocks),
                or to a list of constituent column headers (e.g. `_rlnImageName), and the first and last file lines containing data values of that block (for complex table-style blocks).
        """

        def parse_preamble(filehandle: TextIO,
                           _line_count: int) -> tuple[list[str], str | None, int]:
            """
            Parse a star file preamble (the lines preceeding column header lines and following data rows, as relevant).
            Stop and return when the line initiating a data block is detected, or when end-of-file is detected

            :param filehandle: pre-existing file handle from which to read the star file
            :param _line_count: the currently active line number in the star file
            :return: _preamble: list of lines comprising the preamble section
            :return: _block_name: the name of the data block following the preamble section, or None if no data block follows
            :return: _line_count: the currently active line number in the star file after parsing the preamble
            """
            # parse all lines preceeding column headers (including 'loop_')
            _preamble = []
            while True:
                _line = filehandle.readline()
                _line_count += 1
                if not _line:
                    # end of file detected
                    return _preamble, None, _line_count
                _preamble.append(_line.strip())
                if _line.startswith('data_'):
                    # entering data block, potentially either simple block (to be parsed as dictionary) or loop block (to be parsed as pandas dataframe)
                    _block_name = _line.strip()
                    return _preamble, _block_name, _line_count

        def parse_single_dictionary(_f: TextIO,
                                    _line_count: int,
                                    _line: str) -> tuple[dict, int, bool]:
            """
            Parse and load a dictionary associated with a specific block name from a pre-existing file handle.
            The currently active line and all following lines (until a blank line or end of file) must have two values per line after splitting on white space.
            The first value becomes the dictionary key, and the second value becomes the dictionary value.

            :param _f: pre-existing file handle from which to read the star file
            :param _line_count: the currently active line number in the star file
            :param _line: the current line read from the star file, which begins with `_` as the text beginning the first key in the dictionary to be returned
            :return: dictionary: a dictionary of key:value per row's whitespace-delimited contents
            :return: _line_count: the currently active line number in the star file after parsing the data block
            :return: end_of_file: boolean indicating whether the entire file ends immediately following the data block
            """
            # store the current line in the dictionary, where the current line is known to start with _ and therefore to be the first entry to the dictionary
            key_val = re.split(r'\s+', _line.strip())
            assert len(key_val) == 2, f'The number of whitespace-delimited strings must be 2 to parse a simple STAR block, found {len(key_val)} for line {_line} at row {_line_count}'
            dictionary = {key_val[0]: key_val[1]}

            # iterate through lines until blank line (end of STAR block) or no line (end of file) is reached
            while True:
                _line = _f.readline()
                _line_count += 1

                if not _line:
                    # endo of data block, and end of file detected
                    return dictionary, _line_count, True
                elif _line.strip() == '':
                    # end of data block, not end of file
                    return dictionary, _line_count, False
                else:
                    key_val = re.split(r'\s+', _line.strip())
                    assert len(key_val) == 2, f'The number of whitespace-delimited strings must be 2 to parse a simple STAR block, found {len(key_val)} for line {_line} at row {_line_count}'
                    dictionary[key_val[0]] = key_val[1]

        def parse_single_block(_f: TextIO,
                               _line_count: int) -> tuple[list[str], int, int, bool]:
            """
            Parse, but do not load, the rows defining a dataframe associated with a specific block name from a pre-existing file handle.
            The currently active line is `loop_` when entering this function, and therefore is not part of the dataframe (column names or body).

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
                _line = _f.readline()
                _line_count += 1
                if not _line.strip():
                    # blank line between `loop_` and first header row
                    continue
                elif _line.startswith('_'):
                    # column header
                    _header.append(_line)
                    continue
                elif _line.startswith('#'):
                    # line is a comment, discarding for now
                    utils.log(f'Found comment at STAR file line {_line_count}, will not be preserved if writing star file later')
                    continue
                elif len(_line.split()) == len([column for column in _header if column.startswith('_')]):
                    # first data line
                    _block_start_line = _line_count
                    break
                else:
                    # unrecognized data block format
                    raise RuntimeError
            while True:
                # get length of data block
                _line = _f.readline()
                _line_count += 1
                if not _line:
                    # end of file, therefore end of data block
                    return _header, _block_start_line, _line_count, True
                elif _line.strip() == '':
                    # end of data block
                    return _header, _block_start_line, _line_count, False

        preambles = []
        blocks = {}
        line_count = 0
        with open(sourcefile, 'r') as f:
            # iterates once per preamble/header/block combination, ends when parse_preamble detects EOF
            while True:
                # file cursor is at the beginning of the file (first iteration) or at the end of a data_ block (subsequent iterations); parsing preamble of next data_ block
                preamble, block_name, line_count = parse_preamble(f, line_count)
                if preamble:
                    preambles.append(preamble)
                if block_name is None:
                    return preambles, blocks

                # file cursor is at a `data_*` line; now parsing contents of this block from either simple block (as dictionary) or complex block (as pandas dataframe)
                while True:
                    line = f.readline()
                    line_count += 1
                    if line.startswith('_'):
                        # no loop_ detected, this is a simple STAR block
                        block_dictionary, line_count, end_of_file = parse_single_dictionary(f, line_count, line)
                        blocks[block_name] = block_dictionary
                        break
                    elif line.startswith('loop_'):
                        # this is a complex block
                        preambles[-1].append(line.strip())
                        header, block_start_line, line_count, end_of_file = parse_single_block(f, line_count)
                        blocks[block_name] = [header, block_start_line, line_count]
                        break
                    elif not line:
                        # the data_ block contains no details and end of file reached
                        end_of_file = True
                        blocks[block_name] = {}  # treated as empty simple block
                        break
                    else:
                        # blank lines, comment lines, etc
                        preambles[-1].append(line.strip())

                if end_of_file:
                    return preambles, blocks

    def _load(self,
              blocks: dict[str, [list[str], int, int]]) -> dict[str, pd.DataFrame]:
        """
        Load each table-style data block of a pre-skeletonized star file into a pandas dataframe.

        :param blocks: dict mapping block names (e.g. `data_particles`) to a list of constituent column headers (e.g. `_rlnImageName),
            the first file line containing the data values of that block, and the last file line containing data values of that block
        :return: dict mapping block names (e.g. `data_particles`) to the corresponding data as a pandas dataframe
        """

        def load_single_block(_header: list[str],
                              _block_start_line: int,
                              _block_end_line: int) -> pd.DataFrame:
            """
            Load a single data block of a pre-skeletonized star file into a pandas dataframe.
            Only needs to be called (and should only be called) on blocks containing complex STAR blocks that are to be loaded as pandas dataframes.

            :param _header: list of column headers (e.g. `_rlnImageName) of the data block
            :param _block_start_line: the first file line containing the data values of the data block
            :param _block_end_line: the last file line containing data values of the data block
            :return: pandas dataframe of the data block values
            """
            columns = [line.split(' ')[0].strip() for line in _header if line.startswith('_')]

            # load the first 1 row to get dtypes of columns
            df = pd.read_csv(self.sourcefile,
                             sep=r'\s+',  # raw string to avoid syntaxwarnings when compiling documentation
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
                             sep=r'\s+',
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
            if type(blocks[block_name]) is dict:
                # this is a simple STAR block and was loaded earlier during _skeletonize
                # or this is an empty block (i.e. a `data_` block was declared but no following rows were found)
                pass
            elif type(blocks[block_name]) is list:
                # this list describes the column headers, start row, and end row of a table-style block
                header, block_start_line, block_end_line = blocks[block_name]
                blocks[block_name] = load_single_block(header, block_start_line, block_end_line)
            else:
                raise TypeError(f'Unknown block type {type(blocks[block_name])}; value of self.blocks[block_name] should be a python dictionary '
                                f'or a list of defining rows to be parsed into a pandas dataframe')
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
            Write a dataframe associated with a specific block name to a pre-existing file handle.

            :param _f: pre-existing file handle to which to write this block's contents
            :param _block_name: name of star file block to write (e.g. `data_`, `data_particles`)
            :return: None
            """
            df = self.blocks[_block_name]
            headers = [f'{header} #{i + 1}' for i, header in enumerate(df.columns.values.tolist())]
            _f.write('\n'.join(headers))
            _f.write('\n')
            df.to_csv(_f, index=False, header=False, mode='a', sep='\t')

        def write_single_dictionary(_f: TextIO,
                                    _block_name: str) -> None:
            """
            Write a dictionary associated with a specific block name to a pre-existing file handle.

            :param _f: pre-existing file handle to which to write this block's contents
            :param _block_name: name of star file dictionary to write (e.g. `data_`)
            :return: None
            """
            dictionary = self.blocks[_block_name]
            for key, value in dictionary.items():
                _f.write(f'{key}\t{value}\n')

        with open(outstar, 'w') as f:
            if timestamp:
                f.write('# Created {}\n'.format(dt.now()))

            for preamble, block_name in zip(self.preambles, self.block_names):
                for row in preamble:
                    f.write(row)
                    f.write('\n')
                # check if block is dataframe or dictionary, separate writing methods for each
                if type(self.blocks[block_name]) is pd.DataFrame:
                    write_single_block(f, block_name)
                elif type(self.blocks[block_name]) is dict:
                    write_single_dictionary(f, block_name)
                else:
                    raise TypeError(f'Unknown block type {type(self.blocks[block_name])}; value of self.blocks[block_name] should be a python dictionary or a pandas dataframe')
                f.write('\n')

        utils.log(f'Wrote {os.path.abspath(outstar)}')

    def get_particles_stack(self,
                            particles_block_name: str = None,
                            particles_path_column: str = None,
                            datadir: str = None,
                            lazy: bool = False) -> np.ndarray | list[mrc.LazyImage]:
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

        # group star file by mrcs file and get indices of each image within corresponding mrcs file
        mrcs_files, mrcs_grouped_image_inds = self._group_image_inds_by_mrcs(particles_block_name=particles_block_name,
                                                                             particles_path_column=particles_path_column)

        # confirm where to load MRC file(s) from disk
        if datadir is None:
            # if star file contains relative paths to images, and star file is being loaded from other directory, try setting datadir to starfile abspath
            datadir = os.path.dirname(self.sourcefile)
        mrcs_files = utils.prefix_paths(mrcs_files, datadir)

        # identify key parameters for creating image data array using the first mrcs file
        header = mrc.parse_header(mrcs_files[0])
        boxsize = header.boxsize  # image size along one dimension in pixels
        dtype = header.dtype

        # confirm that all mrcs files match this boxsize and dtype
        for mrcs_file in mrcs_files:
            _h = mrc.parse_header(mrcs_file)
            assert boxsize == _h.boxsize
            assert dtype == _h.dtype

        # calculate the number of bytes corresponding to one image in the mrcs files
        stride = dtype.itemsize * boxsize * boxsize

        if lazy:
            lazyparticles = [mrc.LazyImage(fname=file,
                                           shape=(boxsize, boxsize),
                                           dtype=dtype,
                                           offset=header.total_header_bytes + ind_img * stride)
                             for ind_stack, file in zip(mrcs_grouped_image_inds, mrcs_files)
                             for ind_img in ind_stack]
            return lazyparticles
        else:
            # preallocating numpy array for in-place loading, fourier transform, fourier transform centering, etc.
            # allocating 1 extra pixel along x and y dimensions in anticipation of symmetrizing the hartley transform in-place
            particles = np.zeros((len(self.blocks[particles_block_name]), boxsize + 1, boxsize + 1), dtype=np.float32)
            loaded_images = 0
            for ind_stack, file in zip(mrcs_grouped_image_inds, mrcs_files):
                particles[loaded_images:loaded_images + len(ind_stack), :-1, :-1] = mrc.LazyImageStack(fname=file,
                                                                                                       indices_image=ind_stack).get(low_memory=True)
                loaded_images += len(ind_stack)
            return particles

    def _group_image_inds_by_mrcs(self,
                                  particles_block_name: str = None,
                                  particles_path_column: str = None) -> tuple[list[str], list[np.ndarray]]:
        """
        Group the starfile `particles_path_column` by its referenced mrcs files, then by the indices of images referenced within those mrcs files, respecting star file row order.

        :param particles_block_name: name of star file block containing particle path column (e.g. `data_`, `data_particles`)
        :param particles_path_column: name of star file column containing path to particle images .mrcs (e.g. `_rlnImageName`)
        :return: mrcs_files: list of each mrcs path found in the star file that is unique from the preceeding row.
                 mrcs_grouped_image_inds: list of indices of images within the associated mrcs file which are referenced by the star file
        """
        # get the star file column containing the location of each image on disk
        images = self.blocks[particles_block_name][particles_path_column]
        images = [x.split('@') for x in images]  # assumed format is index@path_to_mrc

        # create new columns for 0-indexed image index and associated mrcs file
        self.blocks[particles_block_name]['_rlnImageNameInd'] = [int(x[0]) - 1 for x in images]  # convert to 0-based indexing of full dataset
        self.blocks[particles_block_name]['_rlnImageNameBase'] = [x[1] for x in images]

        # group image indices by associated mrcs file, respecting star file order
        # i.e. a mrcs file may be referenced discontinously in input star file, and should its images be separately grouped here
        mrcs_files = []
        mrcs_grouped_image_inds = []
        for i, group in self.blocks[particles_block_name].groupby(
                (self.blocks[particles_block_name]['_rlnImageNameBase'].shift() != self.blocks[particles_block_name]['_rlnImageNameBase']).cumsum(), sort=False):
            # mrcs_files = [path1, path2, ...]
            mrcs_files.append(group['_rlnImageNameBase'].iloc[0])
            # grouped_image_inds = [ [0, 1, 2, ..., N], [0, 3, 4, ..., M], ..., ]
            mrcs_grouped_image_inds.append(group['_rlnImageNameInd'].to_numpy())

        return mrcs_files, mrcs_grouped_image_inds

    def identify_particles_data_block(self,
                                      column_substring: str = 'Angle') -> str:
        """
        Attempt to identify the block_name of the data block within the star file for which rows refer to particle data (as opposed to optics or other data).

        :param column_substring: Search pattern to identify as substring within column name for particles block
        :return: the block name of the particles data block (e.g. `data` or `data_particles`)
        """
        block_name = None
        for block_name in self.block_names:
            # find the dataframe containing particle data
            if any(self.blocks[block_name].columns.str.contains(pat=column_substring)):
                return block_name
        if block_name is None:
            raise RuntimeError(f'Could not identify block containing particle data in star file (by searching for column containing text {column_substring}` in all blocks)')


class TiltSeriesStarfile(GenericStarfile):
    """
    Class to parse a particle image-series star file from upstream STA software.
    Each row in the star file must describe an individual image of a particle; groups of related rows describe all images observing one particle.
    """

    def __init__(self,
                 starfile: str,
                 source_software: TILTSERIESSTARFILE_STAR_SOURCES = 'auto'):
        # initialize object from parent class with parent attributes assigned at parent __init__
        super().__init__(starfile)

        # pre-initialize header aliases as None, to be set as appropriate by guess_metadata_interpretation()
        self.block_optics = None
        self.block_particles = None

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

        self.header_image_random_split = '_tomodrgnRandomSubset'
        self.image_ctf_premultiplied = None
        self.image_dose_weighted = None
        self.image_tilt_weighted = None

        self.ind_imgs = None
        self.ind_ptcls = None
        self.sort_ptcl_imgs = 'unsorted'
        self.use_first_ntilts = -1
        self.use_first_nptcls = -1
        self.sourcefile_filtered = None

        self.source_software = source_software

        # infer the upstream metadata format
        if source_software == 'auto':
            self._infer_metadata_mapping()
        elif source_software == TiltSeriesStarfileStarHeaders.warp.name:
            self._warp_metadata_mapping()
        elif source_software == TiltSeriesStarfileStarHeaders.cryosrpnt.name:
            self._cryosrpnt_metadata_mapping()
        elif source_software == TiltSeriesStarfileStarHeaders.nextpyp.name:
            self._nextpyp_metadata_mapping()
        elif source_software == TiltSeriesStarfileStarHeaders.cistem.name:
            self._cistem_metadata_mapping()
        else:
            raise ValueError(f'Unrecognized source_software {source_software} not one of known starfile sources for TiltSeriesStarfile {TILTSERIESSTARFILE_STAR_SOURCES}')

    def _warp_metadata_mapping(self):
        utils.log(f'Using STAR source software: {TiltSeriesStarfileStarHeaders.warp.name}')

        # easy reference to particles data block
        self.block_particles = 'data_'

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
        self.image_ctf_premultiplied = False
        self.image_dose_weighted = False
        self.image_tilt_weighted = False

    def _cryosrpnt_metadata_mapping(self):
        utils.log(f'Using STAR source software: {TiltSeriesStarfileStarHeaders.cryosrpnt.name}')

        # easy reference to particles data block
        self.block_particles = 'data_'

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
        self.image_ctf_premultiplied = False
        self.image_dose_weighted = False
        self.image_tilt_weighted = False

    def _nextpyp_metadata_mapping(self):
        utils.log(f'Using STAR source software: {TiltSeriesStarfileStarHeaders.nextpyp.name}')

        # easy reference to particles data block
        self.block_optics = 'data_optics'
        self.block_particles = 'data_particles'

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
        self.df = self.df.merge(self.blocks[self.block_optics], on='_rlnOpticsGroup', how='inner', validate='many_to_one', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # set additional headers needed by tomodrgn
        self.df[self.header_ptcl_dose] = self.df['_rlnCtfBfactor'] / -4
        self.df[self.header_ptcl_tilt] = np.arccos(self.df['_rlnCtfScalefactor'])
        self.df[self.header_pose_tx] = self.df[self.header_pose_tx_angst] / self.df[self.header_ctf_angpix]
        self.df[self.header_pose_ty] = self.df[self.header_pose_ty_angst] / self.df[self.header_ctf_angpix]

        # image processing applied during particle extraction
        self.image_ctf_premultiplied = False
        self.image_dose_weighted = False
        self.image_tilt_weighted = False

    def _cistem_metadata_mapping(self):
        utils.log(f'Using STAR source software: {TiltSeriesStarfileStarHeaders.cistem.name}')
        raise NotImplementedError

    def _infer_metadata_mapping(self) -> None:
        """
        Infer particle source software and version for key metadata and extraction-time processing corrections

        :return: None
        """

        headers = {block_name: self.blocks[block_name].columns.values.tolist() for block_name in self.block_names}
        match headers:

            case TiltSeriesStarfileStarHeaders.warp.value:
                self._warp_metadata_mapping()

            case TiltSeriesStarfileStarHeaders.cryosrpnt.value:
                self._cryosrpnt_metadata_mapping()

            case TiltSeriesStarfileStarHeaders.nextpyp.value:
                self._nextpyp_metadata_mapping()

            case TiltSeriesStarfileStarHeaders.cistem.value:
                self._cistem_metadata_mapping()

            case _:
                raise NotImplementedError(f'Auto detection of source software failed. '
                                          f'Consider retrying with manually specified `source_software`.'
                                          f'Found STAR file headers: {headers}. '
                                          f'TomoDRGN known STAR file headers: {[e.name for e in TiltSeriesStarfileStarHeaders]}')

    @property
    def headers_rot(self) -> list[str]:
        """
        Shortcut to return headers associated with rotation parameters.

        :return: list of particles dataframe header names for rotations
        """
        return [self.header_pose_phi,
                self.header_pose_theta,
                self.header_pose_psi]

    @property
    def headers_trans(self) -> list[str]:
        """
        Shortcut to return headers associated with translation parameters.

        :return: list of particles dataframe header names for translations
        """
        return [self.header_pose_tx,
                self.header_pose_ty]

    @property
    def headers_ctf(self) -> list[str]:
        """
        Shortcut to return headers associated with CTF parameters.

        :return: list of particles dataframe header names for CTF parameters
        """
        return [self.header_ctf_angpix,
                self.header_ctf_defocus_u,
                self.header_ctf_defocus_v,
                self.header_ctf_defocus_ang,
                self.header_ctf_voltage,
                self.header_ctf_cs,
                self.header_ctf_w,
                self.header_ctf_ps]

    @property
    def df(self) -> pd.DataFrame:
        """
        Shortcut to access the particles dataframe associated with the TiltSeriesStarfile object.

        :return: pandas dataframe of particles metadata
        """
        return self.blocks[self.block_particles]

    @df.setter
    def df(self,
           value: pd.DataFrame) -> None:
        """
        Shortcut to update the particles dataframe associated with the TiltSeriesStarfile object

        :param value: modified particles dataframe
        :return: None
        """
        self.blocks[self.block_particles] = value

    def __len__(self) -> int:
        """
        Return the number of rows (images) in the particles dataframe associated with the TiltSeriesStarfile object.

        :return: the number of rows in the dataframe
        """
        return len(self.df)

    def get_tiltseries_pixelsize(self) -> float | int:
        """
        Returns the pixel size of the extracted particles in Ångstroms.
        Assumes all particles have the same pixel size.

        :return: pixel size in Ångstroms/pixel
        """
        pixel_sizes = self.df[self.header_ctf_angpix].value_counts().index.to_numpy()
        if len(pixel_sizes) > 1:
            print(f'WARNING: found multiple pixel sizes {pixel_sizes} in star file! '
                  f' TomoDRGN does not support this for any volume-space reconstructions (e.g. backproject_voxel, train_vae).'
                  f' Will use the most common pixel size {pixel_sizes[0]}, but this will almost certainly lead to incorrect results.')
        return pixel_sizes[0]

    def get_tiltseries_voltage(self) -> float | int:
        """
        Returns the voltage of the microscope used to image the particles in kV.

        :return: voltage in kV
        """
        voltages = self.df[self.header_ctf_voltage].value_counts().index.to_numpy()
        if len(voltages) > 1:
            print(f'WARNING: found multiple voltages {voltages} in star file! '
                  f' TomoDRGN does not support this for any volume-space reconstructions (e.g. backproject_voxel, train_vae).'
                  f' Will use the most common voltage {voltages[0]}, but this will almost certainly lead to incorrect results.')
        return voltages[0]

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
            stack_path = utils.prefix_paths([stack_path], datadir)[0]
        assert os.path.exists(stack_path), f'{stack_path} not found'
        header = mrc.parse_header(stack_path)
        return header.boxsize

    def filter(self,
               ind_imgs: np.ndarray | str = None,
               ind_ptcls: np.ndarray | str = None,
               sort_ptcl_imgs: Literal['unsorted', 'dose_ascending', 'random'] = 'unsorted',
               use_first_ntilts: int = -1,
               use_first_nptcls: int = -1) -> None:
        """
        Filter the TiltSeriesStarfile in-place by image indices (rows) and particle indices (groups of rows corresponding to the same particle).
        Operations are applied in order: `ind_img -> ind_ptcl -> sort_ptcl_imgs -> use_first_ntilts -> use_first_nptcls`.

        :param ind_imgs: numpy array or path to numpy array of integer row indices to preserve, shape (N)
        :param ind_ptcls: numpy array or path to numpy array of integer particle indices to preserve, shape (N)
        :param sort_ptcl_imgs: sort the star file images on a per-particle basis by the specified criteria
        :param use_first_ntilts: keep the first `use_first_ntilts` images of each particle in the sorted star file.
                Default -1 means to use all. Will drop particles with fewer than this many tilt images.
        :param use_first_nptcls: keep the first `use_first_nptcls` particles in the sorted star file.
                Default -1 means to use all.
        :return: None
        """

        # save inputs as attributes of object for ease of future saving config
        self.ind_imgs = ind_imgs
        self.ind_ptcls = ind_ptcls
        self.sort_ptcl_imgs = sort_ptcl_imgs
        self.use_first_ntilts = use_first_ntilts
        self.use_first_nptcls = use_first_nptcls

        # how many particles does the star file initially contain
        ptcls_unique_list = self.df[self.header_ptcl_uid].unique().to_numpy()
        utils.log(f'Found {len(ptcls_unique_list)} particles in input star file')

        # assign unfiltered indices as column to allow easy downstream identification of preserved particle indices
        self.df['_UnfilteredParticleInds'] = self.df.groupby(self.header_ptcl_uid, sort=False).ngroup()

        # filter by image (row of dataframe) by presupplied indices
        if ind_imgs is not None:
            utils.log('Filtering particle images by supplied indices')

            if type(ind_imgs) is str:
                if ind_imgs.endswith('.pkl'):
                    ind_imgs = utils.load_pkl(ind_imgs)
                else:
                    raise ValueError(f'Expected .pkl file for {ind_imgs=}')

            assert min(ind_imgs) >= 0
            assert max(ind_imgs) <= len(self.df)

            self.df = self.df.iloc[ind_imgs].reset_index(drop=True)

        # filter by particle (group of rows sharing common header_ptcl_uid) by presupplied indices
        if ind_ptcls is not None:
            utils.log('Filtering particles by supplied indices')

            if type(ind_ptcls) is str:
                if ind_ptcls.endswith('.pkl'):
                    ind_ptcls = utils.load_pkl(ind_ptcls)
                else:
                    raise ValueError(f'Expected .pkl file for {ind_ptcls=}')

            assert min(ind_ptcls) >= 0
            assert max(ind_ptcls) <= len(ptcls_unique_list)

            ptcls_unique_list = ptcls_unique_list[ind_ptcls]
            self.df = self.df[self.df[self.header_ptcl_uid].isin(ptcls_unique_list)]
            self.df = self.df.reset_index(drop=True)

            assert len(self.df[self.header_ptcl_uid].unique().to_numpy()) == len(ind_ptcls), 'Make sure particle indices file does not contain duplicates'

        # create temp mapping of input particle order in star file to preserve after sorting
        self.df['_temp_input_ptcl_order'] = self.df.groupby(self.header_ptcl_uid, sort=False).ngroup()
        # sort the star file per-particle by the specified method
        if sort_ptcl_imgs != 'unsorted':
            utils.log(f'Sorting star file per-particle by {sort_ptcl_imgs}')
            if sort_ptcl_imgs == 'dose_ascending':
                # sort by header_ptcl_uid first to keep images of the same particle together, then sort by header_ptcl_dose
                self.df = self.df.sort_values(by=['_temp_input_ptcl_order', self.header_ptcl_dose], ascending=True).reset_index(drop=True)
            elif sort_ptcl_imgs == 'random':
                # group by header_ptcl_uid first to keep images of the same particle together, then shuffle rows within each group
                self.df = self.df.groupby(self.header_ptcl_uid, sort=False).sample(frac=1).reset_index(drop=True)
            else:
                raise ValueError(f'Unsupported value for {sort_ptcl_imgs=}')

        # keep the first ntilts images of each particle
        if use_first_ntilts != -1:
            utils.log(f'Keeping first {use_first_ntilts} images of each particle. Excluding particles with fewer than this many images.')
            self.df = self.df.groupby(self.header_ptcl_uid, sort=False).head(use_first_ntilts).reset_index(drop=True)

            # if a particledoes not have ntilts images, drop it
            rows_to_drop = self.df.loc[self.df.groupby(self.header_ptcl_uid, sort=False)[self.header_ptcl_uid].transform('count') < use_first_ntilts].index
            num_ptcls_to_drop = len(self.df.loc[rows_to_drop, self.header_ptcl_uid].unique())
            if num_ptcls_to_drop > 0:
                utils.log(f'Dropping {num_ptcls_to_drop} from star file due to having fewer than {use_first_ntilts=} tilt images per particle')
            self.df = self.df.drop(rows_to_drop).reset_index(drop=True)

        # keep the first nptcls particles
        if use_first_nptcls != -1:
            utils.log(f'Keeping first {use_first_nptcls=} particles.')
            ptcls_unique_list = self.df[self.header_ptcl_uid].unique().to_numpy()
            ptcls_unique_list = ptcls_unique_list[:use_first_nptcls]
            self.df = self.df[self.df[self.header_ptcl_uid].isin(ptcls_unique_list)]
            self.df = self.df.reset_index(drop=True)

        # order the final star file by input particle order, then by image indices in MRC file for contiguous file I/O
        images = [x.split('@') for x in self.df[self.header_ptcl_image]]  # assumed format is index@path_to_mrc
        self.df['_rlnImageNameInd'] = [int(x[0]) - 1 for x in images]  # convert to 0-based indexing of full dataset
        self.df = self.df.sort_values(by=['_temp_input_ptcl_order', '_rlnImageNameInd'], ascending=True).reset_index(drop=True)
        self.df = self.df.drop(['_temp_input_ptcl_order', '_rlnImageNameInd'], axis=1)

    def make_test_train_split(self,
                              fraction_split1: float = 0.5,
                              show_summary_stats: bool = True) -> None:
        """
        Create indices for tilt images assigned to train vs test split.
        Images are randomly assigned to one set or the other by respecting `fraction_train` on a per-particle basis.
        Random split is stored in `self.df` under the `self.header_image_random_split` column.

        :param fraction_split1: fraction of each particle's tilt images to label split1. All others will be labeled split2.
        :param show_summary_stats: log distribution statistics of particle sampling for test/train splits
        :return: None
        """

        # check required inputs are present
        df_grouped = self.df.groupby(self.header_ptcl_uid, sort=False)
        assert 0 < fraction_split1 <= 1.0

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

            # calculate the number of images to use in train split for this particle
            n_inds_train = np.rint(len(inds_img) * fraction_split1).astype(int)

            # generate sorted indices for images in train split by sampling without replacement
            inds_img_train = np.random.choice(inds_img, size=n_inds_train, replace=False)
            inds_img_train = np.sort(inds_img_train)
            inds_train.append(inds_img_train)

            # assign all other images of this particle to the test split
            inds_img_test = np.array(list(set(inds_img) - set(inds_img_train)))
            inds_test.append(inds_img_test)

        # provide summary statistics
        if show_summary_stats:
            utils.log(f'    Number of tilts sampled by inds_train: {set([len(inds_img_train) for inds_img_train in inds_train])}')
            utils.log(f'    Number of tilts sampled by inds_test: {set([len(inds_img_test) for inds_img_test in inds_test])}')

        # flatten indices
        inds_train = np.asarray([ind_img for inds_img_train in inds_train for ind_img in inds_img_train])
        inds_test = np.asarray([ind_img for inds_img_test in inds_test for ind_img in inds_img_test])

        # sanity check: the intersection of inds_train and inds_test should be empty
        assert len(set(inds_train) & set(inds_test)) == 0, len(set(inds_train) & set(inds_test))
        # sanity check: the union of inds_train and inds_test should be the total number of images in the particles dataframe
        assert len(set(inds_train) | set(inds_test)) == len(self.df), len(set(inds_train) | set(inds_test))

        # store random split in particles dataframe
        self.df[self.header_image_random_split] = np.zeros(len(self.df), dtype=np.uint8)
        self.df.loc[inds_train, self.header_image_random_split] = 1
        self.df.loc[inds_test, self.header_image_random_split] = 2

    def plot_particle_uid_ntilt_distribution(self,
                                             outpath: str) -> None:
        """
        Plot the distribution of the number of tilt images per particle as a line plot (against star file particle index) and as a histogram.

        :param outpath: file name to save the plot
        :return: None
        """
        ptcls_to_imgs_ind = self.get_ptcl_img_indices()
        ntilts_per_particle = np.asarray([len(ptcl_to_imgs_ind) for ptcl_to_imgs_ind in ptcls_to_imgs_ind])
        ntilts, counts_per_ntilt = np.unique(ntilts_per_particle, return_counts=True)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(ntilts_per_particle, linewidth=0.5)
        ax1.set_xlabel('star file particle index')
        ax1.set_ylabel('ntilts per particle')

        ax2.bar(ntilts, counts_per_ntilt)
        ax2.set_xlabel('ntilts per particle')
        ax2.set_ylabel('count')

        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    def get_particles_stack(self,
                            *,
                            datadir: str = None,
                            lazy: bool = False,
                            **kwargs) -> np.ndarray | list[mrc.LazyImage]:
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

    def write(self,
              *args,
              **kwargs) -> None:
        """
        Temporarily removes columns in data_particles dataframe that are present in data_optics dataframe (to restore expected input star file format), then calls parent GenericStarfile write.

        :param args: Passed to parent GenericStarfile write
        :param kwargs: Passed to parent GenericStarfile write
        :return: None
        """
        if self.block_optics is not None:
            # during loading TiltSeriesStarfile, block_optics and block_particles are merged for internal convenience when different upstream software either do or do not include data_optics block
            columns_in_common = self.df.columns.intersection(self.blocks[self.block_optics].columns)
            # need to preserve the optics groups in the data_particles block
            columns_in_common = columns_in_common.drop('_rlnOpticsGroup')
            # drop all other columns in common from the data_particles block
            self.df = self.df.drop(columns_in_common, axis=1)

        # now call parent write method
        super().write(*args, **kwargs)

        if self.block_optics is not None:
            # re-merge data_optics with data_particles so that the starfile object appears unchanged after calling this method
            self.df = self.df.merge(self.blocks[self.block_optics], on='_rlnOpticsGroup', how='inner', validate='many_to_one', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')


class TomoParticlesStarfile(GenericStarfile):
    """
    Class to parse a particle star file from upstream STA software.
    The input star file must be an optimisation set star file from e.g. WarpTools, RELION v5.
    The _rlnTomoParticlesFile referenced in the optimisation set must have each row describing a group of images observing a particular particle.
    This TomoParticlesStarfile is the object which is immediately loaded, though a reference to the parent optimisation set and related _lnTomoTomogramsFile are also stored
    (to reference TomoTomogramsStarfile if loading tomogram-level metadata, and to write a new optimisation set of modified the _rlnTomoParticlesFile contents).
    """

    def __init__(self,
                 starfile: str,
                 source_software: TOMOPARTICLESSTARFILE_STAR_SOURCES = 'auto'):
        # the input star file is the optimisation set; store its path and contents for future writing
        assert is_starfile_optimisation_set(starfile)
        self.optimisation_set_star_path = os.path.abspath(starfile)
        self.optimisation_set_star = GenericStarfile(self.optimisation_set_star_path)

        # the input star also references a TomoTomogramsFile; store its path and contents for future reference
        tomograms_star_rel_path = self.optimisation_set_star.blocks['data_']['_rlnTomoTomogramsFile']
        assert tomograms_star_rel_path != ''
        tomograms_star_path = os.path.join(os.path.dirname(self.optimisation_set_star_path), tomograms_star_rel_path)
        self.tomograms_star_path = tomograms_star_path
        self.tomograms_star = GenericStarfile(self.tomograms_star_path)

        # initialize the main TomoParticlesStarfile object from the _rlnTomoParticlesFile header
        ptcls_star_rel_path = self.optimisation_set_star.blocks['data_']['_rlnTomoParticlesFile']
        ptcls_star_path = os.path.join(os.path.dirname(self.optimisation_set_star_path), ptcls_star_rel_path)
        super().__init__(ptcls_star_path)

        # override the sourcefile attribute set by parent init to point to the optimisation set, since that is the file that must be passed to re-load this object
        self.sourcefile = self.optimisation_set_star_path

        # check that the particles star file references 2D image stacks
        assert self.blocks['data_general']['_rlnTomoSubTomosAre2DStacks'] == '1', 'TomoDRGN is only compatible with tilt series particles extracted as 2D image stacks.'

        # pre-initialize header aliases as None, to be set as appropriate by _infer_metadata_mapping()
        self.block_optics = None
        self.block_particles = None

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

        self.header_coord_x = None
        self.header_coord_y = None
        self.header_coord_z = None

        self.header_ptcl_uid = None
        self.header_ptcl_image = None
        self.header_ptcl_micrograph = None

        self.header_ptcl_random_split = None
        self.header_image_random_split = '_tomodrgnRandomSubset'
        self.image_ctf_premultiplied = None
        self.image_dose_weighted = None
        self.image_tilt_weighted = None

        self.ind_imgs = None
        self.ind_ptcls = None
        self.sort_ptcl_imgs = 'unsorted'
        self.use_first_ntilts = -1
        self.use_first_nptcls = -1
        self.sourcefile_filtered = None

        self.source_software = source_software

        # infer the upstream metadata format
        if source_software == 'auto':
            self._infer_metadata_mapping()
        elif source_software == TomoParticlesStarfileStarHeaders.warptools.name:
            utils.log(f'Using STAR source software: {TomoParticlesStarfileStarHeaders.warptools.name}')
            self._warptools_metadata_mapping()
        elif source_software == TomoParticlesStarfileStarHeaders.relion.name:
            utils.log(f'Using STAR source software: {TomoParticlesStarfileStarHeaders.relion.name}')
            self._relion_metadata_mapping()
        else:
            raise ValueError(f'Unrecognized source_software {source_software} not one of known starfile sources for TomoParticlesStarfile {TOMOPARTICLESSTARFILE_STAR_SOURCES}')

    def _warptools_metadata_mapping(self):
        # in the examples I have seen so far, the warptools metadata and relion metadata are equivalent for the fields required by tomodrgn
        self._relion_metadata_mapping()

    def _relion_metadata_mapping(self):
        # TomoParticlesStarfile optics block and contents
        self.block_optics = 'data_optics'

        self.header_ctf_angpix = '_rlnImagePixelSize'

        # TomoParticlesStarfile particles block and contents
        self.block_particles = 'data_particles'

        self.header_pose_phi = '_rlnAngleRot'
        self.header_pose_theta = '_rlnAngleTilt'
        self.header_pose_psi = '_rlnAnglePsi'

        self.header_pose_tx_angst = '_rlnOriginXAngst'
        self.header_pose_ty_angst = '_rlnOriginYAngst'
        self.header_pose_tz_angst = '_rlnOriginZAngst'
        self.header_pose_tx = '_rlnOriginX'
        self.header_pose_ty = '_rlnOriginY'
        self.header_pose_tz = '_rlnOriginZ'

        self.header_coord_x = '_rlnCoordinateX'
        self.header_coord_y = '_rlnCoordinateY'
        self.header_coord_z = '_rlnCoordinateZ'

        self.header_ptcl_uid = 'index'
        self.header_ptcl_image = '_rlnImageName'
        self.header_ptcl_tomogram = '_rlnTomoName'
        self.header_ptcl_random_split = '_rlnRandomSubset'  # used for random split per-particle (e.g. from RELION)
        self.header_image_random_split = '_tomodrgnRandomSubset'  # used for random split per-particle-image
        self.header_ptcl_visible_frames = '_rlnTomoVisibleFrames'
        self.header_ptcl_box_size = '_rlnImageSize'

        # TomoTomogramsStarfile global block and contents -- NOT accessible from this class directly
        self.header_ctf_voltage = '_rlnVoltage'
        self.header_ctf_cs = '_rlnSphericalAberration'
        self.header_ctf_w = '_rlnAmplitudeContrast'

        # TomoTomogramsStarfile TOMOGRAM_NAME block and contents -- NOT accessible from this class directly
        self.header_tomo_proj_x = '_rlnTomoProjX'
        self.header_tomo_proj_y = '_rlnTomoProjY'
        self.header_tomo_proj_z = '_rlnTomoProjZ'
        self.header_tomo_proj_w = '_rlnTomoProjW'

        self.header_ctf_defocus_u = '_rlnDefocusU'
        self.header_ctf_defocus_v = '_rlnDefocusV'
        self.header_ctf_defocus_ang = '_rlnDefocusAngle'
        self.header_ctf_ps = '_rlnPhaseShift'  # potentially not created yet

        self.header_tomo_dose = '_rlnMicrographPreExposure'
        self.header_tomo_tilt = '_tomodrgnPseudoStageTilt'  # pseudo because arccos returns values in [0,pi] so lose +/- tilt information

        # merge optics groups block with particle data block
        self.df = self.df.merge(self.blocks[self.block_optics], on='_rlnOpticsGroup', how='inner', validate='many_to_one', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # set additional headers needed by tomodrgn
        for tomo_name in self.df[self.header_ptcl_tomogram]:
            # create a temporary column with values of stage tilt in radians
            self.tomograms_star.blocks[f'data_{tomo_name}'][self.header_tomo_tilt] = np.arccos(self.tomograms_star.blocks[f'data_{tomo_name}']['_rlnCtfScalefactor'])
        self.df[self.header_pose_tx] = self.df[self.header_pose_tx_angst] / self.df[self.header_ctf_angpix]
        self.df[self.header_pose_ty] = self.df[self.header_pose_ty_angst] / self.df[self.header_ctf_angpix]
        self.df[self.header_pose_tz] = self.df[self.header_pose_tz_angst] / self.df[self.header_ctf_angpix]
        if self.header_ctf_ps not in self.df.columns:
            self.df[self.header_ctf_ps] = np.zeros(len(self.df), dtype=float)

        # convert the _rlnTomoVisibleFrames column from default dtype inferred by pandas (str of list of int, e.g. '[1,1,0,1,...]' to numpy array of ints
        # more efficient (though less robust) than ast.literal_eval because we know the data structure ahead of time
        self.df[self.header_ptcl_visible_frames] = [np.asarray([include for include in ptcl_frames.replace('[', '').replace(']', '').split(',')], dtype=int)
                                                    for ptcl_frames in self.df[self.header_ptcl_visible_frames]]

        # convert the _rlnTomoProj{X,Y,Z,W} columns from default dtype inferred by pandas (str of list of float, e.g. '[1.0,0.0,0.0,0]' to numpy array of floats
        for tomogram_block_name in self.tomograms_star.block_names:
            if tomogram_block_name == 'data_global':
                # this is global data block, no projection matrices to convert
                continue
            df_tomo = self.tomograms_star.blocks[tomogram_block_name]
            projection_matrices_headers = [self.header_tomo_proj_x, self.header_tomo_proj_y, self.header_tomo_proj_z, self.header_tomo_proj_w]
            for projection_matrices_header in projection_matrices_headers:
                df_tomo[projection_matrices_header] = [np.asarray([proj_element for proj_element in tilt_proj.replace('[', '').replace(']', '').split(',')], dtype=float)
                                                       for tilt_proj in df_tomo[projection_matrices_header]]

        # image processing applied during particle extraction
        self.image_ctf_premultiplied = bool(self.blocks[self.block_optics]['_rlnCtfDataAreCtfPremultiplied'].to_numpy()[0])
        self.image_dose_weighted = True  # warptools applie fixed exposure weights per-frequency for each extracted image
        self.image_tilt_weighted = False

        # note columns added during init, so that we can remove these columns later when writing the star file
        self.tomodrgn_added_headers = [self.header_pose_tx, self.header_pose_ty, self.header_pose_tz, self.header_ctf_ps]

    def _infer_metadata_mapping(self) -> None:
        """
        Infer particle source software and version for key metadata and extraction-time processing corrections

        :return: None
        """

        headers = {block_name: (self.blocks[block_name].columns.values.tolist() if type(self.blocks[block_name]) is pd.DataFrame else list(self.blocks[block_name].keys()))
                   for block_name in self.block_names}
        match headers:

            case TomoParticlesStarfileStarHeaders.warptools.value:
                utils.log(f'Using STAR source software: {TomoParticlesStarfileStarHeaders.warptools.name}')
                self._warptools_metadata_mapping()

            case TomoParticlesStarfileStarHeaders.relion.value:
                utils.log(f'Using STAR source software: {TomoParticlesStarfileStarHeaders.relion.name}')
                self._relion_metadata_mapping()

            case _:
                raise NotImplementedError(f'Auto detection of source software failed. '
                                          f'Consider retrying with manually specified `source_software`.'
                                          f'Found STAR file headers: {headers}. '
                                          f'TomoDRGN known STAR file headers: {TomoParticlesStarfileStarHeaders}')

    @property
    def headers_rot(self) -> list[str]:
        """
        Shortcut to return headers associated with rotation parameters.

        :return: list of particles dataframe header names for rotations
        """
        return [self.header_pose_phi,
                self.header_pose_theta,
                self.header_pose_psi]

    @property
    def headers_trans(self) -> list[str]:
        """
        Shortcut to return headers associated with translation parameters.

        :return: list of particles dataframe header names for translations
        """
        return [self.header_pose_tx,
                self.header_pose_ty,
                self.header_pose_tz]

    @property
    def headers_ctf(self) -> list[str]:
        """
        Shortcut to return headers associated with CTF parameters.

        :return: list of particles dataframe header names for CTF parameters
        """
        return [self.header_ctf_angpix,
                self.header_ctf_defocus_u,
                self.header_ctf_defocus_v,
                self.header_ctf_defocus_ang,
                self.header_ctf_voltage,
                self.header_ctf_cs,
                self.header_ctf_w,
                self.header_ctf_ps]

    @property
    def df(self) -> pd.DataFrame:
        """
        Shortcut to access the particles dataframe associated with the TomoParticlesStarfile object.

        :return: pandas dataframe of particles metadata
        """
        return self.blocks[self.block_particles]

    @df.setter
    def df(self,
           value: pd.DataFrame) -> None:
        """
        Shortcut to update the particles dataframe associated with the TomoParticlesStarfile object

        :param value: modified particles dataframe
        :return: None
        """
        self.blocks[self.block_particles] = value

    def get_tiltseries_pixelsize(self) -> float | int:
        """
        Returns the pixel size of the extracted particles in Ångstroms.
        Assumes all particles have the same pixel size.

        :return: pixel size in Ångstroms/pixel
        """
        pixel_sizes = self.df[self.header_ctf_angpix].value_counts().index.to_numpy()
        if len(pixel_sizes) > 1:
            print(f'WARNING: found multiple pixel sizes {pixel_sizes} in star file! '
                  f' TomoDRGN does not support this for any volume-space reconstructions (e.g. backproject_voxel, train_vae).'
                  f' Will use the most common pixel size {pixel_sizes[0]}, but this will almost certainly lead to incorrect results.')
        return pixel_sizes[0]

    def get_tiltseries_voltage(self) -> float | int:
        """
        Returns the voltage of the microscope used to image the particles in kV.

        :return: voltage in kV
        """
        voltages = self.df[self.header_ctf_voltage].value_counts().index.to_numpy()
        if len(voltages) > 1:
            print(f'WARNING: found multiple voltages {voltages} in star file! '
                  f' TomoDRGN does not support this for any volume-space reconstructions (e.g. backproject_voxel, train_vae).'
                  f' Will use the most common voltage {voltages[0]}, but this will almost certainly lead to incorrect results.')
        return voltages[0]

    def get_ptcl_img_indices(self) -> list[np.ndarray]:
        """
        Returns the indices of each tilt image and associated metadata relative to the pre-filtered subset of all images of all particles in the star file.
        Filtering is done using the ``self.header_ptcl_visible_frames`` column.
        For example, using the first two dataframe rows of this column as ``[[1,1,0,1],[1,0,0,1]]``, this method would return indices ``[np.array([0,1,2]), np.array([3,4])]``.
        The number of tilt images per particle may vary across the STAR file, so returning a list (or object-type numpy array or ragged torch tensor) is required.

        :return: integer indices of each tilt image in the particles dataframe grouped by particle ID
        """
        images_per_ptcl = self.df[self.header_ptcl_visible_frames].apply(np.sum)  # array of number of included images per particle
        cumulative_images_per_ptcl = images_per_ptcl.cumsum().to_list()  # array of cumulative number of included images throughout entire dataframe
        cumulative_images_per_ptcl.insert(0, 0)
        ptcl_to_img_indices = [np.arange(start, stop) for start, stop in pairwise(cumulative_images_per_ptcl)]
        return ptcl_to_img_indices

    def get_image_size(self):
        raise NotImplementedError

    def filter(self,
               ind_imgs: np.ndarray | str = None,
               ind_ptcls: np.ndarray | str = None,
               sort_ptcl_imgs: Literal['unsorted', 'dose_ascending', 'random'] = 'unsorted',
               use_first_ntilts: int = -1,
               use_first_nptcls: int = -1) -> None:
        """
        Filter the TomoParticlesStarfile in-place by image indices (e.g., datafram _rlnTomoVisibleFrames column) and particle indices (dataframe rows).
        Operations are applied in order: `ind_img -> ind_ptcl -> sort_ptcl_imgs -> use_first_ntilts -> use_first_nptcls`.

        :param ind_imgs: numpy array or path to numpy array of integer images to preserve, shape (nimgs),
                Sets values in the _rlnTomoVisibleFrames column to 0 if that image's index is not in ind_imgs.
        :param ind_ptcls: numpy array or path to numpy array of integer particle indices to preserve, shape (nptcls).
                Drops particles from the dataframe if that particle's index is not in ind_ptcls.
        :param sort_ptcl_imgs: sort the star file images on a per-particle basis by the specified criteria.
                This is primarily useful in combination with ``use_first_ntilts`` to get the first ``ntilts`` images of each particle after sorting.
        :param use_first_ntilts: keep the first `use_first_ntilts` images (of those images previously marked to be included by _rlnTomoVisibleFrames) of each particle in the sorted star file.
                Default -1 means to use all. Will drop particles with fewer than this many tilt images.
        :param use_first_nptcls: keep the first `use_first_nptcls` particles in the sorted star file.
                Default -1 means to use all.
        :return: None
        """

        # save inputs as attributes of object for ease of future saving config
        self.ind_imgs = ind_imgs
        self.ind_ptcls = ind_ptcls
        self.sort_ptcl_imgs = sort_ptcl_imgs
        self.use_first_ntilts = use_first_ntilts
        self.use_first_nptcls = use_first_nptcls

        # how many particles does the star file initially contain
        utils.log(f'Found {len(self.df)} particles in input star file')

        # filter by image (element of _rlnTomoVisibleFrames list per row) by presupplied indices
        if ind_imgs is not None:
            utils.log('Filtering particle images by supplied indices')

            if type(ind_imgs) is str:
                if ind_imgs.endswith('.pkl'):
                    ind_imgs = utils.load_pkl(ind_imgs)
                else:
                    raise ValueError(f'Expected .pkl file for {ind_imgs=}')

            assert min(ind_imgs) >= 0, 'The minimum allowable image index is 0'
            nimgs_total = self.df[self.header_ptcl_visible_frames].apply(len).sum()
            assert max(ind_imgs) <= nimgs_total, f'The maximum allowable image index is the total number of images referenced in {self.header_ptcl_visible_frames}: {nimgs_total}'
            unique_ind_imgs, unique_ind_imgs_counts = np.unique(ind_imgs, return_counts=True)
            assert np.all(unique_ind_imgs_counts == 1), f'Repeated image indices are not allowed, found the following repeated image indices: {unique_ind_imgs[unique_ind_imgs_counts != 1]}'

            ind_imgs_mask = np.zeros(nimgs_total)
            ind_imgs_mask[ind_imgs] = 1

            masked_visible_frames = []
            ind_img_cursor = 0
            for ptcl_visible_frames in self.df[self.header_ptcl_visible_frames]:
                # get the number of images in this image as the window width to draw from the ind_imgs_mask
                imgs_this_ptcl = len(ptcl_visible_frames)
                # only preserve images that were both initially marked 1 and are selected by ind_imgs
                masked_ptcl_visible_frames = np.logical_and(ptcl_visible_frames, ind_imgs_mask[ind_img_cursor: ind_img_cursor + imgs_this_ptcl]).astype(int)
                # append to an overall list for all particles
                masked_visible_frames.append(masked_ptcl_visible_frames)
                # increment the global image index offset by the number of images in this particle so that the next iteration's particle is correctly masked
                ind_img_cursor += imgs_this_ptcl

            self.df[self.header_ptcl_visible_frames] = masked_visible_frames

        # filter by particle (df row) by presupplied indices
        if ind_ptcls is not None:
            utils.log('Filtering particles by supplied indices')

            if type(ind_ptcls) is str:
                if ind_ptcls.endswith('.pkl'):
                    ind_ptcls = utils.load_pkl(ind_ptcls)
                else:
                    raise ValueError(f'Expected .pkl file for {ind_ptcls=}')

            assert min(ind_ptcls) >= 0
            assert max(ind_ptcls) <= len(self.df)
            unique_ind_ptcls, unique_ind_ptcls_counts = np.unique(ind_imgs, return_counts=True)
            assert np.all(unique_ind_ptcls_counts == 1), f'Repeated particle indices are not allowed, found the following repeated particle indices: {unique_ind_ptcls[unique_ind_ptcls_counts != 1]}'

            self.df = self.df.iloc[ind_ptcls, :].reset_index(drop=True)

        # sort the star file per-particle by the specified method
        if sort_ptcl_imgs != 'unsorted':
            utils.log(f'Sorting star file per-particle by {sort_ptcl_imgs}')
            sorted_visible_frames = []
            # apply sorting to TomoTomogramsStarfile by sorting rows per-tomogram-block, then apply updated tilt indexing to TomoParticlesStarfile header_ptcl_visible_frames to keep metadata in sync
            for tomo_name, ptcl_group_df in self.df.groupby(self.header_ptcl_tomogram, sort=False):

                if sort_ptcl_imgs == 'dose_ascending':
                    # sort the tilts of this tomo by dose
                    self.tomograms_star.blocks[f'data_{tomo_name}'] = self.tomograms_star.blocks[f'data_{tomo_name}'].sort_values(by=self.header_tomo_dose, ascending=True)
                elif sort_ptcl_imgs == 'random':
                    # sort the tilts of this tomo randomly
                    self.tomograms_star.blocks[f'data_{tomo_name}'] = self.tomograms_star.blocks[f'data_{tomo_name}'].sample(frac=1)
                else:
                    raise ValueError(f'Unsupported value for {sort_ptcl_imgs=}')

                # update the ordering of images via header_ptcl_visible_frames to match the corresponding tomogram df index
                reordered_tilts_this_tomo = self.tomograms_star.blocks[f'data_{tomo_name}'].index.to_numpy()
                for ptcl_visible_frames in ptcl_group_df[self.header_ptcl_visible_frames]:
                    # reindex this image's visible frames
                    sorted_ptcl_visible_frames = ptcl_visible_frames[reordered_tilts_this_tomo]
                    # recast this array of ints to the same input format (str of list) and append to an overall list for all particles
                    sorted_visible_frames.append(sorted_ptcl_visible_frames)

            # update the particles df header_ptcl_visible_frames to the newly sorted visible_frames
            self.df[self.header_ptcl_visible_frames] = sorted_visible_frames

        # keep the first ntilts images of each particle
        if use_first_ntilts != -1:
            utils.log(f'Keeping first {use_first_ntilts} images of each particle. Excluding particles with fewer than this many images.')

            assert use_first_ntilts > 0

            particles_to_drop_insufficient_tilts = []
            masked_visible_frames = []
            for ind_ptcl, ptcl_visible_frames in enumerate(self.df[self.header_ptcl_visible_frames]):
                # preserve the first use_first_ntilts frames that are already marked include (1); set the remainder to not include (0)
                cumulative_ptcl_visible_frames = np.cumsum(ptcl_visible_frames)
                masked_ptcl_visible_frames = np.where(cumulative_ptcl_visible_frames <= use_first_ntilts,
                                                      ptcl_visible_frames,
                                                      0)

                # check how many images are now included for this particle; add this particle to the list to be dropped if fewer than use_first_ntilts
                if sum(masked_ptcl_visible_frames) < use_first_ntilts:
                    particles_to_drop_insufficient_tilts.append(ind_ptcl)

                # append to an overall list for all particles
                masked_visible_frames.append(masked_ptcl_visible_frames)

            # update the particles df header_ptcl_visible_frames to the newly masked visible_frames
            self.df[self.header_ptcl_visible_frames] = masked_visible_frames

            # drop particles (rows) with fewer than use_first_ntilts
            self.df = self.df.drop(particles_to_drop_insufficient_tilts).reset_index(drop=True)

        # keep the first nptcls particles
        if use_first_nptcls != -1:
            utils.log(f'Keeping first {use_first_nptcls=} particles.')

            assert use_first_nptcls > 0

            # recalculate the ptcls_unique_list due to possible upstream filtering invalidating the original list
            self.df = self.df.iloc[:use_first_nptcls, :].reset_index(drop=True)

        # reset indexing of TomoTomogramsStarfile tomogram block rows and of TomoParticlesStarfile header_ptcl_visible_frames to keep image indexing in .mrcs consistent with metadata indexing in .star
        # only necessary if images were sorted
        if sort_ptcl_imgs != 'unsorted':

            unsorted_visible_frames = []
            for tomo_name, ptcl_group_df in self.df.groupby(self.header_ptcl_tomogram, sort=False):

                # create temporary index of sorted images
                self.tomograms_star.blocks[f'data_{tomo_name}']['_reindexed_img_order'] = np.arange(len(self.tomograms_star.blocks[f'data_{tomo_name}']))

                # undo the tilt image sorting applied by sort_ptcl_imgs
                self.tomograms_star.blocks[f'data_{tomo_name}'] = self.tomograms_star.blocks[f'data_{tomo_name}'].sort_index()

                # undo the header_ptcl_visible_frames sorting applied by sort_ptcl_imgs
                reordered_tilts_this_tomo = self.tomograms_star.blocks[f'data_{tomo_name}']['_reindexed_img_order'].to_numpy()
                for ptcl_visible_frames in ptcl_group_df[self.header_ptcl_visible_frames]:
                    # reindex this particle's visible frames
                    unsorted_ptcl_visible_frames = ptcl_visible_frames[reordered_tilts_this_tomo]
                    # append to an overall list for all particles
                    unsorted_visible_frames.append(unsorted_ptcl_visible_frames)

                # remove the temporary index of sorted images
                self.tomograms_star.blocks[f'data_{tomo_name}'] = self.tomograms_star.blocks[f'data_{tomo_name}'].drop(['_reindexed_img_order'], axis=1)

            # update the particles df header_ptcl_visible_frames to the newly unsorted visible_frames
            self.df[self.header_ptcl_visible_frames] = unsorted_visible_frames

    def make_test_train_split(self,
                              fraction_split1: float = 0.5,
                              show_summary_stats: bool = True) -> None:
        """
        Create indices for tilt images assigned to train vs test split.
        Images are randomly assigned to one set or the other by precisely respecting `fraction_train` on a per-particle basis.
        Random split is stored in `self.df` under the `self.header_image_random_split` column as a list of ints in (0, 1, 2) with length `self.header_ptcl_visible_frames`.
        These values map as follows:

        * 0: images marked to not include (value 0) in `self.header_ptcl_visible_frames`.
        * 1: images marked to include (value 1) in `self.header_ptcl_visible_frames`, assigned to image-level half-set 1
        * 2: images marked to include (value 1) in `self.header_ptcl_visible_frames`, assigned to image-level half-set 2

        :param fraction_split1: fraction of each particle's included tilt images to label split1. All other included images will be labeled split2.
        :param show_summary_stats: log distribution statistics of particle sampling for test/train splits
        :return: None
        """

        # check required inputs are present
        assert 0 < fraction_split1 <= 1.0

        # get indices associated with train and test
        train_test_split = []
        for ptcl_visible_frames in self.df[self.header_ptcl_visible_frames]:
            # get the number of included images, and split this set into the number of included assigned to train/test
            ptcl_n_imgs = np.sum(ptcl_visible_frames == 1)
            ptcl_n_imgs_train = np.rint(ptcl_n_imgs * fraction_split1).astype(int)
            ptcl_n_imgs_test = ptcl_n_imgs - ptcl_n_imgs_train

            # the initial array of ptcl_visible_frames already contains values in (0,1); set random ptcl_img_inds_test indices from 1 (include split1) to 2 (include split2)
            ptcl_img_inds = np.flatnonzero(ptcl_visible_frames == 1)
            ptcl_img_inds_test = np.random.choice(a=ptcl_img_inds, size=ptcl_n_imgs_test, replace=False)
            ptcl_visible_frames[ptcl_img_inds_test] = 2

            train_test_split.append(ptcl_visible_frames)

        # store random split in particles dataframe
        self.df[self.header_image_random_split] = train_test_split
        self.tomodrgn_added_headers.append(self.header_image_random_split)

        # provide summary statistics
        if show_summary_stats:
            ntilts_imgs_train = [np.sum(ptcl_imgs == 1) for ptcl_imgs in train_test_split]
            ntilts_imgs_test = [np.sum(ptcl_imgs == 2) for ptcl_imgs in train_test_split]
            utils.log(f'    Number of tilts sampled by inds_train: {sorted(list(set(ntilts_imgs_train)))}')
            utils.log(f'    Number of tilts sampled by inds_test: {sorted(list(set(ntilts_imgs_test)))}')

    def plot_particle_uid_ntilt_distribution(self,
                                             outpath: str) -> None:
        """
        Plot the distribution of the number of visible tilt images per particle as a line plot (against star file particle index) and as a histogram.

        :param outpath: file name to save the plot
        :return: None
        """
        ntilts_per_ptcl = self.df[self.header_ptcl_visible_frames].apply(np.sum)  # number of included images per particle
        unique_ntilts_per_ptcl, ptcl_counts_per_unique_ntilt = np.unique(ntilts_per_ptcl, return_counts=True)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(ntilts_per_ptcl, linewidth=0.5)
        ax1.set_xlabel('star file particle index')
        ax1.set_ylabel('ntilts per particle')

        ax2.bar(unique_ntilts_per_ptcl, ptcl_counts_per_unique_ntilt)
        ax2.set_xlabel('ntilts per particle')
        ax2.set_ylabel('count')

        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    def get_particles_stack(self,
                            *,
                            datadir: str = None,
                            lazy: bool = False,
                            check_headers: bool = False,
                            **kwargs) -> np.ndarray | list[mrc.LazyImageStack]:
        """
        Load the particles referenced in the TomoParticlesStarfile.
        Particles are loaded into memory directly as a numpy array of shape ``(n_images, boxsize+1, boxsize+1)``, or as a list of ``mrc.LazyImageStack`` objects of length ``n_particles``.
        The column specifying the path to images on disk must not specify the image index to load from that file (i.e., syntax like ``1@/path/to/stack.mrcs`` is not supported).
        Instead, specification of which images to load for each particle should be done in the ``_rlnTomoVisibleFrames`` column.

        :param datadir: absolute path to particle images .mrcs to override particles_path_column.
        :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True).
        :param check_headers: whether to parse each file's header to ensure consistency in dtype and array shape in X,Y (True),
                or to use the first .mrc(s) file as representative for the dataset (False).
                Caution that settting ``False`` is faster, but assumes that the first file's header is representative of all files.
        :return: np.ndarray of shape (n_ptcls * n_tilts, D, D) or list of LazyImage objects of length (n_ptcls * n_tilts)
        """
        # assert that no paths include `@` specification of individual images to load from the referenced file
        assert all(~self.df[self.header_ptcl_image].str.contains('@'))

        # validate where to load MRC file(s) from disk
        ptcl_mrcs_files = self.df[self.header_ptcl_image].to_list()
        if datadir is None:
            # if star file contains relative paths to images, and star file is being loaded from other directory, try setting datadir to starfile abspath
            datadir = os.path.dirname(self.sourcefile)
        ptcl_mrcs_files = utils.prefix_paths(ptcl_mrcs_files, datadir)

        # identify which tilt images to load for each particle
        all_ptcls_visible_frames = self.df[self.header_ptcl_visible_frames].to_list()  # [np.array([1, 1, 0]), np.array([1, 1, 1]), ...]
        all_ptcls_visible_frames = [[index for index, include in enumerate(ptcl_visible_frames) if include == 1] for ptcl_visible_frames in all_ptcls_visible_frames]  # [[0, 1], [0, 1, 2], ...]

        # create the LazyImageStack object for each particle
        if check_headers:
            lazyparticles = [mrc.LazyImageStack(fname=ptcl_mrcs_file, indices_image=ptcl_visible_frames, representative_header=None)
                             for ptcl_mrcs_file, ptcl_visible_frames in zip(ptcl_mrcs_files, all_ptcls_visible_frames, strict=True)]

            # assert that all files have the same dtype and same image shape
            assert all([ptcl.dtype_image == lazyparticles[0].dtype_image for ptcl in lazyparticles])
            assert all([ptcl.shape_image == lazyparticles[0].shape_image for ptcl in lazyparticles])

        else:
            representative_header = mrc.parse_header(fname=ptcl_mrcs_files[0])
            lazyparticles = [mrc.LazyImageStack(fname=ptcl_mrcs_file, indices_image=ptcl_visible_frames, representative_header=representative_header)
                             for ptcl_mrcs_file, ptcl_visible_frames in zip(ptcl_mrcs_files, all_ptcls_visible_frames, strict=True)]

        if lazy:
            return lazyparticles
        else:
            # preallocating numpy array for in-place loading, fourier transform, fourier transform centering, etc.
            # allocating 1 extra pixel along x and y dimensions in anticipation of symmetrizing the hartley transform in-place
            all_ptcls_nimgs = [len(ptcl_visible_frames) for ptcl_visible_frames in all_ptcls_visible_frames]
            particles = np.zeros((sum(all_ptcls_nimgs), lazyparticles[0].shape_image[0] + 1, lazyparticles[0].shape_image[1] + 1), dtype=lazyparticles[0].dtype_image)
            loaded_images = 0
            for lazyparticle, ptcl_nimgs in zip(lazyparticles, all_ptcls_nimgs):
                particles[loaded_images:loaded_images + ptcl_nimgs, :-1, :-1] = lazyparticle.get(low_memory=False)
                loaded_images += ptcl_nimgs
            return particles

    def write(self,
              outstar: str,
              *args,
              **kwargs) -> None:
        """
        Temporarily removes columns in data_particles dataframe that are present in data_optics dataframe (to restore expected input star file format), then calls parent GenericStarfile write.
        Writes both the TomoParticlesStar file and the updated Optimisation Set star file pointing to the new TomoParticlesStar file.
        The TomoParticlesStar file is written to the same directory as the optimisation set star file, and has the same name as the optimisation set after removing the string ``_optimisation_set``.

        :param outstar: name of the output optimisation set star file, optionally as absolute or relative path.
                Filename should include the string ``_optimisation_set``, e.g. ``run_optimisation_set.star``.
        :param args: Passed to parent GenericStarfile write
        :param kwargs: Passed to parent GenericStarfile write
        :return: None
        """

        # during loading TomoParticlesStarfile, block_optics and block_particles are merged for internal convenience
        columns_in_common = self.df.columns.intersection(self.blocks[self.block_optics].columns)
        # need to preserve the optics groups in the data_particles block
        columns_in_common = columns_in_common.drop('_rlnOpticsGroup')
        # drop all other columns in common from the data_particles block
        self.df = self.df.drop(columns_in_common, axis=1)

        # temporarily move columns added during __init__ to separate dataframe so that the written file does not contain these new columns
        temp_df = self.df[self.tomodrgn_added_headers].copy()
        self.df = self.df.drop(self.tomodrgn_added_headers, axis=1)

        # temporarily convert self.header_ptcl_visible_frames to dtype str of list of int, as it was at input, for appropriate white-spacing in writing file to disk
        self.df[self.header_ptcl_visible_frames] = [f'[{",".join([str(include) for include in ptcl_visible_frames])}]' for ptcl_visible_frames in self.df[self.header_ptcl_visible_frames]]

        # now call parent write method for the TomoParticlesStar file
        assert '_optimisation_set' in os.path.basename(outstar), f'The name of the output star file must include the string "_optimisation_set", but got {outstar}'
        outstar_particles = f'{os.path.dirname(outstar)}/{os.path.basename(outstar).replace("_optimisation_set", "")}'
        super().write(*args, outstar=outstar_particles, **kwargs)

        # need to copy the tomoTomogramsFile to this new location -- can just copy (not starfile.write) because the file contents do not change
        outstar_tomograms = f'{os.path.dirname(outstar)}/{os.path.basename(self.tomograms_star_path)}'
        shutil.copy(self.tomograms_star_path, outstar_tomograms)

        # also need to update the optimisation set contents and write out the updated optimisation set star file to the same directory
        self.optimisation_set_star.blocks['data_']['_rlnTomoParticlesFile'] = os.path.basename(outstar_particles)
        self.optimisation_set_star.blocks['data_']['_rlnTomoTomogramsFile'] = os.path.basename(outstar_tomograms)
        self.optimisation_set_star.write(outstar=outstar)

        # re-merge data_optics with data_particles so that the starfile object appears unchanged after calling this method
        self.df = self.df.merge(self.blocks[self.block_optics], on='_rlnOpticsGroup', how='inner', validate='many_to_one', suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')

        # re-add the columns added during __init__ to restore the state of self.df from the start of this function call
        self.df = pd.concat([self.df, temp_df], axis=1)

        # re-convert the header_ptcl_visible_frames to dtype np.array of int, as set during __init__
        self.df[self.header_ptcl_visible_frames] = [np.asarray([include for include in ptcl_frames.replace('[', '').replace(']', '').split(',')], dtype=int)
                                                    for ptcl_frames in self.df[self.header_ptcl_visible_frames]]


def is_starfile_optimisation_set(star_path: str) -> bool:
    """
    Infer whether a star file on disk is a RELION optimisation set, or some other type of star file.
    Defining characteristics of an optimisation set star file:

    * the data block name is ``data_``
    * the data block is a simple, two column, dictionary-style block
    * the data block minimally contains the keys ``_rlnTomoTomogramsFile`` and ``_rlnTomoParticlesFile``, as these are needed for tomodrgn

    :param star_path: path to potential optimisation set star file on disk
    :return: bool of whether the input star file matches characteristics of an optimisation set star file
    """
    # only skeletonize the star file for faster processing in case this is a large file (e.g. particle imageseries star file)
    preambles, blocks = GenericStarfile._skeletonize(star_path)

    # the data block named ``data_`` must be present
    if 'data_' not in blocks.keys():
        return False

    # the ``data_`` data block must be a dictionary-style block
    if type(blocks['data_']) is not dict:
        return False

    # the ``data_`` data block must minimally contain keys ``_rlnTomoTomogramsFile`` and ``_rlnTomoParticlesFile``
    if not {'_rlnTomoTomogramsFile', '_rlnTomoParticlesFile'}.issubset(blocks['data_'].keys()):
        return False

    return True


def load_sta_starfile(star_path: str,
                      source_software: KNOWN_STAR_SOURCES = 'auto') -> TiltSeriesStarfile | TomoParticlesStarfile:
    """
    Loads a tomodrgn star file handling class (either ``TiltSeriesStarfile`` or ``TomoParticlesStarfile``) from a star file on disk.
    The input ``star_path`` must point to either a particle imageseries star file (e.g. from Warp v1) or an optimisation set star file (e.g. from RELION v5).
    This is the preferred way of creating a tomodrgn starfile class instance.

    :param star_path: path to star file to load on disk
    :param source_software: type of source software used to create the star file, used to indicate the appropriate star file handling class to instantiate.
            Default of 'auto' tries to infer the appropriate star file handling class based on whether ``star_path`` is an optimisation set star file.
    :return: The created starfile object (either ``TiltSeriesStarfile`` or ``TomoParticlesStarfile``)
    """

    if source_software == 'auto':
        if is_starfile_optimisation_set(star_path):
            return TomoParticlesStarfile(star_path)
        else:
            return TiltSeriesStarfile(star_path)
    else:
        if source_software in get_args(TILTSERIESSTARFILE_STAR_SOURCES):
            return TiltSeriesStarfile(star_path, source_software=source_software)
        elif source_software in get_args(TOMOPARTICLESSTARFILE_STAR_SOURCES):
            return TomoParticlesStarfile(star_path, source_software=source_software)
        else:
            raise ValueError(f'Unrecognized source_software {source_software} not one of known starfile sources {KNOWN_STAR_SOURCES}')
