"""
Classes and functions for loading and writing .mrc(s) headers and data.

Generally adheres to the MRC2014 Update 20141 specification.
    - MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography
    - https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    - https://github.com/ccpem/mrcfile/blob/master/mrcfile/dtypes.py

Deviations from this standard include:
    (1) any data in the extended header is read (and maintained for later writing), but is not interpreted or used in any way;
    (2) the `machst` attribute of the header is unpacked as a byte string (e.g b`DA\x00\x00`) rather than as an array of uint8 (e.g. `[68, 65,  0,  0]`);
    (3) the `label` attribute of the header is unpacked as a concatenated byte string rather than as an array of byte strings for all 10 80-character labels that may exist

Prefer the following uses:
    - interfacing with MRC headers: mrc.MRCHeader.parse() or mrc.parse_header()
    - loading an entire MRC file into memory: mrc.parse_mrc()
    - loading a list of MRC files into memory: mrc.parse_mrc_list()
    - lazily loading a single image from an MRC file: mrc.LazyImage
    - lazily loading a group of images from an MRC file: mrc.LazyImageStack
    - writing an MRC file to disk: mrc.write()
"""

import numpy as np
import numpy.typing as npt
from typing import BinaryIO
import os
import sys
import struct
from collections import OrderedDict
from itertools import groupby
from tomodrgn import utils


class MRCHeader:
    """
    Class to parse the header of an MRC file and to write a new MRC header to disk.
    """
    # define 1024-byte header fields and associated data types to use when decoding bytes
    fieldnames_structformats = (
        ('nx', 'i'),
        ('ny', 'i'),
        ('nz', 'i'),

        ('mode', 'i'),

        ('nxstart', 'i'),
        ('nystart', 'i'),
        ('nzstart', 'i'),

        ('mx', 'i'),
        ('my', 'i'),
        ('mz', 'i'),

        ('cella_xlen', 'f'),
        ('cella_ylen', 'f'),
        ('cella_zlen', 'f'),

        ('cellb_alpha', 'f'),
        ('cellb_beta', 'f'),
        ('cellb_gamma', 'f'),

        ('mapc', 'i'),
        ('mapr', 'i'),
        ('maps', 'i'),

        ('dmin', 'f'),
        ('dmax', 'f'),
        ('dmean', 'f'),

        ('ispg', 'i'),
        ('nsymbt', 'i'),

        ('extra1', '8s'),
        ('exttyp', '4p'),
        ('nversion', 'i'),
        ('extra2', '84s'),

        ('origin_x', 'f'),
        ('origin_y', 'f'),
        ('origin_z', 'f'),

        ('map', '4s'),
        ('machst', '4s'),  # technically this should be 4B but unpacks as an array of `uint8` which is not then mapped to one `machst`

        ('rms', 'f'),

        ('nlabl', 'i'),
        ('label', f'{10 * 80}s')  # technically this should be f'{10*"80s"}', but unpacks as an array of `char which is not then mapped to one `label`
    )

    field_names, struct_formats = zip(*fieldnames_structformats)
    struct_format_string = ''.join(struct_formats)

    # define the MRC standard `mode` values and corresponding numpy dtypes
    dtype_for_mode = {
        0: np.dtype(np.int8),
        1: np.dtype(np.int16),
        2: np.dtype(np.float32),
        3: '2h',  # complex number from 2 shorts
        4: np.dtype(np.complex64),
        6: np.dtype(np.uint16),
        7: np.dtype(np.int32),
        12: np.dtype(np.float16),  # IEEE754
        16: '3B',  # RBG values
    }
    mode_for_dtype = {vv: kk for kk, vv in dtype_for_mode.items()}

    def __init__(self,
                 header_values: tuple,
                 extended_header: bytes = b''):
        """
        Directly initialize a MRCHeader object with pre-parsed header values.

        :param header_values: tuple of values in correct order to be assigned to `MRCHeader.field_names`, derived from standard 1024-byte header of MRC file
        :param extended_header: byte string of data in extended header of MRC file. Read in but not used.
        """
        self.fields = OrderedDict(zip(self.field_names, header_values))
        self.extended_header = extended_header
        self.boxsize = self.fields['nx']
        self.dtype = self.dtype_for_mode[self.fields['mode']]

    def __str__(self):
        return f'Header: {self.fields}\nExtended header: {self.extended_header}'

    @classmethod
    def parse(cls,
              fname: str):
        """
        Constructor method to create an MRCHeader object from an MRC file name.

        :param fname: path to MRC file on disk
        :return: MRCHeader object
        """
        with open(fname, 'rb') as f:
            # load and unpack the header
            header_values = struct.Struct(cls.struct_format_string).unpack(f.read(1024))

            # instantiate the MRCHeader object
            header = cls(header_values)

            # load the extended header if declared present
            extbytes = header.fields['nsymbt']
            extended_header = f.read(extbytes)
            header.extended_header = extended_header

        return header

    @classmethod
    def make_default_header(cls,
                            data: np.ndarray,
                            is_vol: bool = True,
                            angpix: float = 1,
                            origin_x: float = 0,
                            origin_y: float = 0,
                            origin_z: float = 0):
        """
        Constructor method to create an MRCHeader object describing a 3-D data array.

        Automaticaly calculates and sets:
            `nx`,
            `ny`,
            `nz`,
            `mode`,
            `cella_xlen`,
            `cella_ylen`,
            `cella_zlen`
            `dmin`
            `dmax`
            `dmean`
            `ispg`
            `map`
            `machst`

        Assumes default values for:
            `nxstart = nystart = nzstart := 0`,
            `mx := nx`,
            `my := ny`,
            `mz := nz`,
            `cellb_alpha = cellb_beta = cellb_gamma := 90`
            `mapc := 1`
            `mapr := 2`
            `maps := 3`
            `nsymbt := 0`
            `extra1 := b'\x00' * 8`
            `exttyp := b''`
            `nversion := 0`
            `extra2 := b'\x00' * 84`
            `origin_x = origin_y = origin_z = 0`
            `rms := -1`
            `nlabl := 0`
            `labels := b'\x00' * 84`

        :param data: array of data to be described by this header
        :param is_vol: whether the data array is a 3-D volume instead of a stack of 2-D images
        :param angpix: pixel size in angstroms per pixel of the data array
        :param origin_x: coordinate system origin along x-axis in angstroms, or the phase origin of the transformed image in pixels for MODE 3 or 4 data
        :param origin_y: coordinate system origin along y-axis in angstroms, or the phase origin of the transformed image in pixels for MODE 3 or 4 data
        :param origin_z: coordinate system origin along z-axis in angstroms, or the phase origin of the transformed image in pixels for MODE 3 or 4 data
        :return: MRCHeader object
        """
        # get the number of sections, columns, and rows
        nz, ny, nx = data.shape

        # set the space group header for an image stack (0) or a 3-D volume (1) following the MRC2014 standard
        ispg = 1 if is_vol else 0

        # set endianness
        if sys.byteorder == 'little':
            machst = b'\x44\x44\x00\x00'
        elif sys.byteorder == 'big':
            machst = b'\x11\x11\x00\x00'
        else:
            raise RuntimeError(f'Unrecognized byteorder {sys.byteorder}')

        # set data statistics for header
        if is_vol:
            # calculate data statistics directly
            dmin, dmax, dmean, rms = data.min(), data.max(), data.mean(), data.std()
        else:
            # use undefined values for image stacks since the stack could be very large and costly to evaluate min/max/mean/rms
            # in keeping with standard convention, (1) dmax < dmin, (2) dmean < min(dmax, dmin), (3) rms < 0
            dmin, dmax, dmean, rms = -1, -2, -3, -1

        # populate the values in the order defined by MRCHeader.field_names
        vals = (
            nx,  # nx
            ny,  # ny
            nz,  # nz
            cls.mode_for_dtype[data.dtype],  # mode
            0,  # nxstart
            0,  # nystart
            0,  # nzstart
            nx,  # mx
            ny,  # my
            nz,  # mz
            angpix * nx,  # cella_xlen
            angpix * ny,  # cella_ylen
            angpix * nz,  # cella_zlen
            90.,  # cellb_alpha
            90.,  # cellb_beta
            90.,  # cellb_gamma
            1,  # mapc
            2,  # mapr
            3,  # mapz,
            dmin,  # dmin
            dmax,  # dmax
            dmean,  # dmean
            ispg,  # space group header
            0,  # nsymbt, length of extended header
            b'\x00' * 8,  # extra1
            b'',  # exttyp
            0,  # nversion of MRC format
            b'\x00' * 84,  # extra2
            origin_x,  # origin_x
            origin_y,  # origin_y
            origin_z,  # origin_z
            b'MAP ' if is_vol else b'\x00' * 4,  # map
            machst,  # machst
            rms,  # rms
            0,  # nlabl
            b'\x00' * 800,  # labels
        )

        return cls(vals)

    def write(self,
              fh: BinaryIO) -> None:
        """
        Write an MRC file header to the specified file handle.

        :param fh: filehandle to file on disk opened in binary mode
        :return: None
        """
        # check that the actual and declared length of the extended header are in sync
        self.fields['nsymbt'] = len(self.extended_header)

        # pack the header into appropriate binary format
        buf = struct.Struct(self.struct_format_string).pack(*list(self.fields.values()))

        # write the header and extended header
        fh.write(buf)
        fh.write(self.extended_header)

    def get_apix(self) -> float:
        """
        Get the pixel size in angstroms per pixel based on header `cella_xlen` and `nx` ratio

        :return: pixel size in angstroms
        """
        return self.fields['cella_xlen'] / self.fields['nx']

    def update_apix(self,
                    angpix: float) -> None:
        """
        Update the pixel size in angstrom by adjusting the header `cella_xlen`, `cella_ylen`, and `cella_zlen`

        :param angpix:
        :return: None
        """
        self.fields['cella_xlen'] = self.fields['nx'] * angpix
        self.fields['cella_ylen'] = self.fields['ny'] * angpix
        self.fields['cella_zlen'] = self.fields['nz'] * angpix

    def get_origin(self) -> tuple[float, float, float]:
        """
        Get the origin of the data coordinate system.

        :return: origin in (x, y, z)
        """
        return self.fields['origin_x'], self.fields['origin_y'], self.fields['origin_z']

    def update_origin(self,
                      origin_x: float,
                      origin_y: float,
                      origin_z: float) -> None:
        """
        Update the origin of the data coordinate system.

        :param origin_x: new origin along x axis
        :param origin_y: new origin along y axis
        :param origin_z: new origin along z axis
        :return: None
        """
        self.fields['origin_x'] = origin_x
        self.fields['origin_y'] = origin_y
        self.fields['origin_z'] = origin_z

    @property
    def total_header_bytes(self) -> int:
        """
        Calculate the total header length as standard header (1024 bytes) + optional extended header (>= 0)

        :return: total number of bytes in header and extended header
        """
        return 1024 + len(self.extended_header)


class LazyImage:
    """
    Class to lazily load data from an MRC file on-the-fly
    """

    def __init__(self,
                 fname: str,
                 shape: tuple[int, ...],
                 dtype: npt.DTypeLike,
                 offset: int):
        """
        Initialize a LazyImage object. No disk access required or performed by creating the object.

        :param fname: path to the .mrc file on disk
        :param shape: shape of the section of data to be lazily loaded from the .mrc file, e.g. `(128,128)` for a single box128 image
        :param dtype: the data type of the .mrc file contents, as described by the .mrc file header `mode` key
        :param offset: offset from the start of the file to the beginning of the section of data to be lazily loaded, in bytes.
                Should include standard 1024 byte mrc header, optional extended header, and number of data bytes preceeding data to be loaded.
        """
        self.fname = fname
        self.shape = shape
        self.dtype = dtype
        self.offset = offset

    def get(self) -> np.ndarray:
        """
        Load the data from disk to a numpy array

        :return: numpy array of data loaded from disk
        """
        # create a file handle to open the file
        with open(self.fname, 'rb') as f:
            # go to the specified offset from the start of the file
            f.seek(self.offset)
            # load bytes from disk into a numpy array of specified shape
            image = np.fromfile(f, dtype=self.dtype, count=np.prod(self.shape)).reshape(self.shape)

        return image


class LazyImageStack:
    """
    Efficiently load a particle stack from an NxDxD stack.mrcs file on-the-fly.
    Minimizes calls to file opening, file seeking, and file reading in single-process context.
    """

    def __init__(self, fname, indices_image):
        # store the input file name
        self.fname = fname

        # load the input file header as an MRCHeader object
        header = MRCHeader.parse(fname)
        self.header = header

        # store the indices of images to load from this file and group the indices into continuous blocks for contiguous reads
        self.indices_image = indices_image
        self.indices_image_contiguous = self._group_contiguous_indices()

        # calculate information used in determining which bytes to load as each image
        self.header_offset = self.header.total_header_bytes
        self.dtype_image = self.header.dtype
        self.shape_image = (self.header.fields['ny'], self.header.fields['nx'])
        self.stride_image = self.dtype_image.itemsize * self.shape_image[0] * self.shape_image[1]

    def _group_contiguous_indices(self) -> list[np.ndarray[int]]:
        """
        Groups a flat array of input indices by which indices are immediately contiguous.
        For example, input of [0, 1, 2, 4, 5] would return [[0, 1, 2], [4, 5]].

        :return: list of arrays of contiguous indices
        """
        # create list to store contiguous indices
        contiguous_indices = []
        # determine which indices (`j`) are contiguous by virtue of sharing the same offset from natural numbers starting from 0 (`i`)
        contiguous_labels = [j - i for i, j in enumerate(self.indices_image)]

        image_count = 0
        for _, elements in groupby(contiguous_labels):
            length_contiguous_subset = len(list(elements))
            contiguous_indices.append(self.indices_image[image_count: image_count + length_contiguous_subset])
            image_count += length_contiguous_subset

        return contiguous_indices

    def get(self) -> np.ndarray:
        """
        Load the image data from disk to a numpy array

        :return: the numpy array of data
        """

        with open(self.fname, 'rb') as f:
            # preallocate an array to be populated by the loaded images
            stack = np.zeros((len(self.indices_image), *self.shape_image), dtype=self.dtype_image)

            # calculate and apply the offset from the start of the file to the beginning of the first image to be loaded
            f.seek(self.header_offset + self.indices_image[0] * self.stride_image, 0)

            image_count = 0
            for i, inds_contiguous in enumerate(self.indices_image_contiguous):

                if (i > 0) and (inds_contiguous[0] != previous_ind + 1):
                    # np.fromfile advances file pointer by length_contiguous_subset * stride_image bytes on the previous loop
                    # we need to seek the file pointer ahead to the start of the next image to be loaded
                    # by definition of the contiguous indices, this seeking is needed every iteration but the first
                    offset_from_previous_image = (inds_contiguous[0] - (previous_ind + 1)) * self.stride_image
                    # offset by the number of bytes of skipped images, relative to the file pointer's position after reading the previous contiguous images
                    f.seek(offset_from_previous_image, 1)

                length_contiguous_subset = len(inds_contiguous)
                stack[image_count: image_count + length_contiguous_subset] = np.fromfile(f,
                                                                                         dtype=self.dtype_image,
                                                                                         count=self.shape_image[0] * self.shape_image[1] * length_contiguous_subset
                                                                                         ).reshape(length_contiguous_subset,
                                                                                                   self.shape_image[0],
                                                                                                   self.shape_image[1])
                image_count += length_contiguous_subset

                # track the last image loaded of the contiguous block, to be used in adjusting file pointer offset at next iteration
                previous_ind = inds_contiguous[-1]

        return stack


def parse_header(fname: str) -> MRCHeader:
    """
    Convenience function to create an MRCHeader object given an MRC file name

    :param fname: path to MRC file on disk
    :return: MRCHeader object
    """
    return MRCHeader.parse(fname)


def parse_mrc_list(txtfile: str,
                   lazy: bool = False) -> np.ndarray | list[LazyImage]:
    """
    Load the MRC file(s) specified in a text file into memory as either a numpy array or a list of LazyImages.

    :param txtfile: path to newline-delimited text file listing the absolute path to the MRC file(s) to be loaded, or the path to the MRC file(s) relative to `txtfile`.
    :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True)
    :return: numpy array or list of LazyImages of all images in all MRC files
    """
    # get the names of MRC file(s) to load
    with open(txtfile, 'r') as f:
        lines = f.readlines()

    # confirm where to load MRC file(s) from disk
    lines = utils.prefix_paths(mrcs=lines,
                               datadir=os.path.dirname(os.path.abspath(txtfile)))

    # load the particles
    if not lazy:
        particles = np.vstack([parse_mrc(x.strip(), lazy=False)[0] for x in lines])
    else:
        particles = [img for x in lines for img in parse_mrc(x.strip(), lazy=True)[0]]

    return particles


def parse_mrc(fname: str,
              lazy: bool = False) -> tuple[np.ndarray | list[LazyImage], MRCHeader]:
    """
    Load an entire MRC file into memory as either a numpy array or a list of LazyImages

    :param fname: path to MRC file on disk
    :param lazy: whether to load particle images now in memory (False) or later on-the-fly (True)
    :return: array: numpy array of MRC data in header-specified dtype and shape, or list of a LazyImage for each section along the nz axis in the MRC file
            header: an MRCHeader object containing the information encoded in the header and extended header of the MRC file
    """
    # TODO move to method of MRCHeader (which should be renamed MRCFile or inherited from)
    #   -> might make it more clear / easy to have extended data offset for lazyimage / lazyimagestack
    # parse the header
    header = MRCHeader.parse(fname)

    # get information describing the data array size and layout
    dtype = header.dtype
    nz, ny, nx = header.fields['nz'], header.fields['ny'], header.fields['nx']

    # load all data in one block
    if not lazy:
        with open(fname, 'rb') as fh:
            # skip the header + extended header
            fh.seek(header.total_header_bytes)
            # load all bytes remaining in file from disk into a numpy array of specified shape
            array = np.fromfile(fh, dtype=dtype).reshape((nz, ny, nx))

    # or list of LazyImages
    else:
        # calculate the size of each image in bytes
        stride = dtype().itemsize * ny * nx
        array = [LazyImage(fname=fname,
                           shape=(ny, nx),
                           dtype=dtype,
                           offset=header.total_header_bytes + i * stride)
                 for i in range(nz)]
    return array, header


def write(fname: str,
          array: np.ndarray,
          header: MRCHeader | None = None,
          angpix: float = 1,
          origin_x: float = 0,
          origin_y: float = 0,
          origin_z: float = 0,
          is_vol: bool | None = None) -> None:
    """
    Write a data array to disk in MRC format.

    :param fname: Name of the MRC file to write
    :param array: Array of data to write as data block of MRC file
    :param header: MRCFile header object to be written as the header of the new MRC file. `None` means that a default header will be created using the following parameters:
    :param angpix: Pixel size of the data to be used in creating the default header
    :param origin_x: new origin along x axis
    :param origin_y: new origin along y axis
    :param origin_z: new origin along z axis
    :param is_vol: Whether the data array represents a volume (versus an image or image stack). `None` means to infer based on whether data array is a cube (volume-like).
    :return: None
    """
    if header is None:
        # get a default header if not provided
        if is_vol is None:
            is_vol = True if len(set(array.shape)) == 1 else False  # Guess whether data is vol or image stack
        header = MRCHeader.make_default_header(data=array,
                                               is_vol=is_vol,
                                               angpix=angpix,
                                               origin_x=origin_x,
                                               origin_y=origin_y,
                                               origin_z=origin_z)
    else:
        # ensure that the supplied header describes the data correctly
        header.fields['nz'] = array.shape[0]
        header.fields['ny'] = array.shape[1]
        header.fields['nx'] = array.shape[2]
        header.fields['mode'] = header.mode_for_dtype[array.dtype]

    with open(fname, 'wb') as f:
        # write the header (and extended header if present)
        header.write(f)
        # write the data array
        f.write(array.tobytes())
