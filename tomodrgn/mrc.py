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
"""

import numpy as np
import numpy.typing as npt
from typing import BinaryIO
import os
import sys
import struct
from collections import OrderedDict
from itertools import groupby

DTYPE_FOR_MODE = {
    0: np.int8,
    1: np.int16,
    2: np.float32,
    3: '2h',  # complex number from 2 shorts
    4: np.complex64,
    6: np.uint16,
    7: np.int32,
    12: np.float16,  # IEEE754
    16: '3B',  # RBG values
}
MODE_FOR_DTYPE = {vv: kk for kk, vv in DTYPE_FOR_MODE.items()}


class MRCHeader:
    """
    Class to parse the header of an MRC file, with methods to load associated data and to write a new MRC file to disk.
    """
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
        self.dtype = DTYPE_FOR_MODE[self.fields['mode']]

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
            MODE_FOR_DTYPE[data.dtype],  # mode
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
        self.fields['nymbt'] = len(self.extended_header)

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
                      xorg: float,
                      yorg: float,
                      zorg: float) -> None:
        """
        Update the origin of the data coordinate system.
        :param xorg: new origin along x axis
        :param yorg: new origin along y axis
        :param zorg: new origin along z axis
        :return: None
        """
        self.fields['xorg'] = xorg
        self.fields['yorg'] = yorg
        self.fields['zorg'] = zorg


class LazyImage:
    '''On-the-fly image loading'''

    def __init__(self, fname, shape, dtype, offset):
        self.fname = fname
        self.shape = shape
        self.dtype = dtype
        self.offset = offset
    def get(self):
        with open(self.fname) as f:
            f.seek(self.offset)
            image = np.fromfile(f, dtype=self.dtype, count=np.product(self.shape)).reshape(self.shape)
        return image


class LazyImageStack:
    """
    Efficiently load a particle stack from an NxDxD stack.mrcs file on-the-fly.
    Minimizes calls to file opening, file seeking, and file reading in single-process context.

    Attributes
    ----------
    fname : str
        Absolute path to file `stack.mrcs`
    dtype_pixel : dtype
        dtype from MRCHeader.dtype
    shape_image : tuple of ints
        number of pixels along each image axis
    indices_image : list of ints
        0-indexed list of images to read from `stack.mrcs`
    indices_image_contiguous : list of list of ints
        indices_image grouped by continuous value sequences
    stride_image : int
        number of bytes per image

    Methods
    -------
    get:
        loads and returns images specified at class instantiation as (N,D,D) numpy array
    """

    def __init__(self, fname, dtype_pixel, shape_image, indices_image):
        self.fname = fname
        self.dtype_pixel = dtype_pixel
        self.shape_image = shape_image
        self.indices_image = indices_image
        self.indices_image_contiguous = self.group_contiguous_indices()
        self.stride_image = self.dtype_pixel().itemsize * self.shape_image[0] * self.shape_image[1]

    def group_contiguous_indices(self):
        contiguous_indices = []
        positions = [j - i for i, j in enumerate(self.indices_image)]
        offset = 0

        for _, elements in groupby(positions):
            length_contiguous_subset = len(list(elements))
            contiguous_indices.append(self.indices_image[offset: offset + length_contiguous_subset])
            offset += length_contiguous_subset

        return contiguous_indices

    def get(self):

        with open(self.fname, 'rb') as f:
            stack = np.zeros((len(self.indices_image), *self.shape_image), dtype=np.float32)

            offset_file = 1024 + self.indices_image[0] * self.stride_image
            f.seek(offset_file, 0)  # define read to beginning of file including header

            offset_stack = 0
            for i, inds_contiguous in enumerate(self.indices_image_contiguous):

                if (i > 0) and (inds_contiguous[0] != previous_ind + 1):
                    # np.fromfile advances file pointer by one image's stride on the previous loop
                    # further update file pointer if next image is more than one image's stride away
                    offset_file = (inds_contiguous[0] - (previous_ind + 1)) * self.stride_image
                    f.seek(offset_file, 1)

                length_contiguous_subset = len(inds_contiguous)
                stack[offset_stack: offset_stack + length_contiguous_subset] = \
                    np.fromfile(f, dtype=self.dtype_pixel, count=self.shape_image[0] * self.shape_image[1] * length_contiguous_subset) \
                        .reshape(length_contiguous_subset, self.shape_image[0], self.shape_image[1])
                offset_stack += length_contiguous_subset
                previous_ind = inds_contiguous[-1]

        return stack


def parse_header(fname):
    return MRCHeader.parse(fname)


def parse_mrc_list(txtfile, lazy=False):
    lines = open(txtfile, 'r').readlines()

    def abspath(f):
        if os.path.isabs(f):
            return f
        base = os.path.dirname(os.path.abspath(txtfile))
        return os.path.join(base, f)

    lines = [abspath(x) for x in lines]
    if not lazy:
        particles = np.vstack([parse_mrc(x.strip(), lazy=False)[0] for x in lines])
    else:
        particles = [img for x in lines for img in parse_mrc(x.strip(), lazy=True)[0]]
    return particles


def parse_mrc(fname, lazy=False):
    # parse the header
    header = MRCHeader.parse(fname)

    ## get the number of bytes in extended header
    extbytes = header.fields['next']
    start = 1024 + extbytes  # start of image data

    dtype = header.dtype
    nz, ny, nx = header.fields['nz'], header.fields['ny'], header.fields['nx']

    # load all in one block
    if not lazy:
        with open(fname, 'rb') as fh:
            fh.read(start)  # skip the header + extended header
            array = np.fromfile(fh, dtype=dtype).reshape((nz, ny, nx))

    # or list of LazyImages
    else:
        stride = dtype().itemsize * ny * nx
        array = [LazyImage(fname, (ny, nx), dtype, start + i * stride) for i in range(nz)]
    return array, header


def write(fname, array, header=None, Apix=1., xorg=0., yorg=0., zorg=0., is_vol=None):
    # get a default header
    if header is None:
        if is_vol is None:
            is_vol = True if len(set(array.shape)) == 1 else False  # Guess whether data is vol or image stack
        header = MRCHeader.make_default_header(array, is_vol, Apix, xorg, yorg, zorg)
    # write the header
    f = open(fname, 'wb')
    header.write(f)
    f.write(array.tobytes())
