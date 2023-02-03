import numpy as np
import os
import struct
from collections import OrderedDict
from itertools import groupby

# See ref:
# MRC2014: Extensions to the MRC format header for electron cryo-microscopy and tomography
# And:
# https://www.ccpem.ac.uk/mrc_format/mrc2014.php

DTYPE_FOR_MODE = {0:np.int8,
                  1:np.int16,
                  2:np.float32,
                  3:'2h', # complex number from 2 shorts
                  4:np.complex64,
                  6:np.uint16,
                  12:np.float16,
                  16:'3B'} # RBG values
MODE_FOR_DTYPE = {vv:kk for kk,vv in DTYPE_FOR_MODE.items()}

class MRCHeader:
    '''MRC header class'''
    FIELDS = ['nx','ny','nz', # int
              'mode', # int
              'nxstart','nystart','nzstart', # int
              'mx','my','mz', # int
              'xlen','ylen','zlen', # float
              'alpha','beta','gamma', # float
              'mapc','mapr','maps',# int
              'amin','amax','amean', # float
              'ispg','next','creatid', # int, int, short, [pad 30]
              'nint','nreal', # short, [pad 20]
              'imodStamp','imodFlags', # int
              'idtype','lens','nd1','nd2','vd1','vd2', # short
              'tilt_ox','tilt_oy','tilt_oz', # float
              'tilt_cx','tilt_cy','tilt_cz', # float
              'xorg','yorg','zorg', # float
              'cmap','stamp','rms', # char[4], float
              'nlabl','labels'] # int, char[10][80]
    FSTR = '3ii3i3i3f3f3i3f2ih30x2h20x2i6h6f3f4s4sfi800s'
    STRUCT = struct.Struct(FSTR) 

    def __init__(self, header_values, extended_header=b''):
        self.fields = OrderedDict(zip(self.FIELDS,header_values))
        self.extended_header = extended_header
        self.D = self.fields['nx']
        self.dtype = DTYPE_FOR_MODE[self.fields['mode']]

    def __str__(self):
        return f'Header: {self.fields}\nExtended header: {self.extended_header}'

    @classmethod
    def parse(cls, fname):
        with open(fname,'rb') as f:
            header = cls(cls.STRUCT.unpack(f.read(1024)))
            extbytes = header.fields['next']
            extended_header = f.read(extbytes)
            header.extended_header = extended_header
        return header
    
    @classmethod
    def make_default_header(cls, data, is_vol=True, Apix=1., xorg=0., yorg=0., zorg=0.):
        nz, ny, nx = data.shape
        ispg = 1 if is_vol else 0
        if is_vol:
            dmin, dmax, dmean, rms = data.min(), data.max(), data.mean(), data.std()
        else: # use undefined values for image stacks
            dmin, dmax, dmean, rms = -1, -2, -3, -1
        vals = [nx, ny, nz,
                2, # mode = 2 for 32-bit float
                0, 0, 0, # nxstart, nystart, nzstart
                nx, ny, nz, # mx, my, mz
                Apix*nx, Apix*ny, Apix*nz, # cella
                90., 90., 90., # cellb
                1, 2, 3, # mapc, mapr, maps
                dmin, dmax, dmean,
                ispg,
                0, # exthd_size
                0, # creatid
                0, 0, # nint, nreal
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0,
                xorg, yorg, zorg,
                b'MAP ' if is_vol else b'\x00'*4, b'\x00'*4, #cmap, stamp
                rms, # rms
                0, # nlabl
                b'\x00'*800, # labels
                ]
        return cls(vals)

    def write(self, fh):
        buf = self.STRUCT.pack(*list(self.fields.values()))
        fh.write(buf)
        fh.write(self.extended_header)

    def get_apix(self):
        return self.fields['xlen']/self.fields['nx']
    
    def update_apix(self, Apix):
        self.fields['xlen'] = self.fields['nx']*Apix
        self.fields['ylen'] = self.fields['ny']*Apix
        self.fields['zlen'] = self.fields['nz']*Apix
    
    def get_origin(self):
        return self.fields['xorg'], self.fields['yorg'], self.fields['zorg']

    def update_origin(self, xorg, yorg, zorg):
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
            contiguous_indices.append(self.indices_image[offset : offset + length_contiguous_subset])
            offset += length_contiguous_subset

        return contiguous_indices


    def get(self):

        with open(self.fname, 'rb') as f:
            stack = np.zeros((len(self.indices_image), *self.shape_image), dtype=np.float32)

            offset_file = 1024 + self.indices_image[0] * self.stride_image
            f.seek(offset_file, 0) # define read to beginning of file including header

            offset_stack = 0
            for i, inds_contiguous in enumerate(self.indices_image_contiguous):

                if (i > 0) and (inds_contiguous[0] != previous_ind + 1):
                    # np.fromfile advances file pointer by one image's stride on the previous loop
                    # further update file pointer if next image is more than one image's stride away
                    offset_file = (inds_contiguous[0] - (previous_ind+1)) * self.stride_image
                    f.seek(offset_file, 1)

                length_contiguous_subset = len(inds_contiguous)
                stack[offset_stack : offset_stack + length_contiguous_subset] = \
                    np.fromfile(f, dtype=self.dtype_pixel, count=self.shape_image[0]*self.shape_image[1]*length_contiguous_subset)\
                        .reshape(length_contiguous_subset, self.shape_image[0], self.shape_image[1])
                offset_stack += length_contiguous_subset
                previous_ind = inds_contiguous[-1]

        return stack

def parse_header(fname):
    return MRCHeader.parse(fname)

def parse_mrc_list(txtfile, lazy=False):
    lines = open(txtfile,'r').readlines()
    def abspath(f):
        if os.path.isabs(f):
            return f
        base = os.path.dirname(os.path.abspath(txtfile))
        return os.path.join(base,f)
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
    start = 1024+extbytes # start of image data

    dtype = header.dtype
    nz, ny, nx = header.fields['nz'], header.fields['ny'], header.fields['nx']
    
    # load all in one block
    if not lazy:
        with open(fname, 'rb') as fh:
            fh.read(start) # skip the header + extended header
            array = np.fromfile(fh, dtype=dtype).reshape((nz,ny,nx))

    # or list of LazyImages
    else:
        stride = dtype().itemsize*ny*nx
        array = [LazyImage(fname, (ny, nx), dtype, start+i*stride) for i in range(nz)]
    return array, header
   
def write(fname, array, header=None, Apix=1., xorg=0., yorg=0., zorg=0., is_vol=None):
    # get a default header
    if header is None:
        if is_vol is None:
            is_vol = True if len(set(array.shape)) == 1 else False # Guess whether data is vol or image stack
        header = MRCHeader.make_default_header(array, is_vol, Apix, xorg, yorg, zorg)
    # write the header
    f = open(fname,'wb')
    header.write(f)
    f.write(array.tobytes())
