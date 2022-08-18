'''
Create a relion3.0 star file from a particle stack, poses.pkl, ctf.pkl, tilt+dose data
'''

import argparse
import os
import numpy as np
import pandas as pd

from tomodrgn import dataset
from tomodrgn import utils
from tomodrgn import starfile
log = utils.log

IMAGE_HEADERS = ['_rlnImageName']

CTF_HEADERS = ['_rlnDetectorPixelSize',
               '_rlnDefocusU',
               '_rlnDefocusV',
               '_rlnDefocusAngle',
               '_rlnVoltage',
               '_rlnSphericalAberration',
               '_rlnAmplitudeContrast',
               '_rlnPhaseShift']

POSE_HEADERS = ['_rlnAngleRot',
                '_rlnAngleTilt',
                '_rlnAnglePsi',
                '_rlnOriginX',
                '_rlnOriginY']

MICROGRAPH_HEADERS = ['_rlnMicrographName',
                      '_rlnCoordinateX',
                      '_rlnCoordinateY']

MISC_HEADERS = ['_rlnCtfBfactor',
                '_rlnCtfScalefactor',
                '_rlnGroupName']

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('particles', type=os.path.abspath, help='Input particles (.mrcs, .txt, .star, .cs)')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .star file')
    parser.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    parser.add_argument('--ctf', type=os.path.abspath, help='Optionally input ctf.pkl')
    parser.add_argument('--Apix', type=float, help='Override A/px from ctf.pkl (useful if using downsampled stack)')
    parser.add_argument('--poses', type=os.path.abspath, help='Optionally include pose.pkl')
    parser.add_argument('--tilt-series', type=os.path.abspath, help='Path to file (.txt) specifying full tilt series x-axis stage-tilt scheme in degrees. ')
    parser.add_argument('--dose', type=float, help='Dose in e- / A2 / tilt. ')
    parser.add_argument('--group-index', type=int, help='Counter used in _rlnGroupName to identify different sets of particles')
    parser.add_argument('--full-path', action='store_true', help='Write the full path to particles (default: relative paths)')
    return parser


def main(args):
    assert args.o.endswith('.star')

    # read in all data
    particles = dataset.load_particles(args.particles, lazy=True, datadir=args.datadir)
    nimgs = len(particles)
    if args.ctf:
        ctf = utils.load_pkl(args.ctf)
        assert ctf.shape[1] == 9, "Unrecognized CTF pkl format"
        assert nimgs == len(ctf), f'{nimgs} != {len(ctf)}, Number of particles != number of CTF paraameters'
        if args.Apix: ctf[:,1] = args.Apix
    else: assert args.Apix is not None, 'Apix must be provided either via --Apix or indirectly in ctf.pkl'
    if args.poses:
        poses = utils.load_pkl(args.poses)
        if type(poses) == tuple:
            assert nimgs == len(poses[0]), f'{nimgs} != {len(poses[0])}, Number of particles != number of poses'
        else:
            log('--poses input contains only rotations')
            assert type(poses) == np.ndarray, 'Unrecognized pose pkl format'
            assert nimgs == len(poses), f'{nimgs} != {len(poses)}, Number of particles != number of poses'
    if args.tilt_series:
        tilt_series = np.loadtxt(args.tilt_series, dtype=np.float32)
        ntilts = len(tilt_series)
        nptcls = len(particles) // ntilts
        assert nimgs % len(tilt_series) == 0, 'The provided tilt scheme does not correlate with the number of input images'
    else: nptcls = len(particles)
    if args.dose:
        assert args.tilt_series is not None, 'Must supply a tilt series scheme to use dose'
        dose = args.dose
    log(f'Read in data for {nimgs} images')

    # configure _rlnImageName inputs
    ind = np.arange(nimgs)
    ind += 1 # CHANGE TO 1-BASED INDEXING
    image_names = [img.fname for img in particles]
    if args.full_path:
        image_names = [os.path.abspath(img.fname) for img in particles]
    names = [f'{i}@{name}' for i,name in zip(ind, image_names)]

    # configure ctf header inputs
    if args.ctf:
        ctf = ctf[:,2:]  # first col is boxsize, second col is Apix, both of which are read in elsewhere for this script

    # configure pose header inputs
    if args.poses:
        if type(poses) == tuple:
            eulers = utils.R_to_relion_scipy(poses[0])
            D = particles[0].get().shape[0]
            trans = poses[1] * D if poses[1] is not None else None  # convert from fraction to pixels
        else:
            eulers = utils.R_to_relion_scipy(poses)
            trans = None

    # configure b factors
    if args.dose:
        cumulative_doses = np.arange(1, ntilts+1) * dose
        bfactors = np.tile(cumulative_doses * -4, nptcls)

    # configure scale factors
    if args.tilt_series:
        scalefactors = np.tile(np.cos(tilt_series * np.pi / 180), nptcls)

    # configure group names
    if args.group_index is not None:
        group_names = [f'{args.group_index:03d}_{ptcl:06d}' for ptcl in range(nptcls) for tilt in range(ntilts)]

    # populate pandas dataframe with configured data
    df = pd.DataFrame(data=names, columns=IMAGE_HEADERS)
    df[CTF_HEADERS[0]] = np.full(len(particles), args.Apix)
    if args.ctf:
        for i in range(7):
            df[CTF_HEADERS[i+1]] = ctf[:,i]
    if args.poses:
        for i in range(3):
            df[POSE_HEADERS[i]] = eulers[:,i]
        if trans is not None:
            for i in range(2):
                df[POSE_HEADERS[i+3]] = trans[:,i]
    if args.tilt_series:
        if args.dose:
            df[MISC_HEADERS[0]] = bfactors
        df[MISC_HEADERS[1]] = scalefactors
    if args.group_index is not None:
        df[MISC_HEADERS[2]] = group_names

    headers = [f'{header} #{i+1}' for i, header in enumerate(df.columns.values.tolist())]
    df.columns = headers
    s = starfile.Starfile(headers, df)
    s.write(args.o)
    log(f'Wrote: {args.o}')


if __name__ == '__main__':
    main(parse_args().parse_args())
