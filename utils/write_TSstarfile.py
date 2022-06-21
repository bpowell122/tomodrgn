'''
Create a Warp tomography particleseries file from a particle stack, poses, tilt data
'''

import argparse
import numpy as np
import sys, os
import pickle

import pandas as pd

from tomodrgn import dataset
from tomodrgn import utils
from tomodrgn import starfile
from tomodrgn import mrc
log = utils.log

HEADERS = ['_rlnImageName',
           '_rlnDetectorPixelSize']

CTF_HDRS = ['_rlnDefocusU',
            '_rlnDefocusV',
            '_rlnDefocusAngle',
            '_rlnVoltage',
            '_rlnSphericalAberration',
            '_rlnAmplitudeContrast',
            '_rlnPhaseShift']

POSE_HDRS = ['_rlnAngleRot',
             '_rlnAngleTilt',
             '_rlnAnglePsi',
             '_rlnOriginX',
             '_rlnOriginY']

MICROGRAPH_HDRS = ['_rlnMicrographName',
                   '_rlnCoordinateX',
                   '_rlnCoordinateY']

MISC_HEADERS = ['_rlnCtfBfactor',
                '_rlnCtfScalefactor',
                '_rlnGroupName']

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('particles', help='Input particles (.mrcs, .txt)')
    parser.add_argument('--datadir', type=os.path.abspath, help='Path prefix to particle stack if loading relative paths from a .star or .cs file')
    parser.add_argument('--Apix', help='A/px')
    parser.add_argument('--ctf', help='Optionally input ctf.pkl')
    parser.add_argument('--poses', help='Optionally include pose.pkl')
    parser.add_argument('-o', type=os.path.abspath, required=True, help='Output .star file')
    parser.add_argument('--full-path', action='store_true', help='Write the full path to particles (default: relative paths)')
    return parser

def main(args):
    assert args.o.endswith('.star')
    particles = dataset.load_particles(args.particles, lazy=True, datadir=args.datadir)
    ntilts = 41
    nptcls = len(particles) // ntilts
    if args.ctf:
        ctf = utils.load_pkl(args.ctf)
        assert ctf.shape[1] == 9, "Incorrect CTF pkl format"
        assert len(particles) == len(ctf), f'{len(particles)} != {len(ctf)}, Number of particles != number of CTF paraameters'
    if args.poses:
        poses = utils.load_pkl(args.poses)
        assert len(particles) == len(poses[0]), f'{len(particles)} != {len(poses[0])}, Number of particles != number of poses'
    log('{} particles'.format(len(particles)))

    # ind = np.arange(len(particles))
    ind = np.tile(np.arange(20*41), 500)
    ind += 1 # CHANGE TO 1-BASED INDEXING
    image_names = [img.fname for img in particles]
    if args.full_path:
        image_names = [os.path.abspath(img.fname) for img in particles]
    names = [f'{i}@{name}' for i,name in zip(ind, image_names)]

    if args.ctf:
        ctf = ctf[:,2:]

    # convert poses
    if args.poses:
        if type(poses) == tuple:
            eulers = utils.R_to_relion_scipy(poses[0])
            D = particles[0].get().shape[0]
            trans = poses[1] * D # convert from fraction to pixels
        else:
            eulers = utils.R_to_relion_scipy(poses)
            trans = None

    data = {HEADERS[0]:names}
    headers = HEADERS
    data[HEADERS[1]] = float(args.Apix) * np.ones((nptcls * ntilts))

    if args.ctf:
        for i in range(7):
            data[CTF_HDRS[i]] = ctf[:,i]
        headers += CTF_HDRS
    if args.poses:
        for i in range(3):
            data[POSE_HDRS[i]] = eulers[:,i]
        headers += POSE_HDRS[:3]
        if trans is not None:
            for i in range(2):
                data[POSE_HDRS[3+i]] = trans[:,i]
            headers += POSE_HDRS[3:]

    # assumes 41 tilts per particle
    # TODO add option for different tilt scheme, CTF B-factors (dose), etc
    headers += MISC_HEADERS
    ctf_bfactors = np.array([-12.99, -25.97, -38.96, -51.94, -64.93, -77.92, -90.90, -103.89, -116.87, -129.86, -142.85,
                             -155.83, -168.82, -181.80, -194.79, -207.78, -220.76, -233.75, -246.73, -259.72, -272.71,
                             -285.69, -298.68, -311.66, -324.65, -337.63, -350.62, -363.61, -376.59, -389.58, -402.56,
                             -415.55, -428.54, -441.52, -454.51, -467.49, -480.48, -493.47, -506.45, -519.44, -532.42])
    ctf_bfactors = np.repeat(ctf_bfactors.reshape(1, ctf_bfactors.shape[0]), nptcls, axis=0).reshape(-1)
    data[MISC_HEADERS[0]] = ctf_bfactors

    dose_symmetric_tilts = np.array([0, 3, -3, -6, 6, 9, -9, -12, 12, 15, -15, -18, 18, 21, -21, -24, 24, 27, -27,
                                     -30, 30, 33, -33, -36, 36, 39, -39, -42, 42, 45, -45, -48, 48, 51, -51, -54,
                                     54, 57, -57, -60, 60])
    tilt_angles = np.repeat(dose_symmetric_tilts.reshape(1, dose_symmetric_tilts.shape[0]), nptcls, axis=0).reshape(-1)
    tilt_angles = np.abs(np.cos(tilt_angles * np.pi / 180))
    data[MISC_HEADERS[1]] = tilt_angles

    group_names = []
    for ptcl in range(nptcls):
        for tilt in range(ntilts):
            group_names.append(f'000_{ptcl:06d}')
    data[MISC_HEADERS[2]] = group_names

    print([f'{h} : {data[h].shape}' for h in headers if type(data[h]) != list])

    df = pd.DataFrame(data=data)

    s = starfile.Starfile(headers,df)
    s.write(args.o)
    log(f'Wrote: {args.o}')

if __name__ == '__main__':
    main(parse_args().parse_args())
