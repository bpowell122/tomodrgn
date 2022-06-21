'''
Corrupt particle images with structural noise, CTF, digital/shot noise
Written by Ellen Zhong, Emily Navarrete, and Joey Davis
'''

import argparse
import numpy as np
import sys, os
import pickle
from datetime import datetime as dt

from tomodrgn.ctf import compute_ctf_np as compute_ctf
from tomodrgn import mrc
from tomodrgn import utils
from tomodrgn import fft
from tomodrgn import dataset

log = utils.log

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('particles', help='Input MRC stack')
    parser.add_argument('--datadir', help='path to MRC stack')
    parser.add_argument('--snr1', default=1.4, type=float, help='SNR for first pre-CTF application of noise (default: %(default)s)')
    parser.add_argument('--snr2', default=0.05, type=float, help='SNR for second post-CTF application of noise (default: %(default)s)')
    parser.add_argument('--s1', type=float, help='Override --snr1 with gaussian noise stdev')
    parser.add_argument('--s2', type=float, help='Override --snr2 with gaussian noise stdev')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling defocus values (default: %(default)s)')
    parser.add_argument('-o', required=True, type=os.path.abspath, help='Output .mrcs')
    parser.add_argument('--out-star', type=os.path.abspath, help='Output star file (default: [output mrcs filename].star)')
    parser.add_argument('--out-pkl', type=os.path.abspath, help='Output pkl file (default: [output mrcs filename].pkl)')
    parser.add_argument('--Nimg', type=int, default=None, help='Number of particle to use (should be <= number of particles in the input mrc stack')
    parser.add_argument('--mask', default=None, help='Optional mask.mrcs to calculate stdev (output from project3d.py on volume_mask_refine.mrc in refinement)')

    group = parser.add_argument_group('CTF parameters')
    group.add_argument('--Apix', type=float, help='Pixel size (A/pix)')
    group.add_argument('--ctf-pkl', metavar='pkl', help='Use ctf parameters from a cryodrgn ctf pkl')
    group.add_argument('--df-file', metavar='pkl', help='Use defocus parameters from a pkl file of a Nx2 np.array of values')
    group.add_argument('--kv', default=300, type=float, help='Microscope voltage (kV) (default: %(default)s)')
    group.add_argument('--dfu', default=15000, type=float, help='Defocus U (A) (default: %(default)s)')
    group.add_argument('--dfv', default=15000, type=float, help='Defocus V (A) (default: %(default)s)')
    group.add_argument('--ang', default=0, type=float, help='Astigmatism angle (deg) (default: %(default)s)')
    group.add_argument('--cs', default=2, type=float, help='Spherical aberration (mm) (default: %(default)s)')
    group.add_argument('--wgh', default=0.1, type=float, help='Amplitude constrast ratio (default: %(default)s)')
    group.add_argument('--ps', default=0, type=float, help='Phase shift (deg) (default: %(default)s)')
    group.add_argument('-b', default=100, type=float, help='B factor for Gaussian envelope (A^2) (default: %(default)s)')
    group.add_argument('--sample-df', type=float, help='Jiggle defocus per image with this stdev (default: None)')
    group.add_argument('--no-astigmatism', action='store_true', help='Keep dfu and dfv the same per particle')
    group.add_argument('--normalize', action='store_true', help='Normalize particle stack to have a mean 0 and std 1')
    group.add_argument('--noinvert', default=False, action='store_true', help='Do not invert the images. Default is to invert, which is common/correct for cryosparc')

    group = parser.add_argument_group('Tilt series parameters')
    group.add_argument('--tilt-series-weighting', action='store_true', help='weight particles by dose and tilt angle')
    group.add_argument('--dose', default=3.0, type=float, help='Dose in e- / A2 / tilt')

    return parser

# todo - switch to cryodrgn starfile api
def write_starfile(out, mrc, Nimg, df, kv, wgh, cs, Apix, metadata=None):
    header = [ 
    'data_images',
    'loop_',
    '_rlnImageName',
    '_rlnDefocusU',
    '_rlnDefocusV',
    '_rlnDefocusAngle',
    '_rlnVoltage',
    '_rlnAmplitudeContrast',
    '_rlnSphericalAberration',
    '_rlnPhaseShift',
    '_rlnDetectorPixelSize']

    if metadata is not None:
        header.extend(['_rlnEuler1','_rlnEuler2','_rlnEuler3\n'])
        metadata = pickle.load(open(metadata,'rb'))
        assert len(metadata) == Nimg
    else:
        header[-1] += '\n'
    lines = []
    filename = os.path.basename(mrc)
    for i in range(Nimg):
        line = ['{:06d}@{}'.format(i+1,filename),
                '{:1f}'.format(df[i][0]),
                '{:1f}'.format(df[i][1]),
                '{:1f}'.format(df[i][2]),
                kv, wgh, cs,
                '{:1f}'.format(df[i][3]),
                Apix]
        if metadata is not None:
            line.extend(metadata[i])
        lines.append(' '.join([str(x) for x in line]))
    f = open(out, 'w')
    f.write('# Created {}\n'.format(dt.now()))
    f.write('\n'.join(header))
    f.write('\n'.join(lines))
    f.write('\n')

def snr_weight_dose_and_tilt(particles, snr_per_tilt_per_frequency, tilt_scheme):
    '''
    simulate tiltseries of particle's noiseless projections by dose- and tilt- weighting SNR by frequency

    particles: real-space noiseless stack, nptcls*ntilts x D x D
    snr_per_tilt_per_frequency: weighting of frequency components by accumulated dose, ntilts x D x D
    tilts:
    '''

    log('Fourier transforming stack to apply frequency-dependent weighting...')
    ntilts, D, D = snr_per_tilt_per_frequency.shape
    particles = np.asarray([fft.ht2_center(img) for img in particles], dtype=np.float32).reshape(-1, ntilts, D, D)
    particles *= snr_per_tilt_per_frequency
    particles *= np.cos(tilt_scheme * np.pi / 180).reshape(1, ntilts, 1, 1)
    particles = particles.reshape(-1, D, D)
    log('Inverse fourier transforming stack...')
    return np.asarray([fft.iht2_center(img) for img in particles])

def calculate_dose_snr(tilt_scheme, pixel_size, voltage, dose_per_A2_per_tilt, ntilts, nx, ny):
    '''
    code adapted from Grigorieff lab summovie_1.0.2/src/core/electron_dose.f90
    see also Grant and Grigorieff, eLife (2015) DOI: 10.7554/eLife.06980
    assumes even-sized FFT (i.e. non-ht-symmetrized, DC component is bottom-right of central 4 px)
    '''
    cumulative_doses = dose_per_A2_per_tilt * np.arange(tilt_scheme.shape[0])
    snr_per_tilt_per_frequency = np.zeros((ntilts, ny, nx))
    fourier_pixel_sizes = 1.0 / (np.array([nx, ny]))  # in units of 1/px
    box_center_indices = np.array([nx, ny]) // 2
    critical_dose_at_dc = 2 ** 31  # shorthand way to ensure dc component is always weighted ~1
    voltage_scaling_factor = 1.0 if voltage == 300 else 0.8  # 1.0 for 300kV, 0.8 for 200kV microscopes

    for k, dose_at_end_of_tilt in enumerate(cumulative_doses):

        for j in range(ny):
            y = ((j - box_center_indices[1]) * fourier_pixel_sizes[1])

            for i in range(nx):
                x = ((i - box_center_indices[0]) * fourier_pixel_sizes[0])

                if ((i, j) == box_center_indices).all():
                    spatial_frequency_critical_dose = critical_dose_at_dc
                else:
                    spatial_frequency = np.sqrt(x ** 2 + y ** 2) / pixel_size  # units of 1/A
                    spatial_frequency_critical_dose = (0.24499 * spatial_frequency ** (
                        -1.6649) + 2.8141) * voltage_scaling_factor  # eq 3 from DOI: 10.7554/eLife.06980

                snr_per_tilt_per_frequency[k, j, i] = np.exp((-0.5 * dose_at_end_of_tilt) / spatial_frequency_critical_dose)  # eq 5 from DOI: 10.7554/eLife.06980

    assert snr_per_tilt_per_frequency.min() >= 0.0
    assert snr_per_tilt_per_frequency.max() <= 1.0
    return snr_per_tilt_per_frequency

def add_noise(particles, D, sigma):
    particles += np.random.normal(0,sigma,particles.shape)
    return particles

def compute_full_ctf(D, Nimg, args):
    freqs = np.arange(-D/2,D/2)/(args.Apix*D)
    x0, x1 = np.meshgrid(freqs,freqs)
    freqs = np.stack([x0.ravel(),x1.ravel()],axis=1)
    if args.ctf_pkl: # todo: refator
        params = pickle.load(open(args.ctf_pkl,'rb'))
        try: 
            assert len(params) == Nimg
        except AssertionError:
            log('Note that the input ctf.pkl contains '+ str(len(params)) + ' particles, but that you have only chosen to output the first ' + str(Nimg) + ' particle')
            params = params[:Nimg]
        ctf = np.array([compute_ctf(freqs, *x[2:], args.b) for x in params])
        ctf = ctf.reshape((Nimg, D, D))
        df1 = np.array([x[2] for x in params])
        df2 = np.array([x[3] for x in params])
        dfa = np.array([x[4] for x in params])
        dfp = np.array([x[8] for x in params])
        df = np.stack([df1,df2, dfa, dfp], axis=1)
    elif args.df_file:
        raise NotImplementedError

#        df = pickle.load(open(args.df_file,'rb'))
#        try:
#            assert len(df) == Nimg
#        except AssertionError:
#            log('Note that the input df.pkl contains '+ str(len(df)) + ' particles, but that you have only chosen to output the first ' + str(Nimg) + ' particle')
#            df = df[:Nimg]
#        ctf = np.array([compute_ctf(freqs, i, i, args.ang, args.kv, args.cs, args.wgh, args.ps, args.b) \
#                for i in df])
#        ctf = ctf.reshape((Nimg, D, D))
#        df = np.stack([df,df], axis=1)
    elif args.sample_df:
        raise NotImplementedError
#
#        df1 = np.random.normal(args.dfu,args.sample_df,Nimg)
#        if args.no_astigmatism:
#            assert args.dfv == args.dfu, "--dfu and --dfv must be the same"
#            df2 = df1
#        else:
#            df2 = np.random.normal(args.dfv,args.sample_df,Nimg)
#        ctf = np.array([compute_ctf(freqs, i, j, args.ang, args.kv, args.cs, args.wgh, args.ps, args.b) \
#                for i, j in zip(df1, df2)])
#        ctf = ctf.reshape((Nimg, D, D))
#        df = np.stack([df1,df2], axis=1)
    else:
        ctf = compute_ctf(freqs, args.dfu, args.dfv, args.ang, args.kv, args.cs, args.wgh, args.ps, args.b)
        ctf = ctf.reshape((D,D))
        df = np.stack([np.ones(Nimg)*args.dfu, np.ones(Nimg)*args.dfv], np.ones(Nimg)*args.ang, np.ones(Nimg)*args.ps, axis=1)
    return ctf, df

def add_ctf(particles, ctf):
    assert len(particles) == len(ctf)
    particles = np.asarray([fft.ht2_center(img) for img in particles], dtype=np.float32)
    particles *= ctf
    del ctf
    particles = np.asarray([fft.iht2_center(img) for img in particles], dtype=np.float32)
    return particles

def normalize(particles):
    mu, std = np.mean(particles), np.std(particles)
    log('Shifting input images by {}'.format(mu))
    particles -= mu
    log('Scaling input images by {}'.format(std))
    particles /= std
    return particles

def invert(particles):
    particles *=-1
    return particles
 
def main(args):
    np.random.seed(args.seed)
    log('RUN CMD:\n'+' '.join(sys.argv))
    log('Arguments:\n'+str(args))
    if args.Nimg is None:
        log('Loading all particles')
        # particles = mrc.parse_mrc(args.particles, lazy=False)[0]
        particles = dataset.load_particles(args.particles, lazy=False, datadir=args.datadir)
        Nimg = len(particles)
    else:
        log('Lazy loading ' + str(args.Nimg) + ' particles')
        # particle_list = mrc.parse_mrc(args.particles, lazy=True, Nimg=Nimg)[0]
        particle_list = dataset.load_particles(args.particles, lazy=True, datadir=args.datadir)
        Nimg = args.Nimg
        particles = np.array([i.get() for i in particle_list[:Nimg]])
    D, D2 = particles[0].shape
    assert D == D2, 'Images must be square'

    log('Loaded {} images'.format(Nimg))
    #if not args.rad: args.rad = D/2
    #x0, x1 = np.meshgrid(np.arange(-D/2,D/2),np.arange(-D/2,D/2))
    #mask = np.where((x0**2 + x1**2)**.5 < args.rad)

    if args.s1 is not None:
        assert args.s2 is not None, "Need to provide both --s1 and --s2"

    if args.tilt_series_weighting:
        log('Weighting input noiseless projections by simulated dose and tilt angle')
        dose_symmetric_tilt_scheme = np.array([0, 3, -3, -6, 6, 9, -9, -12, 12, 15, -15, -18, 18, 21, -21, -24, 24, 27, -27,
                                               -30, 30, 33, -33, -36, 36, 39, -39, -42, 42, 45, -45, -48, 48, 51, -51, -54,
                                               54, 57, -57, -60, 60])
        ntilts = dose_symmetric_tilt_scheme.shape[0]

        snr_per_tilt_per_frequency = calculate_dose_snr(dose_symmetric_tilt_scheme, args.Apix, args.kv, args.dose, ntilts, D, D)
        particles = snr_weight_dose_and_tilt(particles, snr_per_tilt_per_frequency, dose_symmetric_tilt_scheme)

    if args.s1 is None:
        Nstd = min(100,Nimg)
        if args.mask is not None:
            mask_stack = mrc.parse_mrc(args.mask, lazy=False)[0]
            mask = np.where(mask_stack[:Nstd]>0.01)
        else:
            mask = np.where(particles[:Nstd]>0)
        std = np.std(particles[mask])
        s1 = std/np.sqrt(args.snr1)
    else:
        s1 = args.s1
    if s1 > 0:
        log('Adding noise with stdev {}'.format(s1))
        particles = add_noise(particles, D, s1)
    
    log('Calculating the CTF')
    ctf, defocus_list = compute_full_ctf(D, Nimg, args)
    log('Applying the CTF')
    particles = add_ctf(particles, ctf)

    if args.s2 is None:
        # cascading of noise processes according to Frank and Al-Ali (1975) & Baxter (2009)
        snr2 = (1+1/args.snr1)/(1/args.snr2-1/args.snr1)
        log('SNR2 target {} for total snr of {}'.format(snr2, args.snr2))
        s2 = std/np.sqrt(snr2)
    else:
        s2 = args.s2
    if s2 > 0:
        log('Adding noise with stdev {}'.format(s2))
        particles = add_noise(particles, D, s2)
    
    if args.normalize:
        log('Normalizing particles')
        particles = normalize(particles)

    if not(args.noinvert):
        log('Inverting particles')
        particles = invert(particles)

    log('Writing image stack to {}'.format(args.o))
    mrc.write(args.o, particles.astype(np.float32))

    if args.out_star is None:
        args.out_star = f'{args.o}.star'
    log(f'Writing associated .star file to {args.out_star}')
    if args.ctf_pkl:
        params = pickle.load(open(args.ctf_pkl,'rb'))
        try:
            assert len(params) == Nimg
        except AssertionError:
            log('Note that the input ctf.pkl contains '+ str(len(params)) + ' particles, but that you have only chosen to output the first ' + str(Nimg) + ' particle')
            params = params[:Nimg]
        args.kv = params[0][5]
        args.cs = params[0][6]
        args.wgh = params[0][7]
        args.Apix = params[0][1]
    write_starfile(args.out_star, args.o, Nimg, defocus_list, args.kv, args.wgh, args.cs, args.Apix)

    if not args.ctf_pkl:
        if args.out_pkl is None:
            args.out_pkl = f'{args.o}.pkl'
        log(f'Writing CTF params pickle to {args.out_pkl}')
        params = np.ones((Nimg, 9), dtype=np.float32)
        params[:,0] = D
        params[:,1] = args.Apix
        params[:,2:4] = defocus_list
        params[:,4] = args.ang
        params[:,5] = args.kv
        params[:,6] = args.cs
        params[:,7] = args.wgh
        params[:,8] = args.ps
        log(params[0])
        with open(args.out_pkl,'wb') as f:
            pickle.dump(params,f)

if __name__ == '__main__':
    main(parse_args().parse_args())
