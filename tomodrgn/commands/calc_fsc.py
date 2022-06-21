import numpy as np
from tomodrgn import mrc, fft
import datetime as dt
import time
from scipy import ndimage
from matplotlib import pyplot as plt
import os

try:
    os.remove('/home/bmp/fsc_new.png')
    os.remove('/home/bmp/fsc_bmp.png')
except:
    pass

def load_vols():
    vol1_path = '/nese/mit/group/jhdavis/data001/lkinman/2021_nat_prot/EMPIAR_10076/01_128_8D_256/convergence.49/vols.44/vol_000.mrc'
    vol2_path = '/nese/mit/group/jhdavis/data001/lkinman/2021_nat_prot/EMPIAR_10076/01_128_8D_256/convergence.49/vols.49/vol_000.mrc'
    mask1_path = '/nese/mit/group/jhdavis/data001/lkinman/2021_nat_prot/EMPIAR_10076/01_128_8D_256/convergence.49/vols.44/vol_000.mask.mrc'
    mask2_path = '/nese/mit/group/jhdavis/data001/lkinman/2021_nat_prot/EMPIAR_10076/01_128_8D_256/convergence.49/vols.49/vol_000.mask.mrc'

    # vol1_path = '/home/bmp/groel_stagg_map1.mrc'
    # vol2_path = '/home/bmp/groel_stagg_map2.mrc'

    # vol1_path = '/home/bmp/cryosparc_P42_J17_002_volume_map_half_A.mrc'
    # vol2_path = '/home/bmp/cryosparc_P42_J17_002_volume_map_half_B.mrc'


    vol1, _ = mrc.parse_mrc(vol1_path)
    vol2, _ = mrc.parse_mrc(vol2_path)

    if mask1_path:
        mask1 = mrc.parse_mrc(mask1_path)[0]
        assert mask1.min() >= 0
        assert mask1.max() <= 1
        vol1 *= mask1

    if mask2_path:
        mask2 = mrc.parse_mrc(mask2_path)[0]
        assert mask2.min() >= 0
        assert mask2.max() <= 1
        vol2 *= mask2

    return vol1, vol2

def calc_fsc(vol1, vol2):
    D = vol1.shape[0]
    x = np.arange(-D // 2, D // 2)
    x2, x1, x0 = np.meshgrid(x, x, x, indexing='ij')
    coords = np.stack((x0, x1, x2), -1)
    r = (coords ** 2).sum(-1) ** .5

    #assert r[D // 2, D // 2, D // 2] == 0.0

    vol1 = fft.fftn_center(vol1)
    vol2 = fft.fftn_center(vol2)

    prev_mask = np.zeros((D, D, D), dtype=bool)
    fsc = [1.0]
    for i in range(1, D // 2):
        mask = r < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        p = np.vdot(v1, v2) / (np.vdot(v1, v1) * np.vdot(v2, v2)) ** .5
        fsc.append(p.real)
        prev_mask = mask
    fsc = np.asarray(fsc)
    x = np.arange(D // 2) / D
    res = np.stack((x, fsc), 1)
    return x, fsc

def calc_fsc_new(vol1, vol2):
    Apix = 4.435             #A/px
    df = 1.0/Apix        #px/A
    n = vol1.shape[0]    #px
    qx_ = np.fft.fftfreq(n)*n*df
    qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
    qx_max = qx.max()
    qr = np.sqrt(qx**2+qy**2+qz**2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr>0])
    nbins = int(qmax/qstep)
    qbins = np.linspace(0,nbins*qstep,nbins+1)
    x = np.arange(n//2) / n
    #create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1

    F1 = np.fft.fftn(vol1)
    F2 = np.fft.fftn(vol2)

    numerator = ndimage.sum(np.real(F1*np.conj(F2)), labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    term1 = ndimage.sum(np.abs(F1)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    term2 = ndimage.sum(np.abs(F2)**2, labels=qbin_labels, index=np.arange(0,qbin_labels.max()+1))
    denominator = (term1*term2)**0.5
    FSC = numerator/denominator
    qidx = np.where(qbins < qx_max)
    return (x[qidx], FSC[qidx])

def calc_fsc_bmp(vol1, vol2):
    # define fourier grid and label into shells
    D = vol1.shape[0]
    x = np.arange(-D//2, D//2)
    x0, x1, x2 = np.meshgrid(x, x, x, indexing='ij')
    r = np.sqrt(x0**2 + x1**2 + x2**2)
    r_max = D//2 #sphere inscribed within volume box
    r_step = 1 #int(np.min(r[r>0]))
    bins = np.arange(0, r_max, r_step)
    labels = np.searchsorted(bins, r, side='right')

    # load masked volumes in fourier space
    vol1_ft = fft.fftn_center(vol1)
    vol2_ft = fft.fftn_center(vol2)

    # calculate the FSC via labeled shells
    num = ndimage.sum(np.real(vol1_ft*np.conjugate(vol2_ft)), labels = labels, index = bins+1)
    den1 = ndimage.sum(np.abs(vol1_ft)**2, labels = labels, index = bins+1)
    den2 = ndimage.sum(np.abs(vol2_ft)**2, labels = labels, index = bins+1)
    fsc = num / np.sqrt(den1 * den2)

    x = bins/D # x axis should be spatial frequency in 1/px
    return x, fsc



sizes = [64, 128, 192, 256, 300, 350, 400, 440]
#sizes = [64, 128]
times = np.zeros((len(sizes), 3))

for i, box in enumerate(sizes):
    t1 = time.perf_counter()
    vol1, _ = mrc.parse_mrc(f'/home/bmp/fsc_testing_maps/KvAP_J247_class001.{box}.mrc')
    vol2, _ = mrc.parse_mrc(f'/home/bmp/fsc_testing_maps/KvAP_J247_class002.{box}.mrc')
    t2 = time.perf_counter()
    calc_fsc(vol1, vol2)
    t3 = time.perf_counter()
    calc_fsc_bmp(vol1, vol2)
    t4 = time.perf_counter()

    times[i, 0] = t2-t1 # volume loading
    times[i, 1] = t3-t2 # old FSC implementation
    times[i, 2] = t4-t3 # new fsc implementation
    print(f'done box: {box}')

plt.plot(sizes, times, '-o')
plt.legend(['volume loading', 'calc_fsc (old)', 'calc_fsc_bmp (new)'])
plt.xlabel('box size (voxels)')
plt.ylabel('time (seconds)')
plt.savefig('/home/bmp/fsc_testing_maps/timing.png')



# x_new, fsc_new = calc_fsc_new(vol1, vol2)
# plt.plot(x_new, fsc_new)
# plt.savefig('/home/bmp/fsc_new.png')
# plt.clf()
#
# x_bmp, fsc_bmp = calc_fsc_bmp(vol1, vol2)
# plt.plot(x_bmp, fsc_bmp)
# plt.savefig('/home/bmp/fsc_bmp.png')
# plt.clf()
#
# plt.plot(x_new, fsc_new)
# plt.plot(x_bmp, fsc_bmp)
# plt.savefig('/home/bmp/fsc_superimposed.png')