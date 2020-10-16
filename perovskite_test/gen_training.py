import os
import sys
import ast
import shutil
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
sys.path.append("../data_gen/xrd_simulator/")
import xrd_simulator as xrd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

xrd_simulator = xrd.XRDSimulator(wavelength='CuKa')

root_dir = "./Perovskite_data/"
save_dir = "./test/"
data = pd.read_csv(root_dir+"data_all.csv", header=0, sep=',', index_col=None)
for material_id in data['material_id']:
    with open(root_dir+material_id+".cif") as f:
        cif_struct = Structure.from_str(f.read(), fmt="cif")
    sga = SpacegroupAnalyzer(cif_struct, symprec=0.1)

    # conventional cell
    conventional_struct = sga.get_conventional_standard_structure()
    _, conventional_recip_latt, conventional_features = \
        xrd_simulator.get_pattern(structure=conventional_struct)
    # save conventional reciprocal lattice vector
    np.save(os.path.join(root_dir, material_id+"_conventional_basis.npy"), \
            conventional_recip_latt)
    # save conventional diffraction pattern
    np.save(os.path.join(root_dir, material_id+"_conventional.npy"), \
            np.array(conventional_features))

    feat_file = root_dir + material_id + "_conventional.npy"
    hkl_feat = np.load(feat_file)
            
    conditions = np.where((np.max(hkl_feat[:,:-1], axis=1)<3.1) & \
                          (np.min(hkl_feat[:,:-1], axis=1)>-3.1))
    selected_hkl_feat = hkl_feat[conditions]
    assert(selected_hkl_feat.shape[0] <= 343)

    # convert to Cartesion
    basis_file = root_dir + material_id + "_conventional_basis.npy"
    recip_latt = np.load(basis_file)
    recip_pos = np.dot(selected_hkl_feat[:,:-1], recip_latt)
    # CuKa by default
    max_r = 2 / 1.54184
    recip_pos /= max_r
    assert(np.amax(recip_pos) <= 1.0)
    assert(np.amin(recip_pos) >= -1.0)
    # normalize diffraction intensity
    intensity = np.log(1+selected_hkl_feat[:,-1]) / 3.
    intensity = intensity.reshape(-1, 1)
    assert(np.amax(intensity) <= 1.3)
    assert(np.amin(intensity) >= 0.)
    # generate point cloud and write to file
    point_cloud = np.concatenate((recip_pos, intensity), axis=1)
    np.save(os.path.join(save_dir, material_id), point_cloud)


