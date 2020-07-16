import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pymatgen.core.structure import Structure

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='./', metavar='DATA_DIR')
args = parser.parse_args()

def main():
    filename = os.path.join(args.root, "data_raw/custom_MPdata.csv")
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # read customized data
    MP_data = pd.read_csv(filename, sep=';', header=0, index_col=None)
    
    # add MIT column
    MP_data['MIT'] = (MP_data['band_gap'] > 0.).astype(float)
    
    # target property
    target_list = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom', 'MIT']

    data_dir = "../data/" if args.root=='./' else os.path.join(args.root, "data")
    if os.path.exists(data_dir):
        _ = input("Attention, the existing data dir will be deleted and regenerated.. \
                  \n>> Hit Enter to continue, Ctrl+c to terminate..")
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)

    for target in target_list:
        # specify output
        out_dir = os.path.join(data_dir, target)
        os.mkdir(out_dir)
        
        # write id_prop file
        out_file = os.path.join(out_dir, "id_prop.csv")
        data = MP_data[['material_id', target]]
        data.to_csv(out_file, sep=',', header=None, index=None, mode='w')

        # copy atom_init file
        atom_init = "./atom_init.json"
        assert(os.path.isfile(atom_init))
        shutil.copyfile(atom_init, os.path.join(out_dir, 'atom_init.json'))
        
        # write cif files
        for idx, irow in MP_data.iterrows():
            material_id = irow['material_id']
            cif = irow['cif']
            with open(os.path.join(out_dir, material_id)+'.cif', 'x') as f:
                f.write(cif)

    
if __name__ == "__main__":
    main()


