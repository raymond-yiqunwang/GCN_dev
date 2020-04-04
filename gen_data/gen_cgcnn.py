import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pymatgen.core.structure import Structure


def main():
    filename = "./data_raw/custom_MPdata.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # read customized data
    MP_data = pd.read_csv(filename, sep=';', header=0, index_col=None)
    
    # target property
    target_list = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom']

    root_dir = "../data/"
    if os.path.exists(root_dir):
        _ = input("Attention, the existing data dir will be deleted and regenerated.. \
                  \n>> Hit Enter to continue, Ctrl+c to terminate..")
        shutil.rmtree(root_dir)
    os.mkdir(root_dir)

    for target in target_list:
        # specify output
        out_dir = root_dir + target + "/"
        os.mkdir(out_dir)
        
        # write id_prop file
        out_file = out_dir + "id_prop.csv"
        data = MP_data[['material_id', target]]
        data.to_csv(out_file, sep=',', header=None, index=None, mode='w')

        # copy atom_init file
        atom_init = "./atom_init.json"
        assert(os.path.isfile(atom_init))
        shutil.copyfile(atom_init, out_dir+'atom_init.json')
        
        # write cif files
        for idx, irow in MP_data.iterrows():
            material_id = irow['material_id']
            cif = irow['cif']
            with open(out_dir+material_id+'.cif', 'x') as f:
                f.write(cif)

    
if __name__ == "__main__":
    main()


