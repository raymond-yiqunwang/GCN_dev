import pandas as pd
import json
from pymatgen.core.periodic_table import Element

# raw data
atom_data = pd.read_csv('atomic_props.csv', header=0, index_col=None)
# make a copy of original data
atom_data2 = atom_data.copy()

# electronegativity
neg_data = atom_data['electronegativity']
interval_neg = (neg_data.max() - neg_data.min()) / 10
atom_data2['electronegativity'] = ((neg_data-neg_data.min())/interval_neg-1E-8).astype(int)

# covalent radius
cov_data = atom_data['covalent_radius']
interval_cov = (cov_data.max() - cov_data.min()) / 10
atom_data2['covalent_radius'] = ((cov_data-cov_data.min())/interval_cov-1E-8).astype(int)

# log first ionization energy
ion_data = atom_data['log_first_ionization_energy']
interval_ion = (ion_data.max() - ion_data.min()) / 10
atom_data2['log_first_ionization_energy'] = ((ion_data-ion_data.min())/interval_ion-1E-8).astype(int)

# electron affinity
aff_data = atom_data['electron_affinity']
interval_aff = (aff_data.max() - aff_data.min()) / 10
atom_data2['electron_affinity'] = ((aff_data-aff_data.min())/interval_aff-1E-8).astype(int)

# period, size: 9
period_dummy = pd.get_dummies(atom_data2['period'], prefix='period')
# group, size: 20
group_dummy = pd.get_dummies(atom_data2['group'], prefix='group')
# blocks, size: 4
block_dummy = pd.get_dummies(atom_data2['block'], prefix='block')
# electronegativity, size: 10
electronegativity_dummy = pd.get_dummies(atom_data2['electronegativity'], 
                                         prefix='electronegativity')
# covalent radius, size: 10
covalent_radius_dummy = pd.get_dummies(atom_data2['covalent_radius'], 
                                       prefix='covalent_radius')
# valence electrons, size: 16
valence_electrons_dummy = pd.get_dummies(atom_data2['valence_electrons'],
                                         prefix='valence_electrons')
# log first ionization energy, size: 10 
log_first_ionization_energy_dummy = pd.get_dummies(atom_data2['log_first_ionization_energy'], 
                                                   prefix='log_first_ionization_energy')
# electron affinity, size: 10
electron_affinity_dummy = pd.get_dummies(atom_data2['electron_affinity'], 
                                         prefix='electron_affinity')

data_onehot = pd.concat([period_dummy, group_dummy, block_dummy,
                         electronegativity_dummy, covalent_radius_dummy,
                         valence_electrons_dummy, electron_affinity_dummy,
                         log_first_ionization_energy_dummy], axis=1)

out_dict = {}
for idx, irow in data_onehot.iterrows():
    elem_index = idx + 1
    out_dict[elem_index] = irow.to_list()

with open('atom_init_new.json', 'w') as fout:
    json.dump(out_dict, fout)


