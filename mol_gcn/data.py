from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
import functools
import random
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

def get_train_valid_test_loader(dataset, collate_func, batch_size=64, train_ratio=0.8, valid_ratio=0.1, 
                                test_ratio=0.1, num_workers=1, pin_memory=False):
    total_size = len(dataset)
    indices = list(range(total_size))
    train_split = int(np.floor(total_size * train_ratio))
    train_sampler = SubsetRandomSampler(indices[:train_split])
    valid_split = train_split + int(np.floor(total_size * valid_ratio))
    valid_sampler = SubsetRandomSampler(indices[train_split:valid_split])
    test_sampler = SubsetRandomSampler(indices[valid_split:])

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              collate_fn=collate_func,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=valid_sampler,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=test_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader


def collate_func(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GraphReader(Dataset):
    def __init__(self, root_dir, target, rand_seed=123):
        self.root_dir = root_dir
        self.target = target
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        self.file_names = os.listdir(root_dir)
        random.seed(rand_seed)
        random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def init_graph(properties):
        # read molecular properties
        prop = properties.split()
        tag = prop[0]
        index = int(prop[1])
        rot_A = float(prop[2])
        rot_B = float(prop[3])
        rot_C = float(prop[4])
        dipole = float(prop[5])
        alpha = float(prop[6])
        homo = float(prop[7])
        lumo = float(prop[8])
        gap = float(prop[9])
        r2 = float(prop[10])
        vib = float(prop[11])
        U0 = float(prop[12])
        U_RT = float(prop[13])
        H_RT = float(prop[14])
        G_RT = float(prop[15])
        Cv_RT = float(prop[16])
    
        graph = nx.Graph(tag=tag, index=index, rot_A=rot_A, rot_B=rot_B, rot_C=rot_C, 
                         dipole=dipole, alpha=alpha, homo=homo, lumo=lumo, gap=gap, r2=r2, 
                         vib=vib, U0=U0, U_RT=U_RT, H_RT=H_RT, G_RT=G_RT, Cv_RT=Cv_RT)
        target_dict = {'dipole': dipole, 'alpha': alpha, 'homo': homo, 'lumo': lumo,
                       'gap': gap, 'r2': r2, 'vib': vib, 'U0': U0, 'U_RT': U_RT, 
                       'H_RT': H_RT, 'G_RT': G_RT, 'Cv_RT': Cv_RT}
    
        return graph, target_dict

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        filename = self.file_names[idx]
        with open(os.path.join(self.root_dir, filename), 'r') as f:
            # number of atoms
            natom = int(f.readline())
            # molecular properties
            properties = f.readline()
            graph, target_dict = GraphReader.init_graph(properties)
    
            # atomic properties
            atom_prop = []
            for i in range(natom):
                atom_info = f.readline()
                # TODO
#                atom_info = atom_info.replace('.*^', 'e')
#                atom_info = atom_info.replace('*^', 'e')
                atom_info = atom_info.split()
                atom_prop.append(atom_info)
    
            # skip vibrational frequencies
            f.readline()
   
            # SMILES
            smiles = f.readline()
            smiles = smiles.split()[0]
            mol = Chem.MolFromSmiles(smiles)
            assert(mol)
            mol = Chem.AddHs(mol)
            print(smiles)
    
            fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            features = factory.GetFeaturesForMol(mol)
    
            # create nodes
            for idx in range(mol.GetNumAtoms()):
                iatom = mol.GetAtomWithIdx(idx)
                graph.add_node(idx, atom_type=iatom.GetSymbol(), atom_num=iatom.GetAtomicNum(),
                               acceptor=0, donor=0, aromatic=iatom.GetIsAromatic(),
                               hybridization=iatom.GetHybridization(), num_H=iatom.GetTotalNumHs(),
                               coord=np.array(atom_prop[idx][1:4]).astype(np.float),
                               pcharg=float(atom_prop[idx][4]))
        
            for idx in range(len(features)):
                if features[idx].GetFamily() == 'Donor':
                    donors = features[idx].GetAtomIds()
                    for j in donors:
                        graph.nodes[j]['donor'] = 1
                elif features[idx].GetFamily() == 'Acceptor':
                    acceptors = features[idx].GetAtomIds()
                    for k in acceptors:
                        graph.nodes[k]['acceptor'] = 1
    
            # create edges
            for i in range(mol.GetNumAtoms()):
                for j in range(mol.GetNumAtoms()):
                    e_ij = mol.GetBondBetweenAtoms(i, j)
                    distance = np.linalg.norm(graph.nodes[i]['coord'] - graph.nodes[j]['coord'])
                    if e_ij:
                        graph.add_edge(i, j, bond_type=e_ij.GetBondType(), distance=distance)
                    else:
                        graph.add_edge(i, j, bond_type=None, distance=distance)
        
        # nodes
        nodes = []
        for node, ndata in graph.nodes(data=True):
            nfeat = []
            # atom type HCNOF
            nfeat += [int(ndata['atom_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
            # atomic number
            nfeat.append(ndata['atom_num'])
            # partial charge
            nfeat.append(ndata['pcharg'])
            # acceptor
            nfeat.append(ndata['acceptor'])
            # donor
            nfeat.append(ndata['donor'])
            # aromatic
            nfeat.append(int(ndata['aromatic']))
            # hybrid
            nfeat += [int(ndata['hybridization'] == x) for x in [
                        Chem.rdchem.HybridizationType.SP,
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3]]
            nodes.append(nfeat)

        # edges
        edges = {}
        remove_edges = []
        for n1, n2, edata in graph.edges(data=True):
            efeat = []
            if edata['bond_type'] is None:
                remove_edges.append((n1, n2))
            else:
                efeat.append(edata['distance'])
                efeat += [int(edata['bond_type'] == x) for x in [
                            Chem.rdchem.BondType.SINGLE,
                            Chem.rdchem.BondType.DOUBLE,
                            Chem.rdchem.BondType.TRIPLE,
                            Chem.rdchem.BondType.AROMATIC]]
            if efeat:
                edges[(n1, n2)] = efeat
        for e in remove_edges:
            graph.remove_edge(*e)

        adj_mat = nx.to_numpy_matrix(graph)
        
        return (nodes, edges, adj_mat), target_dict[target], graph.graph['index']


if __name__ == "__main__":
    data_root = '../dataset_QM9/'
    target = 'gap'
    random_seed = 123
    dataset = GraphReader(data_root, target, random_seed)
    """
    (nodes, edges, adj_mat), target, idx = dataset[0]
    print(nodes)
    print(edges)
    print(adj_mat)
    print(target)
    print(idx)
    """
    train, valid, test = get_train_valid_test_loader(dataset, collate_func, batch_size=2)
#    print(next(iter(train)))


