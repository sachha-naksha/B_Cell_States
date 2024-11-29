#!/usr/bin/python3

import logging
import os
from os import makedirs, linesep
from os.path import join as pjoin
import numpy as np
import pandas as pd
from dictys.traj import trajectory, point
import h5py
import sys
import multiprocessing

def process_subset(xi, subsets, rot, noverlap, dmax, dist):
    """
    Process a single subset to determine if it should be removed.

    Parameters:
    xi (int): Index of the subset to process.
    subsets (list): List of all subsets.
    rot (numpy.ndarray): Rotation array for subset indices.
    noverlap (int): Minimum number of overlapping cells required.
    dmax (float): Maximum allowed distance between subsets.
    dist (numpy.ndarray): Distance matrix between cells.

    Returns:
    int or None: Returns xi if the subset should be kept, None if it should be removed.
    """
    t1 = [subsets[x] for x in rot[xi]]
    if len(np.intersect1d(*t1, assume_unique=True, return_indices=False)) < noverlap:
        return None
    if dist[t1[0]][:,t1[1]].mean() - max(dist[t1[0]][:,t1[0]].mean(), dist[t1[1]][:,t1[1]].mean()) >= dmax:
        return None
    return xi

def process_edge(edge, s, ncell, noverlap, dmax):
    """
    Process a single edge to create subsets.

    Parameters:
    edge (int): Index of the edge to process.
    s (point): Point object containing trajectory information.
    ncell (int): Number of cells in each subset.
    noverlap (int): Minimum number of overlapping cells required.
    dmax (float): Maximum allowed distance between subsets.

    Returns:
    tuple: Contains edge_edges, edge_locs, edge_subsets, and edge_neighbors.
    """
    #B1,B2
    ids = np.nonzero(s.edges == edge)[0]
    ids = ids[np.argsort(s.locs[ids])]
    subsets = s.disto[ids][:,:ncell]
    subsets = [s.ans_subsets[s.p.edges[edge,0]]] + list(subsets) + [s.ans_subsets[s.p.edges[edge,1]]]

    #Remove identical subsets
    t1 = np.nonzero([(subsets[x] != subsets[x+1]).any() for x in range(1, len(subsets)-1)])[0]
    subsets = [subsets[x] for x in [0] + list(t1) + [len(subsets)-1]]
    ids = ids[t1]

    #Bidirectional linked list
    rot = np.array([np.arange(len(subsets))-1, np.arange(len(subsets))+1]).T
    rot[0,0] = len(subsets)

    #B3
    t0 = np.arange(1, len(subsets)-1)
    np.random.shuffle(t0)

    with multiprocessing.Pool() as pool:
        to_remove = pool.starmap(process_subset, [(xi, subsets, rot, noverlap, dmax, s.dist) for xi in t0])

    to_remove = [x for x in to_remove if x is not None]

    for xi in to_remove:
        rot[rot[xi,0],1] = rot[xi,1]
        rot[rot[xi,1],0] = rot[xi,0]

    #B4
    ids2 = [0]
    while ids2[-1] < len(subsets)-1:
        ids2.append(rot[ids2[-1],1])
    ids2 = np.array(ids2)
    assert ids2[-1] == len(subsets)-1
    ids2 = ids2[1:-1]

    edge_edges = list(s.edges[ids[ids2-1]])
    edge_locs = list(s.locs[ids[ids2-1]])
    edge_subsets = [subsets[x] for x in ids2]
    edge_neighbors = []

    if len(edge_subsets) > 0:
        edge_neighbors += [[s.p.edges[edge,0], len(s.ans_subsets)]]
        edge_neighbors += [[x,x+1] for x in range(len(s.ans_subsets), len(s.ans_subsets)+len(edge_subsets)-1)]
        edge_neighbors += [[len(s.ans_subsets)+len(edge_subsets)-1, s.p.edges[edge,1]]]
    else:
        edge_neighbors += [[s.p.edges[edge,0], s.p.edges[edge,1]]]

    return edge_edges, edge_locs, edge_subsets, edge_neighbors

def subsets_rna(fi_traj, fi_traj_cell_rna, fi_coord_rna, ncell, noverlap, dmax, edge):
    """
    Process RNA cells subsets for a single edge.

    Parameters:
    fi_traj (str): Path to the trajectory file.
    fi_traj_cell_rna (str): Path to the RNA cell locations file.
    fi_coord_rna (str): Path to the RNA cell coordinates file.
    ncell (int): Number of cells in each subset.
    noverlap (int): Minimum number of overlapping cells required.
    dmax (float): Maximum allowed distance between subsets.
    edge (int): Index of the edge to process.

    Returns:
    tuple: Contains edge_edges, edge_locs, edge_subsets, and edge_neighbors.
    """
    # Load data
    traj = trajectory.from_file(fi_traj)
    points = point.from_file(traj, fi_traj_cell_rna)
    logging.info(f'Reading file {fi_coord_rna}')
    names_cell = np.array(list(pd.read_csv(fi_coord_rna, header=0, index_col=0, sep='\t').index))

    # Step A: Initialize all subsets as node subsets
    points.perturb()
    distn = points.dist.T
    dist = points - points
    disto, distno = [np.argsort(x, axis=1) for x in [dist, distn]]
    points.ans_subsets = list(distno[:, :ncell])
    points.disto = disto

    # Process the single edge
    edge_edges, edge_locs, edge_subsets, edge_neighbors = process_edge(edge, points, ncell, noverlap, dmax)

    # Convert cell indices to cell names
    edge_subsets = [names_cell[subset].tolist() for subset in edge_subsets]

    # Validation steps
    n = len(edge_subsets)
    assert len(edge_neighbors) == n - 1 and len(edge_edges) == n and len(edge_locs) == n
    edge_edges = np.array(edge_edges).astype('u2')
    edge_locs = np.array(edge_locs)
    edge_subsets = np.array(edge_subsets)
    
    t1 = np.zeros((n, n), dtype=bool)
    for i, j in edge_neighbors:
        t1[i, j] = True
        t1[j, i] = True
    edge_neighbors = t1

    assert (edge_neighbors >= 0).all() and (edge_neighbors < n).all()

    return edge_edges, edge_locs, edge_subsets, edge_neighbors


if __name__ == "__main__":
    # Command-line arguments:
    # 1: edge index
    # 2: path to trajectory file
    # 3: path to RNA cell locations file
    # 4: path to RNA cell coordinates file
    # 5: number of cells in each subset
    # 6: minimum number of overlapping cells required
    # 7: maximum allowed distance between subsets
    # 8: path to output file for edge results

    edge = int(sys.argv[1])
    fi_traj = sys.argv[2]
    fi_traj_cell_rna = sys.argv[3]
    fi_coord_rna = sys.argv[4]
    ncell = int(sys.argv[5])
    noverlap = int(sys.argv[6])
    dmax = float(sys.argv[7])
    output_file = sys.argv[8]
    NUM_CORES = int(os.environ.get('SLURM_NTASKS_PER_NODE', os.cpu_count()))
    logging.info(f"Processing edge {edge} using {NUM_CORES} cores")
    edge_edges, edge_locs, edge_subsets, edge_neighbors = subsets_rna(
        fi_traj, fi_traj_cell_rna, fi_coord_rna, ncell, noverlap, dmax, edge
    )

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('edges', data=edge_edges)
        f.create_dataset('locs', data=edge_locs)
        f.create_dataset('subsets', data=edge_subsets)
        f.create_dataset('neighbors', data=edge_neighbors)