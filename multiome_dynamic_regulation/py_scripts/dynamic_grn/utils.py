import numpy as np
import pandas as pd
import h5py
import os
import itertools
from copy import deepcopy
import math
import sys
import re
import networkx as nx
import matplotlib as mpl
import matplotlib.patches as Patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def read_h5_file(file_path):
    """
    Read HDF5 file and return its contents as a dictionary for easier debugging
    """
    data_dict = {}
    with h5py.File(file_path, 'r') as f:
        # Recursively read groups and datasets
        def read_group(group, dict_obj):
            for key in group.keys():
                item = group[key]
                if isinstance(item, h5py.Dataset):
                    # Convert dataset to numpy array for easier inspection
                    dict_obj[key] = item[()]
                elif isinstance(item, h5py.Group):
                    dict_obj[key] = {}
                    read_group(item, dict_obj[key])
        
        read_group(f, data_dict)
    
    return data_dict

def read_adata_from_pkl(pkl_path, workdir):
    """
    Read an AnnData object from a PKL file using stream
    """
    import stream as st
    adata = st.read(file_name=pkl_path, workdir=workdir)
    return adata

def plot_main_trajectory_nodes(adata, n_components=2, comp1=0, comp2=1, fig_size=(8,6), 
                             save_path=None):
    """
    Plot and label the main trajectory nodes (S0, S1, S2, S3) on the elastic principal graph
    """
    # Use 'Agg' backend for non-interactive environments
    import matplotlib
    matplotlib.use('Agg')
    
    # Create figure
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    
    # Plot all EPG edges
    epg = adata.uns['epg']
    epg_node_pos = nx.get_node_attributes(epg, 'pos')
    
    for edge_i in epg.edges():
        start_pos = epg_node_pos[edge_i[0]]
        end_pos = epg_node_pos[edge_i[1]]
        
        ax.plot([start_pos[comp1], end_pos[comp1]], 
                [start_pos[comp2], end_pos[comp2]], 
                'k-', alpha=0.3, linewidth=1)
        ax.plot([start_pos[comp1], end_pos[comp1]], 
                [start_pos[comp2], end_pos[comp2]], 
                'ko', ms=3, alpha=0.3)
    
    # Get flat tree nodes
    flat_tree = adata.uns['flat_tree']
    ft_nodes = list(flat_tree.nodes())
    main_nodes = {
        'S0': ft_nodes[0],
        'S3': ft_nodes[1],
        'S2': ft_nodes[2],
        'S1': ft_nodes[3]
    }
    
    # Plot and label main trajectory nodes
    for label, node_idx in main_nodes.items():
        pos = epg_node_pos[node_idx]
        ax.scatter(pos[comp1], pos[comp2], c='red', s=100, zorder=10)
        ax.text(pos[comp1], pos[comp2], label,
                color='red', fontsize=12, fontweight='bold',
                ha='right', va='bottom')
    
    ax.set_xlabel(f'Dim{comp1+1}')
    ax.set_ylabel(f'Dim{comp2+1}')
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        save_path = 'trajectory_nodes.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")

def parse_motifs(file_content):
    motifs = []
    current_motif = None
    lines = file_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith("MOTIF"):
            # Start new motif
            if current_motif:
                motifs.append(current_motif)
            current_motif = {
                'name': line.split()[1],  # Get motif ID after "MOTIF"
                'matrix': [],
                'w': None
            }
        elif line.startswith("letter-probability matrix"):
            if current_motif:
                # Extract width from the line
                w_match = re.search(r'w=\s*(\d+)', line)
                current_motif['w'] = int(w_match.group(1)) if w_match else None
        elif current_motif and current_motif['w'] is not None:
            # Parse probability lines (skip empty lines and URL lines)
            if line and not line.startswith("URL"):
                try:
                    probs = [float(x) for x in line.split()]
                    if len(probs) == 4:  # Check for 4 probabilities (A,C,G,T)
                        current_motif['matrix'].append(probs)
                except ValueError:
                    continue
    
    # Don't forget the last motif
    if current_motif and current_motif['matrix']:
        motifs.append(current_motif)
    
    # Debug print
    print(f"Parsed {len(motifs)} motifs")
    for i, motif in enumerate(motifs, 1):
        print(f"Motif {i}: {motif['name']}, width={motif['w']}, matrix rows={len(motif['matrix'])}")
    
    return motifs

def calculate_consensus_and_score(matrix):
    bases = ['A', 'C', 'G', 'T']
    consensus = []
    score = 0.0
    
    for row in matrix:
        max_val = max(row)
        max_idx = row.index(max_val)
        consensus.append(bases[max_idx])
        score += math.log(max_val / 0.25)
    
    return ''.join(consensus), round(score, 5)

def process_motif_file(input_file, output_file=None):
    with open(input_file, 'r') as f:
        content = f.read()
    motifs = parse_motifs(content)
    print(f"Found {len(motifs)} motifs")  # Debug print
    # Set default output file if none provided
    if output_file is None:
        output_file = input_file + '.motif'
    with open(output_file, 'w') as f:
        for i, motif in enumerate(motifs, 1):
            # Debug prints
            print(f"Processing motif {i}: {motif['name']}")
            print(f"Matrix size: {len(motif['matrix'])}x{len(motif['matrix'][0]) if motif['matrix'] else 0}")
            if not motif['matrix']:
                print(f"Skipping motif {i}: empty matrix")
                continue
            # Calculate consensus sequence and score
            consensus, score = calculate_consensus_and_score(motif['matrix'])
            print(f"Consensus: {consensus}, Score: {score}")  # Debug print
            # Write header line (HOMER format)
            header = f">{consensus}\t{motif['name']}\t{score:.5f}\t-10\n"
            f.write(header)
            print(f"Wrote header: {header.strip()}")  # Debug print 
            # Write probability matrix
            for row in motif['matrix']:
                line = "\t".join(map(str, row)) + "\n"
                f.write(line)
                print(f"Wrote matrix row: {line.strip()}")  # Debug print
            # Add blank line between motifs
            f.write("\n")
            # Ensure writing to disk
            f.flush()
    # Verify file was written
    if os.path.exists(output_file):
        print(f"File created successfully at: {output_file}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    else:
        print("Error: File was not created!")
    return output_file

if __name__ == "__main__":
    subset_locs = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2/tmp_dynamic/subset_locs.h5'
    data_dict = read_h5_file(subset_locs)
    print(data_dict)