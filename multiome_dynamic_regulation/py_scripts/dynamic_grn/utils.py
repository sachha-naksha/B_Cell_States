import numpy as np
import pandas as pd
import h5py
import os
import itertools
from copy import deepcopy

import networkx as nx
import matplotlib as mpl
import matplotlib.patches as Patches
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import stream as st
import anndata as ad


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
    adata = st.read(file_name=pkl_path, workdir=workdir)
    return adata

def plot_main_trajectory_nodes(adata, n_components=2, comp1=0, comp2=1, fig_size=(8,6), 
                             save_path=None):
    """
    Plot and label the main trajectory nodes (S0, S1, S2, S3) on the elastic principal graph
    
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix with EPG information
    n_components: int (default: 2)
        Number of dimensions to plot
    comp1, comp2: int (default: 0, 1)
        Components to plot on x and y axes
    fig_size: tuple (default: (8,6))
        Size of the figure
    save_path: str (default: None)
        Path to save the figure. If None, saves in current directory
    
    Returns
    -------
    None
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

# Example usage
if __name__ == "__main__":
    # Load data
    stream_outs_path = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/stream_outs/actb1_added'
    adata = read_adata_from_pkl(os.path.join(stream_outs_path, 'stream_traj_v5.pkl'), stream_outs_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(stream_outs_path, 'trajectory_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot main trajectory nodes
    save_path = os.path.join(output_dir, 'trajectory_nodes.png')
    plot_main_trajectory_nodes(adata, save_path=save_path)