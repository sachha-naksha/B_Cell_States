import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import dictys
import matplotlib
from dictys.plot import layout, panel
from utils_custom import *
from multiprocessing import Pool

# Define global constants
BRANCHES = {
    'Plasmablast': (1, 2),
    'Germinal-center': (1, 3)
}

def load_data(data_file, cell_labels_file):
    """Load dictys object and cell labels"""
    dictys_dynamic_object = dictys.net.dynamic_network.from_file(data_file)
    cell_labels_df = pd.read_csv(cell_labels_file)
    cell_labels = cell_labels_df['Cluster']
    cell_type_list = cell_labels.values.tolist() if isinstance(cell_labels, pd.Series) else list(cell_labels)
    dictys_dynamic_object.prop['c']['color'] = cell_type_list
    return dictys_dynamic_object

def create_animation_wrapper(args):
    """Wrapper function for multiprocessing"""
    dictys_dynamic_object, branchname, output_path = args
    return create_animation(dictys_dynamic_object, branchname, output_path)

def create_animation(dictys_dynamic_object, branchname, output_path, 
                    nframe=100, dpi=100, fps_factor=0.10):
    """Create and save animation"""
    fps = fps_factor * nframe
    
    # Select TFs for each row's dynamic subnetwork graph
    tfs_subnet = [['PAX5']]
    
    # Select TFs for each row's other plots
    tfs_ann = [['XBP1', 'PRDM1', 'NRF1', 'PAX5']]
    
    # Select genes to annotate as targets in all rows
    target_ann = ['RUNX2', 'MZB1', 'PRDM1', 'IRF4', 'AFF3']
    
    if branchname not in BRANCHES:
        raise ValueError(f"Branch name must be one of {list(BRANCHES.keys())}")
    
    branch = BRANCHES[branchname]
    
    # initialize layout with dist, n_points, dpi
    layout1 = layout.notch(dist=0.0005, nframe=nframe, dpi=dpi)
    
    pts, fig, panels, animate_ka = layout1.draw(
        dictys_dynamic_object, branch,
        bcde_tfs=tfs_ann,
        e_targets=target_ann,
        f_tfs=tfs_subnet,
        a_ka={'scatterka': {'legend_loc': (0.6, 1)}},
        e_ka={'lim': [-0.02, 0.02]},
    )
    
    ca = panel.animate_generic(pts, fig, panels)
    anim = ca.animate(**animate_ka)
    
    w = matplotlib.animation.writers['ffmpeg_file'](fps=fps, codec='h264')
    w.frame_format = 'jpeg'
    
    output_file = os.path.join(output_path, f'animation-10-{branchname}.mp4')
    anim.save(output_file, writer=w, dpi='figure')
    plt.close(fig)  # Clean up the figure
    return output_file

def main():
    # Define file paths
    data_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/output/dynamic.h5'
    cell_labels_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/data/clusters.csv'
    output_folder = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/output'
    
    # Load data
    dictys_dynamic_object = load_data(data_file, cell_labels_file)
    
    # Prepare arguments for multiprocessing
    args_list = [(dictys_dynamic_object, branchname, output_folder) 
                 for branchname in BRANCHES]
    
    # Create animations in parallel
    with Pool() as pool:
        output_files = pool.map(create_animation_wrapper, args_list)
    
    # Print results
    for branchname, output_file in zip(BRANCHES.keys(), output_files):
        print(f"Animation saved for {branchname} branch: {output_file}")

if __name__ == "__main__":
    matplotlib.use('Agg')  # Required for multiprocessing with matplotlib
    main()