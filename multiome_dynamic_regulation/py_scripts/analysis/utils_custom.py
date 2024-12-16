import numpy as np
import matplotlib.pyplot as plt
import os
from dictys.net import dynamic_network

######################### data loading functions #########################

def load_data(data_file):
    """
    Load the dynamic network data as dictys object and define trajectory branches.
    """
    d0 = dynamic_network.from_file(data_file)
    return d0

def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0] 
    tf_indices = []  
    gene_indices = []  
    for gene in tf_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]  # Get the index in gene_hashmap
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            match = np.where(tf_mappings_to_gene_hashmap == gene_index)[0]
            if match.size > 0:  # If a match is found
                tf_indices.append(int(match[0]))  # Append the position of the match
                gene_indices.append(int(gene_index))  # Also append the gene index 
    return tf_indices, gene_indices

def get_pseudotime_of_windows(dictys_dynamic_object, window_indices):
    """
    Get the pseudotime of specific windows for x-axis in plots
    """
    pseudotime_relative_to_bifurcation = dictys_dynamic_object.point['s'].locs  # Access via dictionary keys
    branch_pseudotime = [float(pseudotime_relative_to_bifurcation[idx]) for idx in window_indices]
    return branch_pseudotime

def get_weights_across_windows(dictys_dynamic_object, tf_indices, window_indices):
    """
    Get the non-zero weights of TFs over specific windows for x-axis in plots
    """
    # Get the actual size of the second dimension
    array_shape = dictys_dynamic_object.prop['es']['w_n'].shape
    n_targets = array_shape[1]
    
    # Access the 3-D array of weights using the correct dimension
    weights_of_tf_target = dictys_dynamic_object.prop['es']['w_n'][np.ix_(tf_indices, range(n_targets), window_indices)]
    return weights_of_tf_target

def get_indirect_weights_across_windows(dictys_dynamic_object, tf_indices, window_indices):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    array_shape = dictys_dynamic_object.prop['es']['w_in'].shape
    n_targets = array_shape[1]
    
    indirect_weights_of_tf_target = dictys_dynamic_object.prop['es']['w_in'][np.ix_(tf_indices, range(n_targets), window_indices)]
    return indirect_weights_of_tf_target

################################ plotting functions #################################

import matplotlib.colors as colors

def plot_log_cpm_tfs(dictys_dynamic_object, gene_indices, window_indices, tf_list, branch_name):
    """
    Plot the log CPM of transcription factors (TFs) as a function of pseudo-time and display the plot in the notebook.
    """
    cpm_values = dictys_dynamic_object.prop['ns']['cpm'][np.ix_(gene_indices, window_indices)]
    log_cpm_values = np.log(cpm_values)
    branch_pseudotime = get_pseudotime_of_windows(dictys_dynamic_object, window_indices)
    plt.figure(figsize=(10, 6))
    for i, tf_name in enumerate(tf_list):
        plt.plot(branch_pseudotime, log_cpm_values[i, :], label=tf_name)  # i corresponds to each TF
    plt.xlabel('Pseudotime')
    plt.ylabel('log(CPM) of TFs')
    plt.title(f'log(CPM) of TFs for {branch_name} Branch')
    plt.legend(title='Transcription Factors')
    plt.show()  

def calculate_and_plot_expression_gradients(dictys_dynamic_object, gene_indices, window_indices, tf_list, branch_name):
    """
    Calculate and plot the gradients (rate of change) of log(CPM) values for transcription factors.
    
    Parameters:
    -----------
    dictys_dynamic_object : dictys object
        The dynamic network object containing expression data
    gene_indices : list
        Indices of genes to analyze
    window_indices : list
        Indices of windows to analyze
    tf_list : list
        List of TF names
    branch_name : str
        Name of the branch for plotting
        
    Returns:
    --------
    dict
        Dictionary mapping TF names to their corresponding gradients across pseudotime
    """
    # Get expression values and pseudotime
    cpm_values = dictys_dynamic_object.prop['ns']['cpm'][np.ix_(gene_indices, window_indices)]
    log_cpm_values = np.log(cpm_values)
    branch_pseudotime = get_pseudotime_of_windows(dictys_dynamic_object, window_indices)
    
    # Calculate gradients for each TF
    gradients_dict = {}
    
    plt.figure(figsize=(10, 6))
    
    for i, tf_name in enumerate(tf_list):
        # Calculate gradients using numpy
        gradients = np.gradient(log_cpm_values[i, :], branch_pseudotime)
        gradients_dict[tf_name] = gradients
        
        # Plot gradients
        plt.plot(branch_pseudotime, gradients, label=tf_name)
    
    # Configure plot
    plt.xlabel('Pseudotime')
    plt.ylabel('Rate of Change (d/dt log(CPM))')
    plt.title(f'Expression Rate of Change - {branch_name} Branch')
    plt.legend(title='Transcription Factors')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)  # Add zero line for reference
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # # Print summary statistics
    # print("\nGradient Summary Statistics:")
    # print("-" * 50)
    # for tf_name in tf_list:
    #     gradients = gradients_dict[tf_name]
    #     print(f"\n{tf_name}:")
    #     print(f"  Mean gradient: {np.mean(gradients):.4f}")
    #     print(f"  Max gradient: {np.max(gradients):.4f}")
    #     print(f"  Min gradient: {np.min(gradients):.4f}")
    #     print(f"  Std deviation: {np.std(gradients):.4f}")
    
    return gradients_dict


def plot_node_degree(dictys_dynamic_object, non_zero_weights_per_tf, window_indices, tf_list, branch_name):
    """
    Plot the node degree of TFs as a function of pseudo-time.
    """
    branch_pseudotime = get_pseudotime_of_windows(dictys_dynamic_object, window_indices)
    plt.figure(figsize=(10, 6))
    for i, tf_name in enumerate(tf_list):
        plt.plot(branch_pseudotime, non_zero_weights_per_tf[i, :], label=tf_name)  # i corresponds to each TF
    plt.xlabel('Pseudotime')
    plt.ylabel('Node Degree of TFs')
    plt.title(f'Node Degree of TFs for {branch_name} Branch')
    plt.legend(title='Transcription Factors')
    plt.show()  

def plot_regulation_heatmap(weights, tf_list, branch_name, global_vmin, global_vmax):
    """
    Plot the regulation heatmap of TFs as a function of pseudo-time.
    """
    n_tfs, n_targets, n_windows = weights.shape
    weights_reshaped = weights.reshape(n_tfs * n_targets, n_windows)
    tf_target_labels = [f"{tf}-{tf_list[i]}" for tf in tf_list for i in range(n_targets)]
    # Remove self-regulation rows
    non_self_reg_indices = [i for i in range(len(tf_target_labels)) if tf_target_labels[i].split('-')[0] != tf_target_labels[i].split('-')[1]]
    weights_reshaped = weights_reshaped[non_self_reg_indices]
    tf_target_labels = [tf_target_labels[i] for i in non_self_reg_indices]
    # Create a custom colormap
    colors_list = ['red', 'white', 'blue']
    n_bins = 100  # Number of bins in the colormap
    cmap = colors.LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(weights_reshaped, aspect='auto', cmap=cmap, interpolation='nearest', vmin=global_vmin, vmax=global_vmax)
    plt.colorbar(im, label='Regulation strength')
    plt.xlabel('Pseudotime')
    plt.ylabel('TF-Target pairs')
    plt.title(f'Dynamic regulation strength - {branch_name} branch')
    plt.yticks(range(len(tf_target_labels)), tf_target_labels)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    ###### CONFIG ######
    data_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/tut_files/skin/output/dynamic.h5'
    output_folder = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/tut_files/skin/output'
    tf_list = ['IRF4', 'PRDM1', 'BACH2', 'BATF']
    pb_window_indices = list(range(30, 46)) + [2]
    abc_window_indices = [1] + list(range(4, 30)) + [0]
    gc_window_indices = list(range(46, 61)) + [3]
    
    ###### RUN ######
    dictys_dynamic_object = load_data(data_file)
    # Get the TF indices
    tf_indices, gene_indices = get_tf_indices(dictys_dynamic_object, tf_list)
    print(f"TF indices: {tf_indices}")
    print(f"Gene indices: {gene_indices}")
    # # get networkweights across PB windows
    # all_target_weights_pb = get_weights_across_windows(dictys_dynamic_object, tf_indices, pb_window_indices)
    # print(f"All target weights across PB windows: {all_target_weights_pb}")

    