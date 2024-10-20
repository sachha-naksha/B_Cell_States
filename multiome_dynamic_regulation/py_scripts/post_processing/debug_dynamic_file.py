import matplotlib.pyplot as plt
from dictys.net import dynamic_network
import numpy as np
import os

def load_data(data_file):
    """
    Load the dynamic network data as dictys object and define trajectory branches.
    Parameters:
        data_file (str): Path to the dynamic network file.
    Returns:
        dynamic_network: Loaded dynamic network data.
    """
    d0 = dynamic_network.from_file(data_file)
    return d0

def get_tf_indices(dictys_dynamic_object, tf_list):
    """
    Get the indices of transcription factors from a list, if present in ndict and nids[0].
    Parameters:
        dictys_dynamic_object: The dynamic network object containing ndict and nids data.
        tf_list (list): List of transcription factors (TFs) to check.
    Returns:
        list: A list of indices of the TFs present in nids[0].
    """
    # Access ndict and nids[0] from the dictys_dynamic_object
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]  # numpy array
    tf_indices = []  # This will store indices in tf_mappings_to_gene_hashmap
    gene_indices = []  # This will store indices in gene_hashmap
    
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

######### dynamic window specific analysis #########
def get_pseudotime_of_windows(dictys_dynamic_object, window_indices):
    """
    Get the pseudotime of specific windows for x-axis in plots
    Parameters:
        dictys_dynamic_object: The dynamic network object containing ndict and nids data.
        window_indices (list): List of indices of the windows to get the pseudotime for.
    Returns:
        list: A list of pseudotime values for the specified windows.
    """
    # Access the pseudotime values from the dictionary
    pseudotime_relative_to_bifurcation = dictys_dynamic_object.point['s'].locs  # Access via dictionary keys
    # Slice the pseudotime values based on the window_indices
    branch_pseudotime = [float(pseudotime_relative_to_bifurcation[idx]) for idx in window_indices]
    return branch_pseudotime

########## Plotting functions ##########
def plot_log_cpm_tfs(dictys_dynamic_object, gene_indices, window_indices, tf_list, branch_name, output_folder, figname):
    """
    Plot and save the log CPM of transcription factors (TFs) as a function of pseudo-time.
    Parameters:
        dictys_dynamic_object: The dynamic network object containing prop as GRNs.
        gene_indices (list): List of indices of the TFs in the gene hashmap.
        window_indices (list): List of indices of the windows to plot.
        tf_list (list): List of TF names to map to the lines for the plot.
        branch_name (str): The name of the branch to be included in the plot title.
        output_folder (str): Folder to save the output figure.
        figname (str): Name of the figure file to save.
    """
    # Access the cpm values from prop.ns['cpm']
    cpm_values = dictys_dynamic_object.prop['ns']['cpm'][np.ix_(gene_indices, window_indices)]
    # Log transform the cpm values
    log_cpm_values = np.log(cpm_values)
    # Get the branch pseudotime for the window indices
    branch_pseudotime = get_pseudotime_of_windows(dictys_dynamic_object, window_indices)
    # Plot the log cpm values with respect to branch pseudotime
    plt.figure(figsize=(10, 6))
    # Plot each TF's log(CPM) line
    for i, tf_name in enumerate(tf_list):
        plt.plot(branch_pseudotime, log_cpm_values[i, :], label=tf_name)  # i corresponds to each TF
    plt.xlabel('Pseudotime')
    plt.ylabel('log(CPM) of TFs')
    # Update the plot title with the branch name
    plt.title(f'log(CPM) of TFs for {branch_name} Branch')
    plt.legend(title='Transcription Factors')
    # Save the figure to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, figname))
    plt.close()


def plot_direct_node_degree(dictys_dynamic_object, tf_indices, window_indices, tf_list, branch_name, output_folder, figname):
    """
    Plot and save the node degree of direct effects of transcription factors (TFs) as a function of pseudo-time.
    Parameters:
        dictys_dynamic_object: The dynamic network object containing prop as GRNs.
        tf_indices (list): List of indices of the transcription factors (TFs) to plot.
        window_indices (list): List of indices of the windows to plot.
        tf_list (list): List of TF names to be used in the legend.
        branch_name (str): The name of the branch to be included in the plot title.
        output_folder (str): Folder to save the output figure.
        figname (str): Name of the figure file to save.
    """
    # Access the direct boolean mask from prop.mask_n
    direct_mask = dictys_dynamic_object.prop['es']['mask_n'][np.ix_(tf_indices, range(15767), window_indices)]
    # Sum over genes axis (axis=1) to get the target counts per TF per window
    direct_node_degree_values = np.sum(direct_mask, axis=1)  # Sum over the genes axis (axis=1)
    # Convert to log target counts
    log_target_counts = np.log(direct_node_degree_values)
    # Get the branch pseudotime for the window indices
    branch_pseudotime = get_pseudotime_of_windows(dictys_dynamic_object, window_indices)
    # Plot the direct node degree values wrt branch pseudotime
    plt.figure(figsize=(10, 6))
    # Plot each TF's direct node degree and add the corresponding TF name to the legend
    for i, tf_name in enumerate(tf_list):
        plt.plot(branch_pseudotime, log_target_counts[i], label=tf_name)  # i corresponds to each TF
    plt.xlabel('Pseudotime')
    plt.ylabel('log(Direct Node Degree)')
    # Update the plot title with the branch name
    plt.title(f'Direct effects node-degree for {branch_name} Branch')
    plt.legend(title='Transcription Factors')
    # Save the figure to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, figname))
    plt.close()


def plot_indirect_node_degree(dictys_dynamic_object, tf_indices, window_indices, tf_list, branch_name, output_folder, figname):
    """
    Plot and save the node degree of indirect effects of transcription factors (TFs) as a function of pseudo-time.
    Parameters:
        dictys_dynamic_object: The dynamic network object containing prop as GRNs.
        tf_indices (list): List of indices of the transcription factors (TFs) to plot.
        window_indices (list): List of indices of the windows to plot.
        tf_list (list): List of TF names to be used in the legend.
        branch_name (str): The name of the branch to be included in the plot title.
        output_folder (str): Folder to save the output figure.
        figname (str): Name of the figure file to save.
    """
    # Access the indirect boolean mask from prop.mask_in and calculate node degree
    indirect_node_degree_values = np.sum(dictys_dynamic_object.prop['es']['mask_in'][np.ix_(tf_indices, range(15767), window_indices)], axis=1)
    # Convert to log target counts
    log_target_counts = np.log(indirect_node_degree_values)
    # Get the branch pseudotime for the window indices
    branch_pseudotime = get_pseudotime_of_windows(dictys_dynamic_object, window_indices)
    # Plot the indirect node degree values wrt branch pseudotime
    plt.figure(figsize=(10, 6))
    # Plot each TF's indirect node degree and add the corresponding TF name to the legend
    for i, tf_name in enumerate(tf_list):
        plt.plot(branch_pseudotime, log_target_counts[i], label=tf_name)  # i corresponds to each TF
    plt.xlabel('Pseudotime')
    plt.ylabel('log(Indirect Node Degree)')
    plt.title(f'Indirect effects node-degree for {branch_name} Branch')
    plt.legend(title='Transcription Factors')
    # Save the figure to the specified output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, figname))
    plt.close()


if __name__ == "__main__":
    # CONFIGURATION
    data_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/output/dynamic.h5'
    output_folder = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/output'
    tf_list = ['IRF4', 'IRF8', 'PRDM1', 'BCL6']
    pb_window_indices = list(range(4, 19)) + [1] # 16 windows
    abc_window_indices = [0] + list(range(19, 67)) + [2] # 50 windows
    gc_window_indices = list(range(67, 93)) + [3] # 25 windows
    
    # ANALYSIS
    d0 = load_data(data_file)
    # Get the TF indices
    tf_indices, gene_indices = get_tf_indices(d0, tf_list)
    # Get the direct node degree and plot
    plot_direct_node_degree(d0, tf_indices, pb_window_indices, tf_list, 'PB', output_folder, 'direct_node_degree_pb.png')
    # plot_direct_node_degree(d0, tf_indices, abc_window_indices, tf_list, 'ABC', output_folder, 'direct_node_degree_abc.png')
    # plot_direct_node_degree(d0, tf_indices, gc_window_indices, tf_list, 'GC', output_folder, 'direct_node_degree_gc.png')
    # # Get the indirect node degree and plot
    # plot_indirect_node_degree(d0, tf_indices, pb_window_indices, tf_list, 'PB', output_folder, 'indirect_node_degree_pb.png')
    # plot_indirect_node_degree(d0, tf_indices, abc_window_indices, tf_list, 'ABC', output_folder, 'indirect_node_degree_abc.png')
    # plot_indirect_node_degree(d0, tf_indices, gc_window_indices, tf_list, 'GC', output_folder, 'indirect_node_degree_gc.png')
    