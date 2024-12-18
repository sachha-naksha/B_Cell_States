import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

######################### Data loading functions #########################

def check_tf_presence(dictys_dynamic_object, tf_list):
    """
    Check if the TFs are present in the dynamic network object.
    """
    gene_hashmap = dictys_dynamic_object.ndict
    tf_mappings_to_gene_hashmap = dictys_dynamic_object.nids[0]
    
    tfs_present_in_dynamic_object = []
    for tf in tf_list:
        # Check if the TF is in the gene_hashmap
        if tf in gene_hashmap:
            gene_index = gene_hashmap[tf]
            # Check if the gene index is present in tf_mappings_to_gene_hashmap
            if np.any(tf_mappings_to_gene_hashmap == gene_index):
                tfs_present_in_dynamic_object.append(tf)
    
    return tfs_present_in_dynamic_object
    
def check_gene_presence(dictys_dynamic_object, gene_list):
    """
    Check if the genes are present as targets in the dynamic network object.
    """
    gene_hashmap = dictys_dynamic_object.ndict
    target_mappings_to_hashmap = dictys_dynamic_object.nids[1]
    
    target_genes_present_in_dynamic_object = []
    for gene in gene_list:
        # Check if the gene is in the gene_hashmap
        if gene in gene_hashmap:
            gene_index = gene_hashmap[gene]
            # Check if the gene index is present in target_mappings_to_hashmap
            if np.any(target_mappings_to_hashmap == gene_index):
                target_genes_present_in_dynamic_object.append(gene)
    
    return target_genes_present_in_dynamic_object

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

def get_target_gene_indices(dictys_dynamic_object, target_list):
    """
    Get the indices of target genes from a list, if present in ndict and nids[1].
    """
    gene_hashmap = dictys_dynamic_object.ndict
    target_indices = []
    for gene in target_list:
        if gene in gene_hashmap:
            target_indices.append(gene_hashmap[gene])
    return target_indices

def get_pseudotime_of_windows(dictys_dynamic_object, window_indices):
    """
    Get the pseudotime of specific windows for x-axis in plots
    """
    pseudotime_relative_to_bifurcation = dictys_dynamic_object.point['s'].locs  # Access via dictionary keys
    branch_pseudotime = [float(pseudotime_relative_to_bifurcation[idx]) for idx in window_indices]
    return branch_pseudotime

def get_grn_weights_across_windows(dictys_dynamic_object, tf_indices, window_indices):
    """
    Get the weights of TFs over specific windows for x-axis in plots
    """
    # Get the actual size of the second dimension
    array_shape = dictys_dynamic_object.prop['es']['w_n'].shape
    n_targets = array_shape[1]
    
    # Access the 3-D array of weights using the correct dimension
    grn_weights_of_tfs = dictys_dynamic_object.prop['es']['w_n'][np.ix_(tf_indices, range(n_targets), window_indices)]
    return grn_weights_of_tfs

def get_weights_for_tf_target_pairs(dictys_dynamic_object, tf_indices, target_indices, window_indices):
    """
    Get the weights for specific TF-target pairs over specific windows.
    """
    # Access the 3-D array of weights using the correct dimensions
    weights_of_tf_target = dictys_dynamic_object.prop['es']['w_n'][np.ix_(tf_indices, target_indices, window_indices)]
    return weights_of_tf_target

def get_indirect_weights_across_windows(dictys_dynamic_object, tf_indices, window_indices):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    array_shape = dictys_dynamic_object.prop['es']['w_in'].shape
    n_targets = array_shape[1]
    
    indirect_weights_of_tf_target = dictys_dynamic_object.prop['es']['w_in'][np.ix_(tf_indices, range(n_targets), window_indices)]
    return indirect_weights_of_tf_target

##################################### Utils ############################################

################################ Plotting functions ####################################

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

    