import numpy as np
import matplotlib.pyplot as plt
import os
from dictys.net import dynamic_network

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
    # Access the 3-D array of weights from prop['es']['w_n'] and slice it using np.ix_
    weights_of_tf_target = dictys_dynamic_object.prop['es']['w_n'][np.ix_(tf_indices, range(15767), window_indices)]
    return weights_of_tf_target

def get_indirect_weights_across_windows(dictys_dynamic_object, tf_indices, window_indices):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    indirect_weights_of_tf_target = dictys_dynamic_object.prop['es']['w_in'][np.ix_(tf_indices, range(15767), window_indices)]
    return indirect_weights_of_tf_target

######### plotting functions #########
def plot_log_cpm_tfs(dictys_dynamic_object, gene_indices, window_indices, tf_list, branch_name, output_folder, fig_name):
    """
    Plot the log CPM of transcription factors (TFs) as a function of pseudo-time and save the plot.
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
    fig_path = os.path.join(output_folder, fig_name)
    plt.savefig(fig_path)
    plt.close()  # Close the figure to free up memory
    print(f"Figure saved as {fig_path}")


if __name__ == "__main__":
    ###### CONFIG ######
    data_file = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/output/dynamic.h5'
    output_folder_grns = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/output/networks'
    output_folder_plots = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/output/figures'
    tf_list = ['IRF4', 'IRF8', 'PRDM1', 'BCL6', 'BATF', 'SPIB']
    pb_window_indices = list(range(4, 19)) + [1]
    abc_window_indices = [0] + list(range(19, 67)) + [2]
    gc_window_indices = list(range(67, 93)) + [3]
    
    ###### RUN ######
    dictys_dynamic_object = load_data(data_file)
    # Get the TF indices
    tf_indices, gene_indices = get_tf_indices(dictys_dynamic_object, tf_list)
    print(f"TF indices: {tf_indices}")
    print(f"Gene indices: {gene_indices}")
    # get networkweights across PB windows
    all_target_weights_pb = get_weights_across_windows(dictys_dynamic_object, tf_indices, pb_window_indices)
    print(f"All target weights across PB windows: {all_target_weights_pb}")

    