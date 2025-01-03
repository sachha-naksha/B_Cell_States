import os
from typing import Optional, Tuple, Union

import dictys
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dictys.net import stat
from dictys.plot.dynamic import _compute_chars_

######################### Data retrieval #########################

def load_dynamic_object(dynamic_object_path):
    """
    Load the dynamic object from the given path
    """
    dynamic_object = dictys.net.dynamic_network.from_file(dynamic_object_path)
    return dynamic_object


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
    pseudotime_relative_to_bifurcation = dictys_dynamic_object.point[
        "s"
    ].locs  # Access via dictionary keys
    branch_pseudotime = [
        float(pseudotime_relative_to_bifurcation[idx]) for idx in window_indices
    ]
    return branch_pseudotime


def get_grn_weights_across_windows(dictys_dynamic_object, tf_indices, window_indices):
    """
    Get the weights of TFs over specific windows for x-axis in plots
    """
    # Get the actual size of the second dimension
    array_shape = dictys_dynamic_object.prop["es"]["w_n"].shape
    n_targets = array_shape[1]

    # Access the 3-D array of weights using the correct dimension
    grn_weights_of_tfs = dictys_dynamic_object.prop["es"]["w_n"][
        np.ix_(tf_indices, range(n_targets), window_indices)
    ]
    return grn_weights_of_tfs


def get_weights_for_tf_target_pairs(
    dictys_dynamic_object, tf_indices, target_indices, window_indices
):
    """
    Get the weights for specific TF-target pairs over specific windows.
    """
    # Access the 3-D array of weights using the correct dimensions
    weights_of_tf_target = dictys_dynamic_object.prop["es"]["w_n"][
        np.ix_(tf_indices, target_indices, window_indices)
    ]
    return weights_of_tf_target


def get_indirect_weights_across_windows(
    dictys_dynamic_object, tf_indices, window_indices
):
    """
    Get the indirect weights of TFs over specific windows for x-axis in plots
    """
    array_shape = dictys_dynamic_object.prop["es"]["w_in"].shape
    n_targets = array_shape[1]

    indirect_weights_of_tf_target = dictys_dynamic_object.prop["es"]["w_in"][
        np.ix_(tf_indices, range(n_targets), window_indices)
    ]
    return indirect_weights_of_tf_target


##################################### Utils ############################################


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


def fig_regulation_heatmap(
    network: dictys.net.dynamic_network,
    start: int,
    stop: int,
    regulations: list[Tuple[str, str]],
    num: int = 100,
    dist: float = 1.5,
    ax: Optional[matplotlib.axes.Axes] = None,
    cmap: Union[str, matplotlib.cm.ScalarMappable] = "coolwarm",
    figsize: Tuple[float, float] = (2, 0.22),
    vmax: Optional[float] = None,
) -> Tuple[
    matplotlib.pyplot.Figure, matplotlib.axes.Axes, matplotlib.cm.ScalarMappable
]:
    """
    Draws pseudo-time dependent heatmap of individual regulation strengths.
    """
    # Get dynamic network edge strength
    # sample equispaced points between start and stop nodes
    pts, fsmooth = network.linspace(start, stop, num, dist)
    stat1_net = fsmooth(stat.net(network))
    # Get pseudotime (pts mapped to start node index so that pseudotime 0 is the start node)
    stat1_x = stat.pseudotime(network, pts)
    tmp=stat1_x.compute(pts)[0]
    dx = pd.Series(tmp)
    # Test regulation existence
    tdict = [dict(zip(x, range(len(x)))) for x in stat1_net.names]
    t1 = [[x[y] for x in regulations if x[y] not in tdict[y]] for y in range(2)]
    if len(t1[0]) > 0:
        raise ValueError(
            "Regulator gene(s) {} not found in network.".format("/".join(t1[0]))
        )
    if len(t1[1]) > 0:
        raise ValueError(
            "Target gene(s) {} not found in network.".format("/".join(t1[1]))
        )
    # Extract regulations to draw
    dnet = stat1_net.compute(pts)
    t1 = np.array([[tdict[0][x[0]], tdict[1][x[1]]] for x in regulations]).T
    dnet = dnet[t1[0], t1[1]]
    # Create figure and axes
    if ax is None:
        figsize = (figsize[0], figsize[1] * dnet.shape[0])
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if figsize is not None:
            raise ValueError("figsize should not be set if ax is set.")
        fig = ax.get_figure()
        figsize = fig.get_size_inches()
    aspect = (figsize[1] / dnet.shape[0]) / (figsize[0] / dnet.shape[1])
    # Determine colormap
    if isinstance(cmap, str):
        if vmax is None:
            vmax = np.quantile(np.abs(dnet).ravel(), 0.95)
        cmap = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), cmap=cmap
        )  # pylint: disable=E1130
    elif vmax is not None:
        raise ValueError(
            "vmax should not be set if cmap is a matplotlib.cm.ScalarMappable."
        )
    if hasattr(cmap, "to_rgba"):
        # Original behavior for dictys default matplotlib colormaps
        ax.imshow(cmap.to_rgba(dnet), aspect=aspect, interpolation="none")
    else:
        # New behavior for customcolormaps
        im = ax.imshow(dnet, aspect=aspect, interpolation="none", cmap=cmap)
        plt.colorbar(
            im, label="Regulation strength"
        )  # Add colorbar for custom colormaps
    # set pseudotime labels
    ax.set_xlabel("Pseudotime")
    num_ticks = 10
    tick_positions = np.linspace(0, dnet.shape[1] - 1, num_ticks, dtype=int)
    tick_labels = dx.iloc[tick_positions]
    # print('dx values: ', tick_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{x:.6f}" for x in tick_labels], rotation=45, ha="right")
    ax.set_yticks(list(range(len(regulations))))
    ax.set_yticklabels(["-".join(x) for x in regulations])
    return fig, ax, cmap


def compute_chars(
    self,
    start: int,
    stop: int,
    num: int = 100,
    dist: float = 1.5,
    mode: str = "regulation",
    sparsity: float = 0.01,
) -> pd.DataFrame:
    """
    Compute curve characteristics for one branch.
    """
    pts, fsmooth = self.linspace(start, stop, num, dist)
    tmp_pt_0 = pts[[1]]
    if mode == "regulation":
        # Log number of targets
        stat1_net = fsmooth(stat.net(self))
        stat1_netbin = stat.fbinarize(stat1_net, sparsity=sparsity)
        stat1_y = stat.flnneighbor(stat1_netbin)
    elif mode == "expression":
        stat1_y = fsmooth(stat.lcpm(self, cut=0))
    else:
        raise ValueError(f"Unknown mode {mode}.")
    # Pseudo time
    stat1_x = stat.pseudotime(self, pts)
    tmp_y = stat1_y.compute(pts)
    tmp_x = stat1_x.compute(pts)
    dy = pd.DataFrame(tmp_y, index=stat1_y.names[0])
    dx = pd.Series(tmp_x[0]) #first gene's pseudotime is taken as all genes have the same pseudotime
    return dy, dx


##################################### Combinatorial control ############################################


def create_tf_target_pairs(dictys_dynamic_object, tf_pairs_df):
    """
    Create a list of TF-Target tuples from combinatorial control data,
    checking presence of both TFs and targets in the dynamic object.
    """
    tf_target_pairs = []

    for _, row in tf_pairs_df.iterrows():
        # Get TF pair and targets
        tf1, tf2 = eval(row["TF"])
        common_targets = eval(row["common"])

        # Check if TFs are present in dynamic object
        present_tfs = check_tf_presence(dictys_dynamic_object, [tf1, tf2])

        # Check if targets are present in dynamic object
        present_targets = check_gene_presence(dictys_dynamic_object, common_targets)

        # Create pairs for present TFs and targets
        for tf in present_tfs:
            for target in present_targets:
                tf_target_pairs.append((tf, target))

    # Remove any duplicates
    tf_target_pairs = list(set(tf_target_pairs))

    print(f"Created {len(tf_target_pairs)} unique TF-target pairs")
    return tf_target_pairs


if __name__ == "__main__":
    dynamic_object_path = "/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added/output/dynamic.h5"
    dynamic_object = load_dynamic_object(dynamic_object_path)
    print("data loaded")
    # #plot the regulation heatmap for the PB branch
    # fig_pb, ax_pb, cmap_pb = fig_regulation_heatmap(
    #     network=dynamic_object,
    #     start=1,  # start node
    #     stop=2,  # end node
    #     regulations=[('BACH2', 'JCHAIN'), ('BACH2', 'MZB1')],
    #     num=100,
    #     dist=0.0005,
    #     ax=None,
    #     cmap='coolwarm'
    # )
    dy, dx = compute_chars(dynamic_object,
    start=1,          # Starting node ID
    stop=2,           # Ending node ID
    num=50,          # Number of points to interpolate
    dist=0.0005,      # dist between sampled points
    mode='expression' # Use expression (CPM) mode
    )
    print(dy)
    print(dx)
