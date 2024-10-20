#!/usr/bin/env python3
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

import pandas as pd
import numpy as np
from scipy.io import mmread


def process_column(xi, names, d):
    td = set()
    ids = []
    dups = []
    for xj in range(len(names[xi])):
        if names[xi][xj] not in td:
            td.add(names[xi][xj])
            ids.append(xj)
        else:
            dups.append(names[xi][xj])
    if len(dups) > 0:
        logging.warning(
            "Skipped duplicate occurrence of {} names: {}".format(
                "gene" if xi == 0 else "cell", ",".join(dups)
            )
        )
    d = d.swapaxes(0, xi)[ids].swapaxes(0, xi)
    names[xi] = names[xi][ids]
    assert len(names[xi]) == len(set(names[xi]))
    return names[xi], d


def main(args):
    diri = args.input_folder
    fo = args.output_file
    colid = args.column
    num_threads = args.threads

    # Read files
    d = mmread(pjoin(diri, "matrix.mtx.gz"))
    d = d.toarray()
    names = [
        pd.read_csv(pjoin(diri, x + ".tsv.gz"), header=None, index_col=None, sep="\t")
        for x in ["features", "barcodes"]
    ]
    assert names[1].shape[1] == 1
    names[0] = names[0][colid].values
    names[1] = names[1][0].values
    assert d.shape == tuple(len(x) for x in names)

    #List of genes to remove
    genes_to_remove = {"TBCE", "LINC01238", "CYB561D2", "MATR3", "LINC01505", 
                       "HSPA14", "GOLGA8M", "GGT1", "ARMCX5-GPRASP2", "TMSB15B"}

    # Filter out these genes from names[0] and corresponding rows in d
    gene_filter = np.array([gene not in genes_to_remove for gene in names[0]])
    names[0] = names[0][gene_filter]
    d = d[gene_filter, :]

    # Ensure that the dimensions match after filtering
    assert d.shape[0] == len(names[0]), "Filtered data dimensions do not match!"

    # Multithreaded processing for selecting unique gene names and cells
    # start and end dimensions of 'd' remains the same after ops
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_column, xi, names, d) for xi in range(2)]
        results = [future.result() for future in as_completed(futures)]

    # Collect results
    for xi, (processed_names, processed_d) in enumerate(results):
        names[xi] = processed_names
        d = processed_d

    d = pd.DataFrame(d, index=names[0], columns=names[1])

    # Remove unneeded featres (gene names with . and peaks)
    t1 = [":" not in x and "." not in x for x in d.index]

    # Output
    d = d.loc[t1]
    d = d.loc[sorted(d.index)]
    d = d[sorted(d.columns)]
    d.to_csv(fo, header=True, index=True, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts mtx.gz format expression file to tsv.gz format."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Input folder that contains matrix.mtx.gz, features.tsv.gz, and barcodes.tsv.gz.",
    )
    parser.add_argument("output_file", type=str, help="Output file in tsv.gz format")
    parser.add_argument(
        "--column",
        default=1,
        type=str,
        help="Column ID in features.tsv.gz for gene name. Starts with 0. Default: 1.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads to use for multithreading. Default: 32.",
    )

    args = parser.parse_args()
    main(args)
