import numpy as np
import pandas as pd
import os
import joblib
import pickle
import math
import ast

class Config:
    """Configuration for B and T Cells analysis paths"""
    
    # Base paths
    _BCELL_BASE = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/multiome_1st_donor_UPMC_aggr/dictys_outs/actb1_added_v2'
    _TCELL_BASE = '/ocean/projects/cis240075p/asachan/datasets/B_Cell/T_cell/outs/dictys/rbpj_ntc'
    OUTPUT_FOLDER = f'{_BCELL_BASE}/output/figures'
    INPUT_FOLDER = f'{_BCELL_BASE}/output/intermediate_tmp_files'
    CELL_LABELS = f'{_BCELL_BASE}/data/clusters.csv'
    
    # State discriminative LFs
    _ENRICHMENT_BASE = f'{INPUT_FOLDER}/direct_effect_enrichment'
    
    PB = {
        'ep1': f'{_ENRICHMENT_BASE}/enrichment_ep1_pb.csv',
        'ep2': f'{_ENRICHMENT_BASE}/enrichment_ep2_pb.csv',
        'ep3': f'{_ENRICHMENT_BASE}/enrichment_ep3_pb.csv',
        'ep4': f'{_ENRICHMENT_BASE}/enrichment_ep4_pb.csv',
    }
    
    GC = {
        'ep1': f'{_ENRICHMENT_BASE}/enrichment_ep1_gc.csv',
        'ep2': f'{_ENRICHMENT_BASE}/enrichment_ep2_gc.csv',
        'ep3': f'{_ENRICHMENT_BASE}/enrichment_ep3_gc.csv',
        'ep4': f'{_ENRICHMENT_BASE}/enrichment_ep4_gc.csv',
    }
    
    # TF KO paths
    
    IRF4 = {f'ep{i}': f'{INPUT_FOLDER}/irf4_ko/gc_98/enrichment_episode_{i}.csv' 
            for i in range(1, 9)}
    
    BLIMP1 = {f'ep{i}': f'{INPUT_FOLDER}/prdm1_ko/gc_98/enrichment_episode_{i}.csv' 
              for i in range(1, 9)}
    
    ETS1_SIG = {f'ep{i}': f'{_TCELL_BASE}/output/ets1/sig_LFs/enrichment_episode_{i}.csv' 
                for i in range(1, 7)}
    
    ETS1_ALL = {f'ep{i}': f'{_TCELL_BASE}/output/ets1/all_lfs/enrichment_episode_{i}.csv' 
                for i in range(1, 7)}
    
    IKZF1_SIG = {f'ep{i}': f'{_TCELL_BASE}/output/ikzf1/sig_lfs/enrichment_episode_{i}.csv' 
                 for i in range(1, 7)}
    
    IKZF1_ALL = {f'ep{i}': f'{_TCELL_BASE}/output/ikzf1/all_lfs/enrichment_episode_{i}.csv' 
                 for i in range(1, 7)}
