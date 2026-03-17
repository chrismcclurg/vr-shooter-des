import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.sparse.csgraph import shortest_path
from src.utils.paths import PARTICIPANT_DIR, ENV_DIR

def get_robot_data(location, dataset, pid):
    pno     = str(pid).zfill(2) # 01, 02...
    file    = PARTICIPANT_DIR / dataset / f"P{pno}.xlsx"
    df      = pd.read_excel(file, index_col = 0)
    t       = np.array(list(df.t))

    rx1, ry1, rx2, ry2 = [], [], [], []
    robs_frm = [literal_eval(xi) for xi in list(df['robot'])]
    for frm in robs_frm:
        x1, y1, _, _ = frm[0]
        x2, y2, _, _ = frm[1]
        rx1.append(x1)
        ry1.append(y1)
        rx2.append(x2)
        ry2.append(y2)

    r1, r2 = dict(), dict()
    r1['px'] = np.array(rx1, dtype='float16')
    r1['py'] = np.array(ry1, dtype='float16')
    r2['px'] = np.array(rx2, dtype='float16')
    r2['py'] = np.array(ry2, dtype='float16')
    return t, r1, r2

def get_robotNodes(location, robotNo=None):
    df = pd.read_excel(ENV_DIR / f"{location}.xlsx", index_col=1)
    df = df[df['access'] == 1]   # keep only accessible nodes

    if robotNo is not None:
        # Return the specific robot's node map
        ans = df[f'R{robotNo}'].astype(int).to_dict()
        return ans

    # robotNo = None → union of R1 and R2
    r1 = df['R1'].astype(int)
    r2 = df['R2'].astype(int)

    # Binary OR across rows
    ans = ((r1 + r2) > 0).astype(int)
    ans = ans.to_dict()
    return ans

def compute_D(A_np, lambda_=0.67, max_hops=3):
    """
    Compute a hop-based diffusion matrix D_ij = exp(-lambda * h_ij).

    Parameters
    ----------
    A_np : np.ndarray
        Binary adjacency matrix (NxN).
    lambda_ : float, default=0.67
        Decay constant per hop (1.5-hop correlation length).
    max_hops : int, default=3
        Truncates diffusion beyond this distance (sets to 0).

    Returns
    -------
    D_np : np.ndarray
        NxN diffusion matrix.
    """
    # compute all-pairs shortest path (in hops)
    hops = shortest_path(A_np, directed=False, unweighted=True)

    # apply exponential decay
    D_np = np.exp(-lambda_ * hops)

    # truncate beyond max_hops or disconnected nodes
    D_np[hops > max_hops] = 0.0
    D_np[np.isinf(hops)] = 0.0

    # ensure self-influence = 1
    np.fill_diagonal(D_np, 1.0)

    return D_np

def get_robot_snapshot(r1_hist, r2_hist, Dij, *, freq_puff=0.5, dt=0.5, nmax_puff=30):

    """
    Physically consistent:  spatial pattern normalized to [0,1],
    temporal magnitude builds linearly until saturation.
    """
    n_nodes = Dij.shape[0]
    R_local = np.zeros(n_nodes, dtype=float)

    # spatial footprint
    valid_nodes = [r for r in r1_hist if r is not None] + \
                  [r for r in r2_hist if r is not None]
    if not valid_nodes:
        return R_local

    # each robot’s presence adds one unit at its node
    for r in valid_nodes:
        R_local[r] += 1.0

    # scale globally so smoke intensity reflects activity but doesn’t explode
    total = np.sum(R_local)
    if total > 0:
        R_local /= total        # keep proportional occupancy, bounded total = 1

    # diffuse spatially
    curr_Rt = Dij @ R_local

    # temporal saturation
    elapsed_time = dt * len(valid_nodes)
    n_puffs = min(elapsed_time * freq_puff, nmax_puff)
    p_puffs = n_puffs / nmax_puff           # ∈[0,1]
    curr_Rt *= p_puffs                      # scale by saturation only once

    # clip for safety
    curr_Rt = np.clip(curr_Rt, 0.0, 1.0)
    return curr_Rt