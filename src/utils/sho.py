import random
import numpy as np
import pandas as pd
from ast import literal_eval
from src.utils.env import get_layout
from src.utils.paths import PARTICIPANT_DIR

def get_participant_data(location, dataset, pid):
    pno         = str(pid).zfill(2) # 01, 02...
    file        = PARTICIPANT_DIR / dataset / f"P{pno}.xlsx"
    df          = pd.read_excel(file, index_col = 0)
    layout, _   = get_layout(location)
    t, player   = get_player(df, layout)
    objects     = get_objects(df)
    return t, player, objects, layout

def get_player(df, layout):
    #time
    pt = list(df.t)
    t = np.array(pt)

    #postion and room history
    px, py, pz = [], [], []
    for temp in list(df.self_pos):
        temp = temp.replace("(","")
        temp = temp.replace(")","")
        temp = temp.replace(",","")
        x = float(temp.split(" ")[0])
        y = float(temp.split(" ")[1])
        z = float(temp.split(" ")[2])
        px.append(x)
        py.append(y)
        pz.append(z)

    # velocity
    dt = 0.5
    vx, vy, vz = [0.0], [0.0], [0.0]
    vx.append((px[1] - px[0]) / dt)  #(O^1 backward)
    vy.append((py[1] - py[0]) / dt)  #(O^1 backward)
    vz.append((pz[1] - pz[0]) / dt)  #(O^1 backward)
    for i in range(2, len(pt)):
        px0, py0, pz0 = px[i], py[i], pz[i]
        px1, py1, pz1 = px[i-1], py[i-1], pz[i-1]
        px2, py2, pz2 = px[i-2], py[i-2], pz[i-2]
        vx.append((3*px0 - 4*px1 + px2) / (2*dt) ) #(O^2 backward)
        vy.append((3*py0 - 4*py1 + py2) / (2*dt) ) #(O^2 backward)
        vz.append((3*pz0 - 4*pz1 + pz2) / (2*dt) ) #(O^2 backward)

    # acceleration
    ax, ay, az = [0.0], [0.0], [0.0]
    for i in range(1, 3):
        ax.append((vx[i] - vx[i-1]) / dt)  #(O^1 backward)
        ay.append((vy[i] - vy[i-1]) / dt)  #(O^1 backward)
        az.append((vz[i] - vz[i-1]) / dt)  #(O^1 backward)
    for i in range(3, len(pt)):
        px0, py0, pz0 = px[i], py[i], pz[i]
        px1, py1, pz1 = px[i-1], py[i-1], pz[i-1]
        px2, py2, pz2 = px[i-2], py[i-2], pz[i-2]
        px3, py3, pz3 = px[i-3], py[i-3], pz[i-3]
        ax.append((2*px0 - 5*px1 + 4*px2 - px3)/(dt**3)) #(O^2 backward)
        ay.append((2*py0 - 5*py1 + 4*py2 - py3)/(dt**3)) #(O^2 backward)
        az.append((2*pz0 - 5*pz1 + 4*pz2 - pz3)/(dt**3)) #(O^2 backward)

    # shots
    nShot = []
    for ix, temp in enumerate(list(df.gun)):
        temp = temp.replace("(","")
        temp = temp.replace(")","")
        temp = temp.replace(",","")
        n_shot = float(temp.split(" ")[0])
        nShot.append(n_shot)

    # assemble result
    player = dict()
    player['px'] = np.array(px, dtype='float16')
    player['py'] = np.array(py, dtype='float16')
    player['pz'] = np.array(pz, dtype='float16')
    player['vx'] = np.array(vx, dtype='float16')
    player['vy'] = np.array(vy, dtype='float16')
    player['vz'] = np.array(vz, dtype='float16')
    player['ax'] = np.array(ax, dtype='float16')
    player['ay'] = np.array(ay, dtype='float16')
    player['az'] = np.array(az, dtype='float16')
    player['ns'] = np.array(nShot, dtype='float16')

    return t, player

def get_objects(df):

    '''modified to only output npcs: unseen (nos), seen (nvs), and shot (nds)'''
    nos = []
    nvs = []
    nds = []

    npcs    = [literal_eval(xi) for xi in list(df['npcs'])]
    npcs_vis = [0 for ix in range(len(npcs[0]))]
    for ts in range(len(npcs)):
        no = []
        nv = []
        nd = []

        for i, ni in enumerate(npcs[ts]):
            if ni[-2] != 1:                                 # dead
                nd.append(tuple((ni[0], ni[1], ni[2])))
            elif ni[-1] == 1:                               # not dead, visible
                npcs_vis[i] = 1
                nv.append(tuple((ni[0], ni[1], ni[2])))
            else:                                           # not dead, not visible
                no.append(tuple((ni[0], ni[1], ni[2])))

        nos.append(no) # NPC XYZs, alive and not visible
        nvs.append(nv) # NPC XYZs, alive and visible
        nds.append(nd) # NPC XYZs, dead

    return [nos, nvs, nds]

def derive_pids(split):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    pidList     = list(range(30))
    np.random.shuffle(pidList)

    if split is not None:
        splitT       = np.clip(split, 0, 4)
        splitV      = (splitT+1) % 5
        testIx      = list(range((6*splitT), 6*(splitT+1)))
        valIx       = list(range((6*splitV), 6*(splitV+1)))
        trainIx     = [xi for xi in range(30) if (xi not in testIx) and (xi not in valIx)]
        pid_train   = [pidList[i] for i in trainIx]
        pid_val     = [pidList[i] for i in valIx]
        pid_test    = [pidList[i] for i in testIx]
    else:
        pid_train   = []
        pid_val     = []
        pid_test    = pidList.copy()

    return pid_train, pid_val, pid_test
