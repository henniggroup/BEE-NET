import torch
import torch.nn.functional as F
import torch_geometric as tg

# crystal structure data
from ase import Atom

import numpy as np
from tqdm import tqdm

import pandas as pd

from ase.neighborlist import neighbor_list

from scipy.signal import savgol_filter

from utils.model import PeriodicNetwork, PeriodicNetworkPhdos

bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


def build_data(
    entry, r_max=4.0, embed_ph_dos=True, embed_e_dos=True, fine=True, avg=False
):
    """
    Construct a data object suitable for graph-based models using PyTorch Geometric.

    Parameters:
    - entry (object): Input object containing structure information about a molecule or crystal.
    - r_max (float, optional): Cutoff radius for neighbor list calculation. Default is 4.0.
    - embed_ph_dos (bool, optional): Flag to indicate if phonon density of states (phDOS) data should be embedded. Default is True.
    - embed_e_dos (bool, optional): Flag to indicate if electronic density of states (eDOS) data should be embedded. Default is True.
    - fine (bool, optional): Flag to determine the granularity of phonon density of states. Default is True.
    - avg (bool, optional): Flag to determine whether to average some properties. Default is False.

    Returns:
    - tg.data.Data: A PyTorch Geometric data object containing various attributes like node features, edge indices, etc., which represent the molecular/crystalline structure.

    Notes:
    - This function assumes certain global variables and helper functions like `neighbor_list`, `type_encoding`, `am_onehot`, `process_edos`, etc. are available in the environment.
    - It is recommended to ensure all dependencies are imported and necessary global variables are initialized before invoking this function.
    """
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=entry.structure, cutoff=r_max, self_interaction=True
    )

    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[
        torch.from_numpy(edge_src)
    ]
    edge_vec = (
        positions[torch.from_numpy(edge_dst)]
        - positions[torch.from_numpy(edge_src)]
        + torch.einsum(
            "ni,nij->nj",
            torch.tensor(edge_shift, dtype=default_dtype),
            lattice[edge_batch],
        )
    )

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)

    x = am_onehot[[type_encoding[specie] for i, specie in enumerate(symbols)]]
    z = type_onehot[
        [type_encoding[specie] for specie in symbols]
    ]  # Not updated at each convolution

    if embed_ph_dos:
        p_ph_dos = process_phdos(entry, fine=fine, avg=avg)

        x = torch.cat((x, torch.ones_like(p_ph_dos)), 1)
        z = torch.cat((z, p_ph_dos), 1)

    data = tg.data.Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x=x,
        z=z,  # atom type (node attribute)
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        ),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec,
        edge_len=edge_len,
        target=torch.from_numpy(np.asarray(entry.target)).unsqueeze(0),
    )
    return data


def get_target(df):
    x = df.Freq_meV
    y = df.a2F
    xl = np.arange(0.25, 101, 0.1)
    y = np.interp(xl, x, y)
    Y = savgol_filter(y, 101, 3, mode="interp")
    Y = np.interp(Freq_final, xl, Y)
    Y = np.asarray([y if y > 0.0 else 0.0 for y in Y])
    return Y  

def process_phdos(entry, fine=True, avg=False):
    def reshape(phdos, sites):
        return np.asarray([np.asarray(phdos) / sites for _ in range(sites)])

    x = entry.freq
    ys = entry.site_phdos

    Y_proc = []
    for y in ys:
        xl = np.arange(0.25, 101, 0.1)
        y = np.interp(xl, x, y)
        Y = savgol_filter(y, 101, 3, mode="interp")
        Y = np.interp(Freq_final, xl, Y)
        Y = [y if y > 0.0 else 0.0 for y in Y]
        Y_proc.append(Y.copy())
    return torch.Tensor(Y_proc)

def get_model(init_dict, lr=0.005, wd=0.05, total=False):
    if total:
        model = PeriodicNetworkPhdos(**init_dict)
    else:
        model = PeriodicNetwork(**init_dict)
        model.pool = True
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    return model, opt, scheduler

def get_avg(df):
    pred = np.zeros(51)
    for i in folds:
        pred += df[f"pred_{i}"]
    return pred / len(folds)


def cal_lamb(freq_w, alpha_F):
    lambdaF = 0
    # try:
    for i in range(1, len(freq_w)):
        if freq_w[i] > 0:
            dw = freq_w[i] - freq_w[i - 1]
            w = freq_w[i]
            alpha_F_w = alpha_F[i]
            lambdaF = lambdaF + ((alpha_F_w / w) * dw)
    return 2 * lambdaF


def cal_w_log(freq_w, alpha_F, lamb):
    w_logF = 0
    try:
        i = 1
        for i in range(1, len(freq_w)):
            if freq_w[i] > 0:
                dw = freq_w[i] - freq_w[i - 1]
                w_logF = w_logF + (alpha_F[i] * np.log(freq_w[i]) * dw / freq_w[i])
        return np.exp(2 * w_logF / lamb) / 0.08617
    except:
        return np.nan


def cal_w_sq(freq_w, alpha_F, lamb):
    w_sqF = 0
    try:
        for i in range(1, len(freq_w)):
            if freq_w[i] > 0:
                dw = freq_w[i] - freq_w[i - 1]
                w_sqF = w_sqF + (alpha_F[i] * freq_w[i] * dw)
        return ((2 * w_sqF / lamb) ** 0.5) / 0.08617
    except:
        return np.nan


def cal_tc(lamb, omega_log, mu=0.1):
    frac = -1.04 * (1 + lamb) / (lamb - mu * (1 + 0.62 * lamb))
    return (omega_log / 1.2) * np.exp(frac)


def cal_tc_ad(lamb, wlog, w2, tc, mu=0.1):
    f1 = (1 + (lamb / (2.46 * (1 + 3.8 * mu))) ** 1.5) ** (1 / 3)
    f2 = 1 + ((lamb ** 2) * ((w2 / wlog) - 1)) / (
        lamb ** 2 + (1.82 * (1 + 6.3 * mu) * (w2 / wlog)) ** 2
    )
    return f1 * f2 * tc


def get_comp(df):
    return df.structure.composition.reduced_formula


df = pd.read_pickle("pkl_files/df_run_unguided_m3gnet_relaxed.pkl")
r_max = 4.0  # cutoff radius
mu = 0.1

Freq_final = np.arange(0.25, 101, 2)
Freq_final_E = np.arange(-50, 50, 1)

tqdm.pandas()


def get_freq(df):
    return np.arange(0.25, 101, 2)


def get_a2f(df):
    return np.ones(51)


df["Freq_meV"] = df.progress_apply(get_freq, axis=1)
df["a2F"] = df.progress_apply(get_a2f, axis=1)
# df['structure'] = df['ase_atom_obj']

df["target"] = df.apply(get_target, axis=1)

df["formula"] = df["structure"].map(lambda x: x.get_chemical_formula())
print("formula")
df["species"] = df["structure"].map(lambda x: list(set(x.get_chemical_symbols())))
species = sorted(list(set(df["species"].sum())))
print("species")
# one-hot encoding atom type and mass
type_encoding = {}
specie_am = []
for Z in tqdm(range(1, 119), bar_format=bar_format):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))

loss_fn = torch.nn.MSELoss()
loss_fn_mae = torch.nn.MSELoss()

max_iter = 500
batch_size = 800

device = "cuda:0" if torch.cuda.is_available() else "cpu"

out_dim = len(Freq_final)
em_dim = 64

init_dict_cpd = dict(
    in_dim=118 + 51,
    em_dim=em_dim,
    irreps_in=str(em_dim) + "x0e",
    irreps_out=str(out_dim) + "x0e",
    irreps_node_attr=str(em_dim) + "x0e",
    layers=2,
    mul=32,
    lmax=3,
    max_radius=r_max,
    num_neighbors=16.879432922468553,  # 17.133204568916714,
    reduce_output=True,
    p=0.0,
)
init_dict_cso = init_dict_cpd.copy()
init_dict_cso["in_dim"] = 118
init_dict_cso["num_neighbors"] = 17.133204568916714

folds = range(100)

# root_relaxed = '/blue/hennig/jasongibson/elemental_sub/materials/'
# df = df.iloc[:int(len(df)/2)]

# df['structure'] = structures_m3g
df["data"] = df.progress_apply(
    build_data, embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=r_max, axis=1
)

model, opt, scheduler = get_model(init_dict_cso, lr=0.05, wd=1e-5)
dataloader = tg.loader.DataLoader(df["data"].values, batch_size=2000)
for i in tqdm(folds):
    df[f"pred_{i}"] = np.empty((len(df), 1)).tolist()
    name = f"model_CSO_derived_EMD_0_{i}"

    run_name = f"/blue/hennig/jasongibson/a2f_v2/models/{name}"
    model.load_state_dict(torch.load(run_name + f".pt1.pt"))
    model.pool = True

    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        i0 = 0
        for z, d in enumerate(dataloader):
            d.to(device)
            output = model(d)
            loss = (
                F.mse_loss(output, d.target, reduction="none")
                .mean(dim=-1)
                .cpu()
                .numpy()
            )
            preds = preds + [k for k in output.cpu().numpy()]
            i0 += len(d.target)

    df[f"pred_{i}"] = preds


def get_lamb(df):
    return cal_lamb(Freq_final, df.pred_avg)


def get_w2(df):
    return cal_w_sq(Freq_final, df.pred_avg, df.lamb_m3g)  # /0.08617


def get_wlog(df):
    return cal_w_log(Freq_final, df.pred_avg, df.lamb_m3g)  # /0.08617


def get_tc(df):
    return cal_tc(df.lamb_m3g, df.wlog_m3g, mu)


def get_tcad(df):
    return cal_tc_ad(df.lamb_m3g, df.wlog_m3g, df.w2_m3g, df.tc_m3g, mu)


df["pred_avg"] = df.progress_apply(get_avg, axis=1)
df["pred_avg_m3g"] = df["pred_avg"]
drops = [f"pred_{i}" for i in folds]
df.drop(drops, axis=1, inplace=True)

df["lamb_m3g"] = df.apply(get_lamb, axis=1)
df["w2_m3g"] = df.apply(get_w2, axis=1)
df["wlog_m3g"] = df.apply(get_wlog, axis=1)
df["tc_m3g"] = df.apply(get_tc, axis=1)
df["tcad_m3g"] = df.apply(get_tcad, axis=1)

df.to_pickle("pkl_files/df_run_unguided_m3gnet_tc.pkl")

