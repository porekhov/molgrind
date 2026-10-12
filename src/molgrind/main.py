#!/usr/bin/env python3
"""
Train and apply a GNN that partitions a molecule into Martini 3 beads.

Training set format, one .npz per molecule in dataset_dir:
    adj          (N, N) heavy-atom adjacency. Any positive entry is a bond.
    x            (N, M) heavy-atom features.
    beads_json   JSON list of lists, the reference bead map. Atom indices match adj.
    mol_bin      RDKit binary of the molecule used to build adj.

Prediction writes <out_pred>.itp, _CG.gro, _CG.pdb, _AA.pdb, .svg, and .npz.
The residue name is the first four characters of --out_pred, uppercased.
SMILES input is ionized with Dimorphite-DL before mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import adjusted_rand_score, f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, LayerNorm
from torch_geometric.utils import dense_to_sparse

try:
    from . import bead_class, draw, topology
except ImportError:
    import bead_class
    import draw
    import topology

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BEAD_MODEL = PACKAGE_DIR / "martini_bead_classifiers.pkl"

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def best_f1_threshold(y_true: np.ndarray, p: np.ndarray, n_steps: int = 41):
    """
    Find threshold in [0.0, 1.0] that maximizes F1 on validation.
    Returns (best_f1, best_t).

    n_steps=41 -> thresholds 0.0,0.025,...,1.0
    """
    if y_true.size == 0:
        return 0.0, 0.5

    ts = np.linspace(0.0, 1.0, n_steps)
    best_f1, best_t = -1.0, 0.5
    y_int = y_true.astype(np.int32)

    for t in ts:
        pred = (p >= t).astype(np.int32)
        f1 = f1_score(y_int, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)

    return best_f1, best_t

# -----------------------------
# IO
# -----------------------------
def load_npz_molecule(path: str) -> Tuple[np.ndarray, np.ndarray, Chem.Mol, Optional[List[List[int]]]]:
    z = np.load(path, allow_pickle=True)
    if "adj" not in z or "x" not in z:
        raise ValueError(f"{path}: must contain keys 'adj' and 'x'.")
    adj = np.asarray(z["adj"])
    x = np.asarray(z["x"])
    mol = Chem.Mol(z["mol_bin"].tobytes())
    bead_map = None
    if "beads_json" in z:
        bead_map = json.loads(str(z["beads_json"]))
    return adj, x, mol, bead_map


def list_npz_files(dataset_dir: str) -> List[str]:
    files = []
    for fn in sorted(os.listdir(dataset_dir)):
        if fn.lower().endswith(".npz"):
            files.append(os.path.join(dataset_dir, fn))
    if not files:
        raise ValueError(f"No .npz files found in {dataset_dir}")
    return files


# -----------------------------
# Graph utilities
# -----------------------------
def adjacency_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Bond list for the GIN. Diagonal entries are dropped; GINConv adds the self term."""
    a = torch.tensor((adj > 0).astype(np.int64))
    a.fill_diagonal_(0)
    edge_index, _ = dense_to_sparse(a)
    return edge_index


def bfs_k_hop_pairs(adj_list: List[List[int]], max_hops: int) -> List[Tuple[int, int]]:
    """
    Return candidate pairs (i,j) with i<j and shortest-path distance <= max_hops.
    """
    pairs, _dist = bfs_k_hop_pairs_with_dist(adj_list, max_hops)
    return pairs


def bfs_k_hop_pairs_with_dist(
    adj_list: List[List[int]], max_hops: int
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Pairs with i<j and their shortest-path distance, both sorted by (i, j)."""
    best: Dict[Tuple[int, int], int] = {}
    for s in range(len(adj_list)):
        seen = {s: 0}
        q = [s]
        for u in q:
            du = seen[u]
            if du == max_hops:
                continue
            for v in adj_list[u]:
                if v not in seen:
                    seen[v] = du + 1
                    q.append(v)
        for t, d in seen.items():
            if t <= s or d == 0 or d > max_hops:
                continue
            prev = best.get((s, t))
            if prev is None or d < prev:
                best[(s, t)] = d
    pairs = sorted(best)
    return pairs, [best[p] for p in pairs]


def make_adj_list_from_adj(adj: np.ndarray) -> List[List[int]]:
    n = adj.shape[0]
    adj_bin = (adj > 0).astype(np.int8)
    lst = []
    for i in range(n):
        nbrs = np.nonzero(adj_bin[i])[0].tolist()
        nbrs = [j for j in nbrs if j != i]
        lst.append(nbrs)
    return lst


def bead_labels_from_map(bead_map: List[List[int]], n_atoms: int) -> np.ndarray:
    """
    Per-atom bead id (0..B-1). Used only to derive pair labels; bead id permutation across molecules is irrelevant.
    """
    labels = -np.ones(n_atoms, dtype=np.int64)
    for b, atoms in enumerate(bead_map):
        for a in atoms:
            labels[a] = b
    if np.any(labels < 0):
        missing = np.where(labels < 0)[0].tolist()
        raise ValueError(f"bead_map does not cover all atoms. Missing atom indices: {missing[:20]}")
    return labels


def build_pair_labels(
    bead_map: List[List[int]],
    pairs: List[Tuple[int, int]],
    n_atoms: int,
) -> np.ndarray:
    lab = bead_labels_from_map(bead_map, n_atoms)
    y = np.zeros(len(pairs), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        y[k] = 1.0 if lab[i] == lab[j] else 0.0
    return y


def subsample_pairs(
    pairs: List[Tuple[int, int]],
    y: np.ndarray,
    hops: np.ndarray,
    neg_pos_ratio: float = 3.0,
    max_pairs: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Keep every bonded pair and every positive pair. The decoder merges along
    bonds, so a random drop of a bonded cut removes the pair that decides a
    bead boundary. Only non-bonded negatives are subsampled, toward
    neg_pos_ratio negatives per positive, and only those are cut by max_pairs.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    bond = hops == 1
    pos = y > 0.5
    must = np.where(bond | pos)[0]
    rest = np.where(~bond & ~pos)[0]

    if len(must) == 0:
        n_keep = min(len(y), max_pairs)
        keep = rng.choice(len(y), size=n_keep, replace=False) if n_keep else np.array([], dtype=int)
        keep = np.sort(keep)
        return [pairs[i] for i in keep], y[keep], hops[keep]

    n_pos = int(np.sum(pos))
    already_neg = int(np.sum(~pos[must])) if len(must) else 0
    extra_neg = max(0, int(round(neg_pos_ratio * max(n_pos, 1))) - already_neg)
    extra_neg = min(extra_neg, len(rest), max(0, int(max_pairs) - len(must)))
    if extra_neg > 0:
        extra = rng.choice(rest, size=extra_neg, replace=False)
        keep = np.concatenate([must, extra])
    else:
        keep = np.array(must, dtype=int)
    rng.shuffle(keep)
    keep = np.sort(keep)
    return [pairs[i] for i in keep], y[keep], hops[keep]


# -----------------------------
# PyG Data (with correct batching of pair indices)
# -----------------------------
class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        # Make sure pair_index is offset by num_nodes when batching.
        if key in ("pair_index",):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def make_mol_data(
    adj: np.ndarray,
    x: np.ndarray,
    bead_map: Optional[List[List[int]]] = None,
    max_hops: int = 3,
    neg_pos_ratio: float = 3.0,
    max_pairs: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> MolData:
    n = adj.shape[0]
    if x.shape[0] != n:
        raise ValueError(f"x has {x.shape[0]} rows but adj is {n}x{n}")

    edge_index = adjacency_to_edge_index(adj)
    x_t = torch.tensor(x, dtype=torch.float32)

    # Candidate pairs within max_hops
    adj_list = make_adj_list_from_adj(adj)
    pairs, hops = bfs_k_hop_pairs_with_dist(adj_list, max_hops=max_hops)
    hops_arr = np.asarray(hops, dtype=np.int64)

    data = MolData(x=x_t, edge_index=edge_index)

    if bead_map is not None:
        y = build_pair_labels(bead_map, pairs, n_atoms=n)
        pairs, y, hops_arr = subsample_pairs(
            pairs, y, hops_arr, neg_pos_ratio=neg_pos_ratio, max_pairs=max_pairs, rng=rng
        )

        pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()  # [2, P]
        y_t = torch.tensor(y, dtype=torch.float32)  # [P]
        data.pair_index = pair_index
        data.pair_y = y_t
        data.pair_hop = torch.tensor(hops_arr, dtype=torch.long)

        # Also store true partition for evaluation if desired
        data.true_beads_json = json.dumps(bead_map)

    return data


# -----------------------------
# Model: GNN node encoder + pairwise head
# -----------------------------
class PairHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        # Features commonly used for pair classification
        feat = torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)
        return self.mlp(feat).squeeze(-1)  # logits


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128, layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dropout_p = float(dropout)
        self.lin_in = nn.Linear(in_dim, emb_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            nn_mlp = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
            )
            # This GINConv already adds (1 + eps) * x_i. Self-loops in the bond
            # list would count that atom a second time, so they are not added.
            self.convs.append(GINConv(nn_mlp))
            self.norms.append(LayerNorm(emb_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.lin_in(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        return h


class CGPartitionGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 128,
        layers: int = 4,
        head_hidden: int = 128,
        head_dropout: float = 0.15,
        encoder_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = GINEncoder(
            in_dim=in_dim, emb_dim=emb_dim, layers=layers, dropout=encoder_dropout
        )
        self.head = PairHead(dim=emb_dim, hidden=head_hidden, dropout=head_dropout)

    def forward(self, data: MolData) -> torch.Tensor:
        z = self.encoder(data.x, data.edge_index)  # [N, D]
        i, j = data.pair_index[0], data.pair_index[1]
        logits = self.head(z[i], z[j])  # [P]
        return logits

    @torch.no_grad()
    def node_embeddings(self, data: MolData) -> torch.Tensor:
        return self.encoder(data.x, data.edge_index)


# -----------------------------
# Clustering (inference)
# -----------------------------
@torch.no_grad()
def pair_probs_for_candidates(
    model: CGPartitionGNN,
    adj: np.ndarray,
    x: np.ndarray,
    max_hops: int,
    device: torch.device,
    batch_pairs: int = 200000,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Compute same-bead probabilities for candidate atom pairs within max_hops.
    """
    data = make_mol_data(adj, x, bead_map=None, max_hops=max_hops)
    data = data.to(device)
    z = model.node_embeddings(data)  # [N,D]

    # Candidate pairs within max_hops
    adj_list = make_adj_list_from_adj(adj)
    pairs = bfs_k_hop_pairs(adj_list, max_hops=max_hops)
    if not pairs:
        return [], np.array([], dtype=np.float32)

    # Score in chunks
    probs = np.zeros(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start : start + batch_pairs]
        idx = torch.tensor(chunk, dtype=torch.long, device=device).t().contiguous()
        i, j = idx[0], idx[1]
        logits = model.head(z[i], z[j])
        p = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        probs[start : start + len(chunk)] = p
    return pairs, probs


##########################################################################################

def _is_single_heavy_substituent(mol, atom_idx: int) -> bool:
    """
    True if atom_idx is a non-ring heavy atom whose only heavy-atom neighbor is exactly one ring atom.
    (Hydrogens don't count. So e.g. F/Cl/OH/Me attached to a ring qualifies.)
    """
    a = mol.GetAtomWithIdx(int(atom_idx))
    if a.GetAtomicNum() <= 1:
        return False
    if a.IsInRing():
        return False

    heavy_nbrs = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(heavy_nbrs) != 1:
        return False

    # the only heavy neighbor must be a ring atom
    return mol.GetAtomWithIdx(heavy_nbrs[0]).IsInRing()


def get_ring_systems(mol):
    """
    Return all ring systems as a list of sorted lists of atom indices.

    A ring system is defined as a connected component in the graph formed by ring bonds only
    (fused, bridged, spiro are included). Additionally, for each ring system we include
    all ring substituents consisting of a single heavy atom (non-ring heavy atom with exactly
    one heavy neighbor, which is a ring atom in that system).
    """
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms:
        return []

    # adjacency restricted to ring bonds (to define ring systems)
    adj = {a: [] for a in ring_atoms}
    for b in mol.GetBonds():
        if not b.IsInRing():
            continue
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()
        if u in ring_atoms and v in ring_atoms:
            adj[u].append(v)
            adj[v].append(u)

    # connected components on ring-bond graph
    seen = set()
    systems = []
    for start in ring_atoms:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp_ring = []
        while q:
            x = q.popleft()
            comp_ring.append(x)
            for y in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    q.append(y)

        comp_set = set(comp_ring)

        # add single-heavy-atom substituents attached to atoms of this ring system
        substituents = set()
        for ra in comp_ring:
            for nb in mol.GetAtomWithIdx(int(ra)).GetNeighbors():
                nb_idx = nb.GetIdx()
                if nb_idx in comp_set:
                    continue
                if _is_single_heavy_substituent(mol, nb_idx):
                    # ensure its only heavy neighbor is in THIS ring system
                    only_heavy_nbr = [n.GetIdx() for n in nb.GetNeighbors() if n.GetAtomicNum() > 1][0]
                    if only_heavy_nbr in comp_set:
                        substituents.add(nb_idx)

        systems.append(sorted(comp_set | substituents))

    # optional: stable ordering by (size, then lexicographic)
    systems.sort(key=lambda s: (len(s), s))
    return systems


def _adjacency_within_atoms(mol, atom_set):
    """Bond adjacency restricted to atom_set."""
    adj = {a: [] for a in atom_set}
    for b in mol.GetBonds():
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()
        if u in atom_set and v in atom_set:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _importance_scores(prob, atoms):
    """importance[a] = sum_j prob[a,j] over j in atoms, excluding self."""
    atoms = list(atoms)
    idx = np.array(atoms, dtype=int)
    sub = prob[np.ix_(idx, idx)]
    # exclude diagonal
    imp = sub.sum(axis=1) - np.diag(sub)
    return {atoms[i]: float(imp[i]) for i in range(len(atoms))}


def _best_free_neighbor(a, candidates, prob, importance):
    """
    Choose neighbor in candidates maximizing prob[a,n].
    Tie-break: higher importance[n], then smaller index.
    """
    if not candidates:
        return None
    best = None
    best_key = None
    for n in candidates:
        key = (float(prob[a, n]), float(importance.get(n, 0.0)), -int(n))  # last term for stable tie-break
        if best_key is None or key > best_key:
            best_key = key
            best = n
    return best


def _max_prob_to_bead(a, bead_atoms, prob, importance):
    """
    For selecting which bead to merge a singleton into:
    score(bead) = max_{b in bead} prob[a,b], tie-break by importance[b].
    Returns (max_prob, best_partner_importance, best_partner_atom).
    """
    best_p = -1.0
    best_imp = -1e18
    best_atom = None
    for b in bead_atoms:
        p = float(prob[a, b])
        imp = float(importance.get(b, 0.0))
        if (p > best_p) or (p == best_p and imp > best_imp) or (p == best_p and imp == best_imp and b < best_atom):
            best_p = p
            best_imp = imp
            best_atom = b
    return best_p, best_imp, best_atom

def cluster_ring_systems_from_prob(prob, ring_systems, mol):
    """
    Clustering based on probability being in one bead and pre-defined rules

    Parameters
    ----------
    prob : (N,N) array-like
        Symmetric matrix where prob[i,j] is probability atoms i and j belong to same CG bead.
    ring_systems : list[list[int]]
        Each element is a ring system atom list (ring atoms + single heavy-atom substituents).
    mol : rdkit.Chem.Mol
        Molecule (atom indices must match prob).
    out_svg_prefix : str
        Prefix for output SVG files (one per ring system).

    Returns
    -------
    all_beads : list[list[list[int]]]
        all_beads[sys_idx] = list of beads (each bead is list of atom indices) for that ring system.
    """
    prob = np.asarray(prob, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != prob.shape[1]:
        raise ValueError("prob must be a square NxN matrix.")
    if mol.GetNumAtoms() != prob.shape[0]:
        raise ValueError("mol.GetNumAtoms() must match prob.shape[0].")

    all_beads = []

    for sys_i, sys_atoms_list in enumerate(ring_systems, start=1):
        atom_set = set(int(a) for a in sys_atoms_list)
        if not atom_set:
            all_beads.append([])
            continue

        # adjacency within this ring system
        adj = _adjacency_within_atoms(mol, atom_set)

        # importance
        importance = _importance_scores(prob, atom_set)

        # sorting
        sorted_atoms = sorted(atom_set, key=lambda a: (importance[a], -a), reverse=True)

        # greedy pairing
        atom_to_bead = {}  # atom -> bead index
        beads = []         # list of sets

        for a in sorted_atoms:
            if a in atom_to_bead:
                continue

            free_neighbors = [n for n in adj.get(a, []) if n not in atom_to_bead]
            # Step 4/5: merge with directly connected free neighbor with highest prob
            nbest = _best_free_neighbor(a, free_neighbors, prob, importance)
            if nbest is not None:
                bead = {a, nbest}
                bead_idx = len(beads)
                beads.append(bead)
                atom_to_bead[a] = bead_idx
                atom_to_bead[nbest] = bead_idx
            else:
                # singleton bead for now
                bead = {a}
                bead_idx = len(beads)
                beads.append(bead)
                atom_to_bead[a] = bead_idx

        # try to merge singletons into neighboring beads with size preference
        # We'll iterate until no singleton can be merged anymore (sizes and neighbors change).
        changed = True
        while changed:
            changed = False

            # collect current singleton atoms (one per singleton bead)
            singleton_atoms = []
            for bi, bead in enumerate(beads):
                if len(bead) == 1:
                    singleton_atoms.append(next(iter(bead)))

            # process higher-importance singletons first
            singleton_atoms.sort(key=lambda a: (importance.get(a, 0.0), -a), reverse=True)

            for s in singleton_atoms:
                # bead might have changed already in this pass
                b_s = atom_to_bead.get(s, None)
                if b_s is None or len(beads[b_s]) != 1:
                    continue

                # candidate neighboring beads = beads containing any neighbor of s
                neigh_atoms = adj.get(s, [])
                candidate_bead_ids = set()
                for n in neigh_atoms:
                    if n in atom_to_bead:
                        candidate_bead_ids.add(atom_to_bead[n])
                candidate_bead_ids.discard(b_s)  # exclude itself

                if not candidate_bead_ids:
                    continue

                # score each candidate bead by max prob(s, atom in bead)
                scored = []
                for bid in candidate_bead_ids:
                    bead_atoms = beads[bid]
                    maxp, bestimp, bestatom = _max_prob_to_bead(s, bead_atoms, prob, importance)
                    scored.append((bid, maxp, bestimp, bestatom, len(bead_atoms)))

                # Preferences:
                #  (A) best bead among those of size 2 (resulting size 3)
                #  (B) then among those size < 3 (i.e. 1 or 2)
                #  (C) else best overall (even if size >= 3)
                def pick_best(rows):
                    # sort by (maxp, bestimp) descending
                    rows = sorted(rows, key=lambda t: (t[1], t[2]), reverse=True)
                    return rows[0] if rows else None

                size2 = [r for r in scored if r[4] == 2]
                lt3 = [r for r in scored if r[4] < 3]

                choice = pick_best(size2)
                if choice is None:
                    choice = pick_best(lt3)
                if choice is None:
                    choice = pick_best(scored)

                if choice is None:
                    continue

                target_bid = choice[0]

                # merge singleton atom s into target bead
                beads[target_bid].add(s)
                # clear singleton bead
                beads[b_s].clear()
                # update mapping
                atom_to_bead[s] = target_bid
                changed = True

            # optional cleanup: remove empty beads & remap indices to keep things tidy
            if changed:
                old_to_new = {}
                new_beads = []
                for i, bead in enumerate(beads):
                    if len(bead) == 0:
                        continue
                    old_to_new[i] = len(new_beads)
                    new_beads.append(bead)
                beads = new_beads

                # rebuild atom_to_bead
                atom_to_bead = {}
                for bi, bead in enumerate(beads):
                    for a in bead:
                        atom_to_bead[a] = bi

        # finalize as sorted lists
        bead_lists = [sorted(list(b)) for b in beads if len(b) > 0]
        bead_lists.sort(key=lambda b: (len(b), b))  # stable order

        all_beads.append(bead_lists)

    return all_beads
    
def cluster_nonring_from_prob(prob, ring_systems, mol, max_bead_size=4):
    """
    Greedy non-ring clustering from an atom-atom probability matrix.

    Parameters
    ----------
    prob : (N,N) array-like
        Symmetric matrix where prob[i,j] is the probability atoms i and j belong to the same CG bead.
    mol : rdkit.Chem.Mol
        RDKit molecule. "Directly connected" means bonded neighbors in this mol.
    ring_systems : list[list[int]]
        Ring systems to EXCLUDE from consideration (atoms in any of these lists are removed).
        (Can include substituents already, as you described.)
    max_bead_size : int
        Target max bead size during the greedy phase (default 4).

    Returns
    -------
    beads : list[list[int]]
        List of CG beads (each bead is a list of atom indices), for NON-RING atoms only.
    """
    prob = np.asarray(prob, dtype=float)
    n = mol.GetNumAtoms()
    if prob.shape != (n, n):
        raise ValueError(f"prob must be shape {(n, n)}; got {prob.shape}.")

    # (2) exclude ring systems
    excluded = set()
    for sys_atoms in ring_systems or []:
        excluded.update(int(a) for a in sys_atoms)

    remaining = [a for a in range(n) if a not in excluded]
    remaining_set = set(remaining)
    if not remaining:
        return []

    # Importance uses only atoms still being clustered. Pairs into ring atoms
    # that were already removed would otherwise promote chain atoms next to a ring.
    importance = prob[:, remaining].sum(axis=1) if remaining else prob.sum(axis=1)

    # adjacency restricted to remaining atoms
    adj = {}
    for a in remaining:
        atom = mol.GetAtomWithIdx(int(a))
        adj[a] = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() in remaining_set]

    # (3) sort by importance (desc)
    sorted_atoms = sorted(remaining, key=lambda a: (float(importance[a]), -a), reverse=True)

    beads = []              # list[set[int]]
    atom_to_bead = {}       # atom -> bead_id

    def new_bead(atom_indices):
        bid = len(beads)
        beads.append(set(atom_indices))
        for x in atom_indices:
            atom_to_bead[int(x)] = bid
        return bid

    def add_to_bead(a, bid):
        beads[bid].add(int(a))
        atom_to_bead[int(a)] = bid

    def best_candidate(a):
        """
        Pick neighbor to merge with.
        Priority: higher prob[a,n], tie-break: higher importance[n], then smaller index.
        Only considers neighbors in remaining_set.
        Allows:
          - create a new bead with a + free neighbor
          - add a into an existing neighbor bead if size < max_bead_size
        """
        a = int(a)
        candidates = []
        for nbh in adj.get(a, []):
            nbh = int(nbh)
            p = float(prob[a, nbh])
            imp = float(importance[nbh])

            if nbh not in atom_to_bead:
                # merge into new bead of size 2
                candidates.append(("new", nbh, p, imp))
            else:
                bid = atom_to_bead[nbh]
                if len(beads[bid]) < max_bead_size:
                    # merge into existing bead (size will be <= max_bead_size)
                    candidates.append(("into", nbh, p, imp))

        if not candidates:
            return None

        candidates.sort(key=lambda t: (t[2], t[3], -t[1]), reverse=True)
        return candidates[0]  # (kind, neighbor, p, imp)

    # (4-6) greedy clustering pass over atoms by importance
    for a in sorted_atoms:
        if a in atom_to_bead:
            continue

        choice = best_candidate(a)
        if choice is None:
            new_bead([a])  # singleton
            continue

        kind, nbh, _, _ = choice
        if kind == "new":
            # create new bead with a and neighbor (neighbor might not be processed yet)
            new_bead([a, nbh])
        else:
            # add a into neighbor's existing bead
            bid = atom_to_bead[nbh]
            add_to_bead(a, bid)

    # (7) now we have beads up to max_bead_size, but singletons may remain

    # (8) merge each singleton into best neighbor bead, preferring beads with <= max_bead_size-1
    changed = True
    while changed:
        changed = False

        # collect singleton atoms (process in descending importance)
        singletons = []
        for bid, bead in enumerate(beads):
            if len(bead) == 1:
                s = next(iter(bead))
                singletons.append((s, bid))
        singletons.sort(key=lambda x: (float(importance[x[0]]), -x[0]), reverse=True)

        for s, sbid in singletons:
            # singleton may have been merged already
            if sbid >= len(beads) or len(beads[sbid]) != 1 or s not in beads[sbid]:
                continue

            # neighbor beads scored by best prob(s, neighbor_atom_in_that_bead) over bonded neighbors
            bead_best = {}  # bid -> (best_prob, best_neighbor_atom)
            for nbh in adj.get(s, []):
                if nbh not in atom_to_bead:
                    continue
                bid = atom_to_bead[nbh]
                if bid == sbid:
                    continue
                p = float(prob[s, nbh])
                if (bid not in bead_best) or (p > bead_best[bid][0]):
                    bead_best[bid] = (p, nbh)

            if not bead_best:
                continue

            # sort beads by best_prob desc, tie-break by importance of the best neighbor atom
            cand = []
            for bid, (p, nbh) in bead_best.items():
                cand.append((bid, p, float(importance[nbh])))
            cand.sort(key=lambda t: (t[1], t[2]), reverse=True)

            # prefer bead with size <= max_bead_size-1 (so size after merge <= max_bead_size)
            target = None
            for bid, _, _ in cand:
                if len(beads[bid]) <= max_bead_size - 1:
                    target = bid
                    break
            if target is None:
                # per your rule: merge into best neighbor bead regardless of resulting size
                target = cand[0][0]

            # merge singleton into target
            beads[target].add(s)
            atom_to_bead[s] = target
            beads[sbid].clear()
            changed = True

        if changed:
            # compact bead list & rebuild mapping
            new_beads = []
            old_to_new = {}
            for i, b in enumerate(beads):
                if b:
                    old_to_new[i] = len(new_beads)
                    new_beads.append(b)
            beads = new_beads

            atom_to_bead = {}
            for i, b in enumerate(beads):
                for a in b:
                    atom_to_bead[a] = i

    # (9) return list of lists
    bead_lists = [sorted(list(b)) for b in beads if b]
    bead_lists.sort(key=lambda lst: (len(lst), lst[0], lst))
    return bead_lists

def singleton_cleanup(adj, cm, bead_map, ring_beads_indices):
    """
    Merge all singleton beads (len==1) into an ADJACENT bead that contains an atom
    with the highest probability cm[s, a] to be in the same bead with the singleton atom s.

    Parameters
    ----------
    adj : (N,N) array-like
        Atomic adjacency matrix (nonzero means directly bonded / connected).
    cm : (N,N) array-like
        Symmetric probability matrix.
    bead_map : list[list[int]]
        Current beads as lists of atom indices.
    ring_beads_indices : list[list[int]]
        Indices of beads belonging to each ring system, e.g. [[0,1,3],[6,7,8],...]

    Returns
    -------
    bead_map_out : list[list[int]]
        Updated bead map after merging singletons and removing empty beads.
    ring_beads_indices_out : list[list[int]]
        Updated ring bead indices after bead list compaction/merges.
    """
    adj = np.asarray(adj)
    cm = np.asarray(cm, dtype=float)

    # work with sets
    beads = [set(map(int, b)) for b in bead_map]

    # atom -> bead_id mapping
    atom_to_bead = {}
    for bi, b in enumerate(beads):
        for a in b:
            atom_to_bead[a] = bi

    # importance only used as a stable tie-break (optional, but reuses existing helper)
    importance = {i: float(cm[i].sum()) for i in range(cm.shape[0])}

    # record bead-index redirects: singleton_bead_id -> target_bead_id (old indices)
    merged_into = {}

    # iterate over original indices; skip if bead already changed
    for bi in range(len(beads)):
        if bi >= len(beads) or len(beads[bi]) != 1:
            continue

        s = next(iter(beads[bi]))  # singleton atom

        # find adjacent beads via atomic neighbors of s
        neigh_atoms = np.where(adj[s] != 0)[0]
        candidate_beads = set()
        for n in neigh_atoms:
            n = int(n)
            if n in atom_to_bead:
                candidate_beads.add(atom_to_bead[n])
        candidate_beads.discard(bi)  # exclude itself

        if not candidate_beads:
            continue  # keep singleton if no adjacent beads

        # pick the adjacent bead whose atoms give max cm[s, a]
        best_bid = None
        best_p = -1.0
        best_imp = -1e18  # tie-break via partner importance

        for bid in candidate_beads:
            if bid < 0 or bid >= len(beads) or not beads[bid]:
                continue
            maxp, best_partner_imp, _ = _max_prob_to_bead(s, beads[bid], cm, importance)
            if (maxp > best_p) or (maxp == best_p and best_partner_imp > best_imp) or (
                maxp == best_p and best_partner_imp == best_imp and (best_bid is None or bid < best_bid)
            ):
                best_p = maxp
                best_imp = best_partner_imp
                best_bid = bid

        if best_bid is None:
            continue

        # merge s into best_bid and clear singleton bead
        beads[best_bid].add(s)
        atom_to_bead[s] = best_bid
        beads[bi].clear()
        merged_into[bi] = best_bid

    # resolve redirects to final targets (in case of chains)
    def _resolve(bid):
        bid = int(bid)
        while bid in merged_into:
            bid = merged_into[bid]
        return bid

    merged_into = {k: _resolve(v) for k, v in merged_into.items()}

    # compact beads and build old->new index mapping
    old_to_new = {}
    new_beads = []
    for old_i, b in enumerate(beads):
        if b:
            old_to_new[old_i] = len(new_beads)
            new_beads.append(b)
    beads = new_beads

    # update ring_beads_indices with redirects + compaction
    ring_beads_indices_out = []
    for sys_ids in (ring_beads_indices or []):
        seen = set()
        new_ids = []
        for old_bid in sys_ids:
            old_bid = _resolve(old_bid)
            if old_bid not in old_to_new:
                continue
            nbid = old_to_new[old_bid]
            if nbid not in seen:
                seen.add(nbid)
                new_ids.append(nbid)
        ring_beads_indices_out.append(new_ids)

    bead_map_out = [sorted(list(b)) for b in beads]
    return bead_map_out, ring_beads_indices_out
    

def cluster_all_from_prob(adj, pairs, probs, mol):
    """
    given adj and mol first cluster rings, then cluster non-ring atoms
    return bead map
    """
    cm = np.zeros((adj.shape[0], adj.shape[0]))
    for i, j in zip(pairs, probs):
        cm[i[0], i[1]] = cm[i[1], i[0]] = j
    
    ring_systems = get_ring_systems(mol)
        
    ring_beads = cluster_ring_systems_from_prob(cm, ring_systems, mol)
    
    nonring_beads = cluster_nonring_from_prob(cm, ring_systems, mol)
    
    bead_map = []
    ring_beads_indices = []
    bead_id = 0
    if ring_beads:
        for ring_system in ring_beads:
            ring_beads_indices.append([])
            for bead in ring_system:
                bead_map.append(bead)
                ring_beads_indices[-1].append(bead_id)
                bead_id += 1
    if nonring_beads:
        for bead in nonring_beads:
            bead_map.append(bead)
            bead_id += 1
            
    bead_map, ring_beads_indices = singleton_cleanup(adj, cm, bead_map, ring_beads_indices)
            
    return bead_map, ring_beads_indices

##########################################################################################


def cg_adjacency_from_beads(atom_adj: np.ndarray, bead_map: List[List[int]]) -> np.ndarray:
    """
    Compute CG adjacency between beads: beads are connected if any atom-atom edge exists between them.
    """
    atom_adj_bin = (atom_adj > 0).astype(np.int8)
    b = len(bead_map)
    A_cg = np.zeros((b, b), dtype=np.int8)
    atom_to_bead = {}
    for bi, atoms in enumerate(bead_map):
        for a in atoms:
            atom_to_bead[a] = bi

    n = atom_adj.shape[0]
    for i in range(n):
        bi = atom_to_bead[i]
        nbrs = np.nonzero(atom_adj_bin[i])[0]
        for j in nbrs:
            bj = atom_to_bead[j]
            if bi != bj:
                A_cg[bi, bj] = 1
                A_cg[bj, bi] = 1
    return A_cg


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class TrainConfig:
    max_hops: int = 3
    neg_pos_ratio: float = 3.0
    max_pairs: int = 20000
    emb_dim: int = 32
    layers: int = 3
    head_hidden: int = 32
    head_dropout: float = 0.15
    encoder_dropout: float = 0.0
    label_smooth: float = 0.05
    bond_weight: float = 3.0
    far_weight: float = 0.35
    community_weight: float = 0.8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    patience: int = 40
    min_epochs: int = 30
    batch_size: int = 8
    seed: int = 0


def _scaffold_key(mol: Chem.Mol) -> str:
    """Murcko scaffold, or the canonical SMILES when the molecule has no rings."""
    try:
        heavy = mol if all(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()) else Chem.RemoveHs(mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=heavy, includeChirality=False)
    except Exception:
        scaffold = ""
    if scaffold:
        return scaffold
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
    except Exception:
        return "unreadable"


def split_files(files: List[str], val_frac: float = 0.1, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Keep every molecule that shares a scaffold on the same side of the split.

    Groups are assigned whole until the validation set reaches about val_frac
    of the files. A random file split would put analogs in both sets.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for path in files:
        _adj, _x, mol, _bead_map = load_npz_molecule(path)
        groups[_scaffold_key(mol)].append(path)

    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]))
    n_val_target = max(1, int(round(val_frac * len(files))))
    train: List[str] = []
    val: List[str] = []
    for key in keys:
        group = groups[key]
        if len(val) < n_val_target and (not val or len(val) + len(group) <= n_val_target):
            val.extend(group)
        else:
            train.extend(group)
    if not val or not train:
        raise RuntimeError(
            "scaffold split left one side empty; the set is one scaffold. "
            "Add molecules with a different scaffold or lower the grouping."
        )
    return train, val


def _mol_for_adjacency(mol: Chem.Mol, n_atoms: int) -> Chem.Mol:
    """Return a molecule whose atom count matches the heavy-atom adjacency."""
    if mol.GetNumAtoms() == n_atoms:
        return mol
    heavy = Chem.RemoveHs(mol)
    if heavy.GetNumAtoms() == n_atoms:
        return heavy
    raise ValueError(
        f"adjacency has {n_atoms} atoms but the stored molecule has {mol.GetNumAtoms()}"
    )


@torch.no_grad()
def mean_adjusted_rand(
    model: CGPartitionGNN,
    files: Sequence[str],
    max_hops: int,
    device: torch.device,
) -> float:
    """Mean adjusted Rand of the decoded bead map against the reference mapping.

    Pair F1 can look strong while the greedy assembly still builds the wrong beads.
    """
    model.eval()
    scores = []
    for path in files:
        adj, x, mol, bead_map = load_npz_molecule(path)
        if bead_map is None:
            continue
        mol_heavy = _mol_for_adjacency(mol, adj.shape[0])
        pairs, probs = pair_probs_for_candidates(
            model, adj, x, max_hops=max_hops, device=device
        )
        predicted, _ring_beads = cluster_all_from_prob(adj, pairs, probs, mol_heavy)
        true_labels = bead_labels_from_map(bead_map, adj.shape[0])
        pred_labels = bead_labels_from_map(predicted, adj.shape[0])
        scores.append(float(adjusted_rand_score(true_labels, pred_labels)))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def build_dataset(files: List[str], cfg: TrainConfig, seed: int) -> List[MolData]:
    rng = np.random.default_rng(seed)
    dataset = []
    for p in files:
        adj, x, mol, bead_map = load_npz_molecule(p)
        if bead_map is None:
            raise ValueError(f"{p}: training requires 'beads_json'.")
        d = make_mol_data(
            adj, x, bead_map=bead_map,
            max_hops=cfg.max_hops,
            neg_pos_ratio=cfg.neg_pos_ratio,
            max_pairs=cfg.max_pairs,
            rng=rng,
        )
        dataset.append(d)
    return dataset


def evaluate_pairwise(model: CGPartitionGNN, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_p = []
    losses = []
    bce = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = bce(logits, batch.pair_y)
            losses.append(float(loss.detach().cpu().item()))

            y = batch.pair_y.detach().cpu().numpy()
            p = torch.sigmoid(logits).detach().cpu().numpy()

            all_y.append(y)
            all_p.append(p)

    y = np.concatenate(all_y) if all_y else np.array([], dtype=np.float32)
    p = np.concatenate(all_p) if all_p else np.array([], dtype=np.float32)

    out = {"loss": float(np.mean(losses)) if losses else float("nan")}

    if y.size == 0:
        out.update({"f1_05": float("nan"), "f1_best": float("nan"), "t_best": 0.5})
        return out

    # F1 at threshold 0.5 (what you used before)
    pred_05 = (p >= 0.5).astype(np.int32)
    out["f1_05"] = float(f1_score(y.astype(np.int32), pred_05, zero_division=0))

    # Best threshold for F1
    f1b, tb = best_f1_threshold(y_true=y, p=p, n_steps=41)
    out["f1_best"] = float(f1b)
    out["t_best"] = float(tb)

    return out

def _pair_weights(hop: torch.Tensor, bond_weight: float, far_weight: float) -> torch.Tensor:
    """Bonded pairs decide the greedy merge. Distance-3 pairs mostly add easy negatives."""
    weight = torch.ones(hop.shape, dtype=torch.float32, device=hop.device)
    weight = torch.where(hop == 1, torch.full_like(weight, bond_weight), weight)
    weight = torch.where(hop >= 3, torch.full_like(weight, far_weight), weight)
    return weight


def community_loss(logits: torch.Tensor, y: torch.Tensor, hop: torch.Tensor, pair_graph: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Push the weakest bond inside a reference bead up, and the strongest cut bond down.

    The decoder joins a free atom to its best bonded neighbor. A mean over all
    pairs can look good while that one bond is still wrong, which is where the
    adjusted Rand index moves.
    """
    terms = []
    for g in pair_graph.unique():
        mask = pair_graph == g
        lg = logits[mask]
        yg = y[mask]
        hg = hop[mask]
        bonded_same = (yg > 0.5) & (hg == 1)
        bonded_cut = (yg <= 0.5) & (hg == 1)
        if bonded_same.any():
            softmin = -tau * torch.logsumexp(-lg[bonded_same] / tau, dim=0)
            terms.append(F.softplus(1.0 - softmin))
        if bonded_cut.any():
            softmax = tau * torch.logsumexp(lg[bonded_cut] / tau, dim=0)
            terms.append(F.softplus(softmax + 1.0))
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def partition_objective(
    logits: torch.Tensor,
    y: torch.Tensor,
    hop: torch.Tensor,
    pair_graph: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    eps = float(cfg.label_smooth)
    y_smooth = y * (1.0 - 2.0 * eps) + eps
    weight = _pair_weights(hop, cfg.bond_weight, cfg.far_weight)
    bce = F.binary_cross_entropy_with_logits(logits, y_smooth, weight=weight, reduction="sum")
    bce = bce / weight.sum().clamp(min=1.0)
    return bce + float(cfg.community_weight) * community_loss(logits, y, hop, pair_graph)


def train_main(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        max_hops=args.max_hops,
        neg_pos_ratio=args.neg_pos_ratio,
        max_pairs=args.max_pairs,
        emb_dim=args.emb_dim,
        layers=args.layers,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        encoder_dropout=args.encoder_dropout,
        label_smooth=args.label_smooth,
        bond_weight=args.bond_weight,
        far_weight=args.far_weight,
        community_weight=args.community_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    files = list_npz_files(args.dataset_dir)
    tr_files, va_files = split_files(files, val_frac=args.val_frac, seed=cfg.seed)

    tr_data = build_dataset(tr_files, cfg, seed=cfg.seed)
    va_data = build_dataset(va_files, cfg, seed=cfg.seed + 1)

    in_dim = tr_data[0].x.size(-1)
    model = CGPartitionGNN(
        in_dim=in_dim,
        emb_dim=cfg.emb_dim,
        layers=cfg.layers,
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
        encoder_dropout=cfg.encoder_dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05
    )

    tr_loader = DataLoader(tr_data, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_data, batch_size=cfg.batch_size, shuffle=False)

    best_ari = float("-inf")
    best_state = None
    best_val_bce = float("nan")
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch in tr_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            pair_graph = batch.batch[batch.pair_index[0]]
            loss = partition_objective(logits, batch.pair_y, batch.pair_hop, pair_graph, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        schedule.step()

        tr_loss = float(np.mean(losses)) if losses else float("nan")
        tr_metrics = evaluate_pairwise(model, tr_loader, device=device)
        va_metrics = evaluate_pairwise(model, va_loader, device=device)
        val_ari = mean_adjusted_rand(model, va_files, cfg.max_hops, device)

        msg = (
            f"[{epoch:03d}] train_obj={tr_loss:.4f}  "
            f"train_bce={tr_metrics.get('loss', float('nan')):.4f}  "
            f"val_bce={va_metrics.get('loss', float('nan')):.4f}  "
            f"val_f1@0.5={va_metrics.get('f1_05', float('nan')):.3f}  "
            f"val_f1@best={va_metrics.get('f1_best', float('nan')):.3f}  "
            f"t_best={va_metrics.get('t_best', 0.5):.3f}  "
            f"val_ari={val_ari:.3f}"
        )

        print(msg)

        # The saved weights are the ones that rebuild the reference bead map.
        # Pair loss is logged so the train/validation gap stays visible, and it
        # does not choose the checkpoint.
        if val_ari > best_ari:
            best_ari = val_ari
            best_val_bce = float(va_metrics.get("loss", float("nan")))
            epochs_without_improvement = 0
            # Clone. state_dict() shares storage with the live weights, so a
            # later epoch would otherwise overwrite the checkpoint we meant to keep.
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "in_dim": in_dim,
                "cfg": cfg.__dict__,
                "val_ari": val_ari,
                "val_bce": best_val_bce,
                "epoch": epoch,
            }
        else:
            if epoch >= cfg.min_epochs:
                epochs_without_improvement += 1
            if epoch >= cfg.min_epochs and epochs_without_improvement >= cfg.patience:
                print(
                    f"Early stop at epoch {epoch}: validation ARI did not improve "
                    f"for {cfg.patience} epochs (best {best_ari:.3f})."
                )
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    torch.save(best_state, args.out)
    print(
        f"Saved model to: {args.out}  "
        f"epoch={best_state['epoch']}  val_ari={best_state['val_ari']:.3f}  "
        f"val_bce={best_state['val_bce']:.4f}"
    )

def build_beads_including_hydrogens(mol_h: Chem.Mol, bead_map):
    """Expand each bead's atom list by including bonded hydrogens (after AddHs)."""
    beads = []
    for bead in bead_map:
        bead_atoms = []
        for atom_idx in bead:
            bead_atoms.append(atom_idx)
            atom = mol_h.GetAtomWithIdx(atom_idx)
            for nb in atom.GetNeighbors():
                if nb.GetAtomicNum() == 1:  # hydrogen
                    bead_atoms.append(nb.GetIdx())
        beads.append(bead_atoms)
    return beads

def mol_from_smiles_or_exit(smi: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        raise SystemExit("\nError: RDKit failed to parse the SMILES string.")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        raise SystemExit("\nError: please check your SMILES code is correct.")
    return mol

def mol_from_mol2_or_exit(mol2: str) -> Chem.Mol:
    mol = Chem.MolFromMol2File(mol2, sanitize=False)
    if mol is None:
        raise SystemExit("\nError: RDKit failed to read mol2 file.")
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        raise SystemExit("\nError: please check your mol2 file is correct.")
    return mol
    

def save_molecule_npz(out_path, am_noh, F, mol, bm=None):
    adj = np.asarray(am_noh)
    x = np.asarray(F)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be (N,N); got {adj.shape}")
    if x.ndim != 2 or x.shape[0] != adj.shape[0]:
        raise ValueError(f"x must be (N,M) with same N as adj; got x={x.shape}, adj={adj.shape}")

    if mol is None or not isinstance(mol, Chem.Mol):
        raise TypeError(f"mol must be an rdkit.Chem.Mol, got {type(mol)}")

    mol_u8 = np.frombuffer(mol.ToBinary(), dtype=np.uint8)

    payload = {
        "adj": adj,
        "x": x,
        "mol_bin": mol_u8,
        # optional, helpful for debugging/humans:
        "smiles": np.array(Chem.MolToSmiles(mol), dtype=object),
    }

    if bm is not None:
        bm_clean = [[int(i) for i in bead] for bead in bm]
        payload["beads_json"] = np.array(json.dumps(bm_clean), dtype=object)

    np.savez_compressed(out_path + '.npz', **payload)

def ionize_smiles(smi: str, ph_min: float, ph_max: float) -> str:
    """Most probable ionization state from Dimorphite-DL, one structure only."""
    from dimorphite_dl import protonate_smiles

    variants = protonate_smiles(
        smi,
        ph_min=ph_min,
        ph_max=ph_max,
        max_variants=1,
        label_states=False,
    )
    if not variants:
        raise SystemExit("Dimorphite-DL returned no ionization state.")
    chosen = str(variants[0]).split()[0]
    print(f"ionization at pH {ph_min:g}–{ph_max:g}: {chosen}")
    return chosen


def _heavy_graph(mol: Chem.Mol):
    """Heavy-atom molecule, its adjacency, and its feature matrix.

    RemoveHs keeps heavy atoms in their original relative order and keeps
    their coordinates. The returned molecule is the one clustering indexes.
    """
    heavy = Chem.RemoveHs(mol)
    if heavy.GetNumAtoms() < 1:
        raise SystemExit("The molecule has no heavy atoms.")
    adjacency = rdmolops.GetAdjacencyMatrix(heavy)
    features = atom_feature_matrix(heavy)
    return heavy, adjacency, features


def _with_appended_hydrogens(heavy: Chem.Mol) -> Chem.Mol:
    """Add hydrogens after the heavy atoms so those heavy indices do not move."""
    return Chem.AddHs(heavy, addCoords=heavy.GetNumConformers() > 0)


def convert_input_mol2(mol2, output_name):
    full = mol_from_mol2_or_exit(mol2)
    heavy, adjacency, features = _heavy_graph(full)
    save_molecule_npz(output_name, adjacency, features, heavy)
    mol_h = _with_appended_hydrogens(heavy)
    return [adjacency, features, heavy, mol_h]


def convert_input_smi(smi, output_name, ph_min: float, ph_max: float):
    smi = ionize_smiles(smi, ph_min, ph_max)
    parsed = mol_from_smiles_or_exit(smi)
    heavy, adjacency, features = _heavy_graph(parsed)
    save_molecule_npz(output_name, adjacency, features, heavy)
    mol_h = _with_appended_hydrogens(heavy)
    return [adjacency, features, heavy, mol_h]


def embed_conformers(mol: Chem.Mol, nconfs: int) -> None:
    """Generate 3D conformers and stop if RDKit did not return any."""
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=nconfs, randomSeed=42)
    n_ok = 0 if ids is None else sum(1 for conf_id in ids if int(conf_id) >= 0)
    if n_ok < 1 or mol.GetNumConformers() < 1:
        raise SystemExit("RDKit did not generate a conformer for this molecule.")
    AllChem.UFFOptimizeMoleculeConfs(mol)

def atom_feature_matrix(
    mol: Chem.Mol,
    elements=["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"],
    ring_sizes=range(3, 9),
    include_other=True,
):
    """Binary atom features. Columns are aromaticity, ring membership, ring size, and element."""
    mol.UpdatePropertyCache(strict=False)
    ring_info = mol.GetRingInfo()

    # Count how many rings each atom is part of (useful for fused ring systems)
    atom_ring_counts = np.zeros(mol.GetNumAtoms(), dtype=int)
    for ring in ring_info.AtomRings():  # each ring is a tuple of atom indices
        for idx in ring:
            atom_ring_counts[idx] += 1

    names = []
    names += ["is_aromatic", "is_in_ring", "is_in_fused_ring"]  # "ring system" proxy
    names += [f"in_ring_size_{k}" for k in ring_sizes]
    names += [f"elem_{e}" for e in elements]
    if include_other:
        names += ["elem_other"]

    X = np.zeros((mol.GetNumAtoms(), len(names)), dtype=np.uint8)

    for i, atom in enumerate(mol.GetAtoms()):
        col = 0

        # Aromaticity
        X[i, col] = 1 if atom.GetIsAromatic() else 0
        col += 1

        # Any ring membership
        in_ring = atom.IsInRing()
        X[i, col] = 1 if in_ring else 0
        col += 1

        # Fused ring / ring system participation: atom belongs to >=2 rings
        X[i, col] = 1 if atom_ring_counts[i] >= 2 else 0
        col += 1

        # Ring size membership
        for k in ring_sizes:
            X[i, col] = 1 if atom.IsInRingSize(int(k)) else 0
            col += 1

        # Element one-hot
        sym = atom.GetSymbol()
        hit = False
        for e in elements:
            if sym == e:
                X[i, col] = 1
                hit = True
            col += 1

        # Other element bucket
        if include_other:
            X[i, col] = 0 if hit else 1

    return X


def predict_main(args: argparse.Namespace) -> None:
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.model, map_location="cpu")

    in_dim = int(ckpt["in_dim"])
    cfg_dict = ckpt.get("cfg", {})
    max_hops = int(cfg_dict.get("max_hops", 3))

    model = CGPartitionGNN(
        in_dim=in_dim,
        emb_dim=int(cfg_dict.get("emb_dim", 32)),
        layers=int(cfg_dict.get("layers", 3)),
        head_hidden=int(cfg_dict.get("head_hidden", 32)),
        head_dropout=float(cfg_dict.get("head_dropout", 0.15)),
        encoder_dropout=float(cfg_dict.get("encoder_dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if args.smi is not None and args.mol2 is not None:
        raise SystemExit("\nError: Both smiles and mol2 are given")
    elif args.smi is not None:
        adj, x, mol, mol_h = convert_input_smi(args.smi, args.out_pred, args.ph_min, args.ph_max)
    elif args.mol2 is not None:
        adj, x, mol, mol_h = convert_input_mol2(args.mol2, args.out_pred)
    else:
        raise SystemExit("\nError: Neither smiles nor mol2 is given")

    pairs, probs = pair_probs_for_candidates(model, adj, x, max_hops=max_hops, device=device)
    
    bead_map, ring_beads = cluster_all_from_prob(adj, pairs, probs, mol)

    A_cg = cg_adjacency_from_beads(adj, bead_map)
    
    # Add explicit hydrogens for bead typing + conformers


    # Expand beads to include hydrogens
    beads = build_beads_including_hydrogens(mol_h, bead_map)

    # Martini bead typing. mol_h was built by appending hydrogens to `mol`,
    # so bead_map indexes the same heavy atoms in both.
    bead_types, charges = bead_class.assign_martini_bead_types(
        mol_h, beads, bead_map, str(DEFAULT_BEAD_MODEL), molecule_name=args.out_pred[:4].upper()
    )

    if args.smi is not None or (args.mol2 is not None and args.gen_3d):
        embed_conformers(mol_h, args.nconfs)
    elif mol_h.GetNumConformers() < 1:
        raise SystemExit("Input structure has no coordinates. Pass --gen_3d to generate them.")

    # Get CG coordinates and write outputs
    coords0 = topology.get_coords(mol_h, beads)

    bead_names = topology.write_itp(
        mol_h,
        args.out_pred[:4].upper(),
        beads,
        bead_types,
        ring_beads,
        coords0,
        charges,
        A_cg,
        f"{args.out_pred}.itp",
    )

    topology.write_gro(
        mol_h,
        args.out_pred[:4].upper(),
        bead_names,
        coords0,
        A_cg,
        f"{args.out_pred}",
        f"{args.out_pred}",
    )

    # Draw a model
    svg = draw.draw_2d(
        mol_h,
        bead_map,
        bead_types,
        size=(500, 500),
        label_font_size_px=12,
        label_bg_opacity=0.5,
        label_margin_px=8
    )

    with open(f"{args.out_pred}.svg", "w", encoding="utf-8") as f:
        f.write(svg)

# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="Train model")
    p_tr.add_argument("--dataset_dir", required=True, help="Directory with .npz molecule files")
    p_tr.add_argument("--out", required=True, help="Output model checkpoint (.pt)")

    p_tr.add_argument("--val_frac", type=float, default=0.1)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--cpu", action="store_true")

    p_tr.add_argument("--max_hops", type=int, default=3)
    p_tr.add_argument("--neg_pos_ratio", type=float, default=3.0)
    p_tr.add_argument("--max_pairs", type=int, default=20000)

    p_tr.add_argument("--emb_dim", type=int, default=32)
    p_tr.add_argument("--layers", type=int, default=3)
    p_tr.add_argument("--head_hidden", type=int, default=32)
    p_tr.add_argument("--head_dropout", type=float, default=0.15)
    p_tr.add_argument("--encoder_dropout", type=float, default=0.0)
    p_tr.add_argument("--label_smooth", type=float, default=0.05)
    p_tr.add_argument("--bond_weight", type=float, default=3.0)
    p_tr.add_argument("--far_weight", type=float, default=0.35)
    p_tr.add_argument("--community_weight", type=float, default=0.8)

    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--weight_decay", type=float, default=1e-4)
    p_tr.add_argument("--epochs", type=int, default=150)
    p_tr.add_argument("--patience", type=int, default=40)
    p_tr.add_argument("--min_epochs", type=int, default=30)
    p_tr.add_argument("--batch_size", type=int, default=8)

    p_pr = sub.add_parser("predict", help="Predict bead partition for one molecule")
    p_pr.add_argument(
        "--model",
        default=str(PACKAGE_DIR / "model.pt"),
        help="Partition checkpoint (.pt). Defaults to the model installed with the package.",
    )
    p_pr.add_argument("--smi", help="Input molecule in smiles format")
    p_pr.add_argument("--mol2", help="Input molecule in mol2 format")
    p_pr.add_argument("--gen_3d", action="store_true", help="Generate conformers instead of input geometry for mol")
    p_pr.add_argument("--out_pred", required=True, help="Prefix for the itp, gro, pdb, svg, and npz outputs")
    p_pr.add_argument("--ph-min", type=float, default=7.0, help="Lower pH bound for SMILES ionization")
    p_pr.add_argument("--ph-max", type=float, default=7.4, help="Upper pH bound for SMILES ionization")
    p_pr.add_argument("--nconfs", type=int, default=10, help="Number of conformations to generate topology")
    p_pr.add_argument("--cpu", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "predict":
        predict_main(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

