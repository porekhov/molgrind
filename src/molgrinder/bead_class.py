from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Any, Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolDescriptors


# ---- Column order required by models ----
DESCRIPTOR_COLUMNS = [
    "MR", "logP", "logP_per_heavy", "MW", "charge",
    "aromatic count", "ring count", "heavy atom count",
    "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "donors", "acceptors",
]

FULL_COLUMN_ORDER = [
    "Molecule", "Bead", "Type", "SMILES",
    *DESCRIPTOR_COLUMNS,
    "Bead_type", "Bead_size", "Bead_subtype",
]

ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
HALOGEN_COLS = ["F", "Cl", "Br", "I"]


def _as_atom_index_list(x: Any) -> List[int]:
    """Flatten RDKit substructure matches to atom indices.

    Lipinski._HDonors and _HAcceptors return GetSubstructMatches: a tuple of
    match tuples, one per donor or acceptor, e.g. ((0,), (3,), (11,)).
    Every match is kept. Match order is preserved and repeated indices are dropped.
    """
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        return []

    out: List[int] = []
    seen = set()
    for match in x:
        atoms = match if isinstance(match, (list, tuple, np.ndarray)) else (match,)
        for v in atoms:
            try:
                idx = int(v)
            except (TypeError, ValueError):
                continue
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def get_descs(mol: Chem.Mol, beads: Sequence[Sequence[int]]) -> List[List[float]]:
    """
    Compute per-bead descriptors in a stable, pythonic way.

    Returns a list of lists with THIS EXACT ORDER:
      MR, logP, logP_per_heavy, MW, charge, aromatic count, ring count, heavy atom count,
      C, N, O, F, P, S, Cl, Br, I, donors, acceptors
    """
    n_atoms = mol.GetNumAtoms()
    atoms = list(mol.GetAtoms())

    # Crippen per-atom contributions: (logP_contrib, MR_contrib)
    crippen = rdMolDescriptors._CalcCrippenContribs(mol)

    atomic_masses = np.array([a.GetMass() for a in atoms], dtype=float)
    aromatic = np.array([1.0 if a.GetIsAromatic() else 0.0 for a in atoms], dtype=float)
    in_ring = np.array([1.0 if a.IsInRing() else 0.0 for a in atoms], dtype=float)
    is_heavy = np.array([0.0 if a.GetAtomicNum() == 1 else 1.0 for a in atoms], dtype=float)

    # Gasteiger charges (ensure props exist, convert safely)
    AllChem.ComputeGasteigerCharges(mol)
    gchg = np.zeros(n_atoms, dtype=float)
    for i, a in enumerate(atoms):
        try:
            gchg[i] = float(a.GetProp("_GasteigerCharge"))
        except Exception:
            gchg[i] = 0.0

    # Atom-level donor/acceptor flags
    donors_flag = np.zeros(n_atoms, dtype=float)
    acceptors_flag = np.zeros(n_atoms, dtype=float)


    d_idx = _as_atom_index_list(Lipinski._HDonors(mol))
    a_idx = _as_atom_index_list(Lipinski._HAcceptors(mol))
    
    for i in d_idx:
        if 0 <= i < n_atoms:
            donors_flag[i] = 1.0
    for i in a_idx:
        if 0 <= i < n_atoms:
            acceptors_flag[i] = 1.0
            

    # Pre-map element symbol -> index in our atom vector
    atom_type_to_col = {sym: j for j, sym in enumerate(ATOM_TYPES)}

    descs: List[List[float]] = []
    for bead in beads:
        bead_idx = [int(i) for i in bead]
        if not bead_idx:
            # Empty bead: emit zeros
            descs.append([0.0] * len(DESCRIPTOR_COLUMNS))
            continue

        if min(bead_idx) < 0 or max(bead_idx) >= n_atoms:
            raise IndexError(
                f"Bead contains atom index outside mol range [0, {n_atoms-1}]: {bead_idx}"
            )

        idx = np.array(bead_idx, dtype=int)

        logP = float(np.sum([crippen[i][0] for i in idx]))
        MR = float(np.sum([crippen[i][1] for i in idx]))
        MW = float(np.sum(atomic_masses[idx]))
        charge = float(np.sum(gchg[idx]))

        aromatic_count = float(np.sum(aromatic[idx]))
        ring_count = float(np.sum(in_ring[idx]))
        heavy_count = float(np.sum(is_heavy[idx]))
        logP_per_heavy = float(logP / heavy_count) if heavy_count > 0 else 0.0

        atom_vector = np.zeros(len(ATOM_TYPES), dtype=float)
        for i in idx:
            sym = atoms[i].GetSymbol()
            j = atom_type_to_col.get(sym)
            if j is not None:
                atom_vector[j] += 1.0

        donor_count = float(np.sum(donors_flag[idx]))
        acceptor_count = float(np.sum(acceptors_flag[idx]))

        desc = [
            MR, logP, logP_per_heavy, MW, charge,
            aromatic_count, ring_count, heavy_count,
            *atom_vector.tolist(),
            donor_count, acceptor_count,
        ]
        descs.append(desc)

    return descs


def get_formal_charges(mol: Chem.Mol, beads: Sequence[Sequence[int]]) -> List[int]:
    """Formal charge per bead (sum of atom formal charges)."""
    charges: List[int] = []
    for bead in beads:
        q = 0
        for i in bead:
            q += int(mol.GetAtomWithIdx(int(i)).GetFormalCharge())
        charges.append(int(q))
    return charges


def _load_bundle(bundle_or_path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(bundle_or_path, dict):
        return bundle_or_path
    # try pickle first, then joblib
    import pickle
    try:
        with open(bundle_or_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(bundle_or_path)


def _predict_with_section(bundle: Dict[str, Any], section: str, X_df: pd.DataFrame) -> np.ndarray:
    """Predict using bundle[section] = {'model':..., 'features':...}."""
    sec = bundle[section]
    model = sec["model"]
    feats = sec["features"]
    return model.predict(X_df[feats].to_numpy())


def _predict_subtype(bundle: Dict[str, Any], bead_type: str, X_df_row: pd.DataFrame) -> str:
    sub = bundle["bead_subtype"][bead_type]
    model = sub["model"]
    feats = sub["features"]
    return str(model.predict(X_df_row[feats].to_numpy())[0])
    
def _get_size(beads):
    sizes = []
    for bead in beads:
        if len(bead) < 3:
            sizes.append('T')
        elif len(bead) == 3:
            sizes.append('S')
        else:
            sizes.append('B')
    return sizes


def assign_martini_bead_types(
    mol: Chem.Mol,
    beads: Sequence[Sequence[int]],
    beads_noh: Sequence[Sequence[int]],
    bundle_or_path: Union[str, Dict[str, Any]],
    molecule_name: str = "MOL",
    bead_names: Optional[Sequence[str]] = None,
    smiles: Optional[str] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Main entry point.

    Returns:
      bead_types: list[str] final martini bead strings per bead
      charges: formal charges of beads
    """
    bundle = _load_bundle(bundle_or_path)

    if bead_names is None:
        bead_names = [f"B{i+1}" for i in range(len(beads))]
    if smiles is None:
        try:
            smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            smiles = ""

    # Compute descriptors + formal charges
    descs = get_descs(mol, beads)
    formal = get_formal_charges(mol, beads)

    # Build dataframe in the required order
    rows: List[Dict[str, Any]] = []
    for i, (bn, drow) in enumerate(zip(bead_names, descs)):
        r: Dict[str, Any] = {
            "Molecule": molecule_name,
            "Bead": bn,
            "Type": "",     # placeholder / compatibility
            "SMILES": smiles,
        }
        r.update({k: float(v) for k, v in zip(DESCRIPTOR_COLUMNS, drow)})
        r.update({"Bead_type": None, "Bead_size": None, "Bead_subtype": None})
        rows.append(r)

    df = pd.DataFrame(rows)
    df = df.reindex(columns=FULL_COLUMN_ORDER)

    # ML predictions (vectorized for speed)
    pred_type = _predict_with_section(bundle, "bead_type", df)
    pred_size = _get_size(beads_noh)

    final_strings: List[str] = []

    for i in range(len(df)):
        q_formal = int(formal[i])

        # halogens present -> X
        has_halogen = any(float(df.loc[i, c]) > 0.0 for c in HALOGEN_COLS)

        # determine bead type
        charge_sign = ""
        if q_formal in (-1, 1):
            # Polarity stays 5. The letter records the sign used by this force field (Q5n, Q5p).
            bead_type = "Q"
            bead_subtype = "5"
            charge_sign = "p" if q_formal > 0 else "n"
        elif abs(q_formal) >= 2:
            # Martini 3 names the divalent bead D, SD, or TD, with no subtype numeral.
            bead_type = "D"
            bead_subtype = ""
        elif has_halogen:
            bead_type = "X"
            bead_subtype = None  # predicted below
        else:
            bead_type = str(pred_type[i])
            bead_subtype = None  # predicted below

        # sizes from number of atoms
        size_raw = str(pred_size[i])
        size_prefix = "" if size_raw in ("B", "", "None", "nan") else size_raw  # B == normal size

        # subtypes from ML models for C/N/P/X
        if bead_subtype is None:
            if bead_type in ("C", "N", "P", "X"):
                bead_subtype = _predict_subtype(bundle, bead_type, df.iloc[[i]])
            elif bead_type == "Q":
                bead_subtype = "5"
            elif bead_type == "D":
                bead_subtype = ""
            else:
                # fallback: no subtype
                bead_subtype = ""

        # add donor/acceptor suffix for N and P particles
        donors = float(df.loc[i, "donors"])
        acc = float(df.loc[i, "acceptors"])
        da = ""
        if bead_type in ("N", "P"):
            if donors > 0 and acc == 0:
                da = "d"
            elif donors == 0 and acc > 0:
                da = "a"
                

        bead_str = f"{size_prefix}{bead_type}{bead_subtype}{da}{charge_sign}"
        final_strings.append(bead_str)

        df.loc[i, "Bead_type"] = bead_type
        df.loc[i, "Bead_size"] = size_raw
        df.loc[i, "Bead_subtype"] = bead_subtype

    return final_strings, formal
