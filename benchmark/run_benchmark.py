"""Time and memory of molgrind 0.1 on one homologous oligomer series.

Each member is para-phenylene joined by a butyl linker: 10 heavy atoms and
6 ring atoms per repeat, so the ring-atom fraction is 0.60 at every size.
Sizes are 50, 100, ..., 400 heavy atoms.

The timed predict is the mol2 path: read a structure that already has
coordinates, run the partition network and bead typing, and write the
topology, structure, and drawing. No conformer is generated and UFF is not
called. Coordinates in the mol2 are a pre-placed extended chain, written
before the timer starts.

Memory is the process resident set, sampled about 20 times a second. Peak
values are the high-water mark during that stage. The baseline is the
resident set after the checkpoint is loaded, before the first molecule.
"""

import csv
import gc
import os
import subprocess
import threading
import time
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

import molgrind.bead_class as bead_class
import molgrind.draw as draw
import molgrind.main as m
import molgrind.topology as topology

OUT = Path(__file__).resolve().parent
RUN = OUT / "out"
MODEL = Path(__file__).resolve().parents[1] / "src" / "molgrind" / "model.pt"
SIZES = list(range(50, 401, 50))


def oligomer_smiles(n_units: int) -> str:
    """para-Phenylene with a butyl linker. 10 heavy atoms and 6 ring atoms per unit."""
    return "c1ccc(cc1)CCCC" * n_units


def rss_mb() -> float:
    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(raw.split()[0]) / 1024.0


class RssProbe:
    """Track the highest resident set seen while a stage runs."""

    def __init__(self) -> None:
        self.peak = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, rss_mb())
            except Exception:
                pass
            self._stop.wait(0.05)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss_mb())
        return self.peak


def timed(fn):
    probe = RssProbe()
    t0 = time.perf_counter()
    try:
        value = fn()
    except BaseException:
        probe.stop()
        raise
    return value, time.perf_counter() - t0, probe.stop()


def write_mol2(mol: Chem.Mol, path: Path, name: str) -> None:
    """TRIPOS mol2. RDKit in this environment has no MolToMol2File."""
    conf = mol.GetConformer()
    lines = [
        "@<TRIPOS>MOLECULE",
        name,
        f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        if symbol == "H":
            atom_type = "H"
        elif atom.GetIsAromatic():
            atom_type = "C.ar"
        else:
            atom_type = "C.3"
        index = atom.GetIdx() + 1
        lines.append(
            f"{index} {symbol}{index} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} {atom_type} 1 UNL 0.0000"
        )
    lines.append("@<TRIPOS>BOND")
    for bond in mol.GetBonds():
        order = "ar" if bond.GetIsAromatic() else "1"
        lines.append(
            f"{bond.GetIdx() + 1} {bond.GetBeginAtomIdx() + 1} {bond.GetEndAtomIdx() + 1} {order}"
        )
    path.write_text("\n".join(lines) + "\n")


def mol2_with_coordinates(smiles: str, path: Path, name: str) -> None:
    """Write a mol2 whose coordinates are already present. Not part of the timer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (i * 1.4, (i % 2) * 1.2, 0.0))
    mol.AddConformer(conf, assignId=True)
    write_mol2(mol, path, name)


def main() -> None:
    check = Chem.MolFromSmiles(oligomer_smiles(5))
    if check.GetNumHeavyAtoms() != 50:
        raise SystemExit("oligomer SMILES does not have 10 heavy atoms per unit")
    ring = sum(1 for atom in check.GetAtoms() if atom.IsInRing())
    if ring != 30:
        raise SystemExit("oligomer SMILES does not keep a 0.60 ring-atom fraction")

    RUN.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    ck = torch.load(MODEL, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = m.CGPartitionGNN(
        ck["in_dim"],
        cfg["emb_dim"],
        cfg["layers"],
        cfg["head_hidden"],
        cfg.get("head_dropout", 0.15),
        cfg.get("encoder_dropout", 0.0),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    gc.collect()
    baseline = rss_mb()
    print(f"baseline RSS after model load: {baseline:.1f} MiB", flush=True)

    rows = []
    for heavy_target in SIZES:
        units = heavy_target // 10
        smi = oligomer_smiles(units)
        name = f"n{heavy_target}"
        prefix = str(RUN / name)
        mol2_path = Path(f"{prefix}.mol2")
        mol2_with_coordinates(smi, mol2_path, name)
        rec = {"smiles": smi, "baseline_mb": round(baseline, 2)}

        (adj, x, heavy, mol_h), rec["t_input"], rec["peak_input_mb"] = timed(
            lambda: m.convert_input_mol2(str(mol2_path), prefix)
        )
        if mol_h.GetNumConformers() < 1:
            raise SystemExit(f"{name} mol2 did not keep a conformer")
        rec["heavy_atoms"] = int(heavy.GetNumHeavyAtoms())
        rec["mw"] = round(float(Descriptors.MolWt(heavy)), 2)
        ring_atoms = sum(1 for atom in heavy.GetAtoms() if atom.IsInRing())
        rec["ring_atoms"] = int(ring_atoms)
        rec["ring_fraction"] = ring_atoms / rec["heavy_atoms"]

        def partition():
            pairs, probs = m.pair_probs_for_candidates(
                model, adj, x, max_hops=cfg["max_hops"], device=device
            )
            bead_map, ring_beads = m.cluster_all_from_prob(adj, pairs, probs, heavy)
            return pairs, bead_map, ring_beads

        (pairs, bead_map, ring_beads), rec["t_partition"], rec["peak_partition_mb"] = timed(partition)
        rec["pairs"] = len(pairs)
        rec["beads"] = len(bead_map)

        def typing():
            beads = m.build_beads_including_hydrogens(mol_h, bead_map)
            bead_types, charges = bead_class.assign_martini_bead_types(
                mol_h, beads, bead_map, str(m.DEFAULT_BEAD_MODEL), molecule_name=name[:4].upper()
            )
            return beads, bead_types, charges

        (beads, bead_types, charges), rec["t_typing"], rec["peak_typing_mb"] = timed(typing)

        def write_outputs():
            coords0 = topology.get_coords(mol_h, beads)
            graph = m.cg_adjacency_from_beads(adj, bead_map)
            bead_names = topology.write_itp(
                mol_h,
                name[:4].upper(),
                beads,
                bead_types,
                ring_beads,
                coords0,
                charges,
                graph,
                f"{prefix}.itp",
            )
            topology.write_gro(
                mol_h, name[:4].upper(), bead_names, coords0, graph, prefix, prefix
            )
            svg = draw.draw_2d(
                mol_h,
                bead_map,
                bead_types,
                size=(500, 500),
                label_font_size_px=12,
                label_bg_opacity=0.5,
                label_margin_px=8,
            )
            Path(f"{prefix}.svg").write_text(svg)

        _, rec["t_write"], rec["peak_write_mb"] = timed(write_outputs)
        rec["t_total"] = rec["t_input"] + rec["t_partition"] + rec["t_typing"] + rec["t_write"]
        rec["peak_mb"] = max(
            rec["peak_input_mb"],
            rec["peak_partition_mb"],
            rec["peak_typing_mb"],
            rec["peak_write_mb"],
        )
        del adj, x, heavy, mol_h, pairs, bead_map, ring_beads, beads, bead_types, charges
        gc.collect()
        rec["rss_after_mb"] = rss_mb()
        print(
            f"{rec['heavy_atoms']:4d}  mw {rec['mw']:.1f}  ring {rec['ring_fraction']:.2f}  "
            f"beads {rec['beads']:4d}  in {rec['t_input']:.3f}  part {rec['t_partition']:.3f}  "
            f"type {rec['t_typing']:.3f}  write {rec['t_write']:.3f}  "
            f"total {rec['t_total']:.3f}s  peak {rec['peak_mb']:.1f} MiB  "
            f"after {rec['rss_after_mb']:.1f} MiB",
            flush=True,
        )
        rows.append(rec)

    fields = [
        "heavy_atoms",
        "mw",
        "ring_atoms",
        "ring_fraction",
        "beads",
        "pairs",
        "t_input",
        "t_partition",
        "t_typing",
        "t_write",
        "t_total",
        "baseline_mb",
        "peak_input_mb",
        "peak_partition_mb",
        "peak_typing_mb",
        "peak_write_mb",
        "peak_mb",
        "rss_after_mb",
        "smiles",
    ]
    with (OUT / "timings.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT / "timings.csv")


if __name__ == "__main__":
    main()
