![molgrind](logo.png)

Molgrind 0.2
============

Coarse-grain a small molecule to Martini 3 from a SMILES string or a mol2 file.

Install
-------

The tested stack is in `environment.yml`. From this directory:

    conda env create -f environment.yml
    conda activate molgrind

That creates the `molgrind` environment and installs this package from
GitHub. Python 3.13.2, NumPy 2.2.4, SciPy 1.16.0, pandas 2.2.3,
scikit-learn 1.7.2, and RDKit 2024.09.6 come from conda-forge. PyTorch
2.6.0, PyTorch Geometric 2.6.1, and Dimorphite-DL 2.1.0 come from pip.
PyTorch Geometric is not published on conda-forge. After `conda activate`,
the command `molgrind` is on PATH. No separate `pip install` is required.

An editable install into an environment you already have is:

    pip install -e .

The partition network (model.pt) and the bead-type classifiers
(martini_bead_classifiers.pkl) are installed with the package.

SMILES
------

SMILES input is passed through Dimorphite-DL first. The default window is
pH 7.0 to 7.4 and a single ionization state is kept.

    molgrind predict --out_pred comp --smi "O=C(C)Oc1ccccc1C(=O)O"

    molgrind predict --out_pred comp --ph-min 7.0 --ph-max 7.4 \
        --smi "CC(=O)O"

--model defaults to the checkpoint installed with the package. Pass a path
to use a different one.

mol2
----

The mol2 may carry hydrogens anywhere in the atom list. They are stripped
for the partition and added back at the end, so bead indices stay aligned.

    molgrind predict --out_pred comp --mol2 mol3D.mol2

    molgrind predict --out_pred comp --mol2 mol3D.mol2 --gen_3d

Mapped topology
---------------

`molgrind topology` does not map the molecule or assign bead types. The JSON
already gives both. Atom indices in `cgnodes` are the heavy atoms in the order
they appear in `smiles`, and `bead_types` is one Martini type per bead. The
SMILES is used as written and is not re-ionized.

    molgrind topology --mapped molecule.json --out_pred imidazole

    {"smiles": "c1c[nH]cn1", "cgnodes": [[1, 0], [4], [3, 2]], "bead_types": ["TC5", "TN5a", "TN6d"]}

A file may contain one object, a JSON list, or one object per line. With more
than one molecule, `--out_pred` is the output directory. Conformers are still
generated so the bonded terms have lengths and angles. `--nconfs` defaults to 10.

Outputs
-------

<out_pred>.itp, _CG.gro, _CG.pdb, _AA.pdb, .svg, and .npz.
The residue name is the first four characters of --out_pred, in upper case.
The gro is centered, and the box is the bead extent plus 1 nm on each side.

Train
-----

`train_dataset/` holds all 148 reference molecules, one `.npz` file each.
It is not itself the training split. `molgrind train` keeps every molecule
that shares a Bemis–Murcko scaffold on the same side of the split. With the
default seed 0 and `--val_frac 0.1` this is the 133 / 15 division reported
in the paper. The 15 validation molecules are used to choose the checkpoint,
so the validation ARI is not an independent test. `martini3_all_beads.csv`
is the bead table for the classifiers: 452 beads, classes C, N, P and X only.
Charge (Q5p, Q5n, D) and bead size are assigned by rules, because those
labels are absent from the table.

    molgrind train --dataset_dir ./train_dataset --out model.pt

Defaults are emb_dim 32, 3 GIN layers, and a pair-head width of 32.
To extend the model, add one `.npz` per curated mapping. Each file needs
`adj`, `x`, `mol_bin` and `beads_json` (the expert atom indices of each bead),
then rerun the same command. Do not add MAD entries without checking that
their mapping and bead names follow the same Martini 3 conventions.
