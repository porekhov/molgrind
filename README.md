![molgrinder](logo.png)

Molgrinder 0.1
==============

Coarse-grain a small molecule to Martini 3 from a SMILES string or a mol2 file.

Install
-------

The tested stack is in `environment.yml`. From this directory:

    conda env create -f environment.yml
    conda activate molgrinder

That creates the `molgrinder` environment and installs this package from
GitHub. Python 3.13.2, NumPy 2.2.4, SciPy 1.16.0, pandas 2.2.3,
scikit-learn 1.7.2, and RDKit 2024.09.6 come from conda-forge. PyTorch
2.6.0, PyTorch Geometric 2.6.1, and Dimorphite-DL 2.1.0 come from pip.
PyTorch Geometric is not published on conda-forge. After `conda activate`,
the command `molgrinder` is on PATH. No separate `pip install` is required.

An editable install into an environment you already have is:

    pip install -e .

The partition network (model.pt) and the bead-type classifiers
(martini_bead_classifiers.pkl) are installed with the package.

SMILES
------

SMILES input is passed through Dimorphite-DL first. The default window is
pH 7.0 to 7.4 and a single ionization state is kept.

    molgrinder predict --out_pred comp --smi "O=C(C)Oc1ccccc1C(=O)O"

    molgrinder predict --out_pred comp --ph-min 7.0 --ph-max 7.4 \
        --smi "CC(=O)O"

--model defaults to the checkpoint installed with the package. Pass a path
to use a different one.

mol2
----

The mol2 may carry hydrogens anywhere in the atom list. They are stripped
for the partition and added back at the end, so bead indices stay aligned.

    molgrinder predict --out_pred comp --mol2 mol3D.mol2

    molgrinder predict --out_pred comp --mol2 mol3D.mol2 --gen_3d

Outputs
-------

<out_pred>.itp, _CG.gro, _CG.pdb, _AA.pdb, .svg, and .npz.
The residue name is the first four characters of --out_pred, in upper case.
The gro is centered, and the box is the bead extent plus 1 nm on each side.

Train
-----

`train_dataset/` holds the 148 molecules used to train the partition network,
one `.npz` file each. `martini3_all_beads.csv` is the bead table used to
train the bead-type classifiers.

    molgrinder train --dataset_dir ./train_dataset --out model.pt

The validation split keeps each Murcko scaffold on one side. The checkpoint
is the epoch with the highest adjusted Rand index of the decoded bead map,
not the epoch with the lowest pair loss. Training stops after 40 epochs
without an improvement in that index.

The pair loss is not uniform. Every bonded pair is kept, bonded pairs are
weighted 3, and pairs three bonds apart are weighted 0.35. A second term,
weighted 0.8, pushes up the weakest bond inside each reference bead and
pushes down the strongest bond that crosses a bead boundary. That is the
decision the greedy merge actually makes, so the objective stays closer to
the adjusted Rand index than a flat pair loss does. Targets are smoothed by
0.05. The pair head uses dropout 0.15, AdamW weight decay is 1e-4, and the
learning rate follows a cosine schedule from 1e-3. The logged train_bce and
val_bce are plain pair cross-entropy, so the train/validation gap is still
visible next to the training objective. The shipped checkpoint is epoch 62
of this procedure: validation adjusted Rand index 0.752, validation pair
cross-entropy 0.709.

Defaults are emb_dim 32, 3 GIN layers, and a pair-head width of 32.
