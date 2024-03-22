# MultiKD-DTA

Enhancing Drug-Target Affinity Prediction through Width and Depth Feature Extraction with Knowledge Distillation

## Get Started

#### Environment

* Python=3.10
* CUDA=12.2
* PyTorch=2.2.0

#### BioToolKit Dependencies

```bash
pip install fair-esm
conda install conda-forge::rdkit
```

#### APEX Dependency

```bash
1. git clone https://github.com/NVIDIA/apex
2. cd apex
3. python setup.py install --cuda_ext --cpp_ext
```

## How to run?

#### Prepare the Embedded Protein file

```bash
python datahelper.py
```

This would consume most of the time, as it involves downloading the parameters of the ESM-2 model and then generating embeddings for the proteins.

#### Convert SMILES to Graph

```bash
python generate_drug_profile.py
```

#### File Structure

```bash
├── README.md
├── data
│   ├── DTI_plot.ipynb
│   ├── davis
│   │   ├── Y
│   │   ├── davis_test_fold0.csv
│   │   ├── davis_test_fold1.csv
│   │   ├── davis_test_fold2.csv
│   │   ├── davis_test_fold3.csv
│   │   ├── davis_test_fold4.csv
│   │   ├── davis_train_fold0.csv
│   │   ├── davis_train_fold1.csv
│   │   ├── davis_train_fold2.csv
│   │   ├── davis_train_fold3.csv
│   │   ├── davis_train_fold4.csv
│   │   ├── drug-drug_similarities_2D.txt
│   │   ├── drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt
│   │   ├── folds
│   │   │   ├── test_fold_setting1.txt
│   │   │   └── train_fold_setting1.txt
│   │   ├── ligands_can.txt
│   │   ├── ligands_iso.txt
│   │   ├── proteins.txt
│   │   └── target-target_similarities_WS.txt
│   ├── davis_test.csv
│   ├── davis_train.csv
│   ├── kiba
│   │   ├── Y
│   │   ├── folds
│   │   │   ├── test_fold_setting1.txt
│   │   │   └── train_fold_setting1.txt
│   │   ├── kiba_binding_affinity_v2.txt
│   │   ├── kiba_drug_sim.txt
│   │   ├── kiba_target_sim.txt
│   │   ├── kiba_test_fold0.csv
│   │   ├── kiba_test_fold1.csv
│   │   ├── kiba_test_fold2.csv
│   │   ├── kiba_test_fold3.csv
│   │   ├── kiba_test_fold4.csv
│   │   ├── kiba_train_fold0.csv
│   │   ├── kiba_train_fold1.csv
│   │   ├── kiba_train_fold2.csv
│   │   ├── kiba_train_fold3.csv
│   │   ├── kiba_train_fold4.csv
│   │   ├── ligands_can.txt
│   │   ├── ligands_iso.txt
│   │   └── proteins.txt
│   ├── kiba_test.csv
│   ├── kiba_train.csv
│   └── processed
│       ├── davis_test.pt
│       ├── davis_train.pt
│       ├── kiba_test.pt
│       └── kiba_train.pt
├── datahelper.py
├── davis.npz
├── generate_drug_profile.py
├── kiba.npz
├── model
├── src
│   ├── ISBRA.py
│   ├── statistics.py
│   └── utils.py
└── train.py
```

#### Start to Train

```bash
python train.py
```

You can change the parameters in the `train.py` and `ISBRA.py` file.





