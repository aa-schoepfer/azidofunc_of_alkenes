import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import ListedColormap

from itertools import product

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from umap import UMAP


def clean_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


# Read and sanitize substrates
reactants = pd.read_csv("reactants.csv")

reactants_clean = (
    reactants.sort_values(by=["price"])
    .drop_duplicates(subset=["smiles"], keep="first")
    .sort_index()
    .reset_index(drop=True)
    .copy()
)
reactants_clean["smiles"] = reactants_clean["smiles"].apply(clean_smiles)
reactants_clean.loc[reactants_clean["price"].isna(), "price"] = 0

# Generate ECFP4 for all possible combinations.
# Faster alternative: compute ECFP4 for each substrate and concatenate at the end
fpgen = AllChem.GetMorganGenerator(radius=4, fpSize=512)

fps = [
    np.asarray(fpgen.GetFingerprint(Chem.MolFromSmiles(e + "." + f)))
    for e, f in product(
        reactants_clean.query("reactivity =='nu'")["smiles"],
        reactants_clean.query("reactivity =='el'")["smiles"],
    )
]

# Get costs
costs = np.array(
    [
        e + f
        for e, f in product(
            reactants_clean.query("reactivity =='nu'")["price"],
            reactants_clean.query("reactivity =='el'")["price"],
        )
    ]
)

costs_nu = np.array(
    [
        e
        for e, f in product(
            reactants_clean.query("reactivity =='nu'")["price"],
            reactants_clean.query("reactivity =='el'")["price"],
        )
    ]
)

costs_el = np.array(
    [
        f
        for e, f in product(
            reactants_clean.query("reactivity =='nu'")["price"],
            reactants_clean.query("reactivity =='el'")["price"],
        )
    ]
)

# Create dataframe for the combinatorial space
combi = pd.DataFrame({"fp": fps, "cost": costs})

# Apply dim. red. on combinatorial FP space
fps_b = np.asarray([np.asarray(l) for l in combi["fp"]]).astype(bool)
pipe = make_pipeline(
    UMAP(n_neighbors=30, n_components=2, metric="jaccard", min_dist=0.1, random_state=0)
)
pca = pipe.fit_transform(fps_b)  # PCA = UMAP = Dimred
combi["pc1"] = pca[:, 0]
combi["pc2"] = pca[:, 1]
combi["cost_nu"] = costs_nu
combi["cost_el"] = costs_el
combi["cost"] = costs

# Cluster reduced space
clu = make_pipeline(
    StandardScaler(), KMeans(n_clusters=15, n_init="auto", random_state=0)
)
clu_fitted = clu.fit_transform(pca)
labels = clu[1].labels_

centers = clu[0].inverse_transform(clu[1].cluster_centers_)

combi["label"] = labels

combi["cc_pc1"] = combi["label"].apply(lambda label: centers[label][0])
combi["cc_pc2"] = combi["label"].apply(lambda label: centers[label][1])

# Compute distance
combi["c_dist"] = np.linalg.norm(
    combi[["pc1", "pc2"]].to_numpy() - combi[["cc_pc1", "cc_pc2"]].to_numpy(), axis=1
)

react_smiles = np.array(
    [
        f"{e}.{f}"
        for e, f in product(
            reactants_clean.query("reactivity =='nu'")["smiles"],
            reactants_clean.query("reactivity =='el'")["smiles"],
        )
    ]
)
combi["r_smiles"] = react_smiles

# Plot
colors = ["#00A79D", "#00748E", "#413D3A", "#CAC7C7"]
custom_cmap = ListedColormap(
    [
        "#00748E",  # blue #4e79a7
        "#413D3A",  # orange #f28e2b
        "#00748E",  # red #e15759
        "#CAC7C7",  # cyan #76b7b2
        "#413D3A",  # green #59a14e
        "#00A79D",  # yellow #edc949
        "#00A79D",  # purple #b07aa2
        "#413D3A",  # pink #ff9da7
        "#CAC7C7",  # brown #9c755f
        "#00A79D",  # grey #bab0ac
        "#00748E",  # red #ff0000
        "#00748E",  # green #00ff00
        "#00A79D",  # #0000ff blue
        "#CAC7C7",  # #00ffff cyan
        "#413D3A",
    ]
)  # #ffff00 yellow

pl.rcParams["font.weight"] = "bold"
pl.rcParams["axes.labelweight"] = "bold"
pl.rcParams["figure.labelweight"] = "bold"
pl.rcParams["axes.linewidth"] = 2

top10 = combi.iloc[[x[1] for x in combi.groupby("label")["c_dist"].nsmallest(10).index]]
top10.copy()[
    [
        "pc1",
        "pc2",
        "cost_nu",
        "cost_el",
        "label",
        "cc_pc1",
        "cc_pc2",
        "c_dist",
        "r_smiles",
    ]
].to_csv(
    "top10_center_c15_pp_1.csv", index=False
)  # save each top 10

fig, ax = pl.subplots(figsize=(6, 6))

ax.scatter(pca[:, 0], pca[:, 1], c=labels, cmap=custom_cmap, s=20)
ax.scatter(centers[:, 0], centers[:, 1], c="#f39869", label="Center", s=20)
ax.scatter(top10["pc1"], top10["pc2"], c="#ff0000", label="Selection", s=20)

ax.set_xticks([])
ax.set_yticks([])

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")

ax.legend()

pl.savefig("umap_c15_pp_v2.png", dpi=300, bbox_inches="tight")

print("Done.")
