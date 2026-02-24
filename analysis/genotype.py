#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from time import perf_counter
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.linalg as spl
import pandas as pd
import dask
import dask.array as da
from sklearn.cluster import KMeans, SpectralClustering
from pandas_plink import read_plink, read_plink1_bin, write_plink1_bin
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns


# In[ ]:


my_bucket = os.getenv('WORKSPACE_BUCKET')
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
plt.rcParams['figure.dpi'] = 800
plt.rcParams['savefig.dpi'] = 800
fig_root = Path('figs/aou/genotype')

genome_dir = Path("genomic_files/genome")
rsvd_dir = genome_dir / "rsvd"


# In[ ]:


# os.system(f"gsutil cp '{my_bucket}/genomic_data/genomic_files.zip' .")
# shutil.unpack_archive("genomic_files.zip", "genomic_files")


# In[ ]:


# bim.snp: SNP IDs (columns)
# fam.iid: sample IDs (indices)
# bed: raw data
bim_full, fam_full, bed_full = read_plink(str(genome_dir / "qc_data_unrelated"))
bed_full = bed_full.T


# In[ ]:


assert bed_full.shape == (len(fam_full), len(bim_full))
bed_full # n_subject, n_snp


# In[ ]:


class BedOperator(sp.sparse.linalg.LinearOperator):
    def __init__(self, bed, mask, env, covariates, manifest):
        if manifest:
            self.A = np.ascontiguousarray(bed[mask].compute().astype(np.float32))
            self.AT = self.A.T
        else:
            self.A = bed[mask].rechunk(('auto', -1))
            self.AT = bed[mask].T.rechunk(('auto', -1))
        
        self.env = env
        self.Q = None
        if covariates is not None:
            if covariates.ndim == 1:
                covariates = covariates[:, np.newaxis]
            self.Q, _ = np.linalg.qr(covariates)
            
        super().__init__(dtype=np.float32, shape=self.A.shape)
        
    def _matmat(self, X):
        ans = self.env[:, np.newaxis] * (self.A @ X)
        if self.Q is not None:
            ans = ans - self.Q @ (self.Q.T @ ans)
        return ans
    
    def _rmatmat(self, X):
        if self.Q is not None:
            X = X - self.Q @ (self.Q.T @ X)
        ans = self.AT @ (self.env[:, np.newaxis] * X)
        return ans


# In[ ]:


def preprocess(bed, mask, env, covariates, manifest):
    col_modes_path = genome_dir / f"col_modes.npy"
    if not col_modes_path.is_file():
        print("Computing column modes...")
        col_modes = da.map_blocks(
            lambda x: sp.stats.mode(x, axis=0, nan_policy='omit', keepdims=True)[0], 
            bed, dtype=bed.dtype, drop_axis=0, new_axis=0
        ).compute()
        np.save(col_modes_path, col_modes)
        print(f"Cached to {col_modes_path}")
    col_modes = np.load(col_modes_path)
    
    imputed_base = genome_dir / f"qc_data_unrelated_imputed"
    if not imputed_base.with_suffix(".bed").is_file():
        print("Computing imputed data...")
        genotype = read_plink1_bin(str(genome_dir / "qc_data_unrelated.bed"), verbose=False)
        genotype_imputed = genotype.fillna(col_modes)
        write_plink1_bin(genotype_imputed, f"{imputed_base}.bed", verbose=True)
        print(f"Cached to {imputed_base}.*")
    bim_imp, fam_imp, bed_imputed = read_plink(str(imputed_base))
    bed_imputed = bed_imputed.T
    
    row_sq_sums_path = genome_dir / f"row_sq_sums.npy"
    if not row_sq_sums_path.is_file():
        print("Computing row squared sums...")
        row_sq_sums = (bed_imputed**2).sum(axis=-1).compute()
        np.save(row_sq_sums_path, row_sq_sums)
        print(f"Cached to {row_sq_sums_path}")
    row_sq_sums = np.load(row_sq_sums_path)
    
    bed_op = BedOperator(bed_imputed, mask, env, covariates, manifest)
    frobenius_sq = np.sum(env**2 * row_sq_sums[mask])
    
    return bed_op, frobenius_sq


# In[ ]:


def manual_rsvd(A, k, n_oversamples=10, n_iter=2, random_state=42, verbose=False):
    rng = np.random.default_rng(random_state)
    m, n = A.shape
    l = k + n_oversamples
    
    if verbose:
        print(f"--- Starting Manual RSVD (k={k}, oversamples={n_oversamples}, iter={n_iter}) ---")
    
    t_start = perf_counter()

    # 1. Generate Omega and Initial Projection
    t0 = perf_counter()
    Omega = rng.standard_normal(size=(n, l)).astype(np.float32)
    Y = A @ Omega
    if verbose:
        print(f"[Step 1-2] Init Projection Y = A @ Omega: {perf_counter() - t0:.4f}s")

    # 2. Power Iterations
    for i in range(n_iter):
        t_iter = perf_counter()
        Y, _ = spl.qr(Y, mode='economic')
        Y_tilde = A.T @ Y
        Y = A @ Y_tilde
        if verbose:
            print(f"  > Power Iteration {i+1}/{n_iter}: {perf_counter() - t_iter:.4f}s")

    # 3. QR Decomposition
    t0 = perf_counter()
    Q, _ = spl.qr(Y, mode='economic')
    if verbose:
        print(f"[Step 4] Final QR: {perf_counter() - t0:.4f}s")

    # 4. Form Small Matrix B
    t0 = perf_counter()
    B = (A.T @ Q).T
    if verbose:
        print(f"[Step 5] Form B (Q.T @ A): {perf_counter() - t0:.4f}s")

    # 5. SVD of B
    t0 = perf_counter()
    U_tilde, S, Vt = spl.svd(B, full_matrices=False)
    if verbose:
        print(f"[Step 6] SVD of B: {perf_counter() - t0:.4f}s")

    # 6. Final Projection
    U = Q @ U_tilde

    # 7. Truncate
    U = U[:, :k]
    S = S[:k]
    Vt = Vt[:k, :]

    if verbose:
        print(f"--- RSVD Finished in {perf_counter() - t_start:.4f}s ---")
    
    return U, S, Vt


# In[ ]:


def compute_rsvd(bed_op, frobenius_sq, n_component_rsvd, cache_key):
    rsvd_dir.mkdir(parents=True, exist_ok=True)

    U_path = rsvd_dir / f"U_{cache_key}_{n_component_rsvd}.npy"
    S_path = rsvd_dir / f"S_{cache_key}_{n_component_rsvd}.npy"
    Vt_path = rsvd_dir / f"Vt_{cache_key}_{n_component_rsvd}.npy"

    if not (U_path.is_file() and S_path.is_file() and Vt_path.is_file()):
        print(f"Running RSVD with {n_component_rsvd} components...")
        U, S, Vt = manual_rsvd(bed_op, k=n_component_rsvd)
        np.save(U_path, U)
        np.save(S_path, S)
        np.save(Vt_path, Vt)
        print(f"Cached to {U_path}, {S_path}, {Vt_path}")
    U, S, Vt = np.load(U_path), np.load(S_path), np.load(Vt_path)
    
    projections = compute_proj(bed_op, Vt, n_component_rsvd, cache_key)
    
    visualize_rsvd(S, frobenius_sq, cache_key)
    visualize_rsvd(S, frobenius_sq, cache_key, top_k=15)
    visualize_rsvd(S, frobenius_sq, cache_key, top_k=20)
    visualize_rsvd(S, frobenius_sq, cache_key, top_k=32)
    visualize_rsvd(S, frobenius_sq, cache_key, top_k=50)
    
    return U, S, Vt, projections


def compute_proj(bed_op, Vt, n_component_rsvd, cache_key):
    proj_path = rsvd_dir / f"proj_{cache_key}_{n_component_rsvd}.npy"
    if not proj_path.is_file():
        print("Running projections...")
        projections = bed_op @ Vt.T
        np.save(proj_path, projections)
        print(f"Cached to {proj_path}")
    projections = np.load(proj_path)
    
    return projections


def visualize_rsvd(S, frobenius_sq, cache_key, top_n = 40, top_k = None):
    x = np.arange(len(S)) + 1
    y = 100 * S[:]**2 / np.sum(S[:top_n]**2)
    title = f"Relative Pct. Var. Explained (w.r.t. Top {top_n} PCs)"
    stem = f"{cache_key}_svdvals"
    if top_k is not None:
        x = x[:top_k]
        y = y[:top_k]
        stem = f"{cache_key}_svdvals_top{top_k}"
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=0.1, right=np.max(x))
    ax.set_ylim((0, None))
    ax.set_xlabel("Component Number", fontsize="xx-large")
    ax.set_ylabel("Variance Explained", fontsize="xx-large")
    
    # fig.suptitle(title, fontsize="xx-large")
    fig.tight_layout()
    fig_root.mkdir(parents=True, exist_ok=True)
    fig_name = fig_root / stem
    fig.savefig(fig_name.with_suffix(".pdf"))
    fig.savefig(fig_name.with_suffix(".png"))
    fig.show()


# In[ ]:


def visualize_clustering(rng, U, S, Vt, projections, n_cluster, n_sample, n_component, transform, init, dim, cache_key, labels_override = None):
    idx = rng.choice(len(projections), size=n_sample, replace=False)
    init_str = init

    data = projections[idx, :n_component]
    
    if transform == "unscale":
        data /= S[:data.shape[-1]]
    elif transform == "row_normalize":
        data /= S[:data.shape[-1]]
        data /= np.linalg.norm(data, axis=-1, keepdims=True)

    algo = KMeans(n_cluster, init=init, random_state=42, max_iter=1)
    if labels_override is None:
        algo.fit(data)
        labels = algo.labels_ + 1
    else:
        labels = labels_override[idx]
    
    bounds = np.arange(n_cluster + 1) + 0.5
    norm = BoundaryNorm(bounds, n_cluster)
    
    if dim == "3d":
        fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
        cmap = ListedColormap(sns.color_palette('husl', n_cluster))
        for i, ax in enumerate(axes.flatten()):
            ax.set_box_aspect(None, zoom=0.85)
            first, second, third = 3*i, 3*i+1, 3*i+2
            sc = ax.scatter(data[:, first], data[:, second], data[:, third],
                            marker='.', cmap=cmap, norm=norm, c=labels, s=0.1, alpha=0.01)
            # ax.set_xlabel(f"PC{first+1}", labelpad=10)
            # ax.set_ylabel(f"PC{second+1}", labelpad=10)
            # ax.set_zlabel(f"PC{third+1}", labelpad=15)
            
            ax.tick_params(axis='both', which='major', labelsize=4, pad=0)
            # ax.tick_params(axis='both', which='minor', labelsize='small')
            ax.grid(color='silver', linestyle='-')

            x_bot, x_top = ax.get_xlim()
            y_bot, y_top = ax.get_ylim()
            z_bot, z_top = ax.get_zlim()
            x_bot, x_top = x_bot - 0.12 * (x_top - x_bot), x_top + 0.12 * (x_top - x_bot)
            y_bot, y_top = y_bot - 0.12 * (y_top - y_bot), y_top + 0.12 * (y_top - y_bot)
            z_bot, z_top = z_bot - 0.12 * (z_top - z_bot), z_top + 0.12 * (z_top - z_bot)
            ax.text(x_bot, y_bot, z_bot, f"PC{first+1}", 
                    ha='center', va='bottom',
                    fontsize='medium',
                    fontweight='normal')
            ax.text(x_top, y_top, z_bot, f"PC{second+1}", 
                    ha='center', va='center',
                    fontsize='medium',
                    fontweight='normal')
            ax.text(x_bot, y_top, z_top, f"PC{third+1}", 
                    ha='center', va='center',
                    fontsize='medium',
                    fontweight='normal')
            ax.set_title(f'PC{first+1} through PC{third+1}', fontsize='large')
    else:
        fig, axes = plt.subplots(2, 3)
        cmap = ListedColormap(sns.color_palette('husl', n_cluster))
        for i, ax in enumerate(axes.flatten()):
            first, second = 2*i, 2*i+1
            sc = ax.scatter(data[:, first], data[:, second],
                            marker='.', cmap=cmap, norm=norm, c=labels, s=0.1, alpha=0.01)
            ax.set_xlabel(f"PC{first+1}")
            ax.set_ylabel(f"PC{second+1}")
            
    # fig.subplots_adjust(top=0.85, bottom=0.2, wspace=0.1, hspace=0.25, right=0.95, left=0.05)
    fig.subplots_adjust(wspace=0)
    
    # Coordinates are [left, bottom, width, height] in figure fraction (0 to 1)
    cax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
    cbar = plt.colorbar(sc, cax=cax, ticks=np.arange(n_cluster) + 1, orientation='horizontal')
    cbar.set_alpha(1)
    cbar.solids.set(alpha=1)
    cbar.set_label("Cluster Membership")
    # fig.suptitle("$K$-Means Clustering", fontsize="xx-large", y=0.90)

    fig_root.mkdir(parents=True, exist_ok=True)
    fig_name = fig_root / f"{cache_key}_{init_str}_{dim}_proj"
    plt.savefig(fig_name.with_suffix(".pdf"), bbox_inches='tight')
    plt.savefig(fig_name.with_suffix(".png"), bbox_inches='tight')
    plt.show()
    
    return labels, idx


# In[ ]:


def analyze(bed, mask, env, covariates, n_component_rsvd, n_cluster, n_sample, n_component, cache_key, manifest, labels_override = None):
    bed_op, frobenius_sq = preprocess(bed, mask, env, covariates, manifest)
    U, S, Vt, projections = compute_rsvd(bed_op, frobenius_sq, n_component_rsvd, cache_key)
    
    rng = np.random.default_rng(42)
    labels_kmeans, idx_kmeans = visualize_clustering(rng, U, S, Vt, projections, n_cluster, n_sample, n_component, None, "k-means++", "3d", cache_key, labels_override)
    return labels_kmeans, idx_kmeans


# In[ ]:


mask = pd.notna(fam_full.iid)
env = covariates = np.ones(bed_full.shape[:-1])
n_component_rsvd = 500
n_cluster = 5
n_sample = np.sum(mask)
n_component = 10
labels, idx = analyze(bed_full, mask, env, covariates, n_component_rsvd, n_cluster, n_sample, n_component, "full", False)


# In[ ]:


split_unhot_doh = pd.read_parquet('ohe_sdoh_data_unhot/split_unhot_doh.parquet')
fam_full['person_id'] = fam_full.iid.astype(int)
environments = fam_full.merge(split_unhot_doh, how='left', on='person_id', validate='one_to_one')


# In[ ]:


env_name = 'qq40192410'
mask = pd.notna(environments[env_name])
env = environments.loc[mask, env_name].cat.codes.to_numpy()
covariates = np.column_stack((np.ones(np.sum(mask)), env))
n_component_rsvd = 500
n_cluster = 5
n_sample = np.sum(mask)
n_component = 10
labels, idx = analyze(bed_full, mask, env, covariates, n_component_rsvd, n_cluster, n_sample, n_component, env_name, True)


# In[ ]:


mask = pd.notna(environments[env_name])
env = np.ones_like(environments.loc[mask, env_name].cat.codes.to_numpy())
covariates = np.ones(np.sum(mask))
n_component_rsvd = 500
n_cluster = 5
n_sample = np.sum(mask)
n_component = 10
labels, idx = analyze(bed_full, mask, env, covariates, n_component_rsvd, n_cluster, n_sample, n_component, f"{env_name}_ctrl", True)


# In[ ]:


mask = pd.notna(environments[env_name])
env = np.ones_like(environments.loc[mask, env_name].cat.codes.to_numpy())
labels_override = environments.loc[mask, env_name].cat.codes.to_numpy()
covariates = np.ones(np.sum(mask))
n_component_rsvd = 500
n_cluster = 5
n_sample = np.sum(mask)
n_component = 10
labels, idx = analyze(bed_full, mask, env, covariates, n_component_rsvd, n_cluster, n_sample, n_component, f"{env_name}_raw", True, labels_override = labels_override)


# In[ ]:


np.sum(mask)


# In[ ]:


bed_full.shape

