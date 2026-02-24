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
import seaborn as sns


# In[ ]:


my_bucket = os.getenv('WORKSPACE_BUCKET')
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
genome_dir = Path("genomic_files/genome")
rng = np.random.default_rng(42)


# In[ ]:


genotype = read_plink1_bin(str(genome_dir / "qc_data_unrelated.bed"), verbose=False)
genotype.variant


# In[ ]:


# 1. Calculate how many new columns we need
target_size = 454207
current_size = genotype.sizes['variant']  # 56961
num_to_add = target_size - current_size

# 2. Generate random indices for the additional columns
# We sample from the existing range [0, current_size)
random_indices = rng.integers(0, current_size, size=num_to_add)

# 3. Create the full index list: Original Indices + Random Indices
# This preserves the original data order at the start
full_indices = np.concatenate([np.arange(current_size), random_indices])

# 4. Use isel to create the extended DataArray
# This operation is lazy; it does not compute the data immediately
extended_genotype = genotype.isel(variant=full_indices)
assert extended_genotype.shape == (genotype.sizes['sample'], target_size)

extended_genotype.variant


# In[ ]:


new_variants = [f"variant{i}" for i in range(target_size)]
extended_genotype = extended_genotype.assign_coords(variant=new_variants)

# 1. Get the new size from your extended object
n_variants = extended_genotype.sizes['variant']

# --- Generate Random Data ---

# Chrom: integers 1-22 converted to string (matching your <U2 dtype)
new_chrom = rng.integers(1, 23, size=n_variants).astype(str)

# Pos: random genomic positions (e.g., 1 to 100 million)
possible_positions = 200_000_000
new_pos = rng.choice(possible_positions, size=target_size, replace=False)

# cm: random float values for centimorgans
new_cm = rng.uniform(0, 100, size=n_variants)

# Alleles: Random A, C, G, T for reconstructing the SNP ID
bases = np.array(['A', 'C', 'G', 'T'])
# Generate random a0
new_a0 = rng.choice(bases, size=n_variants)
# Generate a1 such that it is never equal to a0
# We do this by choosing a random 'offset' (1, 2, or 3) and shifting a0
offsets = rng.integers(1, 4, size=n_variants) # 1, 2, or 3
new_a1_indices = (np.searchsorted(bases, new_a0) + offsets) % 4
new_a1 = bases[new_a1_indices]

# SNP: Reconstruct the ID string to match the format 'chrom:pos:a0:a1'
# (Using a list comprehension is fast enough for ~450k items)
new_snp = [
    f"{c}:{p}:{a0}:{a1}" 
    for c, p, a0, a1 in zip(new_chrom, new_pos, new_a0, new_a1)
]

# --- Assign to the DataArray ---

extended_genotype = extended_genotype.assign_coords({
    "chrom": ("variant", new_chrom),
    "pos":   ("variant", new_pos),
    "cm":    ("variant", new_cm),
    "snp":   ("variant", new_snp),
    "a0":    ("variant", new_a0),
    "a1":    ("variant", new_a1),
})

# Optional: Verify the new coordinates
extended_genotype.variant


# In[ ]:


# --- F. Verification ---
print(f"Total Variants: {extended_genotype.sizes['variant']}")

# 1. Check for duplicates
import pandas as pd
if pd.Series(new_snp).is_unique:
    print("✅ All SNP IDs are unique.")
else:
    print("❌ Duplicate SNP IDs found!")

# 2. Check for invalid alleles
if np.any(new_a0 == new_a1):
    print("❌ Found sites where a0 == a1")
else:
    print("✅ All sites have distinct alleles (a0 != a1).")


# In[ ]:


imputed_base = genome_dir / f"qc_data_unrelated_{target_size}"
write_plink1_bin(extended_genotype, f"{imputed_base}.bed", verbose=True)


# In[ ]:




