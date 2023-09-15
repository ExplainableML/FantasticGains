import numpy as np
import math

# get a matrix with two cols and 4 rows
mat = np.array([[0.0437, 0.676],
                 [0.092, 0.495],
                 [0.556, 0.033],
                 [0.084, 0.838]])
print(f'mat: {mat}')

# standardize the matrix
mat_norm = (mat - mat.mean(axis=0)) / mat.std(axis=0)
print(f'mat_norm: {mat_norm}')
# round mat_norm to 3 decimals
mat_norm = np.round(mat_norm, 3)
print(f'mat_norm: {mat_norm}')
print(f'mat_norm mean: {mat_norm.mean(axis=0)}')
print(f'mat_norm std: {mat_norm.std(axis=0)}')

# get the sample covariance matrix without np
cov_mat = np.dot(mat_norm.T, mat_norm) / (mat_norm.shape[0])
print(f'cov_mat: {cov_mat}')

# get the eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print(f'eig_vals: {eig_vals}')
print(f'eig_vecs: {eig_vecs}')

# get the normalized eigenvalues
norm_eig_vals = eig_vals / sum(eig_vals)
print(f'norm_eig_vals: {norm_eig_vals}')

# get the explained variance
explained_variance = [(i / sum(eig_vals)) * 100 for i in eig_vals]
print(f'explained_variance: {explained_variance}')

# project the data
projected_1 = mat_norm.dot(eig_vecs.T[0])
projected_2 = mat_norm.dot(eig_vecs.T[1])
print(f'projected_1: {projected_1}')
print(f'projected_2: {projected_2}')
