import torch
import math

def B_batch(x, grid, k=0, extend=False, device='cpu'):
    def bernstein_basis(n, v, t):
        return math.comb(n, v) * (t ** v) * ((1 - t) ** (n - v))

    if extend:
        grid = torch.cat([grid[:, :1].repeat(1, k), grid, grid[:, -1:].repeat(1, k)], dim=1)

    basis = torch.zeros((grid.shape[0], grid.shape[1] - 1, x.shape[1]), device=device)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1] - 1):
            basis[i, j, :] = bernstein_basis(grid.shape[1] - 2, j, x[i, :])
            
    # print(f"basis.size(): {basis.size()}")
    return basis

def coef2curve(x_eval, grid, coef, k, device="cpu"):
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval

def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device), driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    return coef.to(device)
