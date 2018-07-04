#pragma once

#define IDX_2D(i, j, Nj) ((i) * (Nj) + (j))
#define UNRAVEL_2D_i(idx, Nj) ((idx) / (Nj))
#define UNRAVEL_2D_j(idx, Nj) ((idx) % (Nj))

#define IDX_2D_colmajor(i, j, Ni) ((j) * (Ni) + (i))
#define UNRAVEL_2D_colmajor_i(idx, Ni) ((idx) % (Ni))
#define UNRAVEL_2D_colmajor_j(idx, Ni) ((idx) / (Ni))

#define IDX_3D(i, j, k, Nj, Nk) ((i) * (Nj) * (Nk) + (j) * (Nk) + (k))
#define UNRAVEL_3D_i(idx, Nj, Nk) ((idx) / (Nj * Nk))
#define UNRAVEL_3D_j(idx, Nj, Nk) ((idx) / (Nk))
#define UNRAVEL_3D_k(idx, Nj, Nk) ((idx) % (Nk))
