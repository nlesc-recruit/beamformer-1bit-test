import sys

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    try:
        fname = sys.argv[1]
    except IndexError:
        print("Provide path to BF bin file")
        sys.exit(1)

    nx, ny, nz = 36, 36, 30
    nframe = 8041
    nsample = 524288

    bf = np.fromfile(fname, dtype='int32')
    print(bf.nbytes)

    bf = bf.reshape(2, nz, ny, nx, nframe) 
    bf = bf[0] + 1j*bf[1]
    bf = np.sum(np.abs(bf)**2, axis=-1)

    bf = 20 * np.log10(bf / bf.max() + 1e-12)

    xy = bf.max(axis=0)
    xz = bf.max(axis=1)
    yz = bf.max(axis=2)

    fig, axes = plt.subplots(figsize=(12, 4), ncols=3)
    axes = axes.flatten()

    cmap = 'gist_heat'

    axes[0].imshow(yz, cmap=cmap)  #, vmin=-5, vmax=0)
    axes[0].set_title('yz')

    axes[2].imshow(xy, cmap=cmap)  #, vmin=-5, vmax=0)
    axes[2].set_title('xy')

    axes[1].imshow(xz, cmap=cmap)  #, vmin=-5, vmax=0)
    axes[1].set_title('xz')

    plt.savefig('bf.png')
    plt.show()
