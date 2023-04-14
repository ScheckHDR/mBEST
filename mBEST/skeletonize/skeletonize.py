#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
# cimport numpy as cnp
# cnp.import_array()


"""
    Taken from scikit-image's implementation:
    https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/morphology/_skeletonize_cy.pyx
    
    Recompiling from source and removal of "nogil" offers considerable speed boost.
"""


def skeletonize(mask):
    """Optimized parts of the Zhang-Suen [1]_ skeletonization.
    Iteratively, pixels meeting removal criteria are removed,
    till only the skeleton remains (that is, no further removable pixel
    was found).

    Parameters
    ----------
    image : numpy.ndarray
        A binary image containing the objects to be skeletonized. '1'
        represents foreground, and '0' represents background.

    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.

    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
           T. Y. Zhang and C. Y. Suen, Communications of the ACM,
           March 1984, Volume 27, Number 3.

    """

    lut = \
      [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
       0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
       0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
       0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
       1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
       0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    # we copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below
    skeleton = np.zeros((mask.shape[0]+2,mask.shape[1]+2), dtype=np.uint8)
    skeleton[1:-1, 1:-1] = mask > 0
    cleaned_skeleton = skeleton.copy()

    # skeleton = _skeleton
    # cleaned_skeleton = _cleaned_skeleton

    pixel_removed = True

    # the algorithm reiterates the thinning until
    # no further thinning occurred (variable pixel_removed set)

    connectivity_kernel = np.array([
        [1,2,4],
        [128,0,8],
        [64,32,16]
    ],dtype=np.uint8)

    while pixel_removed:
        pixel_removed = False

        for first_pass in [True,False]:
            # there are two phases, in the first phase, pixels labeled (see below)
            # 1 and 3 are removed, in the second 2 and 3
            for row,col in np.argwhere(skeleton):
                # Only operate on set pixels.

                neighbors = lut[np.sum(skeleton[row-1:row+2,col-1:col+2] * connectivity_kernel)]

                if ((neighbors == 1 and first_pass) or
                        (neighbors == 2 and not first_pass) or
                        (neighbors == 3)):
                    # Unset the pixel.
                    cleaned_skeleton[row, col] = 0
                    pixel_removed = True

            # once a step has been processed, the original skeleton
            # is overwritten with the cleaned version
            skeleton[:, :] = cleaned_skeleton[:, :]

    return skeleton[1:-1, 1:-1].astype(bool)
