import numpy
import scipy.linalg
from scipy.stats import special_ortho_group

def randomRotationPowers(n, powers, seed=None, unpermute=False):
    RR = special_ortho_group(n, seed = seed).rvs()
    # Reduce the matrix to canonical (block-diag) form using schur decomp
    T, W = scipy.linalg.schur(RR, output='real')
    # Now T is not exactly block diagonal due to numerical approximations.
    # However, its eigenvalues should have the right distribution: let's
    # read the angles off the diagonal and use them to construct a perfectly
    # block diagonal form.
    RA = numpy.arccos(numpy.clip(numpy.diag(T), -1, 1))[0:n:2]
    if unpermute:
        # If unpermuting, first figure out full rotation matrix
        B = [numpy.cos([[a, a + numpy.pi/2], [a - numpy.pi/2, a]]) for a in RA]
        BD = scipy.linalg.block_diag(*B)
        fullrot = numpy.dot(numpy.dot(W, BD), W.transpose())
        # Then sort the rows by max value
        biggest_first = numpy.argsort(-numpy.amax(fullrot, axis=1))
        # In greedy order, assign permutation order
        cols_seen = set()
        permutation = numpy.zeros((n, n))
        for row in biggest_first:
            # For each row, find the max col to which it is closest
            biggest_cols = numpy.argsort(-fullrot[row])
            for col in biggest_cols:
                if col not in cols_seen:
                    # Bring the row back to that slot if not taken
                    permutation[col,row] = 1
                    cols_seen.add(col)
                    break
    else:
        permutation = numpy.eye(n)
    # Now form the requested powers of W * (RA^p) * W'
    result = []
    for p in powers:
        A = [a * p for a in RA]
        B = [numpy.cos([[a, a + numpy.pi/2], [a - numpy.pi/2, a]]) for a in A]
        BD = scipy.linalg.block_diag(*B)
        result.append(numpy.dot(permutation,
            numpy.dot(numpy.dot(W, BD), W.transpose())))
    return result
