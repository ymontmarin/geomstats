"""The vector space of symmetric matrices.

Lead author: Yann Thanwerdas.
"""

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.matrices import Matrices, MatricesMetric


class SymmetricMatrices(VectorSpace):
    """Class for the vector space of symmetric matrices of size n.

    Parameters
    ----------
    n : int
        Integer representing the shapes of the matrices: n x n.
    """

    def __init__(self, n, equip=True):
        super().__init__(dim=int(n * (n + 1) / 2), shape=(n, n), equip=equip)
        self.n = n

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MatricesMetric

    def _create_basis(self):
        """Compute the basis of the vector space of symmetric matrices."""
        indices, values = [], []
        k = -1
        for row in range(self.n):
            for col in range(row, self.n):
                k += 1
                if row == col:
                    indices.append((k, row, row))
                    values.append(1.0)
                else:
                    indices.extend([(k, row, col), (k, col, row)])
                    values.extend([1.0, 1.0])

        return gs.array_from_sparse(indices, values, (k + 1, self.n, self.n))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a matrix is symmetric.

        Parameters
        ----------
        point : array-like, shape=[.., n, n]
            Point to test.
        atol : float
            Tolerance to evaluate equality with the transpose.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        belongs = super().belongs(point)
        if gs.any(belongs):
            is_symmetric = Matrices.is_symmetric(point, atol)
            return gs.logical_and(belongs, is_symmetric)
        return belongs

    def projection(self, point):
        """Make a matrix symmetric, by averaging with its transpose.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        sym : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        return Matrices.to_symmetric(point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample a symmetric matrix.

        Samples from a uniform distribution in a box and then converts to symmetric.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., n, n]
           Sample.
        """
        sample = super().random_point(n_samples, bound)
        return Matrices.to_symmetric(sample)

    @staticmethod
    def to_vector(point):
        """Convert a symmetric matrix into a vector.

        Parameters
        ----------
        mat : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.
        """
        return gs.triu_to_vec(point)

    @staticmethod
    def from_vector(vec):
        """Convert a vector into a symmetric matrix.

        Parameters
        ----------
        vec : array-like, shape=[..., n(n+1)/2]
            Vector.

        Returns
        -------
        mat : array-like, shape=[..., n, n]
            Symmetric matrix.
        """
        vec_dim = vec.shape[-1]
        mat_dim = (gs.sqrt(8.0 * vec_dim + 1) - 1) / 2
        if mat_dim != int(mat_dim):
            raise ValueError(
                "Invalid input dimension, it must be of the form"
                "(n_samples, n * (n + 1) / 2)"
            )
        mat_dim = int(mat_dim)
        shape = (mat_dim, mat_dim)
        mask = 2 * gs.ones(shape) - gs.eye(mat_dim)
        indices = list(zip(*gs.triu_indices(mat_dim)))
        if gs.ndim(vec) == 1:
            upper_triangular = gs.array_from_sparse(indices, vec, shape)
        else:
            upper_triangular = gs.stack(
                [gs.array_from_sparse(indices, data, shape) for data in vec]
            )

        mat = Matrices.to_symmetric(upper_triangular) * mask
        return mat
