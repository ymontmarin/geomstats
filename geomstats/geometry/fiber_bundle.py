"""Class for (principal) fiber bundles.

Lead author: Nicolas Guigui.
"""

import sys
from abc import ABC

from scipy.optimize import minimize

import geomstats.backend as gs
from geomstats.vectorization import get_batch_shape


class FiberBundle(ABC):
    """Class for (principal) fiber bundles.

    This class implements abstract methods for fiber bundles, or more
    generally manifolds, with a submersion map, or a right Lie group action.

    Parameters
    ----------
    group : LieGroup
        Group that acts on the total space by the right.
        Optional. Default : None.
        Either the group or the group action must be given.
    group_action : callable
        Right group action. It must take as input a point of the total space
        and an element of the group, and return a point of the total space.
    """

    def __init__(
        self,
        total_space,
        group=None,
        group_action=None,
        group_dim=None,
    ):
        self.total_space = total_space
        self.group = group

        if group_action is None and group is not None:
            group_action = group.compose
        if group_dim is None and group is not None:
            group_dim = group.dim
        self.group_dim = group_dim
        self.group_action = group_action

    @staticmethod
    def riemannian_submersion(point):
        """Project a point to base manifold.

        This is the projection of the fiber bundle, defined on the total
        space, with values in the base manifold. This map is surjective.
        By default, the base manifold is not explicit but is identified with a
        local section of the fiber bundle, so the submersion is the identity
        map.

        Parameters
        ----------
        point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        projection : array-like, shape=[..., {base_dim, [n, m]}]
            Point of the base manifold.
        """
        return gs.copy(point)

    @staticmethod
    def lift(point):
        """Lift a point to total space.

        This is a section of the fiber bundle, defined on the base manifold,
        with values in the total space. It means that submersion applied after
        lift results in the identity map. By default, the base manifold
        is not explicit but is identified with a section of the fiber bundle,
        so the lift is the identity map.

        Parameters
        ----------
        point : array-like, shape=[..., {base_dim, [n, m]}]
            Point of the base manifold.

        Returns
        -------
        lift : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.
        """
        return gs.copy(point)

    def tangent_riemannian_submersion(self, tangent_vec, base_point):
        """Project a tangent vector to base manifold.

        This is the differential of the projection of the fiber bundle,
        defined on the tangent space of a point of the total space,
        with values in the tangent space of the projection of this point in the
        base manifold. This map is surjective. By default, the base manifold
        is not explicit but is identified with a horizontal section of the
        fiber bundle, so the tangent submersion is the horizontal projection.

        Parameters
        ----------
        tangent_vec :  array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector to the total space at `base_point`.
        base_point: array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        projection: array-like, shape=[..., {base_dim, [n, m]}]
            Tangent vector to the base manifold.
        """
        return self.horizontal_projection(tangent_vec, base_point)

    def align(self, point, base_point, max_iter=25, verbose=False, tol=gs.atol):
        """Align point to base_point.

        Find the optimal group element g such that the base point and
        point.g are well positioned, meaning that the total space distance is
        minimized. This also means that the geodesic joining the base point
        and the aligned point is horizontal. By default, this is solved by a
        gradient descent in the Lie algebra.

        Parameters
        ----------
        point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the manifold.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the manifold.
        max_iter : int
            Maximum number of gradient steps.
            Optional, default : 25.
        verbose : bool
            Verbosity level.
            Optional, default : False.
        tol : float
            Tolerance for the stopping criterion.
            Optional, default : backend atol

        Returns
        -------
        aligned : array-like, shape=[..., {total_space.dim, [n, m]}]
            Action of the optimal g on point.
        """
        group = self.group
        group_action = self.group_action

        batch_shape = get_batch_shape(self.total_space.point_ndim, point, base_point)
        max_shape = batch_shape + (self.group_dim,)

        if group is not None:

            def wrap(param):
                """Wrap a parameter vector to a group element."""
                algebra_elt = gs.reshape(
                    gs.cast(gs.array(param), dtype=base_point.dtype), max_shape
                )
                algebra_elt = group.lie_algebra.matrix_representation(algebra_elt)
                group_elt = group.exp(algebra_elt)
                return self.group_action(point, group_elt)

        elif group_action is not None:

            def wrap(param):
                vector = gs.reshape(gs.array(param), max_shape)
                vector = gs.cast(vector, dtype=base_point.dtype)
                return group_action(vector, point)

        else:
            raise ValueError("Either the group of its action must be known")

        objective_with_grad = gs.autodiff.value_and_grad(
            lambda param: gs.sum(
                self.total_space.metric.squared_dist(wrap(param), base_point)
            ),
            to_numpy=True,
        )

        tangent_vec = gs.flatten(gs.random.rand(*max_shape))
        res = minimize(
            objective_with_grad,
            tangent_vec,
            method="L-BFGS-B",
            jac=True,
            options={"disp": verbose, "maxiter": max_iter},
            tol=tol,
        )

        return wrap(res.x)

    def horizontal_projection(self, tangent_vec, base_point):
        r"""Project to horizontal subspace.

        Compute the horizontal component of a tangent vector at a
        base point by removing the vertical component,
        or by computing a horizontal lift of the tangent projection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the total space.

        Returns
        -------
        horizontal : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal component of `tangent_vec`.
        """
        caller_name = sys._getframe().f_back.f_code.co_name
        if not caller_name == "vertical_projection":
            try:
                return tangent_vec - self.vertical_projection(tangent_vec, base_point)
            except NotImplementedError:
                pass

        return self.horizontal_lift(
            self.tangent_riemannian_submersion(tangent_vec, base_point),
            fiber_point=base_point,
        )

    def vertical_projection(self, tangent_vec, base_point):
        r"""Project to vertical subspace.

        Compute the vertical component of a tangent vector :math:`w` at a
        base point :math:`P` by removing the horizontal component.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector to the total space at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the total space.

        Returns
        -------
        vertical : array-like, shape=[..., {total_space.dim, [n, m]}]
            Vertical component of `tangent_vec`.
        """
        caller_name = sys._getframe().f_back.f_code.co_name
        if caller_name == "horizontal_projection":
            raise NotImplementedError

        return tangent_vec - self.horizontal_projection(tangent_vec, base_point)

    def is_horizontal(self, tangent_vec, base_point, atol=gs.atol):
        """Evaluate if the tangent vector is horizontal at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the manifold.
            Optional, default: None.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol

        Returns
        -------
        is_horizontal : bool
            Boolean denoting if tangent vector is horizontal.
        """
        return gs.all(
            gs.isclose(
                tangent_vec,
                self.horizontal_projection(tangent_vec, base_point),
                atol=atol,
            ),
            axis=(-2, -1),
        )

    def is_vertical(self, tangent_vec, base_point, atol=gs.atol):
        """Evaluate if the tangent vector is vertical at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point on the manifold.
            Optional, default: None.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_vertical : bool
            Boolean denoting if tangent vector is vertical.
        """
        return gs.all(
            gs.isclose(
                0.0,
                self.tangent_riemannian_submersion(tangent_vec, base_point),
                atol=atol,
            ),
            axis=(-2, -1),
        )

    def horizontal_lift(self, tangent_vec, base_point=None, fiber_point=None):
        """Lift a tangent vector to a horizontal vector in the total space.

        It means that horizontal lift is the inverse of the restriction of the
        tangent submersion to the horizontal space at point, where point must
        be in the fiber above the base point. By default, the base manifold
        is not explicit but is identified with a horizontal section of the
        fiber bundle, so the horizontal lift is the horizontal projection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {base_dim, [n, m]}]
        fiber_point : array-like, shape=[..., {ambient_dim, [n, m]}]
            Point of the total space.
            Optional, default : None. The `lift` method is used to compute a
            point at which to compute a tangent vector.
        base_point : array-like, shape=[..., {base_dim, [n, m]}]
            Point of the base space.
            Optional, default : None. In this case, point must be given,
            and `submersion` is used to compute the base_point if needed.

        Returns
        -------
        horizontal_lift : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal tangent vector to the total space at point.
        """
        if base_point is None and fiber_point is None:
            raise ValueError(
                "Either a point (of the total space) or a "
                "base point (of the base manifold) must be "
                "given."
            )

        if fiber_point is None:
            fiber_point = self.lift(base_point)

        return self.horizontal_projection(tangent_vec, fiber_point)

    def integrability_tensor(self, tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the fundamental tensor A of the submersion.

        The fundamental integrability tensor A is defined for tangent vectors
        :math:`X = tangent\_vec\_a` and :math:`Y = tangent\_vec\_b` of the
        total space by [ONeill]_ as
        :math:`A_X Y = ver\nabla_{hor X} (hor Y) + hor \nabla_{hor X}( ver Y)`
        where :math:`hor, ver` are the horizontal and vertical projections
        and :math:`\nabla` is the connection of the total space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        vector : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`, result of the A tensor applied to
            `tangent_vec_a` and `tangent_vec_b`.

        References
        ----------
        .. [ONeill]  O’Neill, Barrett. The Fundamental Equations of a
            Submersion, Michigan Mathematical Journal 13, no. 4
            (December 1966): 459–69. https://doi.org/10.1307/mmj/1028999604.
        """
        raise NotImplementedError

    def integrability_tensor_derivative(
        self,
        horizontal_vec_x,
        horizontal_vec_y,
        nabla_x_y,
        tangent_vec_e,
        nabla_x_e,
        base_point,
    ):
        r"""Compute the covariant derivative of the integrability tensor A.

        The covariant derivative :math:`\nabla_X (A_Y E)` in total space is
        necessary to compute the covariant derivative of the directional
        curvature in a submersion. The components :math:`\nabla_X (A_Y E)`
        and :math:`A_Y E` are computed at base-point :math:`P = base\_point`
        for horizontal vector fields :math:`X, Y` extending the values
        given in argument :math:`X|_P = horizontal\_vec\_x`,
        :math:`Y|_P = horizontal\_vec\_y` and a general vector field
        :math:`E` extending :math:`E|_x = tangent\_vec\_e`
        in a neighborhood of x with covariant derivatives
        :math:`\nabla_X Y |_P = nabla_x y` and
        :math:`\nabla_X E |_P = nabla_x e`.

        Parameters
        ----------
        horizontal_vec_x : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal tangent vector at `base_point`.
        horizontal_vec_y : array-like, shape=[..., {total_space.dim, [n, m]}]
            Horizontal tangent vector at `base_point`.
        nabla_x_y : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        tangent_vec_e : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        nabla_x_e : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., {total_space.dim, [n, m]}]
            Point of the total space.

        Returns
        -------
        nabla_x_a_y_e : array-like, shape=[..., {total_space.dim, [n, m]}]
            Tangent vector at `base_point`, result of :math:`\nabla_X
            (A_Y E)`.
        a_y_e : array-like, shape=[..., {ambient_dim, [n, n]}]
            Tangent vector at `base_point`, result of :math:`A_Y E`.

        References
        ----------
        .. [Pennec] Pennec, Xavier. Computing the curvature and its gradient
            in Kendall shape spaces. Unpublished.
        """
        raise NotImplementedError
