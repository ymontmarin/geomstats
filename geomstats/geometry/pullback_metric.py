"""Classes for the pullback metric.

Lead author: Nina Miolane.
"""
import abc
import itertools
import math
from functools import partial

import joblib

import geomstats.backend as gs
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric


class PullbackMetricOld(RiemannianMetric):
    r"""Pullback metric.

    Let :math:`f` be an immersion :math:`f: M \rightarrow N`
    of one manifold :math:`M` into the Riemannian manifold :math:`N`
    with metric :math:`g`.
    The pull-back metric :math:`f^*g` is defined on :math:`M` for a
    base point :math:`p` as:
    :math:`(f^*g)_p(u, v) = g_{f(p)}(df_p u , df_p v)
    \quad \forall u, v \in T_pM`

    Note
    ----
    The pull-back metric is currently only implemented for an
    immersion into the Euclidean space, i.e. for

    :math:`N=\mathbb{R}^n`.

    Parameters
    ----------
    dim : int
        Dimension of the underlying manifold.
    embedding_dim : int
        Dimension of the embedding Euclidean space.
    immersion : callable
        Map defining the immersion into the Euclidean space.
    """

    def __init__(
        self,
        dim,
        immersion,
        embedding_space=None,
        embedding_dim=None,
        jacobian_immersion=None,
        tangent_immersion=None,
    ):
        super(PullbackMetric, self).__init__(dim=dim)
        if embedding_space is None:
            embedding_space = EuclideanMetric(embedding_dim)

        self.embedding_space = embedding_space
        self.embedding_metric = embedding_space.metric

        self.immersion = immersion

        if jacobian_immersion is None:
            jacobian_immersion = gs.autodiff.jacobian(immersion)

        if tangent_immersion is None:

            def _tangent_immersion(v, x):
                # TODO: More general for shape
                return gs.einsum("...ij,...j->...j", jacobian_immersion(x), v)

            tangent_immersion = _tangent_immersion

        self.tangent_immersion = tangent_immersion

        self.tangent_immersion = _tangent_immersion
        self._hessian_immersion = None

    def metric_matrix(self, base_point=None, n_jobs=1, **joblib_kwargs):
        r"""Metric matrix at the tangent space at a base point.

        Let :math:`f` be the immersion
        :math:`f: M \rightarrow \mathbb{R}^n` of the manifold
        :math:`M` into the Euclidean space :math:`\mathbb{R}^n`.
        The elements of the metric matrix at a base point :math:`p`
        are defined as:
        :math:`(f*g)_{ij}(p) = <df_p e_i , df_p e_j>`,
        for :math:`e_i, e_j` basis elements of :math:`M`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        immersed_base_point = self.immersion(base_point)
        jacobian_immersion = self.jacobian_immersion(base_point)
        basis_elements = gs.eye(self.dim)

        @joblib.delayed
        @joblib.wrap_non_picklable_objects
        def pickable_inner_product(i, j):
            immersed_basis_element_i = gs.squeeze(
                gs.matvec(jacobian_immersion, basis_elements[i])
            )
            immersed_basis_element_j = gs.squeeze(
                gs.matvec(jacobian_immersion, basis_elements[j])
            )
            return self.embedding_metric.inner_product(
                immersed_basis_element_i,
                immersed_basis_element_j,
                base_point=immersed_base_point,
            )

        pool = joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)
        out = pool(
            pickable_inner_product(i, j)
            for i, j in itertools.product(range(self.dim), range(self.dim))
        )
        metric_mat = gs.reshape(gs.array(out), (-1, self.dim, self.dim))
        return metric_mat[0] if base_point.ndim == 1 else metric_mat

    def _hessian_immersion_func(self):
        """Compute the Hessian of the immersion.

        Returns
        -------
        hessian_immersion : list of callable
            List of embedding_dim hessians of the scalar
            functions defining the components of the immersion.
        """

        def _immersion(base_point, a):
            """Compute the component a of the immersion at a base point.

            Parameters
            ----------
            base_point : array-like, shape=[..., dim]
                Base point.
            a : int
                Index of the component of the immersion.
            """
            return gs.array([self.immersion(base_point)[a]])

        hessians = [
            gs.autodiff.hessian(partial(_immersion, a=a))
            for a in range(self.embedding_metric.dim)
        ]
        return hessians

    def hessian_immersion(self, base_point):
        """Compute the Hessian of the immersion.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        hessian_immersion : array-like, shape=[..., embedding_dim, dim, dim]
            Hessian at the base point
        """
        if self._hessian_immersion is None:
            self._hessian_immersion = self._hessian_immersion_func()

        hessian_values = [hes(base_point) for hes in self._hessian_immersion]
        return gs.stack(hessian_values, axis=0)

    def inner_product_derivative_matrix(self, base_point=None):
        r"""Compute the inner-product derivative matrix.

        The derivative of the metrix matrix is given by
        :math:`\partial_k g_{ij}(p)`
        where :math:`p` is the base_point.

        The index k of the derivation is last.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod_deriv_mat : array-like, shape=[..., dim, dim, dim]
            Inner-product derivative matrix, where the index of the derivation
            is last: :math:`mat_{ij}_k = \partial_k g_{ij}`.
        """
        initial_ndim = base_point.ndim
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        inner_prod_deriv_mats = []
        for point in base_point:
            jacobian_ai = self.jacobian_immersion(point)
            if self.dim == 1 and jacobian_ai.ndim > 2:
                jacobian_ai = gs.squeeze(jacobian_ai, axis=-1)

            hessian_aij = self.hessian_immersion(point)
            if self.dim == 1 and hessian_aij.ndim > 3:
                hessian_aij = gs.squeeze(hessian_aij, axis=-1)

            inner_prod_deriv_mat = gs.einsum(
                "aki,aj->ijk", hessian_aij, jacobian_ai
            ) + gs.einsum("akj,ai->ijk", hessian_aij, jacobian_ai)
            inner_prod_deriv_mats.append(inner_prod_deriv_mat)

        inner_prod_deriv_mat = gs.stack(inner_prod_deriv_mats, axis=0)
        return inner_prod_deriv_mat[0] if initial_ndim == 1 else inner_prod_deriv_mat

    def second_fundamental_form(self, base_point):
        r"""Compute the second fundamental form.

        In the case of an immersion f, the second fundamental form is
        given by the formula:
        :math:`\RN{2}(p)_{ij}^\alpha = \partial_{i j}^2 f^\alpha(p)`
        :math:`  -\Gamma_{i j}^k(p) \partial_k f^\alpha(p)`
        at base_point :math:`p`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        second_fundamental_form : array-like, shape=[..., embedding_dim, dim, dim]
            Second fundamental form :math:`\RN{2}(p)_{ij}^\alpha` where the
             :math:`\alpha` index is first.
        """
        initial_ndim = base_point.ndim
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        second_fundamental_forms = []
        for point in base_point:
            christoffels = self.christoffels(point)

            jacobian_ai = self.jacobian_immersion(point)
            if self.dim == 1 and jacobian_ai.ndim > 2:
                jacobian_ai = gs.squeeze(jacobian_ai, axis=-1)

            hessian_aij = self.hessian_immersion(point)
            if self.dim == 1 and hessian_aij.ndim > 3:
                hessian_aij = gs.squeeze(hessian_aij, axis=-1)

            second_fundamental_form_aij = []
            for a in range(self.embedding_metric.dim):
                jacobian_a = jacobian_ai[a]
                hessian_a = hessian_aij[a]

                second_fundamental_form_a = hessian_a - gs.einsum(
                    "kij,k->ij", christoffels, jacobian_a
                )
                second_fundamental_form_aij.append(second_fundamental_form_a)

            second_fundamental_forms_aij = gs.stack(second_fundamental_form_aij, axis=0)
            second_fundamental_forms.append(second_fundamental_forms_aij)
        second_fundamental_forms = gs.stack(second_fundamental_forms, axis=0)
        return (
            second_fundamental_forms[0]
            if initial_ndim == 1
            else second_fundamental_forms
        )

    def mean_curvature_vector(self, base_point):
        r"""Compute the mean curvature vector.

        The mean curvature vector is defined at base point :math:`p` by
        :math:`H_p^\alpha= \frac{1}{d} (f^{*}g)_{p}^{ij} (\partial_{i j}^2 f^\alpha(p)`
        :math:`  -\Gamma_{i j}^k(p) \partial_k f^\alpha(p))`
        where :math:`f^{*}g` is the pullback of the metric :math:`g` by the
        immersion :math:`f`.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        mean_curvature_vector : array-like, shape=[..., embedding_dim]
            Mean curvature vector.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        mean_curvature = []
        for point in base_point:
            second_fund_form = self.second_fundamental_form(point)
            cometric = self.cometric_matrix(point)
            mean_curvature.append(gs.einsum("ij,aij->a", cometric, second_fund_form))

        return gs.stack(mean_curvature, axis=0)


class PullbackMetric(RiemannianMetric, abc.ABC):
    """
    Pullback metric via a diffeomorphism.

    Parameters
    ----------
    dim : int
        Dimension.
    """

    def __init__(self, dim, shape):
        super(PullbackMetric, self).__init__(dim=dim, shape=shape)
        self._embedding_space = None
        self._jacobian = None
        self.embedding_metric = self.embedding_space.metric

    @abc.abstractmethod
    def _create_embedding_space(self):
        pass

    @property
    def embedding_space(self):
        """Lie algebra of the Matrix Lie group.

        Returns
        -------
        mat : array-like, shape=[..., *shape_e, *shape_m]
            Inner-product matrix.
        """
        if self._embedding_space is None:
            self._embedding_space = self._create_embedding_space()
        return self._embedding_space

    @abc.abstractmethod
    def immersion(self, base_point):
        r"""Jacobian of the immersion at base point.

        base_point : array-like, shape=[..., *shape_m]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., *shape_e, *shape_m]
            Inner-product matrix.
        """
        pass

    def jacobian(self, base_point):
        r"""Jacobian of the immersion at base point.

        Let :math:`f` be the immersion
        :math:`f: M \rightarrow N` of the manifold
        :math:`M` into the manifold `N`.

        Parameters
        ----------
        base_point : array-like, shape=[..., *shape_m]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., *shape_e, *shape_m]
            Inner-product matrix.
        """
        if self._jacobian is None:
            # Only compute graph once
            self._jacobian = gs.autodiff.jacobian(self.immersion)

        J = self._jacobian(base_point)
        if base_point.shape != self.shape:
            # We are [...,*shape]
            J = gs.moveaxis(gs.diagonal(J, axis1=0, axis2=len(self.shape) + 1), -1, 0)
        return J

    def tangent_immersion(self, tangent_vec, base_point):
        r"""Tangent immersion."""
        J = self.jacobian_immersion(base_point).reshape(
            -1, self.embedding_space.shape_dim, self.shape_dim
        )
        tv = tangent_vec.reshape(-1, self.shape_dim)
        res = gs.einsum("...ij,...j->...i", J, tv).reshape(
            -1, *self.embedding_space.shape
        )
        if base_point.shape == self.shape:
            return res[0]
        return res

    def metric_matrix(self, base_point):
        r"""Metric matrix."""
        local_basis = self.local_basis(base_point).reshape(
            -1, self.dim * self.shape_dim
        )

        immersed_base_point = self.immersion(base_point)
        immersed_local_basis = self.tangent_immersion(local_basis).reshape(
            -1, self.dim, *self.embedding_space.shape
        )

        return self.embedding_metric.inner_product(
            immersed_local_basis,
            immersed_local_basis,
            base_point=immersed_base_point,
        )


class PullbackDiffeoMetric(RiemannianMetric, abc.ABC):
    """
    Pullback metric via a diffeomorphism.

    Parameters
    ----------
    dim : int
        Dimension.
    shape : tuple of int
        Shape of one element of the underlying manifold.
        Optional, default : None.
    """

    def __init__(self, dim, shape=None):
        super().__init__(dim=dim, shape=shape)

        self._embedding_metric = None
        self._raw_jacobian_diffeomorphism = None
        self._raw_inverse_jacobian_diffeomorphism = None

        self.shape_dim = math.prod(shape)
        self.embedding_space_shape_dim = math.prod(self.embedding_metric.shape)

    @abc.abstractmethod
    def inverse_immersion(self, im_point):
        """Inverse immersion."""
        pass

    def inverse_jacobian(self, im_point):
        """Inverse jacobian."""
        # Can be overwritten or never use if tangent method are
        # How it works when batched with autodiff ??
        if self._inverse_jacobian is None:
            self._inverse_jacobian = gs.autodiff.jacobian(self.inverse_immersion)
        return self._inverse_jacobian(im_point)

    def inverse_tangent_immersion(self, tangent_vec, im_point):
        """Inverse tangent immersion."""
        # Can be overwritten when close form
        # TODO: More general for shape
        return gs.matmul(self.inverse_jacobian(im_point), tangent_vec)

    def exp(self, tangent_vec, base_point, **kwargs):
        """Compute the exponential map via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector.
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., *shape]
            Riemannian exponential of tangent_vec at base_point.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_tangent_vec = self.tangent_diffeomorphism(tangent_vec, base_point)
        image_exp = self.embedding_metric.exp(
            image_tangent_vec, image_base_point, **kwargs
        )
        exp = self.inverse_diffeomorphism(image_exp)
        return exp

    def log(self, point, base_point, **kwargs):
        """Compute the logarithm map via diffeomorphic pullback.

        Parameters
        ----------
        point : array-like, shape=[..., *shape]
            Point.
        base_point : array-like, shape=[..., *shape]
            Point.

        Returns
        -------
        log : array-like, shape=[..., *shape]
            Logarithm of point from base_point.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_point = self.diffeomorphism(point)
        image_log = self.embedding_metric.log(image_point, image_base_point, **kwargs)
        log = self.inverse_tangent_diffeomorphism(image_log, image_base_point)
        return log

    def dist(self, point_a, point_b, **kwargs):
        """Compute the distance via diffeomorphic pullback.

        Parameters
        ----------
        point_a : array-like, shape=[..., *shape]
            Point a.
        point_b : array-like, shape=[..., *shape]
            Point b.

        Returns
        -------
        dist : array-like
            Distance between point_a and point_b.
        """
        image_point_a = self.diffeomorphism(point_a)
        image_point_b = self.diffeomorphism(point_b)
        distance = self.embedding_metric.dist(image_point_a, image_point_b, **kwargs)
        return distance

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        """Compute the curvature via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *shape]
            Tangent vector a.
        tangent_vec_b : array-like, shape=[..., *shape]
            Tangent vector b.
        tangent_vec_c : array-like, shape=[..., *shape]
            Tangent vector c.
        base_point : array-like, shape=[..., *shape]
            Base point.

        Returns
        -------
        curvature : array-like, shape=[..., *shape]
            Curvature in directions tangent_vec a, b, c at base_point.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_tangent_vec_a = self.tangent_diffeomorphism(tangent_vec_a, base_point)
        image_tangent_vec_b = self.tangent_diffeomorphism(tangent_vec_b, base_point)
        image_tangent_vec_c = self.tangent_diffeomorphism(tangent_vec_c, base_point)
        image_curvature = self.embedding_metric.curvature(
            image_tangent_vec_a,
            image_tangent_vec_b,
            image_tangent_vec_c,
            image_base_point,
        )
        curvature = self.inverse_tangent_diffeomorphism(
            image_curvature, image_base_point
        )
        return curvature

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """Compute the parallel transport via diffeomorphic pullback.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *shape]
            Tangent vector.
        base_point : array-like, shape=[..., *shape]
            Base point.
        direction : array-like, shape=[..., *shape]
            Direction.
        end_point : array-like, shape=[..., *shape]
            End point.

        Returns
        -------
        parallel_transport : array-like, shape=[..., *shape]
            Parallel transport.
        """
        image_base_point = self.diffeomorphism(base_point)
        image_tangent_vec = self.tangent_diffeomorphism(tangent_vec, base_point)

        image_direction = (
            None
            if direction is None
            else self.tangent_diffeomorphism(direction, base_point)
        )
        image_end_point = None if end_point is None else self.diffeomorphism(end_point)

        image_parallel_transport = self.embedding_metric.parallel_transport(
            image_tangent_vec,
            image_base_point,
            direction=image_direction,
            end_point=image_end_point,
        )
        parallel_transport = self.inverse_tangent_diffeomorphism(
            image_parallel_transport, image_base_point
        )
        return parallel_transport
