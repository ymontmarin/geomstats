"""Test data for the fisher rao metric."""

import geomstats.backend as gs
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.normal import NormalDistributions, NormalMetric
from tests.data_generation import _RiemannianMetricTestData


class FisherRaoMetricTestData(_RiemannianMetricTestData):
    information_manifolds = [
        NormalDistributions(),
    ]
    supports = [(-10, 10)]
    Metric = FisherRaoMetric
    metric_args_list = list(zip(information_manifolds, supports))

    shape_list = [metric_args[0].shape for metric_args in metric_args_list]
    space_list = [metric_args[0] for metric_args in metric_args_list]
    n_points_list = [1, 2] * 3
    n_tangent_vecs_list = [1, 2] * 3
    n_points_a_list = [1, 2] * 3
    n_points_b_list = [1]
    alpha_list = [1] * 6
    n_rungs_list = [1] * 6
    scheme_list = ["pole"] * 6

    def inner_product_matrix_shape_test_data(self):
        smoke_data = [
            dict(
                information_manifold=NormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_matrix_and_its_inverse_test_data(self):
        smoke_data = [
            dict(
                information_manifold=NormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_and_closed_form_metric_matrix_test_data(self):
        smoke_data = [
            dict(
                information_manifold=NormalDistributions(),
                support=(-10, 10),
                closed_form_metric=NormalMetric(),
                base_point=gs.array([0.1, 0.8]),
            )
        ]
        return self.generate_tests(smoke_data)
