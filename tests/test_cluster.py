import pytest
from sklearn.datasets import make_blobs
from sklearn import cluster

from postlearn.cluster import compute_centers, plot_decision_boundry


@pytest.fixture
def data_labels():
    return make_blobs(random_state=2)


class TestCluster:

    def test_compute_centers(self, data_labels):
        data, _ = data_labels
        ac = cluster.AgglomerativeClustering()
        fit = ac.fit(data)
        result = compute_centers(fit, data)

        assert result.shape == (data.shape[1], len(set(fit.labels_)))


class TestPlotDecisionBoundry:

    @pytest.mark.parametrize('method', [
        cluster.KMeans, cluster.AffinityPropagation,
        cluster.AgglomerativeClustering, cluster.MeanShift,
        cluster.SpectralClustering,
    ])
    def test_smoke(self, data_labels, method):
        data, _ = data_labels
        est = method()
        est.fit(data)
        plot_decision_boundry(data, est)
