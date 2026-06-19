"""Compatibility checks for legacy and package import surfaces."""


def test_legacy_training_utils_import():
    from training_utils import measure_cpu_latency
    from p0.training_utils import measure_cpu_latency as package_measure_cpu_latency

    assert measure_cpu_latency is package_measure_cpu_latency


def test_legacy_train_imports():
    from train import ParticleDataset, evaluate, train_one_epoch
    from p0.data import ParticleDataset as package_particle_dataset
    from p0.training import evaluate as package_evaluate
    from p0.training import train_one_epoch as package_train_one_epoch

    assert ParticleDataset is package_particle_dataset
    assert evaluate is package_evaluate
    assert train_one_epoch is package_train_one_epoch


def test_model_registry_imports():
    from models import create_model
    from p0.models import create_model as package_create_model

    assert create_model is package_create_model


def test_pub_utils_imports():
    from pub_utils import plot_confusion_matrix
    from p0.plotting import plot_confusion_matrix as package_plot_confusion_matrix

    assert plot_confusion_matrix is package_plot_confusion_matrix
