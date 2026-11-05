import numpy as np
import pytest

from src.ai_utils import accuracy, normalize_features, softmax, train_test_split


# ---------- softmax ----------

def test_softmax_returns_valid_probability_distribution():
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs > 0)
    # больший логит -> большая вероятность (монотонность)
    assert probs[2] > probs[1] > probs[0]


@pytest.mark.parametrize("logits,expected", [
    (np.array([0.0, 0.0]), np.array([0.5, 0.5])),
    (np.array([1000.0, 1000.0]), np.array([0.5, 0.5])),
])
def test_softmax_known_and_extreme_cases(logits, expected):
    # случай с 1000 проверяет численную устойчивость (shift по max)
    assert np.allclose(softmax(logits), expected)


def test_softmax_rejects_empty_and_2d_input():
    with pytest.raises(ValueError):
        softmax(np.array([]))
    with pytest.raises(ValueError):
        softmax(np.array([[1.0, 2.0]]))


# ---------- normalize_features ----------

def test_normalize_features_maps_to_unit_range():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Xn = normalize_features(X)
    assert np.all(Xn >= 0.0) and np.all(Xn <= 1.0)
    assert np.isclose(Xn.min(axis=0).min(), 0.0)
    assert np.isclose(Xn.max(axis=0).max(), 1.0)


def test_normalize_features_constant_column_no_division_error():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Xn = normalize_features(X)
    assert np.all(Xn[:, 0] == 0.0)  # константный столбец -> нули, без NaN
    assert not np.isnan(Xn).any()


def test_normalize_features_rejects_1d_input():
    with pytest.raises(ValueError):
        normalize_features(np.array([1.0, 2.0, 3.0]))


# ---------- accuracy ----------

def test_accuracy_perfect_and_random_case():
    y_true = np.array([1, 0, 1, 1, 0])
    assert accuracy(y_true, y_true) == 1.0
    assert accuracy(y_true, np.array([1, 0, 1, 0, 0])) == pytest.approx(0.8)


def test_accuracy_rejects_shape_mismatch_and_empty():
    with pytest.raises(ValueError):
        accuracy(np.array([1, 0]), np.array([1]))
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


# ---------- train_test_split ----------

@pytest.fixture
def dataset():
    X = np.arange(100).reshape(-1, 1).astype(float)
    y = (np.arange(100) % 2).astype(int)
    return X, y


def test_split_sizes_are_consistent(dataset):
    X, y = dataset
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    assert len(X_tr) == 80 and len(X_te) == 20
    assert len(X_tr) == len(y_tr) and len(X_te) == len(y_te)


def test_split_is_deterministic_with_seed(dataset):
    X, y = dataset
    a = train_test_split(X, y, seed=7)
    b = train_test_split(X, y, seed=7)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_split_has_no_label_leakage(dataset):
    # один и тот же объект не должен попасть и в train, и в test
    X, y = dataset
    X_tr, X_te, _, _ = train_test_split(X, y)
    assert set(X_tr.ravel()).isdisjoint(set(X_te.ravel()))


@pytest.mark.parametrize("bad_size", [0.0, 1.0, -0.5, 2.0])
def test_split_rejects_invalid_test_size(dataset, bad_size):
    X, y = dataset
    with pytest.raises(ValueError):
        train_test_split(X, y, test_size=bad_size)
