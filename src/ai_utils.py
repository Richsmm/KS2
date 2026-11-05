import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Преобразует вектор логитов в распределение вероятностей."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("Ожидается непустой одномерный массив логитов")
    shifted = logits - np.max(logits)  # защита от overflow
    exps = np.exp(shifted)
    return exps / exps.sum()


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Min-max нормализация признаков в диапазон [0, 1] по каждому столбцу."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Ожидается двумерный массив (объекты x признаки)")
    mins, maxs = X.min(axis=0), X.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # защита от деления на ноль
    return (X - mins) / ranges


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Временно ломаем: всегда возвращаем 1.0 — это заставит тест упасть
    return 1.0


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     seed: int = 42):
    """Простое перемешанное разбиение выборки (аналог sklearn)."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size должен быть в интервале (0, 1)")
    if len(X) != len(y):
        raise ValueError("X и y должны иметь одинаковую длину")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    n_test = int(np.ceil(len(X) * test_size))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
