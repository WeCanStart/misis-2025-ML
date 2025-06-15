import torch


def sgd(w, dw, config=None):
    """
    Vanilla stochastic gradient descent.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    next_w = w - config["learning_rate"] * dw
    return next_w, config


def sgd_momentum(w, dw, config=None):
    """
    SGD with momentum.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)

    if "velocity" not in config:
        config["velocity"] = torch.zeros_like(w)

    v = config["velocity"]
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    RMSProp update.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)

    if "cache" not in config:
        config["cache"] = torch.zeros_like(w)

    cache = config["cache"]
    cache = config["decay_rate"] * cache + (1 - config["decay_rate"]) * (dw ** 2)
    step = -config["learning_rate"] * dw / (cache.sqrt() + config["epsilon"])
    next_w = w + step
    config["cache"] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Adam optimizer.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", torch.zeros_like(w))
    config.setdefault("v", torch.zeros_like(w))
    config.setdefault("t", 0)

    m, v, t = config["m"], config["v"], config["t"] + 1
    beta1, beta2, eps = config["beta1"], config["beta2"], config["epsilon"]

    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    step = -config["learning_rate"] * m_hat / (v_hat.sqrt() + eps)
    next_w = w + step

    config["m"], config["v"], config["t"] = m, v, t

    return next_w, config
