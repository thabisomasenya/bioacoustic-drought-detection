import numpy as np

from drought_acoustics.augmentation import make_augmenter, make_dataset, mix_at_snr

T = 1001


def measured_snr(clean, mixed):
    noise = mixed - clean
    return 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))


def test_numpy_mixing_hits_target_snr():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((20, T)).astype(np.float32)
    bank = 3 * rng.standard_normal((5, T)).astype(np.float32)
    for snr in [20, 5, 0, -5]:
        mixed = mix_at_snr(clean, bank, snr, seed=1)
        for c, m in zip(clean, mixed):
            assert abs(measured_snr(c, m) - snr) < 0.01


def test_tf_augmenter_snr_within_range():
    import tensorflow as tf
    rng = np.random.default_rng(0)
    x = rng.standard_normal(T).astype(np.float32)
    bank = rng.standard_normal((4, T)).astype(np.float32)
    acfg = {"snr_db": [0.0, 20.0], "max_shift": 0, "noise_prob": 1.0, "polarity_flip_prob": 0.0}
    aug = make_augmenter(bank, acfg, per_clip_peak=False)
    for i in range(20):
        y, _ = aug(tf.constant(x), tf.constant(1.0), tf.constant([i, 7], tf.int64))
        assert -0.01 <= measured_snr(x, y.numpy()) <= 20.01


def test_training_pipeline_is_seeded_and_eval_is_untouched(cfg):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, T)).astype(np.float32)
    y = np.arange(16) % 2
    bank = rng.standard_normal((4, T)).astype(np.float32)
    first = lambda ds: next(iter(ds))[0].numpy()  # noqa: E731
    a = first(make_dataset(X, y, cfg, True, bank, seed=5, batch_size=8))
    b = first(make_dataset(X, y, cfg, True, bank, seed=5, batch_size=8))
    np.testing.assert_allclose(a, b)                         # same seed -> same batch
    ev = first(make_dataset(X, y, cfg, False, batch_size=8))
    np.testing.assert_allclose(ev[..., 0], X[:8])            # no augmentation at eval
    assert a.shape == (8, T, 1)
