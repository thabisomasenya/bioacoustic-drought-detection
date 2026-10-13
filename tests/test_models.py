import numpy as np
import pytest
import tensorflow as tf


@pytest.mark.parametrize("name", ["cnn1d", "cnn2d_mel", "lstm_mel"])
def test_models_build_and_output_probabilities(cfg, name, tmp_path):
    from drought_acoustics.models import build_model
    cfg["model"]["name"] = name
    model = build_model(cfg)
    x = np.random.default_rng(0).standard_normal((3, 1001, 1)).astype(np.float32)
    p = model.predict(x, verbose=0)
    assert p.shape == (3, 1) and np.all((p >= 0) & (p <= 1))
    path = tmp_path / f"{name}.keras"                    # save/load round trip
    model.save(path)
    reloaded = tf.keras.models.load_model(path)
    np.testing.assert_allclose(reloaded.predict(x, verbose=0), p, atol=1e-5)


def test_log_mel_matches_librosa_shape(cfg):
    from drought_acoustics.models import LogMelSpectrogram
    m = cfg["model"]["mel"]
    layer = LogMelSpectrogram(**m)
    out = layer(tf.zeros((2, 1001, 1)))
    frames = 1 + (1001 - m["n_fft"]) // m["hop_length"]
    assert out.shape == (2, frames, m["n_mels"])
