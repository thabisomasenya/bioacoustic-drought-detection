import numpy as np
import pytest

from drought_acoustics.preprocessing import AmplitudeScaler, design_bandpass, filter_signals

SR, T = 500_000, 1001


def tone(f):
    t = np.arange(4 * T) / SR
    return np.sin(2 * np.pi * f * t).astype(np.float32)


def gain_db(cfg, f):
    x = tone(f)[None]
    y = filter_signals(x, cfg)
    core = slice(T, 3 * T)  # ignore edge transients
    return 20 * np.log10(np.std(y[0, core]) / np.std(x[0, core]))


@pytest.mark.parametrize("f,expect_pass", [(5e3, False), (50e3, True), (80e3, True), (200e3, False)])
def test_bandpass_response(cfg, f, expect_pass):
    g = gain_db(cfg, f)
    if expect_pass:
        assert g > -1.0, f"{f} Hz attenuated by {g:.1f} dB"
    else:
        assert g < -15.0, f"{f} Hz only attenuated by {g:.1f} dB"


def test_zero_phase_keeps_peak_position(cfg):
    from conftest import click
    x = click(55e3, np.random.default_rng(1))[None]
    y = filter_signals(x, cfg)
    assert abs(int(np.argmax(np.abs(y))) - 500) <= 3


def test_invalid_band_rejected():
    with pytest.raises(ValueError):
        design_bandpass(100e3, 20e3, SR)


def test_scaler_fits_on_train_only():
    rng = np.random.default_rng(0)
    train, test = 0.1 * rng.standard_normal((50, T)), 5.0 * rng.standard_normal((50, T))
    s = AmplitudeScaler("global_rms").fit(train)
    assert np.isclose(np.sqrt(np.mean(s.transform(train) ** 2)), 1.0, atol=1e-3)
    # test statistics must not influence the fitted scale
    assert np.sqrt(np.mean(s.transform(test) ** 2)) > 10
    assert AmplitudeScaler.from_dict(s.to_dict()).scale_ == s.scale_

