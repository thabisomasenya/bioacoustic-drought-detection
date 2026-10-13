import numpy as np
import soundfile as sf

from drought_acoustics.data import build_manifest, dataset_fingerprint, load_waveforms, select_signals


def test_manifest_labels_groups_and_validity(cfg, tiny_dataset):
    m = build_manifest(cfg)
    assert len(m) == 4 * 30 + 2 * 20
    assert m["valid"].all()
    sig = select_signals(cfg, m)
    assert set(sig["label"]) == {0, 1}
    # plant ids repeat across folders, groups must not
    assert sig["group"].nunique() == 24


def test_invalid_file_is_flagged(cfg, tiny_dataset):
    sf.write(tiny_dataset / "Tomato Dry" / "id_99_sound_0.wav", np.zeros(500, np.float32),
             500_000, subtype="PCM_16")
    m = build_manifest(cfg)
    bad = m[~m["valid"]]
    assert len(bad) == 1 and "length=500" in bad["issue"].iloc[0]


def test_fingerprint_detects_changed_audio(cfg, tiny_dataset):
    fp1 = dataset_fingerprint(build_manifest(cfg))
    path = tiny_dataset / "Tomato Cut" / "id_0_sound_0.wav"
    x, sr = sf.read(path, dtype="int16")
    x[10] += 1
    sf.write(path, x, sr, subtype="PCM_16")
    assert dataset_fingerprint(build_manifest(cfg)) != fp1


def test_waveforms_shape_and_range(cfg, tiny_dataset):
    m = build_manifest(cfg)
    X = load_waveforms(cfg, m)
    assert X.shape == (len(m), 1001) and X.dtype == np.float32
    assert np.abs(X).max() <= 1.0
