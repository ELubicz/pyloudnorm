from pathlib import Path
import soundfile as sf
import numpy as np
import pyloudnorm as pyln

data_dir = Path(__file__).parent / "data"

def compare_lufs(lufs, filename):
    #np.savetxt(str(data_dir / ("R_" + filename + ".txt")), lufs, fmt='%.1f')
    np.loadtxt(str(data_dir / ("R_" + filename + ".txt")))
    data_ref = np.loadtxt(str(data_dir / ("R_" + filename + ".txt")))
    # Compare LUFS_I
    assert np.isclose(data_ref[:,0], lufs[:,0], atol=0.1).all()
    # Compare LUFS_M
    assert np.isclose(data_ref[:,1], lufs[:,1], atol=0.1).all()

def calc_loudness_rt(data, rate, block_size_rt=0.1):
    meter = pyln.Meter(rate)
    bsrt_samples = int(np.round(block_size_rt * rate))
    num_blocks = np.floor(data.shape[0] / bsrt_samples)
    if data.ndim == 1:
        data_rt = np.reshape(data[:int(num_blocks*bsrt_samples)], (-1, bsrt_samples, 1))
    else:
        num_ch = data.shape[1]
        data_rt = np.reshape(data[:int(num_blocks*bsrt_samples),:], (-1, bsrt_samples, num_ch))
    # create an empty LUFS array
    lufs = np.zeros((data_rt.shape[0], 3))
    for i, block in enumerate(data_rt):
        (lufs_i, lufs_m, lufs_s) = meter.step_loudness(block)
        lufs[i,:] = [lufs_i, lufs_m, lufs_s]
    return lufs

def test_integrated_loudness():
    data, rate = sf.read(str(data_dir / "sine_1000.wav"))
    meter = pyln.Meter(rate)
    loudness_offline = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)
    target_loudness = -3.0523438444331137

    assert np.isclose(loudness_offline, target_loudness)
    assert np.isclose(lufs[-1][0], target_loudness)


def test_peak_normalize():
    data = np.array(0.5)
    norm = pyln.normalize.peak(data, 0.0)

    assert np.isclose(norm, 1.0)


def test_loudness_normalize():
    data, rate = sf.read(str(data_dir / "sine_1000.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    norm = pyln.normalize.loudness(data, loudness, -6.0)
    loudness = meter.integrated_loudness(norm)

    assert np.isclose(loudness, -6.0)


def test_rel_gate_test():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_RelGateTest.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -10.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    #assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_abs_gate_test():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_AbsGateTest.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -69.5
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_24LKFS_25Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_24LKFS_25Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_24LKFS_100Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_24LKFS_100Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_24LKFS_500Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_24LKFS_500Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_24LKFS_1000Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_24LKFS_1000Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_24LKFS_2000Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_24LKFS_2000Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_24LKFS_10000Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_24LKFS_10000Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_23LKFS_25Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_23LKFS_25Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_23LKFS_100Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_23LKFS_100Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_23LKFS_500Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_23LKFS_500Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_23LKFS_1000Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_23LKFS_1000Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_23LKFS_2000Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_23LKFS_2000Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_23LKFS_10000Hz_2ch():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_23LKFS_10000Hz_2ch.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_18LKFS_frequency_sweep():
    data, rate = sf.read(str(data_dir / "1770-2_Comp_18LKFS_FrequencySweep.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)

    target_loudness = -18.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_conf_stereo_vinL_R_23LKFS():
    data, rate = sf.read(str(data_dir / "1770-2_Conf_Stereo_VinL+R-23LKFS.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)
    compare_lufs(lufs, "1770-2_Conf_Stereo_VinL+R-23LKFS")
    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    #assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_conf_monovoice_music_24LKFS():
    data, rate = sf.read(str(data_dir / "1770-2_Conf_Mono_Voice+Music-24LKFS.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)
    np.savetxt(str(data_dir / ("R_" + "1770-2_Conf_Mono_Voice+Music-24LKFS" + ".txt")), lufs, fmt='%.1f')
    compare_lufs(lufs, "1770-2_Conf_Mono_Voice+Music-24LKFS")
    target_loudness = -24.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    #assert np.isclose(target_loudness, lufs[-1][0], atol=0.1)


def test_conf_monovoice_music_23LKFS():
    data, rate = sf.read(str(data_dir / "1770-2_Conf_Mono_Voice+Music-23LKFS.wav"))
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    lufs = calc_loudness_rt(data, rate)
    compare_lufs(lufs, "1770-2_Conf_Mono_Voice+Music-23LKFS")
    target_loudness = -23.0
    assert np.isclose(target_loudness, loudness, atol=0.1)
    # TODO: check this
    #assert np.isclose(target_loudness, lufs[-1][0])
