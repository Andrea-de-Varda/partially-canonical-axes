"""
pereira_data.load_pereira(): the public npz gives exactly the arrays and sentence
order of the original pickle (skipped when either file is missing), and the
derived targets are those the paper used.

    python -m pytest other/test_pereira_data.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pereira_data as PD            # noqa: E402
import pereira_targets as PT         # noqa: E402

HAVE_NPZ = os.path.exists(PD.NPZ)
HAVE_PKL = os.path.exists(PD.PKL)


@pytest.mark.skipif(not (HAVE_NPZ and HAVE_PKL), reason="needs both the npz and the original pickle")
def test_npz_identical_to_pickle():
    a = PD.load_pereira(npz_path=PD.NPZ)
    b = PD.load_pereira(npz_path="/nonexistent")
    for exp in PD.EXPERIMENTS:
        assert a["net_sent_orders"][exp] == list(b["net_sent_orders"][exp])
        assert len(a["net_structs"][exp]) == len(b["net_structs"][exp])
        for x, y in zip(a["net_structs"][exp], b["net_structs"][exp]):
            assert np.array_equal(x[PT.NETWORK_KEY], y[PT.NETWORK_KEY])


@pytest.mark.skipif(not (HAVE_NPZ or HAVE_PKL), reason="no brain data in data/")
def test_targets_shape():
    d = PD.load_pereira()
    t2 = PT.build_targets(d["net_structs"]["PereiraE2_96pass"], "top20rel")
    t3 = PT.build_targets(d["net_structs"]["PereiraE3_72pass"], "top20rel")
    assert [t["part"] for t in t2] == [0, 1, 2, 3, 4, 6, 7, 8]      # single-session participant dropped
    assert len(t3) == 6
    assert all(t["Y"].shape == (384, 273) for t in t2) and all(t["Y"].shape == (243, 273) for t in t3)
    assert len(d["net_sent_orders"]["PereiraE2_96pass"]) == 384
