from evolving_classifier.EvolvingClassifier import *

def test_ec_init():
    ec = EvolvingClassifier()
    dhrange = get_default_hrange()

    assert ec.hrange.layerCountMin == dhrange.layerCountMin
    assert ec.hrange.layerCountMax == dhrange.layerCountMax
    assert ec.hrange.neuronCountMin == dhrange.neuronCountMin
    assert ec.hrange.neuronCountMax == dhrange.neuronCountMax
    assert len(ec.hrange.actFunSet) == len(dhrange.actFunSet)
    for i in range(len(ec.hrange.actFunSet)):
        assert ec.hrange.actFunSet[i].to_string() == dhrange.actFunSet[i].to_string()
    assert ec.hrange.weiAbs == 1
    assert ec.hrange.biaAbs == 1

