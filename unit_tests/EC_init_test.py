from evolving_classifier.EvolvingClassifier import *

def test_ec_init():
    ec = EvolvingClassifier()
    dhrange = get_default_hrange()

    assert ec.hrange.layerCountMin == dhrange.layerCountMin
    assert ec.hrange.layerCountMax == dhrange.layerCountMax
    assert ec.hrange.neuronCountMin == dhrange.neuronCountMin
    assert ec.hrange.neuronCountMax == dhrange.neuronCountMax
    assert len(ec.hrange.actFunSet) == len(dhrange.actFunSet)
    assert len(ec.hrange.aggrFunSet) == len(dhrange.aggrFunSet)
    assert len(ec.hrange.lossFunSet) == len(dhrange.lossFunSet)
    for i in range(len(ec.hrange.actFunSet)):
        assert ec.hrange.actFunSet[i].to_string() == dhrange.actFunSet[i].to_string()
    for i in range(len(ec.hrange.aggrFunSet)):
        assert ec.hrange.aggrFunSet[i].to_string() == dhrange.aggrFunSet[i].to_string()
    for i in range(len(ec.hrange.lossFunSet)):
        assert ec.hrange.lossFunSet[i].to_string() == dhrange.lossFunSet[i].to_string()
    assert ec.hrange.learningRateMin == dhrange.learningRateMin
    assert ec.hrange.learningRateMax == dhrange.learningRateMax
    assert ec.hrange.momentumCoeffMin == dhrange.momentumCoeffMin
    assert ec.hrange.momentumCoeffMax == dhrange.momentumCoeffMax
    assert ec.hrange.batchSizeMin == dhrange.batchSizeMin
    assert ec.hrange.batchSizeMax == dhrange.batchSizeMax

