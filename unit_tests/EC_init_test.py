from evolving_classifier.EvolvingClassifier import *

def test_ec_init():
    #TODO - B - zbadać jakie zostały stworzone sieci?
    ec = EvolvingClassifier()
    ec.prepare(2, [np.array([[0], [1]]), np.array([[1], [2]])], 1001)
    dhrange = get_default_hrange()

    assert ec.hrange.min_init_wei == dhrange.min_init_wei
    assert ec.hrange.max_init_wei == dhrange.max_init_wei
    assert ec.hrange.min_init_bia == dhrange.min_init_bia
    assert ec.hrange.max_init_bia == dhrange.max_init_bia
    assert ec.hrange.min_it == dhrange.min_it
    assert ec.hrange.max_it == dhrange.max_it
    assert ec.hrange.min_hidden == dhrange.min_hidden
    assert ec.hrange.max_hidden == dhrange.max_hidden

    assert len(ec.hrange.actFunSet) == len(dhrange.actFunSet)
    for i in range(len(ec.hrange.actFunSet)):
        assert ec.hrange.actFunSet[i].to_string() == dhrange.actFunSet[i].to_string()


    assert ec.hrange.min_mut_radius == dhrange.min_mut_radius
    assert ec.hrange.max_mut_radius == dhrange.max_mut_radius
    assert ec.hrange.min_sqr_mut_prob == dhrange.min_sqr_mut_prob
    assert ec.hrange.max_sqr_mut_prob == dhrange.max_sqr_mut_prob
    assert ec.hrange.min_lin_mut_prob == dhrange.min_lin_mut_prob
    assert ec.hrange.max_lin_mut_prob == dhrange.max_lin_mut_prob
    assert ec.hrange.min_p_mut_prob == dhrange.min_p_mut_prob
    assert ec.hrange.max_p_mut_prob == dhrange.max_p_mut_prob
    assert ec.hrange.min_c_prob == dhrange.min_c_prob
    assert ec.hrange.max_c_prob == dhrange.max_c_prob
    assert ec.hrange.min_dstr_mut_prob == dhrange.min_dstr_mut_prob
    assert ec.hrange.max_dstr_mut_prob == dhrange.max_dstr_mut_prob

    assert len(ec.population) == 2
    assert ec.pop_size == 2

    assert np.array_equal(ec.trainInputs, np.array([[0], [1]]))
    assert np.array_equal(ec.trainOutputs, np.array([[1], [2]]))

    # assert ec.hrange.hiddenLayerCountMax == dhrange.hiddenLayerCountMax
    # assert ec.hrange.neuronCountMin == dhrange.neuronCountMin
    # assert ec.hrange.neuronCountMax == dhrange.neuronCountMax
    # assert len(ec.hrange.actFunSet) == len(dhrange.actFunSet)
    # for i in range(len(ec.hrange.actFunSet)):
    #     assert ec.hrange.actFunSet[i].to_string() == dhrange.actFunSet[i].to_string()
    # assert ec.hrange.weiAbs == 1
    # assert ec.hrange.biaAbs == 1

# test_ec_init()

