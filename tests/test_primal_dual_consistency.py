import pytest
import numpy as np
from kooplearn._src import primal, dual
from kooplearn.data.datasets import mock_trajectory

@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi'])
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_reduced_rank(dt, svd_solver):
    num_features = 50
    num_test_pts = 200
    rank = 10
    tikhonov_reg = 1e-3

    X, Y = mock_trajectory(num_features=num_features)
    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg, svd_solver = svd_solver)

    dual_predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, _, _ = dual.low_rank_eig(U, V, K_X, K_Y, K_YX)

    assert dual_predict.shape == (num_test_pts, num_features)

    #Primal
    U = primal.fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg, svd_solver = svd_solver)

    primal_predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, _ = primal.low_rank_eig(U, C_XY)

    assert dual_predict.shape == (num_test_pts, num_features)

    assert np.allclose(primal_predict, dual_predict)
    assert np.allclose(np.sort(dual_eig.real), np.sort(primal_eig.real))
    assert np.allclose(np.sort(dual_eig.imag), np.sort(primal_eig.imag))

@pytest.mark.parametrize('rank', [10, None])
@pytest.mark.parametrize('svd_solver', ['full', 'arnoldi'])
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_tikhonov(dt, svd_solver, rank):
    num_features = 50
    num_test_pts = 200
    tikhonov_reg = 1e-3

    X, Y = mock_trajectory(num_features=num_features)
    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V = dual.fit_tikhonov(K_X, tikhonov_reg, rank = rank, svd_solver = svd_solver)

    dual_predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, _, _ = dual.low_rank_eig(U, V, K_X, K_Y, K_YX)

    assert dual_predict.shape == (num_test_pts, num_features)

    #Primal
    U = primal.fit_tikhonov(C_X, tikhonov_reg, rank = rank, svd_solver = svd_solver)

    primal_predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, _ = primal.low_rank_eig(U, C_XY)

    assert dual_predict.shape == (num_test_pts, num_features)

    assert np.allclose(primal_predict, dual_predict)
    if rank is not None:
        assert np.allclose(np.sort(dual_eig.real), np.sort(primal_eig.real))
        assert np.allclose(np.sort(dual_eig.imag), np.sort(primal_eig.imag))

@pytest.mark.skip()
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_rand_reduced_rank(dt):
    num_features = 50
    num_test_pts = 200
    rank = 3
    tikhonov_reg = 1e-3
    n_oversamples = 20
    iterated_power = 2

    X, Y = mock_trajectory(num_features=num_features)
    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    K_X = X@(X.T)
    C_X = rdim*((X.T)@X)

    K_Y = Y@(Y.T)

    K_YX = Y@(X.T)
    C_XY = rdim*((X.T)@Y)

    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_rand_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg, n_oversamples, False, iterated_power)

    dual_predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    dual_eig, _, _ = dual.low_rank_eig(U, V, K_X, K_Y, K_YX)

    assert dual_predict.shape == (num_test_pts, num_features)

    #Primal
    U = primal.fit_rand_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg, n_oversamples, iterated_power)

    primal_predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    primal_eig, _ = primal.low_rank_eig(U, C_XY)

    assert dual_predict.shape == (num_test_pts, num_features)
    assert np.allclose(primal_predict, dual_predict)
    assert np.allclose(np.sort(dual_eig.real), np.sort(primal_eig.real))
    assert np.allclose(np.sort(dual_eig.imag), np.sort(primal_eig.imag))

@pytest.mark.skip()
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_rand_reduced_rank_primal(dt):
    num_features = 50
    num_test_pts = 200
    rank = 5
    tikhonov_reg = 1e-3
    n_oversamples = 20
    iterated_power = 3

    X, Y = mock_trajectory(num_features=num_features)
    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    rdim = np.true_divide(1, X.shape[0])
    C_X = rdim*((X.T)@X)
    C_XY = rdim*((X.T)@Y)
    
    #Primal
    U = primal.fit_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg)

    predict = primal.low_rank_predict(dt, U, C_XY, X_test, X, Y)
    eig, _ = primal.low_rank_eig(U, C_XY)
    
    #Rand
    U_rand = primal.fit_rand_reduced_rank_regression_tikhonov(C_X, C_XY, rank, tikhonov_reg, n_oversamples, iterated_power)

    rand_predict = primal.low_rank_predict(dt, U_rand, C_XY, X_test, X, Y)
    rand_eig, _ = primal.low_rank_eig(U_rand, C_XY)

    diff = np.ravel(rand_predict - predict)
    mean = 0.5*np.ravel(rand_predict + predict)
    print(np.linalg.norm(diff, ord=np.inf)/np.linalg.norm(mean, ord = np.inf))

    assert predict.shape == (num_test_pts, num_features)
    assert rand_predict.shape == (num_test_pts, num_features)

    assert np.allclose(predict, rand_predict)
    assert np.allclose(np.sort(rand_eig.real), np.sort(eig.real))
    assert np.allclose(np.sort(rand_eig.imag), np.sort(eig.imag))

@pytest.mark.skip()
@pytest.mark.parametrize('dt', [1, 5, 10])
def test_rand_reduced_rank_dual(dt):
    num_features = 50
    num_test_pts = 200
    rank = 10
    tikhonov_reg = 1e-3
    n_oversamples = 10
    iterated_power = 2

    X, Y = mock_trajectory(num_features=num_features)
    rng = np.random.default_rng(42)
    X_test = rng.random((num_test_pts, num_features))
    
    K_X = X@(X.T)
    K_Y = Y@(Y.T)
    K_YX = Y@(X.T)
    K_testX = X_test@(X.T)
    
    #Dual
    U, V, _ = dual.fit_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg)

    predict = dual.low_rank_predict(dt, U, V, K_YX, K_testX, Y)
    eig, _, _ = dual.low_rank_eig(U, V, K_X, K_Y, K_YX)

    #Rand
    U_rand, V_rand, _ = dual.fit_rand_reduced_rank_regression_tikhonov(K_X, K_Y, rank, tikhonov_reg, n_oversamples, False, iterated_power)

    rand_predict = dual.low_rank_predict(dt, U_rand, V_rand, K_YX, K_testX, Y)
    rand_eig, _, _ = dual.low_rank_eig(U_rand, V_rand, K_X, K_Y, K_YX)

    assert predict.shape == (num_test_pts, num_features)
    assert rand_predict.shape == (num_test_pts, num_features)

    assert np.allclose(predict, rand_predict)
    assert np.allclose(np.sort(rand_eig.real), np.sort(eig.real))
    assert np.allclose(np.sort(rand_eig.imag), np.sort(eig.imag))