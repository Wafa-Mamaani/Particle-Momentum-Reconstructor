import pytest
import numpy as np
from preprocessing import TrackPreprocessor

def test_preprocessor_fit_ignores_padding():
    '''
    WHAT: Verifies that the fit method calculates statistics strictly excluding the pad value.
    WHY: Including -9999.0 in the mean/std calculations will catastrophically skew the feature scaling.
    '''
    processor = TrackPreprocessor(filepath = 'dummy.csv', pad_val = -9999.0)

    X_train = np.array([
        [1.0, 3.0],
        [3.0, -9999.0],
        [-9999.0, 1.0]
    ])
    y_train = np.array([10.0, 20.0, 30.0])

    processor.fit(X_train, y_train)

    np.testing.assert_allclose(processor.x_mean, np.array([2.0, 2.0]))

def test_preprocessor_transform_preserves_padding():
    '''
    WHAT: Ensures the transform method scales valid floats but leaves the sentinel padding values completely untouched.
    WHY: The PyTorch neural network relies on the exact value of -9999.0 to generate its boolean masks.
    '''
    processor = TrackPreprocessor(filepath = 'dummy.csv', pad_val = -9999.0)

    processor.x_mean = np.array([0.0, 0.0])
    processor.x_std = np.array([1.0, 1.0])
    processor.y_mean = 0.0
    processor.y_std = 1.0

    X_unscaled = np.array([[5.0, -9999.0]])
    y_unscaled = np.array([[10.0]])

    X_scaled, _ = processor.transform(X_unscaled, y_unscaled)

    np.testing.assert_allclose(X_scaled[0, 1], -9999.0)

def test_preprocessor_transform_without_fit_raises_error():
    '''
    WHAT: Tests that calling transform before fit raises a RuntimeError.
    WHY: Prevents accidental scaling with uninitialized (NoneType) statistics.
    '''
    processor = TrackPreprocessor(filepath = 'dummy.csv')
    X_dummy = np.ones((5, 72))
    y_dummy = np.ones((5, 1))

    with pytest.raises(RuntimeError):
        processor.transform(X_dummy, y_dummy)
