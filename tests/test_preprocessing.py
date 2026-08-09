import os
import pytest
import numpy as np
import pandas as pd
from preprocessing import TrackPreprocessor

def test_preprocessor_fit_ignores_padding():
    '''
    WHAT: Verifies that the fit method calculates statistics strictly excluding the pad value.
    WHY: Including -9999.0 in the mean/std calculations will catastrophically skew the feature scaling.
    '''
    processor = TrackPreprocessor(filepath = 'dummy.csv', pad_val = -9999.0)
    
    #Create a 3x2 dummy array where valid hits average to 2.0
    X_train = np.array([
        [1.0, 3.0], 
        [3.0, -9999.0], 
        [-9999.0, 1.0]
    ])
    y_train = np.array([10.0, 20.0, 30.0])
    
    processor.fit(X_train, y_train)
    
    #The mean of [1.0, 3.0] is 2.0, and so is the mean of its transpose.
    np.testing.assert_allclose(processor.x_mean, np.array([2.0, 2.0]))

def test_preprocessor_transform_preserves_padding():
    '''
    WHAT: Ensures the transform method scales valid floats but leaves the sentinel padding values completely untouched.
    WHY: The PyTorch neural network relies on the exact value of -9999.0 to generate its boolean masks.
    '''
    processor = TrackPreprocessor(filepath = 'dummy.csv', pad_val = -9999.0)
    
    #Inject dummy fitted statistics
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

def test_preprocessor_full_pipeline_io(tmp_path):
    '''
    WHAT: Tests the file I/O methods (load_and_split, save_tensors, run_pipeline).
    WHY: Ensures the preprocessor can correctly read from disk, split data, and save the resulting PyTorch-ready tensors without crashing.
    '''
    #1. Create a dummy CSV inside the temporary test directory
    dummy_csv = tmp_path / 'dummy_tracks.csv'
    dummy_outdir = tmp_path / 'processed'
    
    #We need enough rows so train_test_split doesn't fail on empty datasets
    df = pd.DataFrame({
        'pt_true': np.random.uniform(65, 105, 10),
        'hit_0_x': np.random.uniform(-0.5, 0.5, 10),
        'hit_0_y': np.random.uniform(-0.5, 0.5, 10)
    })
    df.to_csv(dummy_csv, index = False)
    
    #2. Initialize the preprocessor with the temporary file
    processor = TrackPreprocessor(filepath = str(dummy_csv), random_state = 13)
    
    #3. Execute the full pipeline, instructing it to save to the temporary output dir
    processor.run_pipeline(str(dummy_outdir))
    
    #4. Assert that the files were physically created on disk
    assert os.path.exists(dummy_outdir / 'X_train.npy')
    assert os.path.exists(dummy_outdir / 'y_test.npy')
    assert os.path.exists(dummy_outdir / 'y_stats.npz')

def test_preprocessor_file_not_found():
    '''
    WHAT: Tests the load_and_split method's exception handling for missing files.
    WHY: Provides a clear error to the user if they forget to run simulation.py.
    '''
    processor = TrackPreprocessor(filepath = 'non_existent_file.csv')
    with pytest.raises(FileNotFoundError):
        processor.load_and_split()


def test_preprocessor_rejects_zero_target_variance():
    """Check that constant training targets cannot be standardized."""
    processor = TrackPreprocessor(filepath='dummy.csv')

    X_train = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
        ]
    )
    y_train = np.array(
        [
            [85.0],
            [85.0],
            [85.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match='Target standard deviation is zero',
    ):
        processor.fit(X_train, y_train)


#Property of Wafa Mamani. May 2026.