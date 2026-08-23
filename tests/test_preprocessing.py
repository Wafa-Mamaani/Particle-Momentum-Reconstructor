import os
import pytest
import numpy as np
import pandas as pd
from preprocessing import TrackPreprocessor

def test_preprocessor_fit_ignores_padding():
    '''Padding values should not affect the calculated feature statistics.'''
    processor = TrackPreprocessor(filepath='dummy.csv', pad_val=-9999.0)
    
    # Create a 3x2 dummy array where valid hits average to 2.0
    X_train = np.array([
        [1.0, 3.0], 
        [3.0, -9999.0], 
        [-9999.0, 1.0]
    ])
    y_train = np.array([10.0, 20.0, 30.0])
    
    processor.fit(X_train, y_train)

    # Both feature columns have a valid-value mean of 2.0.
    np.testing.assert_allclose(processor.x_mean, np.array([2.0, 2.0]))

def test_preprocessor_transform_preserves_padding():
    '''Scaling should leave the -9999.0 padding values unchanged.'''
    processor = TrackPreprocessor(filepath='dummy.csv', pad_val=-9999.0)
    
    # Inject dummy fitted statistics
    processor.x_mean = np.array([0.0, 0.0])
    processor.x_std = np.array([1.0, 1.0])
    processor.y_mean = 0.0
    processor.y_std = 1.0
    
    X_unscaled = np.array([[5.0, -9999.0]])
    y_unscaled = np.array([[10.0]])
    
    X_scaled, _ = processor.transform(X_unscaled, y_unscaled)
    
    np.testing.assert_allclose(X_scaled[0, 1], -9999.0)

def test_preprocessor_transform_without_fit_raises_error():
    '''Transform should not run before the training statistics are fitted.'''
    processor = TrackPreprocessor(filepath='dummy.csv')
    X_dummy = np.ones((5, 72))
    y_dummy = np.ones((5, 1))
    
    with pytest.raises(RuntimeError):
        processor.transform(X_dummy, y_dummy)


def test_preprocessor_transform_rejects_mismatched_rows():
    '''Features and targets must contain the same number of samples.'''

    preprocessor = TrackPreprocessor('unused.csv')

    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ])
    y_train = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    preprocessor.fit(X_train, y_train)

    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
    ])
    y = np.array([[1.0]])

    with pytest.raises(
        ValueError,
        match='same number of rows',
    ):
        preprocessor.transform(X, y)


def test_preprocessor_full_pipeline_io(tmp_path):
    '''The complete preprocessing pipeline should create the expected output files.'''
    # 1. Create a dummy CSV inside the temporary test directory
    dummy_csv = tmp_path / 'dummy_tracks.csv'
    dummy_outdir = tmp_path / 'processed'
    
    # We need enough rows so train_test_split doesn't fail on empty datasets
    rng = np.random.default_rng(13)

    df = pd.DataFrame({
        'pt_true': rng.uniform(65, 105, 10),
        'hit_0_x': rng.uniform(-0.5, 0.5, 10),
        'hit_0_y': rng.uniform(-0.5, 0.5, 10),
    })
    df.to_csv(dummy_csv, index=False)
    
    # 2. Initialize the preprocessor with the temporary file
    processor = TrackPreprocessor(filepath=str(dummy_csv), random_state=13)
    
    # 3. Execute the full pipeline, instructing it to save to the temporary output dir
    processor.run_pipeline(str(dummy_outdir))
    
    # 4. Assert that the files were physically created on disk
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


def test_preprocessor_reports_missing_target_column(tmp_path):
    """Check that a CSV without pt_true produces a clear error."""
    csv_path = tmp_path / 'missing_target.csv'

    df = pd.DataFrame(
        {
            'hit_0_x': [0.1, 0.2, 0.3],
            'hit_0_y': [0.4, 0.5, 0.6],
        }
    )
    df.to_csv(csv_path, index=False)

    processor = TrackPreprocessor(filepath=str(csv_path))

    with pytest.raises(
        ValueError,
        match="required column 'pt_true'",
    ):
        processor.load_and_split()


def test_preprocessor_rejects_non_numeric_features(tmp_path):
    """Check that tracker feature columns contain only numeric values."""
    csv_path = tmp_path / 'non_numeric_features.csv'

    df = pd.DataFrame(
        {
            'hit_0_x': [0.1, 0.2, 0.3, 0.4],
            'hit_0_y': [0.5, 'invalid', 0.7, 0.8],
            'pt_true': [80.0, 85.0, 90.0, 95.0],
        }
    )
    df.to_csv(csv_path, index=False)

    processor = TrackPreprocessor(filepath=str(csv_path))

    with pytest.raises(
        ValueError,
        match='Feature columns must contain only numeric values',
    ):
        processor.load_and_split()


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
def test_preprocessor_rejects_non_finite_features(tmp_path, bad_value):
    '''Feature columns should not contain NaN or infinite values.'''

    df = pd.DataFrame({
        'pt_true': [70.0, 75.0, 80.0, 85.0, 90.0],
        'hit_0_x': [0.1, 0.2, bad_value, 0.4, 0.5],
        'hit_0_y': [0.2, 0.3, 0.4, 0.5, 0.6],
    })

    filepath = tmp_path / 'tracks.csv'
    df.to_csv(filepath, index=False)

    preprocessor = TrackPreprocessor(str(filepath))

    with pytest.raises(
        ValueError,
        match='Feature columns must contain only finite values',
    ):
        preprocessor.load_and_split()


def test_preprocessor_rejects_dataset_too_small_to_split(tmp_path):
    """Check that enough rows exist for train, validation, and test sets."""
    csv_path = tmp_path / 'too_small.csv'

    df = pd.DataFrame(
        {
            'hit_0_x': [0.1, 0.2, 0.3, 0.4],
            'hit_0_y': [0.5, 0.6, 0.7, 0.8],
            'pt_true': [80.0, 85.0, 90.0, 95.0],
        }
    )
    df.to_csv(csv_path, index=False)

    processor = TrackPreprocessor(filepath=str(csv_path))

    with pytest.raises(
        ValueError,
        match='at least 5 rows',
    ):
        processor.load_and_split()


def test_preprocessor_rejects_non_numeric_target(tmp_path):
    """Check that pt_true contains only numeric values."""
    csv_path = tmp_path / 'non_numeric_target.csv'

    df = pd.DataFrame(
        {
            'hit_0_x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'hit_0_y': [0.5, 0.6, 0.7, 0.8, 0.9],
            'pt_true': [80.0, 85.0, 'invalid', 95.0, 100.0],
        }
    )
    df.to_csv(csv_path, index=False)

    processor = TrackPreprocessor(filepath=str(csv_path))

    with pytest.raises(
        ValueError,
        match="Target column 'pt_true' must contain only numeric values",
    ):
        processor.load_and_split()


@pytest.mark.parametrize(
    'invalid_target',
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_preprocessor_rejects_non_finite_target(
    tmp_path,
    invalid_target,
):
    """Check that pt_true does not contain NaN or infinite values."""
    csv_path = tmp_path / 'non_finite_target.csv'

    df = pd.DataFrame(
        {
            'hit_0_x': [0.1, 0.2, 0.3, 0.4, 0.5],
            'hit_0_y': [0.5, 0.6, 0.7, 0.8, 0.9],
            'pt_true': [80.0, 85.0, invalid_target, 95.0, 100.0],
        }
    )
    df.to_csv(csv_path, index=False)

    processor = TrackPreprocessor(filepath=str(csv_path))

    with pytest.raises(
        ValueError,
        match="Target column 'pt_true' must contain only finite values",
    ):
        processor.load_and_split()


# Property of Wafa Mamaani. May 2026.
