import pytest
import numpy as np
import pandas as pd
from simulation import generate_toy_tracks

def test_generate_toy_tracks_expected_shape():
    '''
    WHAT: Tests that the simulation outputs a dataframe with the correct dimensions.
    WHY: Downstream preprocessing expects exactly 1 target column and 72 coordinate columns (36 layers * 2).
    '''
    df = generate_toy_tracks(num_samples = 15)
    
    #1 target (pt_true) + 72 features (36 pairs of X and Y coordinates) = 73 columns
    assert df.shape == (15, 73)

def test_generate_toy_tracks_respects_random_seed():
    '''
    WHAT: Verifies that executing the generator twice with the same seed yields identical data.
    WHY: Reproducibility is required for scientific analysis and debugging.
    '''
    df1 = generate_toy_tracks(num_samples = 5, random_seed = 13)
    df2 = generate_toy_tracks(num_samples = 5, random_seed = 13)
    
    #Utilizing pandas testing tools for robust float comparisons
    pd.testing.assert_frame_equal(df1, df2)

def test_generate_toy_tracks_pT_within_physical_bounds():
    '''
    WHAT: Checks that the generated true transverse momentum falls within the requested range.
    WHY: Ensures the kinematic sampling function is operating in the correct physical regime (MeV/c).
    '''
    df = generate_toy_tracks(num_samples = 100)
    pt_values = df['pt_true'].values
    
    assert np.all(pt_values >= 65.0)
    assert np.all(pt_values <= 105.0)

def test_generate_toy_tracks_zero_samples_raises_error():
    '''
    WHAT: Tests that requesting zero or negative samples triggers a ValueError.
    WHY: Prevents downstream initialization errors with empty arrays.
    '''
    with pytest.raises(ValueError):
        generate_toy_tracks(num_samples = 0)


@pytest.mark.parametrize(
    'kwargs, expected_message',
    [
        (
            {'b_field': 0.0},
            'magnetic field strength must be positive',
        ),
        (
            {'efficiency': 1.5},
            'Efficiency must be between 0 and 1',
        ),
        (
            {'efficiency': -0.1},
            'Efficiency must be between 0 and 1',
        ),
        (
            {'resolution': -0.001},
            'Resolution cannot be negative',
        ),
        (
            {'layers': []},
            'At least one detector layer is required',
        ),
        (
            {'layers': [0.4, -0.5]},
            'All detector layer radii must be positive',
        ),
    ],
)
def test_generate_toy_tracks_rejects_invalid_parameters(
    kwargs,
    expected_message,
):
    """Check that physically invalid simulation inputs are rejected."""
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        generate_toy_tracks(
            num_samples=5,
            **kwargs,
        )

#Property of Wafa Mamani. May 2026.