import numpy as np
import pandas as pd
import pytest

from simulation import generate_toy_tracks


def test_generate_toy_tracks_returns_requested_number_of_tracks():
    '''The generator should return one row for every requested track.'''
    df = generate_toy_tracks(num_samples=15)
    
    assert len(df) == 15


def test_generate_toy_tracks_respects_random_seed():
    '''The same random seed should reproduce the same simulated tracks.'''

    df1 = generate_toy_tracks(num_samples=5, random_seed=13)
    df2 = generate_toy_tracks(num_samples=5, random_seed=13)
    
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_toy_tracks_pt_within_sampling_range():
    '''Generated transverse momenta should remain inside the chosen sampling range.'''
    df = generate_toy_tracks(num_samples=100)
    pt_values = df['pt_true'].values
    
    assert np.all(pt_values >= 65.0)
    assert np.all(pt_values <= 105.0)


def test_generate_toy_tracks_zero_samples_raises_error():
    '''Requesting zero tracks should raise a ValueError.'''
    with pytest.raises(ValueError):
        generate_toy_tracks(num_samples=0)


def test_hits_lie_on_detector_layer_without_smearing():
    '''Without smearing, recorded hits should lie on the detector layer.'''

    layer_radius = 0.4

    df = generate_toy_tracks(
        num_samples=20,
        layers=[layer_radius],
        efficiency=1.0,
        resolution=0.0,
        random_seed=13,
    )

    hit_radius = np.sqrt(df['hit_0_x'] ** 2 + df['hit_0_y'] ** 2)

    assert np.allclose(hit_radius, layer_radius)


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
    '''Invalid simulation inputs should be rejected.'''
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        generate_toy_tracks(
            num_samples=5,
            **kwargs,
        )

# Property of Wafa Mamaani. May 2026.
