import torch

from model import TrackMomentumRegressor


def test_model_forward_pass_output_shape():
    '''The model should return one pT prediction for every input track.'''
    model = TrackMomentumRegressor(input_dim=72, pad_val=-9999.0)

    dummy_input = torch.randn(16, 72)
    output = model(dummy_input)
    
    assert output.shape == (16, 1)


def test_model_masks_padding_before_prediction():
    '''Padding should have the same effect as replacing it with zero.'''

    model = TrackMomentumRegressor(input_dim=2, pad_val=-9999.0)

    padded_input = torch.tensor(
        [[5.0, -9999.0]],
        dtype=torch.float32,
    )
    zeroed_input = torch.tensor(
        [[5.0, 0.0]],
        dtype=torch.float32,
    )

    padded_output = model(padded_input)
    zeroed_output = model(zeroed_input)

    assert torch.allclose(padded_output, zeroed_output)


# Property of Wafa Mamaani. May 2026.
