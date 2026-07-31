import pytest
import torch
from model import TrackMomentumRegressor

def test_model_forward_pass_output_shape():
    '''
    WHAT: Verifies that the neural network returns a tensor of shape (batch_size, 1).
    WHY: PyTorch loss functions require strict dimensional alignment between predictions and targets.
    '''
    model = TrackMomentumRegressor(input_dim = 72, pad_val = -9999.0)
    
    #Simulate a batch of 16 tracks
    dummy_input = torch.randn(16, 72) 
    output = model(dummy_input)
    
    assert output.shape == (16, 1)

def test_model_dynamic_masking_zeroes_padding():
    '''
    WHAT: Verifies the masking mechanism physically neutralizes padding inside the forward pass.
    WHY: A -9999.0 hitting the first linear layer will cause gradient explosion unless explicitly masked to 0.0.
    '''
    model = TrackMomentumRegressor(input_dim = 2, pad_val = -9999.0)
    
    #Override the network to just return the first linear layer's inputs for testing
    model.network = torch.nn.Identity()
    
    dummy_input = torch.tensor([[5.0, -9999.0]], dtype = torch.float32)
    output = model(dummy_input)
    
    #The 5.0 should be preserved, the -9999.0 should become 0.0
    expected_tensor = torch.tensor([[5.0, 0.0]], dtype = torch.float32)
    
    assert torch.allclose(output, expected_tensor)

#Property of Wafa Mamani. May 2026.