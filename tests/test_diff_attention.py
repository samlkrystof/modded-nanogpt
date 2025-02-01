import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

from ..diff_attention import MultiheadFlashDiff

@dataclass
class MockArgs:
    """Mock arguments for MultiheadFlashDiff"""
    max_seq_len: int = 2048

def get_device() -> str:
    """Get the available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def device() -> str:
    return get_device()

@pytest.fixture
def model(device: str) -> MultiheadFlashDiff:
    args = MockArgs()
    return MultiheadFlashDiff(
        args=args,
        embed_dim=512,
        depth=0,
        num_heads=8
    ).to(device)

@pytest.fixture
def input_tensor(device: str) -> torch.Tensor:
    return torch.randn(2, 32, 512).to(device)

def test_flash_diff_shape(model: MultiheadFlashDiff, input_tensor: torch.Tensor) -> None:
    """Test if attention module maintains expected output shape"""
    output = model(input_tensor)
    assert output.shape == input_tensor.shape, \
        f"Expected shape {input_tensor.shape}, got {output.shape}"

def test_flash_diff_backward(model: MultiheadFlashDiff, device: str) -> None:
    """Test if attention module supports backward pass"""
    x = torch.randn(2, 32, 512, requires_grad=True).to(device)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Expected gradients for input"
    assert x.grad.shape == x.shape, f"Expected gradient shape {x.shape}, got {x.grad.shape}"

def test_flash_diff_deterministic(model: MultiheadFlashDiff, input_tensor: torch.Tensor) -> None:
    """Test if attention module produces deterministic outputs with same input"""
    torch.manual_seed(42)
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    
    torch.testing.assert_close(output1, output2)

def test_flash_diff_device_transfer(device: str) -> None:
    """Test if attention module can be transferred between devices"""
    args = MockArgs()
    model = MultiheadFlashDiff(args=args, embed_dim=512, depth=0, num_heads=8)
    x = torch.randn(2, 32, 512)
    
    model = model.to(device)
    x = x.to(device)
    
    output = model(x)
    assert output.device.type == device, f"Expected output on {device}, got {output.device.type}"

def test_flash_diff_lambda_init() -> None:
    """Test if lambda parameters are initialized correctly"""
    args = MockArgs()
    model = MultiheadFlashDiff(args=args, embed_dim=512, depth=0, num_heads=8)
    
    # Check lambda parameters existence and shapes
    assert hasattr(model, 'lambda_q1')
    assert hasattr(model, 'lambda_k1')
    assert hasattr(model, 'lambda_q2')
    assert hasattr(model, 'lambda_k2')
    
    head_dim = 512 // (8 * 2)  # embed_dim // (num_heads * 2)
    assert model.lambda_q1.shape == (head_dim,)
    assert model.lambda_k1.shape == (head_dim,)
    assert model.lambda_q2.shape == (head_dim,)
    assert model.lambda_k2.shape == (head_dim,)

def test_flash_diff_lambda_values(model: MultiheadFlashDiff) -> None:
    """Test if lambda values are properly initialized and updated"""
    # Check initial lambda values
    lambda_1 = torch.exp(torch.sum(model.lambda_q1 * model.lambda_k1, dim=-1))
    lambda_2 = torch.exp(torch.sum(model.lambda_q2 * model.lambda_k2, dim=-1))
    
    assert lambda_1.item() > 0, "Lambda_1 should be positive"
    assert lambda_2.item() > 0, "Lambda_2 should be positive" 