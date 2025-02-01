import pytest
import torch
import torch.nn as nn

from ..latent_attention import MultiheadLatentAttention

def get_device() -> str:
    """Get the available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def device() -> str:
    return get_device()

@pytest.fixture
def model(device: str) -> MultiheadLatentAttention:
    return MultiheadLatentAttention(
        d_model=512,
        n_heads=8,
        kv_dim=256,
        q_dim=256,
        rope_dim=32,
        max_seq_len=2048
    ).to(device)

@pytest.fixture
def input_tensor(device: str) -> torch.Tensor:
    return torch.randn(2, 32, 512).to(device)

def test_latent_attention_shape(model: MultiheadLatentAttention, input_tensor: torch.Tensor) -> None:
    """Test if attention module maintains expected output shape"""
    output = model(input_tensor)
    assert output.shape == input_tensor.shape, \
        f"Expected shape {input_tensor.shape}, got {output.shape}"

def test_latent_attention_backward(model: MultiheadLatentAttention, device: str) -> None:
    """Test if attention module supports backward pass"""
    x = torch.randn(2, 32, 512, requires_grad=True).to(device)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Expected gradients for input"
    assert x.grad.shape == x.shape, f"Expected gradient shape {x.shape}, got {x.grad.shape}"

def test_latent_attention_deterministic(model: MultiheadLatentAttention, input_tensor: torch.Tensor) -> None:
    """Test if attention module produces deterministic outputs with same input"""
    torch.manual_seed(42)
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    
    torch.testing.assert_close(output1, output2)

def test_latent_attention_device_transfer(device: str) -> None:
    """Test if attention module can be transferred between devices"""
    model = MultiheadLatentAttention(
        d_model=512,
        n_heads=8,
        kv_dim=256,
        q_dim=256,
        rope_dim=32
    )
    x = torch.randn(2, 32, 512)
    
    model = model.to(device)
    x = x.to(device)
    
    output = model(x)
    assert output.device.type == device, f"Expected output on {device}, got {output.device.type}"

def test_latent_attention_dimensions() -> None:
    """Test if different dimension configurations are handled correctly"""
    test_configs = [
        {"d_model": 512, "n_heads": 8, "kv_dim": 256, "q_dim": 256, "rope_dim": 32},
        {"d_model": 768, "n_heads": 12, "kv_dim": 384, "q_dim": 384, "rope_dim": 64},
        {"d_model": 1024, "n_heads": 16, "kv_dim": 512, "q_dim": 512, "rope_dim": 128},
    ]
    
    for config in test_configs:
        model = MultiheadLatentAttention(**config)
        x = torch.randn(2, 32, config["d_model"])
        output = model(x)
        assert output.shape == x.shape, \
            f"Failed for config {config}. Expected shape {x.shape}, got {output.shape}"

def test_latent_attention_projection_shapes(model: MultiheadLatentAttention) -> None:
    """Test if internal projection layers maintain correct shapes"""
    assert model.rope_key.out_features == model.rope_dim, \
        "Rope key projection dimension mismatch"
    assert model.rope_query.out_features == model.rope_dim, \
        "Rope query projection dimension mismatch"
    assert model.down_kv.out_features == model.kv_dim, \
        "KV dimension reduction mismatch"
    assert model.down_query.out_features == model.q_dim, \
        "Query dimension reduction mismatch" 