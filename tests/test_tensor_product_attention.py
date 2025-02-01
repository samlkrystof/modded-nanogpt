import pytest
import torch
import torch.nn as nn

from ..tensor_product_attention import TensorProductAttention

def get_device() -> str:
    """Get the available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def device() -> str:
    return get_device()

@pytest.fixture
def model(device: str) -> TensorProductAttention:
    return TensorProductAttention(
        d_model=512,
        n_heads=8,
        q_rank=32,
        kv_rank=32,
        max_seq_len=2048
    ).to(device)

@pytest.fixture
def input_tensor(device: str) -> torch.Tensor:
    return torch.randn(2, 32, 512).to(device)

def test_tensor_product_shape(model: TensorProductAttention, input_tensor: torch.Tensor) -> None:
    """Test if attention module maintains expected output shape"""
    output = model(input_tensor)
    assert output.shape == input_tensor.shape, \
        f"Expected shape {input_tensor.shape}, got {output.shape}"

def test_tensor_product_backward(model: TensorProductAttention, device: str) -> None:
    """Test if attention module supports backward pass"""
    x = torch.randn(2, 32, 512, requires_grad=True).to(device)
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Expected gradients for input"
    assert x.grad.shape == x.shape, f"Expected gradient shape {x.shape}, got {x.grad.shape}"

def test_tensor_product_deterministic(model: TensorProductAttention, input_tensor: torch.Tensor) -> None:
    """Test if attention module produces deterministic outputs with same input"""
    torch.manual_seed(42)
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    
    torch.testing.assert_close(output1, output2)

def test_tensor_product_device_transfer(device: str) -> None:
    """Test if attention module can be transferred between devices"""
    model = TensorProductAttention(
        d_model=512,
        n_heads=8,
        q_rank=32,
        kv_rank=32,
        max_seq_len=2048
    )
    x = torch.randn(2, 32, 512)
    
    model = model.to(device)
    x = x.to(device)
    
    output = model(x)
    assert output.device.type == device, f"Expected output on {device}, got {output.device.type}"

def test_tensor_product_dimensions() -> None:
    """Test if different dimension configurations are handled correctly"""
    test_configs = [
        {"d_model": 512, "n_heads": 8, "q_rank": 32, "kv_rank": 32},
        {"d_model": 768, "n_heads": 12, "q_rank": 48, "kv_rank": 48},
        {"d_model": 1024, "n_heads": 16, "q_rank": 64, "kv_rank": 64},
    ]
    
    for config in test_configs:
        model = TensorProductAttention(max_seq_len=2048, **config)
        x = torch.randn(2, 32, config["d_model"])
        output = model(x)
        assert output.shape == x.shape, \
            f"Failed for config {config}. Expected shape {x.shape}, got {output.shape}"

def test_tensor_product_projection_shapes(model: TensorProductAttention) -> None:
    """Test if projection layers maintain correct shapes"""
    head_dim = model.d_model // model.n_heads
    
    # Test A projections
    assert model.aq_proj.out_features == model.q_rank * model.n_heads, \
        "AQ projection dimension mismatch"
    assert model.ak_proj.out_features == model.kv_rank * model.n_heads, \
        "AK projection dimension mismatch"
    assert model.av_proj.out_features == model.kv_rank * model.n_heads, \
        "AV projection dimension mismatch"
    
    # Test B projections
    assert model.bq_proj.out_features == model.q_rank * head_dim, \
        "BQ projection dimension mismatch"
    assert model.bk_proj.out_features == model.kv_rank * head_dim, \
        "BK projection dimension mismatch"
    assert model.bv_proj.out_features == model.kv_rank * head_dim, \
        "BV projection dimension mismatch"

def test_tensor_product_initialization() -> None:
    """Test if weights are properly initialized"""
    model = TensorProductAttention(
        d_model=512,
        n_heads=8,
        q_rank=32,
        kv_rank=32,
        max_seq_len=2048
    )
    model._init_weights()
    
    # Test A projections initialization
    aq_tensor = model.aq_proj.weight.view(512, 8, 32)
    ak_tensor = model.ak_proj.weight.view(512, 8, 32)
    av_tensor = model.av_proj.weight.view(512, 8, 32)
    
    assert not torch.allclose(aq_tensor, torch.zeros_like(aq_tensor)), \
        "AQ weights not properly initialized"
    assert not torch.allclose(ak_tensor, torch.zeros_like(ak_tensor)), \
        "AK weights not properly initialized"
    assert not torch.allclose(av_tensor, torch.zeros_like(av_tensor)), \
        "AV weights not properly initialized"
    
    # Test B projections initialization
    head_dim = 512 // 8
    bq_tensor = model.bq_proj.weight.view(512, 32, head_dim)
    bk_tensor = model.bk_proj.weight.view(512, 32, head_dim)
    bv_tensor = model.bv_proj.weight.view(512, 32, head_dim)
    
    assert not torch.allclose(bq_tensor, torch.zeros_like(bq_tensor)), \
        "BQ weights not properly initialized"
    assert not torch.allclose(bk_tensor, torch.zeros_like(bk_tensor)), \
        "BK weights not properly initialized"
    assert not torch.allclose(bv_tensor, torch.zeros_like(bv_tensor)), \
        "BV weights not properly initialized"

def test_tensor_product_rotary_embeddings(model: TensorProductAttention) -> None:
    """Test if rotary embeddings are properly applied"""
    # Check if rotary embeddings are precomputed
    assert hasattr(model, 'cos'), "Rotary cosine embeddings not found"
    assert hasattr(model, 'sin'), "Rotary sine embeddings not found"
    
    # Check shapes of rotary embeddings
    head_dim = model.d_model // model.n_heads
    assert model.cos.shape[1] == head_dim // 2, \
        f"Expected cosine embedding dimension {head_dim // 2}, got {model.cos.shape[1]}"
    assert model.sin.shape[1] == head_dim // 2, \
        f"Expected sine embedding dimension {head_dim // 2}, got {model.sin.shape[1]}"

def test_tensor_product_causal_attention(model: TensorProductAttention, device: str) -> None:
    """Test if causal attention mask is properly applied"""
    x = torch.randn(2, 32, 512).to(device)
    output = model(x)
    
    # Create a second input with modified later positions
    x_modified = x.clone()
    x_modified[:, 16:] = torch.randn_like(x_modified[:, 16:])
    output_modified = model(x_modified)
    
    # Check if early positions are unaffected by changes in later positions
    torch.testing.assert_close(
        output[:, :16], 
        output_modified[:, :16],
        msg="Causal masking not working properly - early positions affected by later ones"
    ) 