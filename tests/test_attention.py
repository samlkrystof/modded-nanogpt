import pytest
import torch
import torch.nn as nn
from typing import Type, Any
from dataclasses import dataclass

from ..diff_attention import MultiheadFlashDiff
from ..latent_attention import MultiheadLatentAttention

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
def batch_size() -> int:
    return 2

@pytest.fixture
def seq_len() -> int:
    return 32

@pytest.fixture
def embed_dim() -> int:
    return 512

class TestAttentionModules:
    
    @pytest.mark.parametrize(
        "attention_cls,kwargs",
        [
            (
                MultiheadFlashDiff,
                {
                    "args": MockArgs(),
                    "embed_dim": 512,
                    "depth": 0,
                    "num_heads": 8
                }
            ),
            (
                MultiheadLatentAttention,
                {
                    "d_model": 512,
                    "n_heads": 8,
                    "kv_dim": 256,
                    "q_dim": 256,
                    "rope_dim": 32,
                    "max_seq_len": 2048
                }
            )
        ]
    )
    def test_attention_shape(
        self,
        attention_cls: Type[nn.Module],
        kwargs: dict[str, Any],
        device: str,
        batch_size: int,
        seq_len: int,
        embed_dim: int
    ) -> None:
        """Test if attention modules maintain expected output shapes"""
        
        model = attention_cls(**kwargs).to(device)
        x = torch.randn(batch_size, seq_len, embed_dim).to(device)
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, embed_dim), \
            f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"

    @pytest.mark.parametrize(
        "attention_cls,kwargs",
        [
            (
                MultiheadFlashDiff,
                {
                    "args": MockArgs(),
                    "embed_dim": 512,
                    "depth": 0,
                    "num_heads": 8
                }
            ),
            (
                MultiheadLatentAttention,
                {
                    "d_model": 512,
                    "n_heads": 8,
                    "kv_dim": 256,
                    "q_dim": 256,
                    "rope_dim": 32,
                    "max_seq_len": 2048
                }
            )
        ]
    )
    def test_attention_backward(
        self,
        attention_cls: Type[nn.Module],
        kwargs: dict[str, Any],
        device: str,
        batch_size: int,
        seq_len: int,
        embed_dim: int
    ) -> None:
        """Test if attention modules support backward pass"""
        
        model = attention_cls(**kwargs).to(device)
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True).to(device)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Expected gradients for input"
        assert x.grad.shape == x.shape, f"Expected gradient shape {x.shape}, got {x.grad.shape}"

    @pytest.mark.parametrize(
        "attention_cls,kwargs",
        [
            (
                MultiheadFlashDiff,
                {
                    "args": MockArgs(),
                    "embed_dim": 512,
                    "depth": 0,
                    "num_heads": 8
                }
            ),
            (
                MultiheadLatentAttention,
                {
                    "d_model": 512,
                    "n_heads": 8,
                    "kv_dim": 256,
                    "q_dim": 256,
                    "rope_dim": 32,
                    "max_seq_len": 2048
                }
            )
        ]
    )
    def test_attention_deterministic(
        self,
        attention_cls: Type[nn.Module],
        kwargs: dict[str, Any],
        device: str,
        batch_size: int,
        seq_len: int,
        embed_dim: int
    ) -> None:
        """Test if attention modules produce deterministic outputs with same input"""
        
        torch.manual_seed(42)
        model = attention_cls(**kwargs).to(device)
        x = torch.randn(batch_size, seq_len, embed_dim).to(device)
        
        output1 = model(x)
        output2 = model(x)
        
        torch.testing.assert_close(output1, output2)

    @pytest.mark.parametrize(
        "attention_cls,kwargs",
        [
            (
                MultiheadFlashDiff,
                {
                    "args": MockArgs(),
                    "embed_dim": 512,
                    "depth": 0,
                    "num_heads": 8
                }
            ),
            (
                MultiheadLatentAttention,
                {
                    "d_model": 512,
                    "n_heads": 8,
                    "kv_dim": 256,
                    "q_dim": 256,
                    "rope_dim": 32,
                    "max_seq_len": 2048
                }
            )
        ]
    )
    def test_attention_device_transfer(
        self,
        attention_cls: Type[nn.Module],
        kwargs: dict[str, Any],
        device: str,
        batch_size: int,
        seq_len: int,
        embed_dim: int
    ) -> None:
        """Test if attention modules can be transferred between devices"""
        
        model = attention_cls(**kwargs)
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Move to target device
        model = model.to(device)
        x = x.to(device)
        
        output = model(x)
        assert output.device.type == device, f"Expected output on {device}, got {output.device.type}"

def test_flash_diff_lambda_init() -> None:
    """Test if MultiheadFlashDiff initializes lambda parameters correctly"""
    args = MockArgs()
    model = MultiheadFlashDiff(args=args, embed_dim=512, depth=0, num_heads=8)
    
    # Check if lambda parameters exist and have correct shapes
    assert hasattr(model, 'lambda_q1')
    assert hasattr(model, 'lambda_k1')
    assert hasattr(model, 'lambda_q2')
    assert hasattr(model, 'lambda_k2')
    
    head_dim = 512 // (8 * 2)  # embed_dim // (num_heads * 2)
    assert model.lambda_q1.shape == (head_dim,)
    assert model.lambda_k1.shape == (head_dim,)
    assert model.lambda_q2.shape == (head_dim,)
    assert model.lambda_k2.shape == (head_dim,)

def test_latent_attention_dimensions() -> None:
    """Test if MultiheadLatentAttention handles different dimension configurations"""
    model = MultiheadLatentAttention(
        d_model=512,
        n_heads=8,
        kv_dim=256,
        q_dim=256,
        rope_dim=32
    )
    
    # Test if the model properly handles the dimension reduction and expansion
    x = torch.randn(2, 32, 512)
    output = model(x)
    
    assert output.shape == (2, 32, 512), \
        f"Expected output shape (2, 32, 512), got {output.shape}" 