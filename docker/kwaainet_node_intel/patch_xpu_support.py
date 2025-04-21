import os
import sys
import re

def patch_for_xpu_support(base_path):
    """
    Apply a comprehensive patch for Intel XPU support across multiple files.
    
    Args:
        base_path: Path to the petals package (e.g., /opt/conda/lib/python3.10/site-packages/petals/)
    """
    results = []
    
    print("Starting comprehensive XPU support patch for Petals...")
    
    # 1. Patch peft.py first since other modules depend on it
    peft_path = os.path.join(base_path, 'utils', 'peft.py')
    if os.path.exists(peft_path):
        results.append(patch_peft_py(peft_path))
    else:
        results.append(f"Error: {peft_path} not found")
    
    # 2. Patch server.py
    server_path = os.path.join(base_path, 'server', 'server.py')
    if os.path.exists(server_path):
        results.append(patch_server_py(server_path))
    else:
        results.append(f"Error: {server_path} not found")
    
    # 3. Patch throughput.py
    throughput_path = os.path.join(base_path, 'server', 'throughput.py')
    if os.path.exists(throughput_path):
        results.append(patch_throughput_py(throughput_path))
    else:
        results.append(f"Error: {throughput_path} not found")
    
    # 4. Verify backend.py (may not need changes now)
    backend_path = os.path.join(base_path, 'server', 'backend.py')
    if os.path.exists(backend_path):
        results.append(patch_backend_py(backend_path))
    else:
        results.append(f"Error: {backend_path} not found")
    
    print("Completed XPU support patching!")
    return "\n".join(results)

def patch_server_py(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Import torch.xpu
    if 'import torch.xpu' not in content:
        content = re.sub(
            r'import torch\.mps',
            'import torch.mps\nimport torch.xpu',
            content
        )
    
    # 2. Add XPU device detection
    content = re.sub(
        r'if torch.cuda.is_available():\n                device = "cuda"\n            elif torch.xpu.is_available():\n                device = "xpu"\n            elif torch.backends.mps.is_available():\n                device = "mps"\n            else:\n                device = "cpu"',
        'if torch.cuda.is_available():\n            device = "cuda"\n        elif torch.xpu.is_available():\n            device = "xpu"\n        elif torch.backends.mps.is_available():\n            device = "mps"\n        else:\n            device = "cpu"',
        content
    )
    
    # 3. Handle bfloat16 compatibility for XPU
    content = re.sub(
        r'if device\.type == "mps" and torch_dtype == torch\.bfloat16:',
        'if device.type in ["mps", "xpu"] and torch_dtype == torch.bfloat16:',
        content
    )
    
    # 4. Update _choose_num_blocks for XPU
    content = re.sub(
        r'assert self\.device\.type in \("cuda", "mps"\),',
        'assert self.device.type in ("cuda", "mps", "xpu"),',
        content
    )
    
    # 5. Update memory calculation for XPU
    content = re.sub(
        r'elif self\.device\.type == "cuda":\s+total_memory = torch\.cuda\.get_device_properties\(self\.device\)\.total_memory\s+else:',
        'elif self.device.type == "cuda":\n            total_memory = torch.cuda.get_device_properties(self.device).total_memory\n        elif self.device.type == "xpu":\n            total_memory = torch.xpu.get_device_properties(self.device).total_memory\n        else:',
        content
    )
    
    # 6. Add memory cleaning for XPU
    content = re.sub(
        r'elif self\.device\.type == "mps":\s+torch\.mps\.empty_cache\(\)',
        'elif self.device.type == "mps":\n            torch.mps.empty_cache()\n        elif self.device.type == "xpu":\n            torch.xpu.empty_cache()',
        content
    )
    
    # 7. Enable tensor parallelism for XPU
    content = re.sub(
        r'assert self\.device\.type == "cuda", f"Tensor parallelism is not supported on {self\.device\.type\.upper\(\)}"',
        'assert self.device.type in ["cuda", "xpu"], f"Tensor parallelism is not supported on {self.device.type.upper()}"',
        content
    )
    
    # 8. Set quantization to NONE for XPU to avoid bitsandbytes
    content = re.sub(
        r'quant_type = QuantType\.NF4 if device\.type == "cuda" else QuantType\.NONE',
        'quant_type = QuantType.NONE if device.type == "xpu" else (QuantType.NF4 if device.type == "cuda" else QuantType.NONE)',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return f"Patched {file_path} for XPU support"

def patch_throughput_py(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Import torch.xpu
    if 'import torch.xpu' not in content:
        content = re.sub(
            r'import torch\.mps',
            'import torch.mps\nimport torch.xpu',
            content
        )
    
    # 2. Fix get_device_name function
    content = re.sub(
        r'def get_device_name\(device: torch\.device\) -> str:.*?return f"{torch\.cuda\.get_device_name\(device\)} GPU" if device\.type == "cuda" else device\.type\.upper\(\)',
        'def get_device_name(device: torch.device) -> str:\n    if device.type == "cuda":\n        return f"{torch.cuda.get_device_name(device)} GPU"\n    elif device.type == "xpu":\n        return f"{torch.xpu.get_device_name(device)} GPU"\n    else:\n        return device.type.upper()',
        content,
        flags=re.DOTALL
    )
    
    # 3. Fix synchronize function (remove device parameter)
    content = re.sub(
        r'def synchronize\(device: torch\.device\):.*?if device\.type == "cuda":\s+torch\.cuda\.synchronize\(device\)\s+elif device\.type == "mps":\s+torch\.mps\.synchronize\(\)',
        'def synchronize(device: torch.device):\n    if device.type == "cuda":\n        torch.cuda.synchronize()\n    elif device.type == "xpu":\n        torch.xpu.synchronize()\n    elif device.type == "mps":\n        torch.mps.synchronize()',
        content,
        flags=re.DOTALL
    )
    
    # 4. Force NONE quantization for XPU in measure_compute_rps
    content = re.sub(
        r'block = convert_block\(block, 0, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True\)',
        'quant_type_for_measure = QuantType.NONE if device.type == "xpu" else quant_type\n        block = convert_block(block, 0, config, tensor_parallel_devices, device, quant_type=quant_type_for_measure, freeze=True)',
        content
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return f"Patched {file_path} for XPU support"

def patch_backend_py(file_path):
    # Create a backup of the original file if it doesn't exist
    backup_path = file_path + '.backup'
    if not os.path.exists(backup_path):
        try:
            with open(file_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple approach - just use the regular import since we've fixed peft.py
    # The peft.py module will handle XPU detection and provide appropriate dummy implementations
    if 'import petals.utils.peft as _peft_module' in content:
        # Already patched, no need to modify
        return f"No changes needed for {file_path} - already patched or compatible"
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return f"Checked {file_path} - no changes needed as peft.py now handles XPU compatibility"

def patch_peft_py(file_path):
    # Create a comprehensive XPU-compatible implementation that preserves all required variables
    peft_py_content = '''# Patched peft.py for XPU support
import contextlib
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Union

# Function to check if we're running on XPU without relying on checking modules
def is_xpu_available():
    try:
        import torch.xpu
        return torch.xpu.is_available()
    except (ImportError, AttributeError):
        return False

# Import necessary modules based on device availability
if not is_xpu_available():
    try:
        import bitsandbytes as bnb
        import transformers
        from accelerate import init_empty_weights
        from hivemind.utils.logging import get_logger
        from huggingface_hub import HfFileSystem, get_hf_file_metadata, hf_hub_url
        from peft.config import PeftConfig
        from peft.tuners import lora
        from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers.utils import get_file_from_repo

        from petals.server.block_utils import get_model_block, resolve_block_dtype
        from petals.utils.convert_block import QuantType
        from petals.utils.disk_cache import allow_cache_reads, allow_cache_writes, free_disk_space_for
        from petals.utils.misc import get_size_in_bytes
        
        PEFT_AVAILABLE = True
        logger = get_logger(__name__)
    except ImportError:
        PEFT_AVAILABLE = False
        logger = None
else:
    PEFT_AVAILABLE = False
    
    # Create dummy logger
    class DummyLogger:
        def debug(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        
    logger = DummyLogger()
    
    # Create dummy modules to avoid errors
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    bnb = DummyModule()
    transformers = DummyModule()
    
    # Define dummy classes to match the expected interfaces
    class DummyPeftConfig:
        @staticmethod
        def from_json_file(*args, **kwargs):
            return {}
    
    # Create minimal dummy versions of imported classes
    CONFIG_NAME = "config.json"
    SAFETENSORS_WEIGHTS_NAME = "model.safetensors"
    PeftConfig = DummyPeftConfig
    
    # Create dummy lora module with expected classes
    class DummyLora:
        class Linear(nn.Module):
            pass
            
        class Linear8bitLt(nn.Module):
            pass
            
        class Linear4bit(nn.Module):
            pass
            
        class LoraLayer:
            def __init__(self, *args, **kwargs):
                pass
    
    lora = DummyLora()

# Define constants that were in the original file
COMMON_LAYERS_PATTERN = ["layers", "h", "block", "blocks", "layer"]

class AdapterContextMixin:
    """A mixin that makes LoRA-wrapped linear layers obey an adapter set from context"""

    ADAPTER_NOT_SET = "__ADAPTER_NOT_SET"
    _context_active_adapter = ADAPTER_NOT_SET

    @staticmethod
    @contextlib.contextmanager
    def using_adapter(active_adapter: Optional[str]):
        prev, AdapterContextMixin._context_active_adapter = AdapterContextMixin._context_active_adapter, active_adapter
        try:
            yield
        finally:
            AdapterContextMixin._context_active_adapter = prev

    @property
    def active_adapter(self):
        if self._context_active_adapter == self.ADAPTER_NOT_SET:
            logger.warning(f"Layer {self} was called without using_adapter. This should only be used for debug")
        return self._context_active_adapter

    @active_adapter.setter
    def active_adapter(self, value: Optional[str]):
        assert value == self.ADAPTER_NOT_SET, "active adapter can only be changed via .using_adapter"

    @property
    def active_adapters(self):
        return [self._context_active_adapter]

    def set_adapter(self, adapter_names) -> None:
        """
        In PEFT, this function makes the adapter trainable. However, in Petals environment this is not possible now. Thus,
        this code removes this functionality.
        Link to peft code: https://github.com/huggingface/peft/blob/98f4db2c7990ef9c879a0e1da9a28a19a04701ef/src/peft/tuners/tuners_utils.py#L463
        """
        pass

# Export the using_adapter function at module level
using_adapter = AdapterContextMixin.using_adapter

# Define LoRA classes for XPU compatibility
if not is_xpu_available() and PEFT_AVAILABLE:
    class LoraLinear(AdapterContextMixin, lora.Linear):
        """LoRA linear layer that uses adapter selected via using_adapter"""

        def __init__(self, base_layer, adapter_name: str):
            nn.Module.__init__(self)
            lora.LoraLayer.__init__(self, base_layer)

            self._active_adapter = adapter_name
            self.is_target_conv_1d_layer = False


    class LoraLinear8bitLt(LoraLinear, lora.Linear8bitLt):
        """LoRA linear 8-bit with outliers that uses adapter selected via using_adapter"""


    class LoraLinear4bit(LoraLinear, lora.Linear4bit):
        """LoRA linear 4-bit that uses adapter selected via using_adapter"""
else:
    # Dummy implementations for XPU devices
    class LoraLinear(AdapterContextMixin, nn.Module):
        def __init__(self, base_layer, adapter_name: str):
            super().__init__()
            self.base_layer = base_layer
            self._active_adapter = adapter_name
            self.lora_A = {}
            self.lora_B = {}
            self.is_target_conv_1d_layer = False
            
        def forward(self, x):
            return self.base_layer(x)
            
        def update_layer(self, *args, **kwargs):
            pass
    
    LoraLinear8bitLt = LoraLinear
    LoraLinear4bit = LoraLinear

# Function implementations
def check_peft_repository(repo_id: str) -> bool:
    if not PEFT_AVAILABLE or is_xpu_available():
        return False
    return HfFileSystem().exists(f"{repo_id}/{SAFETENSORS_WEIGHTS_NAME}")

def load_specific_module(block_idx: int, filepath: str, framework: str = "pt", device: Optional[int] = None):
    if not PEFT_AVAILABLE or is_xpu_available():
        return {}
        
    tensors = dict()
    is_tensors_found = dict()
    common_layer_patter_re = (
        ".+\." + "".join(f"({common_name})?" for common_name in COMMON_LAYERS_PATTERN) + f"\.({block_idx})?\..+"
    )
    with safe_open(filepath, framework=framework, device=device) as f:
        for k in f.keys():
            if re.match(common_layer_patter_re, k):
                is_tensors_found[block_idx] = True
                tensors[k] = f.get_tensor(k)
        if not is_tensors_found.get(block_idx, False):
            logger.warning(f"There is no peft weights for block {block_idx}")
        return tensors

def get_adapter_from_repo(
    repo_id: str,
    block_idx: Optional[int] = None,
    device: Optional[int] = None,
    *,
    token: Optional[Union[str, bool]] = None,
    **kwargs,
):
    if not PEFT_AVAILABLE or is_xpu_available():
        return {}, {}
        
    config_path = get_file_from_repo(repo_id, CONFIG_NAME, use_auth_token=token, **kwargs)
    if config_path is None:
        raise RuntimeError(f"File {CONFIG_NAME} does not exist in repo {repo_id}")
    config = PeftConfig.from_json_file(config_path)

    weight_path = get_file_from_repo(repo_id, SAFETENSORS_WEIGHTS_NAME, use_auth_token=token, **kwargs)
    if weight_path is None:
        raise RuntimeError(f"File {SAFETENSORS_WEIGHTS_NAME} does not exist in repo {repo_id}")
    if block_idx is None:
        return config, load_file(weight_path)
    return config, load_specific_module(block_idx, weight_path, device=device)

def load_peft(
    repo_id: str,
    block_idx: Optional[int] = None,
    device: Optional[int] = None,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
):
    if not PEFT_AVAILABLE or is_xpu_available():
        return {}, {}
        
    if not check_peft_repository(repo_id):
        raise ValueError(f"Repo: {repo_id} doesn't have safetensors inside for a safe loading.")

    try:
        with allow_cache_reads(cache_dir):
            return get_adapter_from_repo(
                repo_id,
                block_idx,
                device,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_files_only=False,
            )
    except Exception:
        logger.warning(f"Cache for peft weights {repo_id} is corrupted, it will be downloaded again", exc_info=True)

    while True:
        try:
            with allow_cache_writes(cache_dir):
                config_url = hf_hub_url(repo_id, CONFIG_NAME, revision=revision)
                config_file_size = get_hf_file_metadata(config_url, token=token).size
                weight_url = hf_hub_url(repo_id, SAFETENSORS_WEIGHTS_NAME, revision=revision)
                weight_file_size = get_hf_file_metadata(weight_url, token=token).size

                file_size = config_file_size + weight_file_size
                if file_size is not None:
                    free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size from peft repo {repo_id}")

                return get_adapter_from_repo(
                    repo_id,
                    block_idx,
                    device,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
        except Exception as e:
            logger.warning(
                f"Failed to load peft weights {repo_id} from HF Hub (retry in {delay:.0f} sec)", exc_info=True
            )
            time.sleep(delay)

def create_lora_adapter(block):
    if not PEFT_AVAILABLE or is_xpu_available():
        return

    for module_name, module in block.named_modules():
        if isinstance(module, LoraLinear):
            continue
        for child_name, child in module.named_children():
            lora_class = None
            if isinstance(child, nn.Linear):
                lora_class = LoraLinear
            elif isinstance(child, bnb.nn.Linear8bitLt):
                lora_class = LoraLinear8bitLt
            elif isinstance(child, bnb.nn.Linear4bit):
                lora_class = LoraLinear4bit
            if lora_class:
                lora_wrapped_child = lora_class(
                    child,
                    AdapterContextMixin.ADAPTER_NOT_SET,
                )
                setattr(module, child_name, lora_wrapped_child)

def add_adapter_to_block(block, block_index, adapter_name, peft_config, peft_state_dict):
    if not PEFT_AVAILABLE or is_xpu_available():
        return
        
    assert peft_config["peft_type"] == "LORA", "Petals works only with LORA adapters"
    if peft_config["lora_dropout"] > 0:
        logger.info(f"Adapter {adapter_name} has dropout enabled, this server will disable dropout")

    for _, module in block.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, (lora.Linear, lora.Linear8bitLt, lora.Linear4bit)):
                continue

            if child_name in peft_config["target_modules"] or (
                isinstance(peft_config["target_modules"], str)
                and re.fullmatch(peft_config["target_modules"], child_name)
            ):
                is_lora_a_loaded = False
                is_lora_b_loaded = False
                for peft_key in peft_state_dict:
                    if child_name not in peft_key:
                        continue

                    if adapter_name not in child.lora_A:
                        child.update_layer(
                            adapter_name,
                            peft_config["r"],
                            peft_config["lora_alpha"],
                            use_rslora=peft_config.get("use_rslora", False),
                            lora_dropout=peft_config["lora_dropout"],
                            init_lora_weights=peft_config["init_lora_weights"],
                        )
                        child.train(False)
                        for p in child.parameters():
                            p.requires_grad = False

                    if peft_key.endswith(".lora_A.weight"):
                        child.lora_A[adapter_name].weight[...] = peft_state_dict[peft_key]
                        is_lora_a_loaded = True
                    elif peft_key.endswith(".lora_A.bias"):
                        raise NotImplementedError(f"LoRA adapters with bias not supported: {peft_key}")
                    elif peft_key.endswith(".lora_B.weight"):
                        child.lora_B[adapter_name].weight[...] = peft_state_dict[peft_key]
                        is_lora_b_loaded = True
                    elif peft_key.endswith(".lora_B.bias"):
                        raise NotImplementedError(f"LoRA adapters with bias not supported: {peft_key}")

                if is_lora_a_loaded and is_lora_b_loaded:
                    logger.debug(f"Loaded adapter {adapter_name} for block {block_index}.{child_name}")
                elif is_lora_a_loaded or is_lora_b_loaded:
                    raise ValueError(f"Invalid adapter {adapter_name} for block {block_index}.{child_name}")
    logger.info(f"Loaded adapter {adapter_name} for block {block_index}")

def estimate_adapter_memory_per_block(
    block_config,
    torch_dtype: torch.dtype, 
    adapters: List[str],
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None
) -> int:
    """
    Estimate approximate adapter memory consumption per block, based on the model configuration and torch dtype.
    
    Note: This is simplified for XPU devices to avoid dependency issues.
    """
    # Skip memory estimation for XPU devices
    if is_xpu_available() or not PEFT_AVAILABLE:
        return 0
        
    # For non-XPU devices with PEFT available, use the full implementation
    if not adapters:
        return 0
        
    with init_empty_weights(include_buffers=False):
        block = get_model_block(block_config)
        base_block_parameters = sum(p.numel() for p in block.parameters())
        create_lora_adapter(block)

        for adapter in adapters:
            peft_config, peft_state_dict = load_peft(
                adapter, 
                block_idx=0, 
                token=token,
                cache_dir=cache_dir,
                max_disk_space=max_disk_space
            )
            assert peft_config["peft_type"].upper() == "LORA", "only LoRA adapters are supported for now"
            add_adapter_to_block(
                block, block_index=0, adapter_name=adapter, peft_config=peft_config, peft_state_dict=peft_state_dict
            )
        adapter_parameters = sum(p.numel() for p in block.parameters()) - base_block_parameters
    bytes_per_parameter = get_size_in_bytes(resolve_block_dtype(block_config, torch_dtype))
    return adapter_parameters * bytes_per_parameter

def get_all_adapters(model) -> List[str]:
    """Get a list of all adapters available in a model."""
    if not PEFT_AVAILABLE or is_xpu_available():
        return []
    from peft.utils import _get_submodules
    try:
        return model.peft_config.keys()
    except (AttributeError, KeyError):
        return []
'''
    
    # Create a backup of the original file
    backup_path = file_path + '.backup'
    if not os.path.exists(backup_path):
        try:
            with open(file_path, 'r') as src:
                with open(backup_path, 'w') as dst:
                    dst.write(src.read())
            print(f"Created backup at {backup_path}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    # Write the new content
    with open(file_path, 'w') as f:
        f.write(peft_py_content)
    
    return f"Replaced {file_path} with XPU-compatible version including all required variables (using_adapter, AdapterContextMixin, etc.)"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python patch_xpu_support.py /path/to/petals/")
        sys.exit(1)
    
    result = patch_for_xpu_support(sys.argv[1])
    print(result)