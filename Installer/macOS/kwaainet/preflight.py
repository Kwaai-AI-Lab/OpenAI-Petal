"""
Pre-flight checks for KwaaiNet startup
"""
import os
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model size estimates in GB (conservative estimates including cache overhead)
MODEL_SIZE_ESTIMATES = {
    # Popular models with size estimates
    'unsloth/Llama-3.1-8B-Instruct': 16.0,
    'unsloth/Llama-3.1-70B-Instruct': 140.0,
    'meta-llama/Llama-2-7b-hf': 14.0,
    'meta-llama/Llama-2-13b-hf': 26.0,
    'meta-llama/Llama-2-70b-hf': 140.0,
    'bigscience/bloom-560m': 1.1,
    'bigscience/bloom-1b1': 2.2,
    'bigscience/bloom-3b': 6.0,
    'bigscience/bloom-7b1': 14.0,
    'facebook/opt-1.3b': 2.6,
    'facebook/opt-2.7b': 5.4,
    'facebook/opt-6.7b': 13.4,
    'facebook/opt-13b': 26.0,
    'facebook/opt-30b': 60.0,
    'EleutherAI/gpt-j-6b': 12.0,
    'EleutherAI/gpt-neox-20b': 40.0,
}

def get_disk_space_gb(path="/"):
    """Get available disk space in GB for the given path"""
    try:
        _, _, free_bytes = shutil.disk_usage(path)
        return free_bytes / (1024**3)  # Convert to GB
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return None

def estimate_model_size_gb(model_name):
    """Estimate model size in GB based on model name"""
    # Check exact matches first
    if model_name in MODEL_SIZE_ESTIMATES:
        return MODEL_SIZE_ESTIMATES[model_name]

    # Try to infer from model name patterns
    model_lower = model_name.lower()

    # Look for size indicators in model name
    if '70b' in model_lower or '65b' in model_lower:
        return 140.0
    elif '30b' in model_lower:
        return 60.0
    elif '20b' in model_lower:
        return 40.0
    elif '13b' in model_lower:
        return 26.0
    elif '8b' in model_lower or '7b' in model_lower:
        return 16.0
    elif '6b' in model_lower:
        return 12.0
    elif '3b' in model_lower:
        return 6.0
    elif '1b' in model_lower or '1.3b' in model_lower:
        return 2.5
    elif '560m' in model_lower:
        return 1.1

    # Default conservative estimate for unknown models
    return 10.0

def get_cache_directory():
    """Get the Hugging Face cache directory path"""
    # Check environment variable first
    cache_dir = os.environ.get('HF_HOME')
    if cache_dir:
        return Path(cache_dir)

    cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE')
    if cache_dir:
        return Path(cache_dir)

    # Default cache location
    return Path.home() / '.cache' / 'huggingface'

def check_disk_space_requirements(model_name, required_margin_gb=2.0):
    """
    Check if there's sufficient disk space for the model

    Args:
        model_name: Name of the model to check
        required_margin_gb: Extra space margin required in GB

    Returns:
        dict with check results
    """
    cache_dir = get_cache_directory()

    # Get available space
    available_gb = get_disk_space_gb(str(cache_dir.parent))
    if available_gb is None:
        return {
            'success': False,
            'error': 'Could not determine available disk space',
            'available_gb': 0,
            'required_gb': 0
        }

    # Estimate required space
    estimated_model_size = estimate_model_size_gb(model_name)
    required_gb = estimated_model_size + required_margin_gb

    success = available_gb >= required_gb

    result = {
        'success': success,
        'available_gb': round(available_gb, 1),
        'required_gb': round(required_gb, 1),
        'model_size_gb': round(estimated_model_size, 1),
        'cache_dir': str(cache_dir),
        'model_name': model_name
    }

    if not success:
        shortage_gb = required_gb - available_gb
        result['error'] = f"Insufficient disk space. Need {shortage_gb:.1f} GB more space."
        result['shortage_gb'] = round(shortage_gb, 1)

    return result

def check_network_connectivity():
    """Check if we can reach Hugging Face"""
    try:
        import urllib.request
        import socket

        # Test connectivity to Hugging Face
        socket.setdefaulttimeout(10)
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        return {'success': True}
    except Exception as e:
        return {
            'success': False,
            'error': f"Network connectivity issue: {e}"
        }

def run_preflight_checks(model_name):
    """
    Run all pre-flight checks before starting KwaaiNet

    Returns:
        dict with overall results and individual check results
    """
    results = {
        'overall_success': True,
        'checks': {}
    }

    # Check disk space
    logger.info("Checking disk space requirements...")
    disk_check = check_disk_space_requirements(model_name)
    results['checks']['disk_space'] = disk_check

    if not disk_check['success']:
        results['overall_success'] = False
        logger.error(f"❌ Disk space check failed: {disk_check.get('error', 'Unknown error')}")
        logger.error(f"   Available: {disk_check['available_gb']:.1f} GB")
        logger.error(f"   Required: {disk_check['required_gb']:.1f} GB")
        logger.error(f"   Cache directory: {disk_check['cache_dir']}")
    else:
        logger.info(f"✅ Disk space check passed: {disk_check['available_gb']:.1f} GB available, {disk_check['required_gb']:.1f} GB required")

    # Check network connectivity
    logger.info("Checking network connectivity...")
    network_check = check_network_connectivity()
    results['checks']['network'] = network_check

    if not network_check['success']:
        results['overall_success'] = False
        logger.error(f"❌ Network check failed: {network_check.get('error', 'Unknown error')}")
    else:
        logger.info("✅ Network connectivity check passed")

    return results

def suggest_solutions(check_results):
    """Suggest solutions based on failed checks"""
    suggestions = []

    disk_check = check_results['checks'].get('disk_space', {})
    if not disk_check.get('success', True):
        shortage = disk_check.get('shortage_gb', 0)
        cache_dir = disk_check.get('cache_dir', '~/.cache/huggingface')
        model_name = disk_check.get('model_name', 'current model')

        suggestions.extend([
            f"💡 Solutions for disk space issue:",
            f"   1. Free up {shortage:.1f} GB of disk space",
            f"   2. Use a smaller model (e.g., 'bigscience/bloom-560m' needs only 1.1 GB)",
            f"   3. Set HF_HOME environment variable to external storage:",
            f"      export HF_HOME=/path/to/external/storage/.cache/huggingface",
            f"   4. Clear existing model cache: rm -rf {cache_dir}",
            f"   5. Change model in config: kwaainet config set model bigscience/bloom-560m",
            f"",
            f"📊 Model size estimates:",
            f"   • bigscience/bloom-560m: 1.1 GB (recommended for testing)",
            f"   • bigscience/bloom-3b: 6.0 GB",
            f"   • {model_name}: {disk_check.get('model_size_gb', 'unknown')} GB"
        ])

    network_check = check_results['checks'].get('network', {})
    if not network_check.get('success', True):
        suggestions.extend([
            f"💡 Solutions for network issue:",
            f"   1. Check internet connection",
            f"   2. Check firewall settings",
            f"   3. Try using VPN if in restricted network",
            f"   4. Set HTTP proxy if required: export HTTP_PROXY=http://proxy:port"
        ])

    return suggestions