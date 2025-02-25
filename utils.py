from typing import Dict, List, Tuple, Union

import hivemind
import torch
from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

import config
from data_structures import ModelConfig

logger = hivemind.get_logger(__file__)


def load_models() -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer, ModelConfig]]:
    models = {}
    for family in config.MODEL_FAMILIES.values():
        for model_config in family:
            backend_config = model_config.backend

            logger.info(f"Loading tokenizer for {backend_config.repository}")
            tokenizer = AutoTokenizer.from_pretrained(backend_config.repository, add_bos_token=False, use_fast=False)

            logger.info(
                f"Loading model {backend_config.repository} with adapter {backend_config.adapter} in {config.TORCH_DTYPE}"
            )
            # We set use_fast=False since LlamaTokenizerFast takes a long time to init
            model = AutoDistributedModelForCausalLM.from_pretrained(
                backend_config.repository,
                active_adapter=backend_config.adapter,
                torch_dtype=config.TORCH_DTYPE,
                initial_peers=config.INITIAL_PEERS,
                max_retries=3,
            )
            model = model.to(config.DEVICE)

            for key in [backend_config.key] + list(backend_config.aliases):
                print(model)
                models[key] = model, tokenizer, backend_config
    return models


def unload_models(models_dict: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer, ModelConfig]]) -> None:
    """
    Properly unload models from GPU memory.
    
    :param models_dict: Dictionary mapping model names to (model, tokenizer, config) tuples
    """
    if not models_dict:
        logger.info("No models to unload")
        return
        
    logger.info(f"Unloading {len(models_dict)} models from memory")
    
    # Track which models we've unloaded (since the same model may appear under multiple keys)
    unloaded_models = set()
    
    for model_name, (model, _, _) in models_dict.items():
        # Skip if we've already unloaded this model instance
        model_id = id(model)
        if model_id in unloaded_models:
            continue
            
        try:
            logger.info(f"Unloading model: {model_name}")
            
            # Close any active inference sessions
            if hasattr(model, 'active_sessions'):
                for session in model.active_sessions:
                    try:
                        session.close()
                    except Exception as e:
                        logger.warning(f"Error closing model session: {str(e)}")
            
            # For Petals models, disconnect from the swarm
            if hasattr(model, 'disconnect'):
                try:
                    model.disconnect()
                    logger.info(f"Disconnected model {model_name} from swarm")
                except Exception as e:
                    logger.warning(f"Error disconnecting model from swarm: {str(e)}")
            
            # Move model to CPU first
            try:
                model.to('cpu')
            except Exception as e:
                logger.warning(f"Error moving model to CPU: {str(e)}")
            
            # Clear CUDA cache for this model's parameters more safely
            try:
                if hasattr(model, 'parameters'):
                    for param in model.parameters():
                        if hasattr(param, 'data') and param.data is not None:
                            # Just detach and move to CPU instead of setting to None
                            if param.data.device.type == 'cuda':
                                param.data = param.data.detach().cpu()
            except Exception as e:
                logger.warning(f"Error clearing model parameters: {str(e)}")
                
            # Mark as unloaded
            unloaded_models.add(model_id)
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {str(e)}")
    
    # Clear any remaining CUDA cache
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing CUDA cache: {str(e)}")
        
    logger.info("Model unloading complete")


def safe_decode(tokenizer: PreTrainedTokenizer, outputs: Union[torch.Tensor, List[int]]) -> str:
    # Workaround to make SentencePiece .decode() keep leading spaces in a token
    fake_token = tokenizer("^")["input_ids"][0]
    outputs = outputs.tolist() if isinstance(outputs, torch.Tensor) else outputs
    
    # Always decode with skip_special_tokens=False to preserve special tokens
    # that the streaming logic may depend on
    result = tokenizer.decode([fake_token] + outputs, skip_special_tokens=False)

    # We use .lstrip() since SentencePiece may add leading spaces, e.g. if the outputs are "</s>"
    return result.lstrip()[1:]