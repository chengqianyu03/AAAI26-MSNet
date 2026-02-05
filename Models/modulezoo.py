import torch
import torch.nn as nn
from segment_anything import sam_model_registry




from Models.backboneSAM00 import SimpleGlassPromptGenerator as M44
from Models.backboneSAM16v6e import MultiSemanticPromptGenerator as M53
from Models.backboneSAM16_woSE import MultiSemanticPromptGenerator as M55
from Models.backboneSAM16_woASFM import MultiSemanticPromptGenerator as M56
from Models.backboneSAM16_woSurr import MultiSemanticPromptGenerator as M57
from Models.backboneSAM16_woASFM_S import MultiSemanticPromptGenerator as M58
from Models.backboneSAM16_woASFM_ST import MultiSemanticPromptGenerator as M59


class ModuleZoo:
    """
    Factory class for creating different prompt generation modules.
    """
    @staticmethod
    def get_module(module_name, **kwargs):
        """
        Factory method to get a module by name with the specified parameters.
        
        Args:
            module_name (str): Name of the module to create
            **kwargs: Additional arguments to pass to the module constructor
            
        Common kwargs:
            num_points (int): Number of sparse prompts to generate (default: 1)
            d_model (int): Hidden dimension size (default: 256)
            
        Returns:
            nn.Module: The initialized module
            
        Raises:
            ValueError: If the module name is not recognized
        """
        # Extract common parameters with defaults
        num_points = kwargs.get('num_points', 1)
        d_model = kwargs.get('d_model', 256)
        clip_module_name = kwargs.get('clip_module_name', "ViT-B/16")
            
        if  module_name == "sam00":
            return M44(
                d_model=d_model
            )
        
        elif module_name == "sam1" or module_name == "sam166e":
            return M53(
                d_model=d_model,
                clip_model_name = clip_module_name 
            )

        elif module_name == "samwoSE":
            return M55(
                d_model=d_model,
                clip_model_name = clip_module_name 
            )
        elif module_name == "samwoASFM":
            return M56(
                d_model=d_model,
                clip_model_name = clip_module_name 
            )
        elif module_name == "samwoSurr":
            return M57(
                d_model=d_model,
                clip_model_name = clip_module_name 
            )
        elif module_name == "samwoASurr":
            return M58(
                d_model=d_model,
                clip_model_name = clip_module_name 
            )
        elif module_name == "samwoAST":
            return M59(
                d_model=d_model,
                clip_model_name = clip_module_name 
            )
        else:
            available_models = ModuleZoo.list_available_modules().keys()
            raise ValueError(f"Unknown module name: {module_name}. Available options: "
                           f"{', '.join(available_models)}")
    
    @staticmethod
    def list_available_modules():
        """
        Lists all available module types with descriptions.
        
        Returns:
            dict: Dictionary mapping module names to their descriptions
        """
        modules = {
            "sam00": "Simple Glass Prompt Generator",
            "sam1": "Multi-Semantic Prompt Generator with CLIP features and adaptive fusion (default)",
            "sam166e": "Alias for sam1 - Multi-Semantic Prompt Generator",
            "samwoSE": "Multi-Semantic Prompt Generator without Semantic Elimination",
            "samwoASFM": "Multi-Semantic Prompt Generator without Adaptive Semantic Fusion",
            "samwoSurr": "Multi-Semantic Prompt Generator without Surrounding features",
            "samwoASurr": "Multi-Semantic Prompt Generator - Ablation variant",
            "samwoAST": "Multi-Semantic Prompt Generator - Ablation variant"
        }
        return modules


def get_prompt_module(module_name="sam1", **kwargs):
    """
    Convenience function to get a prompt generation module by name.
    
    Args:
        module_name (str): Name of the module (default: "sam1")
        **kwargs: Additional parameters for the module:
            - num_points (int): Number of sparse prompts to generate (default: 1)
            - d_model (int): Hidden dimension size (default: 256)
            - Other module-specific parameters
        
    Returns:
        nn.Module: The instantiated module
    """
    return ModuleZoo.get_module(module_name, **kwargs)