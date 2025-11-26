import os
import xml.etree.ElementTree as ET
from typing import List
import torch

# Import your NeoMind code
# from neomind_model import NeoMindConfig, NeoMindModel  

def parse_list(text: str) -> List[float]:
    """Parse a string like '[1.0, 1.0, 1.0]' into a Python list of floats."""
    text = text.strip("[] ")
    return [float(x.strip()) for x in text.split(",") if x.strip()]

def load_cfml_config(cfml_path: str):
    tree = ET.parse(cfml_path)
    root = tree.getroot()
    model_node = root.find("model")

    rope_node = model_node.find("rope_scaling")
    rope_scaling = {
        "type": rope_node.find("type").text,
        "short_factor": parse_list(rope_node.find("short_factor").text),
        "long_factor": parse_list(rope_node.find("long_factor").text)
    } if rope_node is not None else None

    config = NeoMindConfig(
        vocab_size=int(model_node.find("vocab_size").text),
        hidden_size=int(model_node.find("hidden_size").text),
        intermediate_size=int(model_node.find("intermediate_size").text),
        num_hidden_layers=int(model_node.find("num_hidden_layers").text),
        num_attention_heads=int(model_node.find("num_attention_heads").text),
        num_key_value_heads=int(model_node.find("num_key_value_heads").text),
        resid_pdrop=float(model_node.find("resid_pdrop").text),
        embd_pdrop=float(model_node.find("embd_pdrop").text),
        attention_dropout=float(model_node.find("attention_dropout").text),
        hidden_act=model_node.find("hidden_act").text,
        max_position_embeddings=int(model_node.find("max_position_embeddings").text),
        original_max_position_embeddings=int(model_node.find("original_max_position_embeddings").text),
        use_cache=model_node.find("use_cache").text.lower() == "true",
        tie_word_embeddings=model_node.find("tie_word_embeddings").text.lower() == "true",
        rope_theta=float(model_node.find("rope_theta").text),
        rope_scaling=rope_scaling,
        bos_token_id=int(model_node.find("bos_token_id").text),
        eos_token_id=int(model_node.find("eos_token_id").text),
        pad_token_id=int(model_node.find("pad_token_id").text),
        sliding_window=int(model_node.find("sliding_window").text),
    )
    return config

# -------------------------------
# Load multiple NeoMind models from folder
# -------------------------------
def load_models_from_folder(folder_path: str):
    models = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".cfml"):
            model_name = os.path.splitext(file_name)[0]
            cfml_path = os.path.join(folder_path, file_name)
            config = load_cfml_config(cfml_path)
            models[model_name] = NeoMindModel(config)
    return models

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    folder = "neomind_models/configs"
    neo_models = load_models_from_folder(folder)

    # Example: select mini model
    model = neo_models["neomind_mini"]

    # Test with dummy input
    input_ids = torch.randint(0, model.config.vocab_size, (1, 512))
    output, _ = model(input_ids)
    print(f"Output shape for mini model: {output.shape}")
