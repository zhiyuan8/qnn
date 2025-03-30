# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import os
import json
import gguf
import sys

from transformers.integrations.ggml import generate_fp32_encodings
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from . import permute_weights


class GGUFParser:
    """
    Class to parse GGUF files and export de-quantized weights and encodings.
    """
    def __init__(self, input_model: str, filename_prefix: str, output_dir: str = None):
        """
        Constructor
        :param input_model: Path to GGUF file.
        :param filename_prefix: Prefix for the exported encodings file.
        :param output_dir: Path to the output directory for saving the processed files.
        """
        self.arch = None
        self.input_model = os.path.abspath(input_model)
        self.output_dir = output_dir if output_dir else os.path.dirname(input_model)
        self.filename_prefix = filename_prefix
        self.param_info = dict()
        self.param_encodings = list()

    def parse_gguf(self):
        """
        Parse the GGUF file and loads parameter tensors information.
        """
        self.param_info = load_gguf_checkpoint(self.input_model,
                                               return_tensors=True,
                                               extract_encodings=True,
                                               preserve_original_names=True)
        self.arch = self.param_info["config"]["model_type"]
        if self.arch != "llama":
            sys.exit("ONLY LLAMA ARCHITECTURE IS SUPPORTED IN GGUF! Exiting code execution")


    def export_dequantized_weights(self):
        """
        Export de-quantized weights to a GGUF file.
        """
        dequantized_weights_path = os.path.join(self.output_dir, self.filename_prefix + "_dequantized.gguf")
        writer = gguf.GGUFWriter(dequantized_weights_path, self.arch)

        metadata_info = self.param_info["ignore"]
        config_info = self.param_info["config"]
        tokenizer_info = self.param_info["tokenizer"]

        # add metadata
        writer.add_file_type(metadata_info["file_type"])
        writer.add_quantization_version(metadata_info["quantization_version"])
        writer.add_name(config_info["_model_name_or_path"])

        # add model config
        writer.add_block_count(config_info["num_hidden_layers"])
        writer.add_context_length(config_info["max_position_embeddings"])
        writer.add_embedding_length(config_info["hidden_size"])
        writer.add_feed_forward_length(config_info["intermediate_size"])
        writer.add_head_count(config_info["num_attention_heads"])
        writer.add_head_count_kv(config_info["num_key_value_heads"])
        writer.add_layer_norm_rms_eps(config_info["rms_norm_eps"])
        writer.add_vocab_size(config_info["vocab_size"])

        # add tokenizer_config
        writer.add_tokenizer_model(tokenizer_info["tokenizer_type"])
        writer.add_token_list(tokenizer_info["tokens"])
        writer.add_token_scores(tokenizer_info["scores"])
        writer.add_token_types(tokenizer_info["token_type"])

        writer.add_bos_token_id(tokenizer_info["bos_token_id"])
        writer.add_eos_token_id(tokenizer_info["eos_token_id"])

        # add tensors
        for tensor_name, tensor in self.param_info["tensors"].items():
            if self.arch == "llama":
                if ".attn_q." in tensor_name:
                    tensor = permute_weights(tensor,
                                             config_info["num_attention_heads"],
                                             config_info["num_attention_heads"])
                elif ".attn_k." in tensor_name:
                    tensor = permute_weights(tensor,
                                             config_info["num_attention_heads"],
                                             config_info["num_key_value_heads"])

            writer.add_tensor(tensor_name, tensor)

        # write to gguf
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        return dequantized_weights_path

    def _generate_param_encodings(self):
        """
        Generate parameter encodings from the parsed GGUF.
        """
        for tensor, encodings in self.param_info["encodings_map"].items():
            if tensor == 'token_embd.weight':
                gather_encodings = generate_fp32_encodings(32, 'FLOAT', 'PER_TENSOR')
                gather_encodings['name'] = tensor
                self.param_encodings.append(gather_encodings)
            else:
                self.param_encodings.append(encodings)

    def export_encodings(self):
        """
        Export parameter encodings to a JSON file.
        """
        activation_encodings = []
        excluded_layer_names = []
        if not self.param_encodings:
            self._generate_param_encodings()

        encoding_file = {"version": "1.0.0",
                         "activation_encodings": activation_encodings,
                         "param_encodings": self.param_encodings,
                         "excluded_layers": excluded_layer_names}

        # export weight encodings to output json file
        encoding_file_path = os.path.join(self.output_dir, self.filename_prefix + ".encodings")

        with open(encoding_file_path, "w") as encoding_fp_json:
            json.dump(encoding_file, encoding_fp_json, sort_keys=True, indent=4)

        return encoding_file_path
