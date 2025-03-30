# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import onnx
import math
import transformers
import numpy as np

from transformers.integrations.ggml import generate_fp32_encodings
from onnx import helper, numpy_helper, TensorProto, ModelProto

MODEL_TYPE_TO_ARCH = {"llama": "LlamaForCausalLM"}

MODEL_TYPE_TO_TOKENIZER = {"llama": transformers.LlamaTokenizerFast}

GGUF_TO_ONNX_TENSOR = {
    "llama": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj.MatMul",
        "ffn_down": "mlp.down_proj.MatMul",
        "ffn_gate": "mlp.gate_proj.MatMul",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "attn.q_proj.MatMul",
        "attn_v": "attn.v_proj.MatMul",
        "attn_k": "attn.k_proj.MatMul",
        "attn_output": "attn.o_proj.MatMul",
        "output.weight": "lm_head.MatMul.weight",
        "output_norm": "model.layers.{max_block}.final_norm_layernorm",
    }
}

ONNX_TENSOR_NAME_STRINGS = {
    "llama_final_layernorm": "final_norm_layernorm",
    "llama_SkipLayerNorm": "SkipLayerNorm",
    "llama_LayerNorm": "LayerNorm",
    "llama_name_seqlens_k": "/model/attn_mask_reformat/attn_mask_subgraph/Sub/Cast/output_0",
    "llama_GroupQueryAttention": "GroupQueryAttention",
    "llama_qkv_proj": "qkv_proj",
}

ONNX_ENCODING_STRINGS = ["sin_cache", "cos_cache", "attention_mask_matrix_value"]


def update_symbolic_shape_with_value(model: ModelProto, model_config: dict, batch_size: int):
    """
        Utility function to update an onnx model inputs, outputs and intermediate value_infos shapes.
        All symbolic shapes for the above mentioned tensors are replaced with constant values.
    """
    batch_size_value = batch_size
    sequence_length_value = 1
    total_sequence_length_value = model_config["max_position_embeddings"]
    past_sequence_length_value = model_config["max_position_embeddings"]
    model_input_shape_dict = {
        "batch_size": batch_size_value,
        "sequence_length": sequence_length_value,
        "total_sequence_length": total_sequence_length_value,
        "past_sequence_length": past_sequence_length_value
    }

    def update_symbolic_values(value_infos):
        for value_info in value_infos:
            if value_info.type.HasField("tensor_type") and value_info.type.tensor_type.HasField("shape"):
                for cur_dim in value_info.type.tensor_type.shape.dim:
                    if cur_dim.HasField("dim_param") and cur_dim.dim_param in model_input_shape_dict.keys():
                        dict_key = cur_dim.dim_param
                        cur_dim.Clear()
                        cur_dim.dim_value = model_input_shape_dict[dict_key]

    update_symbolic_values(model.graph.input)
    update_symbolic_values(model.graph.output)
    update_symbolic_values(model.graph.value_info)


def decompose_layernorms(model: ModelProto, model_config: dict):
    """
        Utility function to decompose the layernorm ops in ort-genai onnx model with constituent onnx ops.
        1. SkipSimplifiedLayerNormalization -> Elementwise Add + SimplifiedLayerNormalization
        2. SimplifiedLayerNormalization -> x/Sqrt(RedSum(x^2) + eps) * gamma_weight
    """
    hidden_size = model_config["hidden_size"]

    # Update SkipSimplifiedLayerNorm = Elementwise Add and SimplifiedLayerNorm
    for node in model.graph.node:
        if node.op_type == 'SkipSimplifiedLayerNormalization':
            # Get input output weight info
            name_sln = node.name
            is_last_layernorm = False
            # Final SkipSLN only has one output
            if ONNX_TENSOR_NAME_STRINGS["llama_final_layernorm"] in name_sln:
                is_last_layernorm = True
            input_name_data_0 = node.input[0]
            input_name_data_1 = node.input[1]
            weight_name_scale = node.input[2]
            eps_data = node.attribute[0].f
            output_name_data_0 = node.output[0]

            # Set Node Names to be added to graph
            split_name_skip_sln= name_sln.split(ONNX_TENSOR_NAME_STRINGS["llama_SkipLayerNorm"])
            node_name_elementwise_add = split_name_skip_sln[0] + 'elementwise_add'
            node_name_simplifiedlayernorm = split_name_skip_sln[0] + 'LayerNorm'
            if not is_last_layernorm:
                output_name_data_3 = node.output[3]
                node_name_identity_out = split_name_skip_sln[0] + 'identity_out'
            # Set Output names for nodes
            out_name_elementwise_add = node_name_elementwise_add + '/output_0'

            # Create Decomposed Skip_SLN Nodes
            node_elementwise_add = helper.make_node('Add', name=node_name_elementwise_add, inputs=[input_name_data_0, input_name_data_1],
                                                    outputs=[out_name_elementwise_add])
            if not is_last_layernorm:
                node_identity_out = helper.make_node('Identity', name=node_name_identity_out, inputs=[out_name_elementwise_add],
                                                     outputs=[output_name_data_3])
            node_simplifiedlayernorm = helper.make_node('SimplifiedLayerNormalization', name=node_name_simplifiedlayernorm, inputs=[out_name_elementwise_add, weight_name_scale],
                                                        outputs=[output_name_data_0])

            # Add required attributes to SimplfiedLayerNorm
            eps_attribute = helper.make_attribute("epsilon", eps_data)
            axis_attribute = helper.make_attribute("axis", -1)
            node_simplifiedlayernorm.attribute.extend([eps_attribute, axis_attribute])

            # Create intermediate output tensors and add to graph: Value_Info
            vi_elementwise_add = helper.make_tensor_value_info(out_name_elementwise_add, TensorProto.FLOAT,
                                                               ['batch_size', 'sequence_length', hidden_size])

            model.graph.value_info.extend([vi_elementwise_add])

            # Add created nodes to graph
            if not is_last_layernorm:
                model.graph.node.extend([node_elementwise_add, node_identity_out, node_simplifiedlayernorm])
            else:
                model.graph.node.extend([node_elementwise_add, node_simplifiedlayernorm])

            # Remove Node
            model.graph.node.remove(node)

    # Decompose SimplifiedLayerNormalization into constituent onnx ops
    for node in model.graph.node:
        if node.op_type == 'SimplifiedLayerNormalization':

            # Get input output weight info
            name_sln = node.name
            input_name_data = node.input[0]
            weight_name_scale = node.input[1]
            eps_data = node.attribute[0].f
            output_name_data = node.output[0]

            # Set Node Names to be added to graph
            split_name_sln= name_sln.split(ONNX_TENSOR_NAME_STRINGS["llama_LayerNorm"])
            node_name_pow_value = split_name_sln[0] + 'pow_value'
            node_name_eps_value = split_name_sln[0] + 'eps_value'
            node_name_square_inp = split_name_sln[0] + 'square_inp'
            node_name_reduce_mean = split_name_sln[0] + 'reduce_mean'
            node_name_add_eps = split_name_sln[0] + 'add_eps'
            node_name_sqrt = split_name_sln[0] + 'sqrt'
            node_name_elementwise_div = split_name_sln[0] + 'elementwise_div'
            node_name_elementwise_mul_gamma = split_name_sln[0] + 'elementwise_mul_gamma'

            # Set constants Input
            pow_value = np.array([2], dtype=np.float32)
            eps_value = np.array([eps_data], dtype=np.float32)

            # Set Output names for nodes
            out_name_pow_value = node_name_pow_value + '/output_0'
            out_name_eps_value = node_name_eps_value + '/output_0'
            out_name_square_inp = node_name_square_inp + '/output_0'
            out_name_reduce_mean = node_name_reduce_mean + '/output_0'
            out_name_add_eps = node_name_add_eps + '/output_0'
            out_name_sqrt = node_name_sqrt + '/output_0'
            out_name_elementwise_div = node_name_elementwise_div + '/output_0'

            # Create Decomposed RMSNorm Nodes
            node_constant_pow_value = helper.make_node('Constant', inputs=[], outputs=[out_name_pow_value],
                                                       value=helper.make_tensor(
                                                           name=node_name_pow_value,
                                                           data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                           dims=pow_value.shape,
                                                           vals=pow_value.flatten()))
            node_constant_eps_value = helper.make_node('Constant', inputs=[], outputs=[out_name_eps_value],
                                                       value=helper.make_tensor(
                                                           name=node_name_eps_value,
                                                           data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                           dims=eps_value.shape,
                                                           vals=eps_value.flatten()))
            node_square_inp = helper.make_node('Pow', name=node_name_square_inp, inputs=[input_name_data, out_name_pow_value],
                                               outputs=[out_name_square_inp])
            node_reduce_mean = helper.make_node('ReduceMean', name=node_name_reduce_mean, inputs=[out_name_square_inp], axes=[2],
                                                outputs=[out_name_reduce_mean])
            node_add_eps = helper.make_node('Add', name=node_name_add_eps, inputs=[out_name_reduce_mean, out_name_eps_value],
                                            outputs=[out_name_add_eps])
            node_sqrt = helper.make_node('Sqrt', name=node_name_sqrt, inputs=[out_name_add_eps],
                                         outputs=[out_name_sqrt])
            node_elementwise_div = helper.make_node('Div', name=node_name_elementwise_div, inputs=[input_name_data, out_name_sqrt],
                                                    outputs=[out_name_elementwise_div])
            node_elementwise_mul_gamma = helper.make_node('Mul', name=node_name_elementwise_mul_gamma, inputs=[out_name_elementwise_div, weight_name_scale],
                                                          outputs=[output_name_data])

            # Create intermediate output tensors and add to graph: Value_Info
            vi_square_inp = helper.make_tensor_value_info(out_name_square_inp, TensorProto.FLOAT,
                                                          ['batch_size', 'sequence_length', hidden_size])
            vi_reduce_mean = helper.make_tensor_value_info(out_name_reduce_mean, TensorProto.FLOAT,
                                                           ['batch_size', 'sequence_length', 1])
            vi_add_eps = helper.make_tensor_value_info(out_name_add_eps, TensorProto.FLOAT,
                                                       ['batch_size', 'sequence_length', 1])
            vi_sqrt = helper.make_tensor_value_info(out_name_sqrt, TensorProto.FLOAT,
                                                    ['batch_size', 'sequence_length', 1])
            vi_elementwise_div = helper.make_tensor_value_info(out_name_elementwise_div, TensorProto.FLOAT,
                                                               ['batch_size', 'sequence_length', hidden_size])

            model.graph.value_info.extend([vi_square_inp, vi_reduce_mean, vi_add_eps,
                                           vi_sqrt, vi_elementwise_div])

            # Add created nodes to graph
            model.graph.node.extend([node_constant_pow_value, node_constant_eps_value,
                                     node_square_inp, node_reduce_mean, node_add_eps, node_sqrt,
                                     node_elementwise_div, node_elementwise_mul_gamma])

            # Remove Node
            model.graph.node.remove(node)


def decompose_gqa(model: ModelProto, model_config: dict, batch_size: int):
    """
        Utility function that decomposes ort-genai generated GroupQueryAttention(GQA) op into constituent ops.
    """
    def update_o_proj_input(model: ModelProto, node_name: str, output_tensor_name: str):
        o_proj_node_name = node_name.replace("reshape_attnv", "o_proj/MatMul")

        for node in model.graph.node:
            if node.name == o_proj_node_name:
                node.input[0] = output_tensor_name

    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    total_seq_len = model_config["max_position_embeddings"]
    num_kv_heads = model_config["num_key_value_heads"] if "num_key_value_heads" in model_config else num_heads
    head_dim = hidden_size // num_heads
    n_rep = num_heads // num_kv_heads
    # Creating a matrix of -INF, 0 to be used as attention mask addition after QK Matmul.
    # Layout of matrix for eg . max_seq_len = 2048
    # [-INF -INF ...... -INF 0] -INF(2047) 0(1)
    # [-INF -INF .....-INF 0 0] -INF(2046) 0(2)
    # [-INF -INF ...-INF 0 0 0] -INF(2045) 0(3)
    #            ...
    # [0 0 0 ....... 0 0 0 0 0] -INF(0) 0(2048)
    attention_mask_ones = np.reshape(np.flipud(np.tril(np.ones((total_seq_len, total_seq_len),dtype=np.float32),-1)),
                                     (1, total_seq_len, total_seq_len))
    attention_mask_matrix = np.where(attention_mask_ones == 1, -np.inf, attention_mask_ones)
    tensor_name_attention_mask_matrix_value = "attention_mask_matrix_value"
    tensor_attention_mask_matrix_value = helper.make_tensor(
        name=tensor_name_attention_mask_matrix_value,
        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
        dims=attention_mask_matrix.shape,
        vals=attention_mask_matrix.flatten()
    )

    model.graph.initializer.extend([tensor_attention_mask_matrix_value])

    input_name_seqlens_k = ONNX_TENSOR_NAME_STRINGS["llama_name_seqlens_k"]
    node_name_gather_attention_mask_row = "gather_attention_mask_row"
    out_name_gather_attention_mask_row = node_name_gather_attention_mask_row + "/output_0"
    node_gather_attention_mask_row = helper.make_node('Gather',
                                                      name=node_name_gather_attention_mask_row,
                                                      inputs=[tensor_name_attention_mask_matrix_value,
                                                              input_name_seqlens_k],
                                                      outputs=[out_name_gather_attention_mask_row], axis=1)
    vi_gather_attention_mask_row = helper.make_tensor_value_info(out_name_gather_attention_mask_row, TensorProto.FLOAT,
                                                                 [batch_size, 1, 'sequence_length', total_seq_len])
    model.graph.value_info.extend([vi_gather_attention_mask_row])
    model.graph.node.extend([node_gather_attention_mask_row])

    # Iterate over nodes
    for node in model.graph.node:
        # check for matmul that has q, k, v weights packed into one tensor
        if node.op_type == 'GroupQueryAttention':

            cur_seq_len = 1
            # Set Reshape/constants Input
            shape_q = np.array([batch_size, -1, num_heads, head_dim], dtype=np.int64)
            shape_k = np.array([batch_size, -1, num_kv_heads, head_dim], dtype=np.int64)
            shape_v = np.array([batch_size, -1, num_kv_heads, head_dim], dtype=np.int64)
            div_value = np.array([math.sqrt(head_dim)], dtype=np.float32)
            mul_sin_cache_value = np.array([-1], dtype=np.float32)
            split_past_data_shape = np.array([cur_seq_len, total_seq_len - cur_seq_len], dtype=np.int64)
            unsqueeze_axis = np.array([2], dtype=np.int64)
            unsqueeze_cache_axis = np.array([1], dtype=np.int64)
            num_rep = np.array([n_rep], dtype=np.int64)
            shape_current_k = np.array([-1, num_kv_heads * n_rep, head_dim, total_seq_len], dtype=np.int64)
            shape_current_v = np.array([-1, num_kv_heads * n_rep, total_seq_len, head_dim], dtype=np.int64)
            shape_attnv = np.array([-1, cur_seq_len, num_heads * head_dim], dtype=np.int64)
            # Get input output weight info
            name_gqa = node.name
            input_name_q = node.input[0]
            input_name_k = node.input[1]
            input_name_v = node.input[2]

            input_name_past_key = node.input[3]
            input_name_past_value = node.input[4]
            weight_name_cos_cache = node.input[7]
            weight_name_sin_cache = node.input[8]
            output_name_gqa = node.output[0]
            output_name_present_key = node.output[1]
            output_name_present_value = node.output[2]

            # Set Node Names to be added to graph
            split_name_gqa = name_gqa.split(ONNX_TENSOR_NAME_STRINGS["llama_GroupQueryAttention"])
            node_name_shape_q = split_name_gqa[0] + 'shape_q'
            node_name_shape_k = split_name_gqa[0] + 'shape_k'
            node_name_shape_v = split_name_gqa[0] + 'shape_v'
            node_name_div_value = split_name_gqa[0] + 'div_value'
            node_name_mul_sin_cache = split_name_gqa[0] + 'mul_sin_cache'
            node_name_split_past_data_shape = split_name_gqa[0] + 'split_past_data_shape'
            node_name_unsqueeze_axis = split_name_gqa[0] + 'unsqueeze_axis'
            node_name_unsqueeze_cache_axis = split_name_gqa[0] + 'unsqueeze_cache_axis'
            node_name_num_rep = split_name_gqa[0] + 'num_rep'
            node_name_shape_current_k = split_name_gqa[0] + 'shape_current_k'
            node_name_shape_current_v = split_name_gqa[0] + 'shape_current_v'
            node_name_shape_attnv = split_name_gqa[0] + 'shape_attnv'
            node_name_reshape_q = split_name_gqa[0] + 'q_reshape'
            node_name_reshape_k = split_name_gqa[0] + 'k_reshape'
            node_name_reshape_v = split_name_gqa[0] + 'v_reshape'
            node_name_transpose_q = split_name_gqa[0] + 'q_transpose'
            node_name_transpose_k = split_name_gqa[0] + 'k_transpose'
            node_name_transpose_v = split_name_gqa[0] + 'v_transpose'
            node_name_split_q = split_name_gqa[0] + 'split_q'
            node_name_gather_sin_cache_row = split_name_gqa[0] + 'gather_sin_cache_row'
            node_name_gather_cos_cache_row = split_name_gqa[0] + 'gather_cos_cache_row'
            node_name_unsqueeze_sin_cache = split_name_gqa[0] + 'unsqueeze_sin_cache'
            node_name_unsqueeze_cos_cache = split_name_gqa[0] + 'unsqueeze_cos_cache'
            node_name_mul_q1_sin = split_name_gqa[0] + 'mul_q1_sin'
            node_name_mul_q1_cos = split_name_gqa[0] + 'mul_q1_cos'
            node_name_mul_q_neg1_sin_cache = split_name_gqa[0] + 'mul_q_neg1_sin_cache'
            node_name_mul_q2_sin = split_name_gqa[0] + 'mul_q2_sin'
            node_name_mul_q2_cos = split_name_gqa[0] + 'mul_q2_cos'
            node_name_add_q1_sin_q2_cos = split_name_gqa[0] + 'add_q1_sin_q2_cos'
            node_name_add_q1_cos_q2_sin = split_name_gqa[0] + 'add_q1_cos_q2_sin'
            node_name_concat_q1_q2 = split_name_gqa[0] + 'concat_q1_q2'
            node_name_split_k = split_name_gqa[0] + 'split_k'
            node_name_mul_k1_sin = split_name_gqa[0] + 'mul_k1_sin'
            node_name_mul_k1_cos = split_name_gqa[0] + 'mul_k1_cos'
            node_name_mul_k_neg1_sin_cache = split_name_gqa[0] + 'mul_k_neg1_sin_cache'
            node_name_mul_k2_sin = split_name_gqa[0] + 'mul_k2_sin'
            node_name_mul_k2_cos = split_name_gqa[0] + 'mul_k2_cos'
            node_name_add_k1_sin_k2_cos = split_name_gqa[0] + 'add_k1_sin_k2_cos'
            node_name_add_k1_cos_k2_sin = split_name_gqa[0] + 'add_k1_cos_k2_sin'
            node_name_concat_k1_k2 = split_name_gqa[0] + 'concat_k1_k2'
            node_name_split_past_key = split_name_gqa[0] + 'split_past_key'
            node_name_split_past_value = split_name_gqa[0] + 'split_past_value'
            node_name_concat_k = split_name_gqa[0] + 'k_concat'
            node_name_identity_k = split_name_gqa[0] + 'output_present_key'
            node_name_concat_v = split_name_gqa[0] + 'v_concat'
            node_name_identity_v = split_name_gqa[0] + 'output_present_value'
            node_name_transpose_current_k = split_name_gqa[0] + 'current_k_transpose'
            node_name_unsqueeze_k = split_name_gqa[0] + 'unsqueeze_k'
            node_name_expand_k = split_name_gqa[0] +'expand_k'
            node_name_reshape_current_k = split_name_gqa[0] + 'current_k_reshape'
            node_name_unsqueeze_v = split_name_gqa[0] + 'unsqueeze_v'
            node_name_expand_v = split_name_gqa[0] +'expand_v'
            node_name_reshape_current_v = split_name_gqa[0] + 'current_v_reshape'
            node_name_matmul_qk = split_name_gqa[0] + 'matmul_qk'
            node_name_add_qk_attn_mask = split_name_gqa[0] + 'add_qk_attn_mask'
            node_name_div_qk = split_name_gqa[0] + 'div_qk'
            node_name_softmax_qk = split_name_gqa[0] + 'softmax_qk'
            node_name_matmul_attnv = split_name_gqa[0] + 'matmul_attnv'
            node_name_transpose_attnv = split_name_gqa[0] + 'transpose_attnv'
            node_name_reshape_attnv = split_name_gqa[0] + 'reshape_attnv'

            # Set Output names for nodes
            out_name_shape_q = node_name_shape_q + '/output_0'
            out_name_shape_k = node_name_shape_k + '/output_0'
            out_name_shape_v = node_name_shape_v + '/output_0'
            out_name_div_value = node_name_div_value + '/output_0'
            out_name_mul_sin_cache_value = node_name_mul_sin_cache + '/output_0'
            out_name_split_past_data_shape = node_name_split_past_data_shape + '/output_0'
            out_name_unsqueeze_axis = node_name_unsqueeze_axis + '/output_0'
            out_name_unsqueeze_cache_axis = node_name_unsqueeze_cache_axis + '/output_0'
            out_name_num_rep = node_name_num_rep + '/output_0'
            out_name_shape_current_k = node_name_shape_current_k + '/output_0'
            out_name_shape_current_v = node_name_shape_current_v + '/output_0'
            out_name_shape_attnv = node_name_shape_attnv + '/output_0'
            out_name_reshape_q = node_name_reshape_q + '/output_0'
            out_name_reshape_k = node_name_reshape_k + '/output_0'
            out_name_reshape_v = node_name_reshape_v + '/output_0'

            out_name_transpose_q = node_name_transpose_q + '/output_0'
            out_name_transpose_k = node_name_transpose_k + '/output_0'
            out_name_transpose_v = node_name_transpose_v + '/output_0'

            out_name_split_q1 = node_name_split_q + '/output_0'
            out_name_split_q2 = node_name_split_q + '/output_1'
            out_name_gather_sin_cache_row = node_name_gather_sin_cache_row + '/output_0'
            out_name_gather_cos_cache_row = node_name_gather_cos_cache_row + '/output_0'
            out_name_sin_cache_row = node_name_unsqueeze_sin_cache + '/output_0'
            out_name_cos_cache_row = node_name_unsqueeze_cos_cache + '/output_0'
            out_name_mul_q1_sin = node_name_mul_q1_sin + '/output_0'
            out_name_mul_q1_cos = node_name_mul_q1_cos + '/output_0'
            out_name_q_weight_neg1_sin_cache = node_name_mul_q_neg1_sin_cache + '/output_0'
            out_name_mul_q2_sin = node_name_mul_q2_sin + '/output_0'
            out_name_mul_q2_cos = node_name_mul_q2_cos +'/output_0'
            out_name_add_q1_sin_q2_cos = node_name_add_q1_sin_q2_cos + '/output_0'
            out_name_add_q1_cos_q2_sin = node_name_add_q1_cos_q2_sin + '/output_0'
            out_name_rope_q = node_name_concat_q1_q2 + '/output_0'

            out_name_split_k1 = node_name_split_k + '/output_0'
            out_name_split_k2 = node_name_split_k + '/output_1'
            out_name_mul_k1_sin = node_name_mul_k1_sin + '/output_0'
            out_name_mul_k1_cos = node_name_mul_k1_cos + '/output_0'
            out_name_k_weight_neg1_sin_cache = node_name_mul_k_neg1_sin_cache + '/output_0'
            out_name_mul_k2_sin = node_name_mul_k2_sin + '/output_0'
            out_name_mul_k2_cos = node_name_mul_k2_cos +'/output_0'
            out_name_add_k1_sin_k2_cos = node_name_add_k1_sin_k2_cos + '/output_0'
            out_name_add_k1_cos_k2_sin = node_name_add_k1_cos_k2_sin + '/output_0'
            out_name_rope_k = node_name_concat_k1_k2 + '/output_0'
            out_name_split_past_key_discard = node_name_split_past_key + '/output_0'
            out_name_split_past_key_retain = node_name_split_past_key + '/output_1'
            out_name_split_past_value_discard = node_name_split_past_value + '/output_0'
            out_name_split_past_value_retain = node_name_split_past_value + '/output_1'

            out_name_current_k = node_name_concat_k + '/output_0'
            out_name_current_v = node_name_concat_v + '/output_0'
            out_name_transpose_current_k = node_name_transpose_current_k + '/output_0'
            out_name_unsqueeze_k = node_name_unsqueeze_k + '/output_0'
            out_name_expand_k = node_name_expand_k + '/output_0'
            out_name_reshape_current_k = node_name_reshape_current_k + '/output_0'
            out_name_unsqueeze_v = node_name_unsqueeze_v + '/output_0'
            out_name_expand_v = node_name_expand_v + '/output_0'
            out_name_reshape_current_v = node_name_reshape_current_v + '/output_0'

            out_name_matmul_qk = node_name_matmul_qk + '/output_0'
            out_name_add_qk_attn_mask = node_name_add_qk_attn_mask + '/output_0'
            out_name_div_qk = node_name_div_qk + '/output_0'
            out_name_softmax_qk = node_name_softmax_qk + '/output_0'
            out_name_matmul_attnv = node_name_matmul_attnv + '/output_0'
            out_name_transpose_attnv = node_name_transpose_attnv + '/output_0'
            out_name_reshape_attnv = node_name_reshape_attnv + '/output_0'

            # Create Decomposed GQA Nodes
            # Reshape Nodes
            node_constant_shape_q = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_q],
                                                     value=helper.make_tensor(
                                                         name=node_name_shape_q,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=shape_q.shape,
                                                         vals=shape_q.flatten()))
            node_constant_shape_k = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_k],
                                                     value=helper.make_tensor(
                                                         name=node_name_shape_k,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=shape_k.shape,
                                                         vals=shape_k.flatten()))
            node_constant_shape_v = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_v],
                                                     value=helper.make_tensor(
                                                         name=node_name_shape_v,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=shape_v.shape,
                                                         vals=shape_v.flatten()))
            node_reshape_q = helper.make_node('Reshape', name=node_name_reshape_q, inputs=[input_name_q, out_name_shape_q],
                                              outputs=[out_name_reshape_q])
            node_reshape_k = helper.make_node('Reshape', name=node_name_reshape_k, inputs=[input_name_k, out_name_shape_k],
                                              outputs=[out_name_reshape_k])
            node_reshape_v = helper.make_node('Reshape', name=node_name_reshape_v, inputs=[input_name_v, out_name_shape_v],
                                              outputs=[out_name_reshape_v])

            # Transpose Nodes
            node_transpose_q = helper.make_node('Transpose', name=node_name_transpose_q, inputs=[out_name_reshape_q],
                                                outputs=[out_name_transpose_q], perm=[0, 2, 1, 3])
            node_transpose_k = helper.make_node('Transpose', name=node_name_transpose_k, inputs=[out_name_reshape_k],
                                                outputs=[out_name_transpose_k], perm=[0, 2, 1, 3])
            node_transpose_v = helper.make_node('Transpose', name=node_name_transpose_v, inputs=[out_name_reshape_v],
                                                outputs=[out_name_transpose_v], perm=[0, 2, 1, 3])

            # RoPE Nodes Q
            node_constant_unsqueeze_cache_axis = helper.make_node('Constant', inputs=[], outputs=[out_name_unsqueeze_cache_axis],
                                                                  value=helper.make_tensor(
                                                                      name=node_name_unsqueeze_cache_axis,
                                                                      data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                      dims=unsqueeze_cache_axis.shape,
                                                                      vals=unsqueeze_cache_axis.flatten()))
            node_split_q = helper.make_node('Split', name=node_name_split_q, inputs=[out_name_transpose_q],
                                            outputs=[out_name_split_q1, out_name_split_q2], axis = -1)
            node_gather_sin_cache_row = helper.make_node('Gather', name=node_name_gather_sin_cache_row, inputs=[weight_name_sin_cache, input_name_seqlens_k],
                                                         outputs=[out_name_gather_sin_cache_row], axis = 0)
            node_gather_cos_cache_row = helper.make_node('Gather', name=node_name_gather_cos_cache_row, inputs=[weight_name_cos_cache, input_name_seqlens_k],
                                                         outputs=[out_name_gather_cos_cache_row], axis = 0)
            node_unsqueeze_sin_cache = helper.make_node('Unsqueeze', name=node_name_unsqueeze_sin_cache, inputs=[out_name_gather_sin_cache_row, out_name_unsqueeze_cache_axis],
                                                        outputs=[out_name_sin_cache_row])
            node_unsqueeze_cos_cache = helper.make_node('Unsqueeze', name=node_name_unsqueeze_cos_cache, inputs=[out_name_gather_cos_cache_row, out_name_unsqueeze_cache_axis],
                                                        outputs=[out_name_cos_cache_row])
            node_mul_q1_sin = helper.make_node('Mul', name=node_name_mul_q1_sin, inputs=[out_name_split_q1, out_name_sin_cache_row],
                                               outputs=[out_name_mul_q1_sin])
            node_mul_q1_cos = helper.make_node('Mul', name=node_name_mul_q1_cos, inputs=[out_name_split_q1, out_name_cos_cache_row],
                                               outputs=[out_name_mul_q1_cos])
            # Node for -1 Mul
            node_constant_neg1_q = helper.make_node('Constant', inputs=[], outputs=[out_name_mul_sin_cache_value],
                                                    value=helper.make_tensor(
                                                        name=node_name_mul_sin_cache,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                        dims=mul_sin_cache_value.shape,
                                                        vals=mul_sin_cache_value.flatten()))
            node_mul_q_neg1_sin_cache = helper.make_node('Mul', name=node_name_mul_q_neg1_sin_cache, inputs=[out_name_sin_cache_row, out_name_mul_sin_cache_value],
                                                         outputs=[out_name_q_weight_neg1_sin_cache])
            node_mul_q2_sin = helper.make_node('Mul', name=node_name_mul_q2_sin, inputs=[out_name_split_q2, out_name_q_weight_neg1_sin_cache],
                                               outputs=[out_name_mul_q2_sin])
            node_mul_q2_cos = helper.make_node('Mul', name=node_name_mul_q2_cos, inputs=[out_name_split_q2, out_name_cos_cache_row],
                                               outputs=[out_name_mul_q2_cos])
            node_add_q1_sin_q2_cos = helper.make_node('Add', name=node_name_add_q1_sin_q2_cos, inputs=[out_name_mul_q1_sin, out_name_mul_q2_cos],
                                                      outputs=[out_name_add_q1_sin_q2_cos])
            node_add_q1_cos_q2_sin = helper.make_node('Add', name=node_name_add_q1_cos_q2_sin, inputs=[out_name_mul_q1_cos, out_name_mul_q2_sin],
                                                      outputs=[out_name_add_q1_cos_q2_sin])
            node_concat_q1_q2 = helper.make_node('Concat', name=node_name_concat_q1_q2, inputs=[out_name_add_q1_cos_q2_sin, out_name_add_q1_sin_q2_cos],
                                                 axis=-1, outputs=[out_name_rope_q])

            # RoPE Nodes K
            node_split_k = helper.make_node('Split', name=node_name_split_k, inputs=[out_name_transpose_k],
                                            outputs=[out_name_split_k1, out_name_split_k2], axis = -1)
            node_mul_k1_sin = helper.make_node('Mul', name=node_name_mul_k1_sin, inputs=[out_name_split_k1, out_name_sin_cache_row],
                                               outputs=[out_name_mul_k1_sin])
            node_mul_k1_cos = helper.make_node('Mul', name=node_name_mul_k1_cos, inputs=[out_name_split_k1, out_name_cos_cache_row],
                                               outputs=[out_name_mul_k1_cos])
            # Node for -1 Mul
            node_mul_k_neg1_sin_cache = helper.make_node('Mul', name=node_name_mul_k_neg1_sin_cache, inputs=[out_name_sin_cache_row, out_name_mul_sin_cache_value],
                                                         outputs=[out_name_k_weight_neg1_sin_cache])
            node_mul_k2_sin = helper.make_node('Mul', name=node_name_mul_k2_sin, inputs=[out_name_split_k2, out_name_k_weight_neg1_sin_cache],
                                               outputs=[out_name_mul_k2_sin])
            node_mul_k2_cos = helper.make_node('Mul', name=node_name_mul_k2_cos, inputs=[out_name_split_k2, out_name_cos_cache_row],
                                               outputs=[out_name_mul_k2_cos])
            node_add_k1_sin_k2_cos = helper.make_node('Add', name=node_name_add_k1_sin_k2_cos, inputs=[out_name_mul_k1_sin, out_name_mul_k2_cos],
                                                      outputs=[out_name_add_k1_sin_k2_cos])
            node_add_k1_cos_k2_sin = helper.make_node('Add', name=node_name_add_k1_cos_k2_sin, inputs=[out_name_mul_k1_cos, out_name_mul_k2_sin],
                                                      outputs=[out_name_add_k1_cos_k2_sin])
            node_concat_k1_k2 = helper.make_node('Concat', name=node_name_concat_k1_k2, inputs=[out_name_add_k1_cos_k2_sin, out_name_add_k1_sin_k2_cos],
                                                 axis=-1, outputs=[out_name_rope_k])
            # Split Past Key and Value. Retain  max_seq_len - current_seq_len elements from end
            # Example if maxlen = 2048, curr_len=1, retain 1:2048 and discard 0 element
            node_constant_split_past_data = helper.make_node('Constant', inputs=[], outputs=[out_name_split_past_data_shape],
                                                             value=helper.make_tensor(
                                                                 name=node_name_split_past_data_shape,
                                                                 data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                 dims=split_past_data_shape.shape,
                                                                 vals=split_past_data_shape.flatten()))
            node_split_past_key = helper.make_node('Split', name=node_name_split_past_key, inputs=[input_name_past_key, out_name_split_past_data_shape],
                                                   outputs=[out_name_split_past_key_discard, out_name_split_past_key_retain], axis = 2)
            node_split_past_value = helper.make_node('Split', name=node_name_split_past_value, inputs=[input_name_past_value, out_name_split_past_data_shape],
                                                     outputs=[out_name_split_past_value_discard, out_name_split_past_value_retain], axis = 2)
            # Concat Nodes
            node_concat_k = helper.make_node('Concat', name=node_name_concat_k, inputs=[out_name_split_past_key_retain, out_name_rope_k],
                                             axis=2, outputs=[out_name_current_k])
            node_identity_k = helper.make_node('Identity', name=node_name_identity_k, inputs=[out_name_current_k],
                                               outputs=[output_name_present_key])
            node_transpose_current_k = helper.make_node('Transpose', name=node_name_transpose_current_k, inputs=[out_name_current_k],
                                                        outputs=[out_name_transpose_current_k], perm=[0, 1, 3, 2])
            node_concat_v = helper.make_node('Concat', name=node_name_concat_v, inputs=[out_name_split_past_value_retain, out_name_transpose_v],
                                             axis=2, outputs=[out_name_current_v])
            node_identity_v = helper.make_node('Identity', name=node_name_identity_v, inputs=[out_name_current_v],
                                               outputs=[output_name_present_value])
            # K Repetition Node
            node_constant_unsqueeze_axis = helper.make_node('Constant', inputs=[], outputs=[out_name_unsqueeze_axis],
                                                            value=helper.make_tensor(
                                                                name=node_name_unsqueeze_axis,
                                                                data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                dims=unsqueeze_axis.shape,
                                                                vals=unsqueeze_axis.flatten()))
            node_constant_num_rep = helper.make_node('Constant', inputs=[], outputs=[out_name_num_rep],
                                                     value=helper.make_tensor(
                                                         name=node_name_num_rep,
                                                         data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                         dims=num_rep.shape,
                                                         vals=num_rep.flatten()))
            node_constant_shape_current_k = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_current_k],
                                                             value=helper.make_tensor(
                                                                 name=node_name_shape_current_k,
                                                                 data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                 dims=shape_current_k.shape,
                                                                 vals=shape_current_k.flatten()))
            node_constant_shape_current_v = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_current_v],
                                                             value=helper.make_tensor(
                                                                 name=node_name_shape_current_v,
                                                                 data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                                 dims=shape_current_v.shape,
                                                                 vals=shape_current_v.flatten()))
            node_unsqueeze_k = helper.make_node('Unsqueeze', name=node_name_unsqueeze_k, inputs=[out_name_transpose_current_k, out_name_unsqueeze_axis],
                                                outputs=[out_name_unsqueeze_k])
            node_expand_k = helper.make_node('Expand', name=node_name_expand_k, inputs=[out_name_unsqueeze_k, out_name_num_rep],
                                             outputs=[out_name_expand_k])
            node_reshape_current_k = helper.make_node('Reshape', name=node_name_reshape_current_k, inputs=[out_name_expand_k, out_name_shape_current_k],
                                                      outputs=[out_name_reshape_current_k])
            # V Repetition Node
            node_unsqueeze_v = helper.make_node('Unsqueeze', name=node_name_unsqueeze_v, inputs=[out_name_current_v, out_name_unsqueeze_axis],
                                                outputs=[out_name_unsqueeze_v])
            node_expand_v = helper.make_node('Expand', name=node_name_expand_v, inputs=[out_name_unsqueeze_v, out_name_num_rep],
                                             outputs=[out_name_expand_v])
            node_reshape_current_v = helper.make_node('Reshape', name=node_name_reshape_current_v, inputs=[out_name_expand_v, out_name_shape_current_v],
                                                      outputs=[out_name_reshape_current_v])

            # Q * K'
            node_matmul_qk = helper.make_node('MatMul', name=node_name_matmul_qk, inputs=[out_name_rope_q, out_name_reshape_current_k],
                                              outputs=[out_name_matmul_qk])
            node_add_qk_attn_mask = helper.make_node('Add', name=node_name_add_qk_attn_mask, inputs=[out_name_matmul_qk, out_name_gather_attention_mask_row],
                                                     outputs=[out_name_add_qk_attn_mask])
            # Constant Div
            node_constant_div_qk = helper.make_node('Constant', inputs=[], outputs=[out_name_div_value],
                                                    value=helper.make_tensor(
                                                        name=node_name_div_value,
                                                        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                                                        dims=div_value.shape,
                                                        vals=div_value.flatten()))
            node_div_qk = helper.make_node('Div', name=node_name_div_qk, inputs=[out_name_add_qk_attn_mask, out_name_div_value],
                                           outputs=[out_name_div_qk])
            # Softmax Node
            node_softmax_qk = helper.make_node('Softmax', name=node_name_softmax_qk, inputs=[out_name_div_qk],
                                               outputs=[out_name_softmax_qk])
            # Attn * V
            node_matmul_attnv = helper.make_node('MatMul', name=node_name_matmul_attnv, inputs=[out_name_softmax_qk, out_name_reshape_current_v],
                                                 outputs=[out_name_matmul_attnv])
            node_transpose_attnv = helper.make_node('Transpose', name=node_name_transpose_attnv, inputs=[out_name_matmul_attnv],
                                                    outputs=[out_name_transpose_attnv], perm=[0, 2, 1, 3])
            node_constant_shape_attnv = helper.make_node('Constant', inputs=[], outputs=[out_name_shape_attnv],
                                                         value=helper.make_tensor(
                                                             name=node_name_shape_attnv,
                                                             data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('int64')],
                                                             dims=shape_attnv.shape,
                                                             vals=shape_attnv.flatten()))
            node_reshape_attnv = helper.make_node('Reshape', name=node_name_reshape_attnv, inputs=[out_name_transpose_attnv, out_name_shape_attnv],
                                                  outputs=[out_name_reshape_attnv])

            # Create intermediate output tensors and add to graph: Value_Info
            vi_reshape_q = helper.make_tensor_value_info(out_name_reshape_q, TensorProto.FLOAT,
                                                         [batch_size, 'sequence_length', shape_q[2].item(), shape_q[3].item()])
            vi_reshape_k = helper.make_tensor_value_info(out_name_reshape_k, TensorProto.FLOAT,
                                                         [batch_size, 'sequence_length', shape_k[2].item(), shape_k[3].item()])
            vi_reshape_v = helper.make_tensor_value_info(out_name_reshape_v, TensorProto.FLOAT,
                                                         [batch_size, 'sequence_length', shape_v[2].item(), shape_v[3].item()])
            vi_transpose_q = helper.make_tensor_value_info(out_name_transpose_q, TensorProto.FLOAT,
                                                           [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item()])
            vi_transpose_k = helper.make_tensor_value_info(out_name_transpose_k, TensorProto.FLOAT,
                                                           [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item()])
            vi_transpose_v = helper.make_tensor_value_info(out_name_transpose_v, TensorProto.FLOAT,
                                                           [batch_size, shape_v[2].item(), 'sequence_length',  shape_v[3].item()])
            vi_split_q1 = helper.make_tensor_value_info(out_name_split_q1, TensorProto.FLOAT,
                                                        [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_split_q2 = helper.make_tensor_value_info(out_name_split_q2, TensorProto.FLOAT,
                                                        [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_gather_sin_cache_row = helper.make_tensor_value_info(out_name_gather_sin_cache_row, TensorProto.FLOAT,
                                                                    [batch_size, 'sequence_length', shape_q[3].item() // 2 ])
            vi_gather_cos_cache_row = helper.make_tensor_value_info(out_name_gather_cos_cache_row, TensorProto.FLOAT,
                                                                    [batch_size, 'sequence_length', shape_q[3].item() // 2 ])
            vi_unsqueeze_sin_cache_row = helper.make_tensor_value_info(out_name_sin_cache_row, TensorProto.FLOAT,
                                                                       [batch_size, 1, 'sequence_length', shape_q[3].item() // 2 ])
            vi_unsqueeze_cos_cache_row = helper.make_tensor_value_info(out_name_cos_cache_row, TensorProto.FLOAT,
                                                                       [batch_size, 1, 'sequence_length', shape_q[3].item() // 2 ])
            vi_mul_q1_sin = helper.make_tensor_value_info(out_name_mul_q1_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_mul_q1_cos = helper.make_tensor_value_info(out_name_mul_q1_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_q_weight_neg1_sin_cache = helper.make_tensor_value_info(out_name_q_weight_neg1_sin_cache, TensorProto.FLOAT,
                                                                       [batch_size, 'sequence_length',  1, shape_q[3].item() // 2 ])
            vi_mul_q2_sin = helper.make_tensor_value_info(out_name_mul_q2_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_mul_q2_cos = helper.make_tensor_value_info(out_name_mul_q2_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_add_q1_sin_q2_cos = helper.make_tensor_value_info(out_name_add_q1_sin_q2_cos, TensorProto.FLOAT,
                                                                 [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_add_q1_cos_q2_sin = helper.make_tensor_value_info(out_name_add_q1_cos_q2_sin, TensorProto.FLOAT,
                                                                 [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item() // 2 ])
            vi_rope_q = helper.make_tensor_value_info(out_name_rope_q, TensorProto.FLOAT,
                                                      [batch_size, shape_q[2].item(), 'sequence_length',  shape_q[3].item()])
            vi_split_k1 = helper.make_tensor_value_info(out_name_split_k1, TensorProto.FLOAT,
                                                        [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_split_k2 = helper.make_tensor_value_info(out_name_split_k2, TensorProto.FLOAT,
                                                        [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_mul_k1_sin = helper.make_tensor_value_info(out_name_mul_k1_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_mul_k1_cos = helper.make_tensor_value_info(out_name_mul_k1_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_k_weight_neg1_sin_cache = helper.make_tensor_value_info(out_name_k_weight_neg1_sin_cache, TensorProto.FLOAT,
                                                                       [batch_size, 'sequence_length', 1, shape_k[3].item() // 2 ])
            vi_mul_k2_sin = helper.make_tensor_value_info(out_name_mul_k2_sin, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_mul_k2_cos = helper.make_tensor_value_info(out_name_mul_k2_cos, TensorProto.FLOAT,
                                                          [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_add_k1_sin_k2_cos = helper.make_tensor_value_info(out_name_add_k1_sin_k2_cos, TensorProto.FLOAT,
                                                                 [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_add_k1_cos_k2_sin = helper.make_tensor_value_info(out_name_add_k1_cos_k2_sin, TensorProto.FLOAT,
                                                                 [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() // 2 ])
            vi_rope_k = helper.make_tensor_value_info(out_name_rope_k, TensorProto.FLOAT,
                                                      [batch_size, shape_k[2].item(), 'sequence_length',  shape_k[3].item() ])
            vi_split_past_k1 = helper.make_tensor_value_info(out_name_split_past_key_discard, TensorProto.FLOAT,
                                                             [batch_size, shape_k[2].item(), cur_seq_len, shape_k[3].item() ])
            vi_split_past_k2 = helper.make_tensor_value_info(out_name_split_past_key_retain, TensorProto.FLOAT,
                                                             [batch_size, shape_k[2].item(), total_seq_len - cur_seq_len, shape_k[3].item() ])
            vi_split_past_v1 = helper.make_tensor_value_info(out_name_split_past_value_discard, TensorProto.FLOAT,
                                                             [batch_size, shape_v[2].item(), cur_seq_len, shape_v[3].item() ])
            vi_split_past_v2 = helper.make_tensor_value_info(out_name_split_past_value_retain, TensorProto.FLOAT,
                                                             [batch_size, shape_v[2].item(), total_seq_len - cur_seq_len, shape_v[3].item() ])
            vi_current_k = helper.make_tensor_value_info(out_name_current_k, TensorProto.FLOAT,
                                                         [batch_size, shape_k[2].item(), total_seq_len,  shape_k[3].item() ])
            vi_current_v = helper.make_tensor_value_info(out_name_current_v, TensorProto.FLOAT,
                                                         [batch_size, shape_v[2].item(), total_seq_len,  shape_v[3].item() ])
            vi_transpose_current_k = helper.make_tensor_value_info(out_name_transpose_current_k, TensorProto.FLOAT,
                                                                   [batch_size, shape_k[2].item(), shape_k[3].item(), total_seq_len])
            vi_unsqueeze_k = helper.make_tensor_value_info(out_name_unsqueeze_k, TensorProto.FLOAT,
                                                           [batch_size, shape_k[2].item(), 1, shape_k[3].item(), total_seq_len])
            vi_expand_k = helper.make_tensor_value_info(out_name_expand_k, TensorProto.FLOAT,
                                                        [batch_size, shape_k[2].item(), num_rep.item(), shape_k[3].item(), total_seq_len])
            vi_reshape_current_k = helper.make_tensor_value_info(out_name_reshape_current_k, TensorProto.FLOAT,
                                                                 [batch_size, shape_k[2].item() * num_rep.item(), shape_k[3].item(), total_seq_len])
            vi_unsqueeze_v = helper.make_tensor_value_info(out_name_unsqueeze_v, TensorProto.FLOAT,
                                                           [batch_size, shape_v[2].item(), 1, total_seq_len, shape_v[3].item()])
            vi_expand_v = helper.make_tensor_value_info(out_name_expand_v, TensorProto.FLOAT,
                                                        [batch_size, shape_v[2].item(), num_rep.item(),  total_seq_len, shape_v[3].item()])
            vi_reshape_current_v = helper.make_tensor_value_info(out_name_reshape_current_v, TensorProto.FLOAT,
                                                                 [batch_size, shape_v[2].item() * num_rep.item(), total_seq_len, shape_v[3].item()])
            vi_matmul_qk = helper.make_tensor_value_info(out_name_matmul_qk, TensorProto.FLOAT,
                                                         [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_add_qk_attn_mask = helper.make_tensor_value_info(out_name_add_qk_attn_mask, TensorProto.FLOAT,
                                                                [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_div_qk = helper.make_tensor_value_info(out_name_div_qk, TensorProto.FLOAT,
                                                      [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_softmax_qk = helper.make_tensor_value_info(out_name_softmax_qk, TensorProto.FLOAT,
                                                          [batch_size, shape_q[2].item(), 'sequence_length', total_seq_len])
            vi_matmul_attnv = helper.make_tensor_value_info(out_name_matmul_attnv, TensorProto.FLOAT,
                                                            [batch_size, shape_q[2].item(), 'sequence_length', shape_q[3].item()])
            vi_transpose_attnv = helper.make_tensor_value_info(out_name_transpose_attnv, TensorProto.FLOAT,
                                                               [batch_size, 'sequence_length', shape_q[2].item(), shape_q[3].item()])
            vi_reshape_attnv = helper.make_tensor_value_info(out_name_reshape_attnv, TensorProto.FLOAT,
                                                             [batch_size, 'sequence_length', shape_attnv[2].item()])
            gqa_vi = [vi for vi in model.graph.value_info if vi.name == output_name_gqa][0]
            model.graph.value_info.extend([vi_reshape_q, vi_reshape_k, vi_reshape_v,
                                           vi_transpose_q, vi_transpose_k, vi_transpose_v,
                                           vi_split_q1, vi_split_q2, vi_gather_sin_cache_row,
                                           vi_gather_cos_cache_row,
                                           vi_unsqueeze_sin_cache_row, vi_unsqueeze_cos_cache_row, vi_mul_q1_sin,
                                           vi_mul_q1_cos, vi_q_weight_neg1_sin_cache, vi_mul_q2_sin, vi_mul_q2_cos,
                                           vi_add_q1_sin_q2_cos, vi_add_q1_cos_q2_sin, vi_rope_q,
                                           vi_split_k1, vi_split_k2, vi_mul_k1_sin, vi_mul_k1_cos,
                                           vi_k_weight_neg1_sin_cache, vi_mul_k2_sin, vi_mul_k2_cos,
                                           vi_add_k1_sin_k2_cos, vi_add_k1_cos_k2_sin, vi_rope_k,
                                           vi_split_past_k1, vi_split_past_k2, vi_split_past_v1, vi_split_past_v2,
                                           vi_current_k, vi_current_v, vi_transpose_current_k,
                                           vi_unsqueeze_k, vi_expand_k, vi_reshape_current_k,
                                           vi_unsqueeze_v, vi_expand_v, vi_reshape_current_v,
                                           vi_matmul_qk, vi_add_qk_attn_mask, vi_div_qk, vi_softmax_qk,
                                           vi_matmul_attnv, vi_transpose_attnv, vi_reshape_attnv])
            model.graph.value_info.remove(gqa_vi)

            # Add created nodes to graph
            model.graph.node.extend([node_constant_shape_q, node_constant_shape_k, node_constant_shape_v,
                                     node_reshape_q, node_reshape_k, node_reshape_v,
                                     node_transpose_q, node_transpose_k, node_transpose_v,
                                     node_gather_sin_cache_row, node_gather_cos_cache_row,
                                     node_constant_unsqueeze_cache_axis, node_unsqueeze_sin_cache, node_unsqueeze_cos_cache,
                                     node_split_q, node_mul_q1_sin, node_mul_q1_cos,
                                     node_constant_neg1_q, node_mul_q_neg1_sin_cache, node_mul_q2_sin, node_mul_q2_cos,
                                     node_add_q1_sin_q2_cos, node_add_q1_cos_q2_sin, node_concat_q1_q2,
                                     node_split_k, node_mul_k1_sin, node_mul_k1_cos,
                                     node_mul_k_neg1_sin_cache, node_mul_k2_sin, node_mul_k2_cos,
                                     node_add_k1_sin_k2_cos, node_add_k1_cos_k2_sin, node_concat_k1_k2, node_constant_split_past_data,
                                     node_split_past_key, node_concat_k, node_identity_k,
                                     node_split_past_value, node_concat_v, node_identity_v,
                                     node_transpose_current_k, node_constant_unsqueeze_axis, node_constant_num_rep,
                                     node_constant_shape_current_k, node_constant_shape_current_v,
                                     node_unsqueeze_k, node_expand_k, node_reshape_current_k,
                                     node_unsqueeze_v, node_expand_v, node_reshape_current_v,
                                     node_matmul_qk, node_add_qk_attn_mask, node_constant_div_qk,
                                     node_div_qk, node_softmax_qk,
                                     node_matmul_attnv, node_transpose_attnv,
                                     node_constant_shape_attnv, node_reshape_attnv])
            # Remove GQA Node
            model.graph.node.remove(node)

            update_o_proj_input(model, node_name_reshape_attnv, out_name_reshape_attnv)


def unpack_qkv(model: ModelProto, model_config: dict):
    """
        Utility function to subdivide ort-genai generated combined QKV FullyConnected(FC).
        Combined QKV op is split into 3 FC Ops for Q,K and V respectively.
    """
    def update_gqa_inputs(model: ModelProto, node_name: str, output_tensor_name: str):
        gqa_node_name = node_name.replace("q_proj/MatMul", "GroupQueryAttention")

        for node in model.graph.node:
            if node.name == gqa_node_name:
                node.input[0] = output_tensor_name
                node.input[1] = output_tensor_name.replace("q_proj", "k_proj")
                node.input[2] = output_tensor_name.replace("q_proj", "v_proj")

    hidden_size = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config["num_key_value_heads"] if "num_key_value_heads" in model_config else num_heads
    head_dim = hidden_size // num_heads
    n_q = num_heads * head_dim
    n_k = num_kv_heads * head_dim
    n_v = num_kv_heads * head_dim

    # Iterate over nodes
    for node in model.graph.node:

        # check for matmul that has q, k, v weights packed into one tensor
        if node.op_type == "MatMul" and "qkv_proj" in node.name:

            # get input, output and weight names
            name_qkv = node.name
            input_name_qkv = node.input[0]
            output_name_qkv = node.output[0]
            weight_name_qkv = node.input[1]

            # get weight initializer
            weight_init_qkv = [initializer for initializer in model.graph.initializer if weight_name_qkv == initializer.name][0]

            # get weight tensor
            weight_tensor_qkv = numpy_helper.to_array(weight_init_qkv)

            # split packed tensor into wq, wk, wv
            weight_tensor_q = np.copy(weight_tensor_qkv[:, : n_q])
            weight_tensor_k = np.copy(weight_tensor_qkv[:, n_q: n_q + n_k])
            weight_tensor_v = np.copy(weight_tensor_qkv[:, n_q + n_k: n_q + n_k + n_v])

            # remove packed tensor
            model.graph.initializer.remove(weight_init_qkv)

            # set wq, wk, wv tensor names
            split_w_qkv = weight_name_qkv.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
            weight_name_q = split_w_qkv[0] + "q_proj" + split_w_qkv[1]
            weight_name_k = split_w_qkv[0] + "k_proj" + split_w_qkv[1]
            weight_name_v = split_w_qkv[0] + "v_proj" + split_w_qkv[1]

            # set names for q, k, v Matmul nodes
            split_name_qkv = name_qkv.split(ONNX_TENSOR_NAME_STRINGS["llama_qkv_proj"])
            node_name_q = split_name_qkv[0] + "q_proj" + split_name_qkv[1]
            node_name_k = split_name_qkv[0] + "k_proj" + split_name_qkv[1]
            node_name_v = split_name_qkv[0] + "v_proj" + split_name_qkv[1]

            # create initializers from split tensors
            init_q = numpy_helper.from_array(weight_tensor_q, weight_name_q)
            init_k = numpy_helper.from_array(weight_tensor_k, weight_name_k)
            init_v = numpy_helper.from_array(weight_tensor_v, weight_name_v)

            # add split tensors to initializers
            model.graph.initializer.extend([init_q, init_k, init_v])

            # set output names for split nodes
            out_name_q = node_name_q + "/output_0"
            out_name_k = node_name_k + "/output_0"
            out_name_v = node_name_v + "/output_0"

            # create split nodes
            node_matmul_q = helper.make_node("MatMul", name=node_name_q, inputs=[input_name_qkv, weight_name_q],
                                             outputs=[out_name_q])

            node_matmul_k = helper.make_node("MatMul", name=node_name_k, inputs=[input_name_qkv, weight_name_k],
                                             outputs=[out_name_k])

            node_matmul_v = helper.make_node("MatMul", name=node_name_v, inputs=[input_name_qkv, weight_name_v],
                                             outputs=[out_name_v])

            # create output tensors and add to graph
            q_vi = helper.make_tensor_value_info(out_name_q, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_q.shape[-1]])
            k_vi = helper.make_tensor_value_info(out_name_k, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_k.shape[-1]])
            v_vi = helper.make_tensor_value_info(out_name_v, TensorProto.FLOAT,
                                                 ["batch_size", "sequence_length", weight_tensor_v.shape[-1]])

            qkv_vi = [vi for vi in model.graph.value_info if vi.name == output_name_qkv][0]

            model.graph.value_info.extend([q_vi, k_vi, v_vi])
            model.graph.value_info.remove(qkv_vi)

            # add split nodes to graph
            model.graph.node.extend([node_matmul_q, node_matmul_k, node_matmul_v])
            model.graph.node.remove(node)

            update_gqa_inputs(model, node_name_q, out_name_q)


def append_onnx_gqa_encodings(param_encodings: dict):
    """
        Utility Function that appends Fp32 encodings for ONNX GQA Gather Ops Data (sin_cache, cos_cache, etc).
        This is required so that Quantizer does not quantize these data inputs.
    """
    for onnx_name in ONNX_ENCODING_STRINGS:
        gather_onnx_encodings = generate_fp32_encodings(32, 'FLOAT', 'PER_TENSOR')
        gather_onnx_encodings['name'] = onnx_name
        param_encodings.append(gather_onnx_encodings)


def update_encodings(param_encodings: dict, num_layers: int, model_type: str):
    """
        Utility Function that updates the names of tensors in encodings.json file.
        The tensor names in encodings corresponding to GGUF are updated with ONNX model tensor names.
    """
    onnx_tensor_name_map = GGUF_TO_ONNX_TENSOR[model_type]

    for param_encoding in param_encodings:
        tensor_name = param_encoding["name"]

        for onnx_tensor_key in onnx_tensor_name_map:
            if onnx_tensor_key in tensor_name:
                onnx_tensor_value = onnx_tensor_name_map[onnx_tensor_key]
                if onnx_tensor_key == "output_norm":
                    onnx_tensor_value = onnx_tensor_value.format(max_block=num_layers)
                tensor_name = tensor_name.replace(onnx_tensor_key, onnx_tensor_value)
            param_encoding["name"] = tensor_name


def permute_weights(weights, num_heads: int, num_kv_heads: int):
    if num_kv_heads is not None and num_heads != num_kv_heads:
        num_heads = num_kv_heads
    return (weights.reshape(num_heads, 2, weights.shape[0] // num_heads // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))