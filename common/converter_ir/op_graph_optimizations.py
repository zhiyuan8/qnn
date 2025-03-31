# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import collections
import copy
import math
import numpy as np
import random
from functools import reduce

from qti.aisw.converters.common import ir_graph
from qti.aisw.converters.common.converter_ir import translation, op_adapter, op_graph
from qti.aisw.converters.common.converter_ir.op_graph import InputEncodings, InputLayout
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker, AxisOrder, CaffeAxisOrder, SpatialLastAxisOrder, RelayAxisOrder
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.converters.common.utils.converter_utils import *
from qti.aisw.converters.common.utils.argparser_util import ArgParserWrapper
from qti.aisw.converters.common.utils import code_to_message, translation_utils
import qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm as lstm_props
import qti.aisw.converters.common.converter_ir.op_properties.stateful_gru as gru_props
try:
    from qti.aisw.converters.common.passes.layout_transform.layout_manager import PyIrGraphLayoutManager
except ImportError:
    PyIrGraphLayoutManager = None
else:
    from qti.aisw.converters.common.passes.layout_transform import layout_inferer

# ------------------------------
#   Module Level enum/Functions
# ------------------------------
INJECT_CAST_FOR_GATHER = "INJECT_CAST_FOR_GATHER"
REMOVE_IDENTITY = "REMOVE_IDENTITY"
REMOVE_CAST_IDENTITY = "REMOVE_CAST_IDENTITY"
REMOVE_DISCONNECTED = "REMOVE_DISCONNECTED"
MATCH_CHANNELSHUFFLE = "MATCH_CHANNELSHUFFLE"
MATCH_GATHERND = "MATCH_GATHERND"
MATCH_GELU = "MATCH_GELU"
MATCH_GELU_APPROX = "MATCH_GELU_APPROX"
MATCH_HARDSWISH = "MATCH_HARDSWISH"
MATCH_LAYERNORM = "MATCH_LAYERNORM"
MATCH_RMSNORM = "MATCH_RMSNORM"
ADJUST_NORM_OP_BUFFERS = "ADJUST_NORM_OP_BUFFERS"
MATCH_CAFFE_SSD_TO_TF = "MATCH_CAFFE_SSD_TO_TF"
MATCH_SPACETODEPTH = "MATCH_SPACETODEPTH"
MATCH_DEPTHTOSPACE = "MATCH_DEPTHTOSPACE"
SQUASH_BATCHNORM = "SQUASH_BATCHNORM"
SQUASH_SCALE = "SQUASH_SCALE"
SQUASH_BOX_DECODER = "SQUASH_BOX_DECODER"
SQUASH_SUM = "SQUASH_SUM"
SQUASH_PROD = "SQUASH_PROD"
SQUASH_DIV = "SQUASH_DIV"
SQUASH_SUB = "SQUASH_SUB"
SQUASH_PAD = "SQUASH_PAD"
SQUASH_RESHAPE = "SQUASH_RESHAPE"
# Common function to handle both Transpose-Reshape and Reshape-Transpose sequence
SQUASH_TRANSPOSE_RESHAPE = "SQUASH_TRANSPOSE_RESHAPE"
FOLD_CAST = "FOLD_CAST"
FOLD_CONCATS = "FOLD_CONCATS"
FOLD_RESHAPES = "FOLD_RESHAPES"
FOLD_MULTIPLE_TRANSPOSE = "FOLD_MULTIPLE_TRANSPOSE"
FOLD_SOFTMAX = "FOLD_SOFTMAX"
AXES_TO_SPATIAL_FIRST_ORDER = "AXES_TO_SPATIAL_FIRST_ORDER"
ADD_QPARAMS = "ADD_QPARAMS"
ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE = "ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE"
ADJUST_NMS_FEATURE_DIMS = "ADJUST_NMS_FEATURE_DIMS"
EXTRACT_COLOR_TRANSFROM = "EXTRACT_COLOR_TRANSFROM"
OPTIMIZE_NEG = "OPTIMIZE_NEG"
PREPROCESS_ROI_POOL_INPUTS = "PREPROCESS_ROI_POOL_INPUTS"
UNROLL_LSTM_TIME_STEPS = "UNROLL_LSTM_TIME_STEPS"
EXPAND_LSTM_OP_STRUCTURE = "EXPAND_LSTM_OP_STRUCTURE"
MULTI_TIME_STEPS_LSTM = "MULTI_TIME_STEPS_LSTM"
UNROLL_GRU_TIME_STEPS = "UNROLL_GRU_TIME_STEPS"
EXPAND_GRU_OP_STRUCTURE = "EXPAND_GRU_OP_STRUCTURE"
MULTI_TIME_STEPS_GRU = "MULTI_TIME_STEPS_GRU"
SQUASH_CONSTANT_INPUT = "SQUASH_CONSTANT_INPUT"
MERGE_LOW_LEVEL_OPS_TO_LAYERS = "MERGE_LOW_LEVEL_OPS_TO_LAYERS"
MATMUL_TO_FC = "MATMUL_TO_FC"
MATMUL_ADD_FUSION = "MATMUL_ADD_FUSION"
REMOVE_QUANT_NODES = "REMOVE_QUANT_NODES"
SQUASH_QUANT_NODES = "SQUASH_QUANT_NODES"
ALIGN_MATMUL_RANKS = "ALIGN_MATMUL_RANKS"
PREPARE_INPUTS_AS_PARAMS = "PREPARE_INPUTS_AS_PARAMS"
MASKED_SOFTMAX = "MASKED_SOFTMAX"
HANDLE_GATHER_NEGATIVE_INDICES = "HANDLE_GATHER_NEGATIVE_INDICES"
PREPARE_BIASES = "PREPARE_BIASES"
REPLACE_6D_OPERATION = "REPLACE_6D_OPERATION"
CAST_FP16_TO_FP32 = "CAST_FP16_TO_FP32"
EXPAND_SPARSE_OP_STRUCTURE = "EXPAND_SPARSE_OP_STRUCTURE"
expand_1d_spatial_nn_nodes = "expand_1d_spatial_nn_nodes"
expand_thresholded_relu = "expand_thresholded_relu"
EXPAND_REDUCE_L2_OP_STRUCTURE = "EXPAND_REDUCE_L2_OP_STRUCTURE"
TRANSFORM_LAYOUT = "TRANSFORM_LAYOUT"
SINK_TRANSPOSE_BELOW_SUM = "SINK_TRANSPOSE_BELOW_SUM"
expand_elementwise_mean = "expand_elementwise_mean"
EXPAND_INVERSE = "EXPAND_INVERSE"
EXPAND_TO_TILE = "EXPAND_TO_TILE"

supported_opt_list = [SQUASH_SCALE, SQUASH_PROD, SQUASH_DIV, SQUASH_SUM, SQUASH_SUB, SQUASH_RESHAPE, SQUASH_BATCHNORM, FOLD_CAST, FOLD_CONCATS, FOLD_RESHAPES,
                      FOLD_SOFTMAX, MATCH_CHANNELSHUFFLE, MATCH_GATHERND, MATCH_GELU, MATCH_GELU_APPROX, MATCH_HARDSWISH, MATCH_LAYERNORM, AXES_TO_SPATIAL_FIRST_ORDER,
                      REMOVE_IDENTITY, REMOVE_CAST_IDENTITY, ADD_QPARAMS, ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE, MATCH_DEPTHTOSPACE,
                      ADJUST_NMS_FEATURE_DIMS, EXTRACT_COLOR_TRANSFROM, OPTIMIZE_NEG, MATCH_SPACETODEPTH,
                      PREPROCESS_ROI_POOL_INPUTS, UNROLL_LSTM_TIME_STEPS, EXPAND_LSTM_OP_STRUCTURE, MULTI_TIME_STEPS_LSTM, SQUASH_PAD,
                      MERGE_LOW_LEVEL_OPS_TO_LAYERS, MATMUL_TO_FC, SQUASH_CONSTANT_INPUT, INJECT_CAST_FOR_GATHER, REMOVE_QUANT_NODES, SQUASH_QUANT_NODES,
                      ALIGN_MATMUL_RANKS, PREPARE_INPUTS_AS_PARAMS, HANDLE_GATHER_NEGATIVE_INDICES, PREPARE_BIASES, REPLACE_6D_OPERATION,
                      expand_1d_spatial_nn_nodes, UNROLL_GRU_TIME_STEPS, EXPAND_GRU_OP_STRUCTURE, MULTI_TIME_STEPS_GRU, MASKED_SOFTMAX, CAST_FP16_TO_FP32,
                      EXPAND_SPARSE_OP_STRUCTURE, FOLD_MULTIPLE_TRANSPOSE, MATCH_RMSNORM, ADJUST_NORM_OP_BUFFERS, expand_thresholded_relu,
                      TRANSFORM_LAYOUT, SQUASH_TRANSPOSE_RESHAPE, SINK_TRANSPOSE_BELOW_SUM, EXPAND_REDUCE_L2_OP_STRUCTURE,
                      expand_elementwise_mean, EXPAND_TO_TILE, EXPAND_INVERSE, MATMUL_ADD_FUSION]

spatial_first_format_to_channel_first_permute_order = {'NDHWC': AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                                       'NHWC': AxisTracker.AxisFormat.NSC_TO_NCS,
                                                       'NFC': AxisTracker.AxisFormat.NFC_TO_NCF,
                                                       'NTF': AxisTracker.AxisFormat.NTF_TO_TNF}
spatial_first_format_to_channel_first_format = {'NDHWC': AxisTracker.AxisFormat.NCDHW,
                                                'NHWC': AxisTracker.AxisFormat.NCS,
                                                'NFC': AxisTracker.AxisFormat.NCF,
                                                'NTF': AxisTracker.AxisFormat.TNF}
OptimizationTranslations = translation.TranslationBank()


class IROptimizations(object):
    class ArgParser(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(IROptimizations.ArgParser, self).__init__(**kwargs)
            self.add_optional_argument("--dumpIR", action="store_true",
                                       help=argparse.SUPPRESS,
                                       default=False)
            self.add_optional_argument("--disable_batchnorm_folding",
                                       default=False,
                                       action="store_true")
            self.add_optional_argument("--squash_box_decoder",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--match_caffe_ssd_to_tf",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--adjust_nms_features_dims",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--extract_color_transform",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--preprocess_roi_pool_inputs",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_layout_transformation",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_axes_to_spatial_first_order",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_lstm_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_lstm_op_structure",
                                       default=False,
                                       help="Enables optimization that breaks the LSTM op to equivalent math ops",
                                       action="store_true")
            self.add_optional_argument("--multi_time_steps_lstm",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_gru_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_gru_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--multi_time_steps_gru",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--force_prune_cast_ops",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--inject_cast_for_gather",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--use_convert_quantization_nodes",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--align_matmul_ranks",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--prepare_inputs_as_params",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--handle_gather_negative_indices",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--enable_match_gathernd",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_sparse_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--keep_disconnected_nodes",
                                       default=False,
                                       help="Disable Optimization that removes Ops not connected to the main graph.\n"
                                            "This optimization uses output names provided over commandline OR\n"
                                            "inputs/outputs extracted from the Source model to determine the main graph",
                                       action="store_true")

            masked_softmax_group = self.add_argument_group(
                title='Masked Softmax Optimization Options')

            masked_softmax_group.add_argument("--apply_masked_softmax",
                                        default="uncompressed",
                                        choices=["compressed", "uncompressed"],
                                        type=str,
                                        help="This flag enables the pass that creates a MaskedSoftmax Op and\n" \
                                            "rewrites the graph to include this Op. MaskedSoftmax Op may not\n" \
                                            "be supported by all the QNN backends. Please check the\n" \
                                            "supplemental backend XML for the targeted backend.\n" \
                                            "This argument takes a string parameter input that selects\n" \
                                            "the mode of MaskedSoftmax Op.\n" \
                                            "'compressed' value rewrites the graph with the compressed version of MaskedSoftmax Op.\n" \
                                            "'uncompressed' value rewrites the graph with the uncompressed version of MaskedSoftmax Op.\n")
            masked_softmax_group.add_argument("--packed_masked_softmax_inputs",
                                        nargs='+',
                                        default=[],
                                        help="Mention the input ids tensor name which will be packed in the single inference.\n" \
                                            "This is applicable only for Compressed MaskedSoftmax Op.\n"
                                            "This will create a new input to the graph named 'position_ids'\n" \
                                            "with same shape as the provided input name in this flag.\n" \
                                            "During runtime, this input shall be provided with the token\n" \
                                            "locations for individual sequences so that the same will be\n" \
                                            "internally passed to positional embedding layer.\n" \
                                            "E.g. If 2 sequences of length 20 and 30 are packed together\n" \
                                            "in single batch of 64 tokens then this new input 'position_ids' should have\n" \
                                            "value [0, 1, ..., 19, 0, 1, ..., 29, 0, 0, 0, ..., 0]\n" \
                                            "Usage: --packed_masked_softmax input_ids\n" \
                                            "Packed model will enable the user to pack multiple sequences into\n" \
                                            "single batch of inference.\n")
            masked_softmax_group.add_argument("--packed_max_seq",
                                        default=1,
                                        type=int,
                                        help="Number of sequences packed in the single input ids and \n" \
                                            "single attention mask inputs. Applicable only for \n" \
                                            "Compressed MaskedSoftmax Op.")

    class ArgParserv2(ArgParserWrapper):
        def __init__(self, **kwargs):
            super(IROptimizations.ArgParserv2, self).__init__(**kwargs)
            self.add_optional_argument("--dumpIR", action="store_true",
                                       help=argparse.SUPPRESS,
                                       default=False)
            self.add_optional_argument("--disable_batchnorm_folding",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--squash_box_decoder",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--match_caffe_ssd_to_tf",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--adjust_nms_features_dims",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--extract_color_transform",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--preprocess_roi_pool_inputs",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_layout_transformation",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--perform_axes_to_spatial_first_order",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_lstm_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_lstm_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--multi_time_steps_lstm",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_gru_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--unroll_gru_time_steps",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--multi_time_steps_gru",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--force_prune_cast_ops",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--inject_cast_for_gather",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--use_convert_quantization_nodes",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--align_matmul_ranks",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--prepare_inputs_as_params",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--handle_gather_negative_indices",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--enable_match_gathernd",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--expand_sparse_op_structure",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")
            self.add_optional_argument("--keep_disconnected_nodes",
                                       default=False,
                                       help=argparse.SUPPRESS,
                                       action="store_true")

            masked_softmax_group = self.add_argument_group(
                title='Masked Softmax Optimization Options')

            masked_softmax_group.add_argument("--apply_masked_softmax",
                                              default="uncompressed",
                                              choices=["compressed", "uncompressed"],
                                              type=str,
                                              help=argparse.SUPPRESS)
            masked_softmax_group.add_argument("--packed_masked_softmax_inputs",
                                              nargs='+',
                                              default=[],
                                              help=argparse.SUPPRESS)
            masked_softmax_group.add_argument("--packed_max_seq",
                                              default=1,
                                              type=int,
                                              help=argparse.SUPPRESS)


    def __init__(self, args):
        self.dump_ir_graph = args.dumpIR
        self.dump_qairt_io_config_yaml = ""
        if hasattr(args, 'dump_qairt_io_config_yaml'):
            self.args = args
            self.dump_qairt_io_config_yaml = args.dump_qairt_io_config_yaml
        self.enable_batchnorm_folding = not args.disable_batchnorm_folding
        self.squash_box_decoder = args.squash_box_decoder
        self.match_caffe_ssd_to_tf = args.match_caffe_ssd_to_tf
        self.adjust_nms_features_dims = args.adjust_nms_features_dims
        self.extract_color_transform = args.extract_color_transform
        self.perform_layout_transformation = args.perform_layout_transformation
        self.perform_axes_to_spatial_first_order = args.perform_axes_to_spatial_first_order
        self.preprocess_roi_pool_inputs = args.preprocess_roi_pool_inputs
        self.multi_time_steps_lstm = args.multi_time_steps_lstm
        if self.multi_time_steps_lstm:
            self.unroll_lstm_time_steps = False
            self.expand_lstm_op_structure = False
        else:
            self.unroll_lstm_time_steps = args.unroll_lstm_time_steps
            self.expand_lstm_op_structure = args.expand_lstm_op_structure
        self.multi_time_steps_gru = args.multi_time_steps_gru
        if self.multi_time_steps_gru:
            self.unroll_gru_time_steps = False
            self.expand_gru_op_structure = False
        else:
            self.unroll_gru_time_steps = args.unroll_gru_time_steps
            self.expand_gru_op_structure = args.expand_gru_op_structure
        self.force_prune_cast_ops = args.force_prune_cast_ops
        self.inject_cast_for_gather = args.inject_cast_for_gather
        self.use_convert_quantization_nodes = args.use_convert_quantization_nodes
        self.align_matmul_ranks = args.align_matmul_ranks
        self.prepare_inputs_as_params = args.prepare_inputs_as_params
        self.handle_gather_negative_indices = args.handle_gather_negative_indices
        self.enable_match_gathernd = args.enable_match_gathernd
        self.expand_sparse_op_structure = args.expand_sparse_op_structure
        self.keep_disconnected_nodes = args.keep_disconnected_nodes
        self.apply_masked_softmax = args.apply_masked_softmax if "--apply_masked_softmax" in sys.argv else False
        self.packed_masked_softmax_inputs = args.packed_masked_softmax_inputs
        self.packed_max_seq = args.packed_max_seq

    def optimize(self, graph, backend_info_obj=None):
        # apply graph transformations
        log_debug2("Applying graph Optimizations...")

        # Dump the IR for debug before or after an optimization using graph.dump_json(<filename>)
        if self.dump_ir_graph:
            log_info("Dumping IR graph before all optimizations as IRGraph_before_optimizations.json")
            graph.dump_json("IRGraph_before_optimizations.json")

        # A dict containing the IO tensor name and corresponding layout obtained from the original model.
        # This information is taken from the intial unoptimized IR graph. Hence, this line of code should
        # be before any optimization.
        original_io_layouts = self.get_original_io_layouts(graph) if graph.preserve_io_layout_passed else {}

        # Remove nodes disconnected from the main graph
        # This function should be in the beginning and the end.
        if not self.keep_disconnected_nodes:
            remove_disconnected_nodes(graph)

        # First attempt to match and fold quant nodes, then remove any remaining
        if graph.keep_quant_nodes:
            if self.use_convert_quantization_nodes:
                OptimizationTranslations.apply_method_to_graph(SQUASH_QUANT_NODES, graph, fail_if_no_method=False)
        else:
            OptimizationTranslations.apply_method_to_all_ops(REMOVE_QUANT_NODES, graph, fail_if_no_method=False)

        if graph.has_user_quantization_overrides():
            self.populate_quantization_params(graph)

        # TODO Remove this preparation once backends are able to consume optional bias tensors
        # prepares bias tensors from frontends for consumption by optimizations and backends
        OptimizationTranslations.apply_method_to_all_ops(PREPARE_BIASES, graph, fail_if_no_method=False)

        # this optimization needs to be run first before any other optimizations
        OptimizationTranslations.apply_method_to_all_ops(CAST_FP16_TO_FP32, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_all_ops(SQUASH_CONSTANT_INPUT, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATMUL_TO_FC, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MERGE_LOW_LEVEL_OPS_TO_LAYERS, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PAD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_CAST, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_SOFTMAX, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_CHANNELSHUFFLE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_GELU, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_HARDSWISH, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_SPACETODEPTH, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_DEPTHTOSPACE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_all_ops(REPLACE_6D_OPERATION, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(FOLD_RESHAPES, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_LAYERNORM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(MATCH_RMSNORM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(ADJUST_NORM_OP_BUFFERS, graph, fail_if_no_method=False)

        if self.enable_match_gathernd:
            OptimizationTranslations.apply_method_to_graph(MATCH_GATHERND, graph, fail_if_no_method=False)

        # Element-wise squashing optimizations. This shall be done after matching larger sequences as they single-op
        # squashing into previous layer
        OptimizationTranslations.apply_method_to_graph(SQUASH_SCALE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_PROD, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_DIV, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUM, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_graph(SQUASH_SUB, graph, fail_if_no_method=False)

        if self.enable_batchnorm_folding:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)
        if self.squash_box_decoder:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BOX_DECODER, graph, fail_if_no_method=False)
        if self.match_caffe_ssd_to_tf:
            OptimizationTranslations.apply_method_to_graph(MATCH_CAFFE_SSD_TO_TF, graph, fail_if_no_method=False)
        if self.adjust_nms_features_dims:
            OptimizationTranslations.apply_method_to_graph(ADJUST_NMS_FEATURE_DIMS, graph, fail_if_no_method=False)
        if self.extract_color_transform:
            OptimizationTranslations.apply_method_to_graph(EXTRACT_COLOR_TRANSFROM, graph, fail_if_no_method=False)

        OptimizationTranslations.apply_method_to_graph(SQUASH_RESHAPE, graph, fail_if_no_method=False)
        # ------------------------------------------------------------------------------
        #   PRE-PROCESSING
        # TODO: Move once optimizations are split into backend specific sections
        # ------------------------------------------------------------------------------
        # pre-process roi inputs
        if self.preprocess_roi_pool_inputs:
            OptimizationTranslations.apply_method_to_graph(PREPROCESS_ROI_POOL_INPUTS, graph, fail_if_no_method=False)

        # Performs pruning of cast Ops that are identity, if force_prune is set then all cast ops are pruned
        # TODO: remove separate identity call for casts when Cast supported by all backends
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_CAST_IDENTITY, graph, force_prune=self.force_prune_cast_ops,
                                                         fail_if_no_method=False)

        OptimizationTranslations.apply_method_to_all_ops(expand_1d_spatial_nn_nodes, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_all_ops(expand_thresholded_relu, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_all_ops(EXPAND_REDUCE_L2_OP_STRUCTURE, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_all_ops(expand_elementwise_mean, graph, fail_if_no_method=False)
        OptimizationTranslations.apply_method_to_all_ops(EXPAND_INVERSE, graph, fail_if_no_method=False)

        # Ensure matmul dims are handled/squashed as needed.
        if self.align_matmul_ranks:
            OptimizationTranslations.apply_method_to_all_ops(ALIGN_MATMUL_RANKS, graph, fail_if_no_method=False)
            OptimizationTranslations.apply_method_to_graph(FOLD_RESHAPES, graph, fail_if_no_method=False)

        if self.dump_ir_graph and get_log_level() == logging.VERBOSE:
            graph.dump_json("IRGraph_before_layout_change.json")

        # Save source axis formats
        graph.save_src_axis_formats()

        # Populate Preserve IO Layout Tensors before calling AXES_TO_SPATIAL_FIRST_ORDER
        self.populate_preserve_io_layout_tensors(graph)

        # transition to NSC
        if self.perform_layout_transformation is True and PyIrGraphLayoutManager is not None:
            if self.dump_ir_graph:
                # Custom_IO layout is intergrated into layout-transform,
                # so IRGraph_before_customIO.json needs to be dumped before layout-transform
                log_info("Dumping IR graph before custom IO as IRGraph_before_customIO.json")
                graph.dump_json("IRGraph_before_customIO.json")

            layout_manager = PyIrGraphLayoutManager(graph)
            with LayoutTransformContext(layout_manager):
                OptimizationTranslations.apply_method_to_all_ops(
                    TRANSFORM_LAYOUT, graph, layout_manager
                )

            graph = layout_manager.layout_mutator.new_graph
        elif self.perform_axes_to_spatial_first_order:
            OptimizationTranslations.apply_method_to_all_ops(AXES_TO_SPATIAL_FIRST_ORDER, graph)

        # Moving similar transposes from input to a single transpose after output
        OptimizationTranslations.apply_method_to_graph(SINK_TRANSPOSE_BELOW_SUM, graph, fail_if_no_method=False)

        if self.dump_ir_graph and get_log_level() == logging.VERBOSE:
            graph.dump_json("IRGraph_after_layout_change.json")

        self.squash_multiple_permute(graph)

        if self.dump_ir_graph and get_log_level() == logging.VERBOSE:
            graph.dump_json("IRGraph_after_removing_multiple_permutes.json")

        # Optimize negations which typically apply to binary eltwise operations.
        OptimizationTranslations.apply_method_to_graph(OPTIMIZE_NEG, graph, fail_if_no_method=False)

        # add transpose after output reshape
        OptimizationTranslations.apply_method_to_all_ops(ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE, graph, fail_if_no_method=False)

        # remove IDENTITYs, which may include trivial permutes at this point
        # This may happen because some ops result in constant attributes that are absorbed by the layers
        OptimizationTranslations.apply_method_to_all_ops(REMOVE_IDENTITY, graph, fail_if_no_method=False)

        # Apply fold_concats optimization
        OptimizationTranslations.apply_method_to_graph(FOLD_CONCATS, graph, fail_if_no_method=False)

        # remove redundant transpose ops introduced after other optimizations
        OptimizationTranslations.apply_method_to_all_ops(FOLD_MULTIPLE_TRANSPOSE, graph, fail_if_no_method=False)

        # remove redundant transpose-reshape and reshape-transpose ops introduced after other optimizations
        OptimizationTranslations.apply_method_to_graph(SQUASH_TRANSPOSE_RESHAPE, graph, fail_if_no_method=False)

        # this is to squash batchnorm for the case [FC, Reshape, BN] --> [FC, Reshape]
        if self.enable_batchnorm_folding:
            OptimizationTranslations.apply_method_to_graph(SQUASH_BATCHNORM, graph, fail_if_no_method=False)

        # apply the approx gelu fusion once all the above optimizations are removed.
        # so approx. gelu pass will be of few numbers of patterns only.
        OptimizationTranslations.apply_method_to_graph(MATCH_GELU_APPROX, graph, True, fail_if_no_method=False)

        # add op-specific quantization encodings to QParams Record.
        OptimizationTranslations.apply_method_to_all_ops(ADD_QPARAMS, graph, fail_if_no_method=False)

        # Convert RolledLstmOp to multi-time step LstmOp.
        if self.multi_time_steps_lstm:
            log_warning("Multi-timestep LSTM Op is not supported by every QNN inference backend.\n" \
                        "Please check the supplemental backend XML for the targeted backend.")
            OptimizationTranslations.apply_method_to_graph(MULTI_TIME_STEPS_LSTM, graph, fail_if_no_method=False)
        # Apply expansion to RolledLstmOp
        elif self.expand_lstm_op_structure:
            OptimizationTranslations.apply_method_to_graph(EXPAND_LSTM_OP_STRUCTURE, graph, fail_if_no_method=False)
        # Apply unrolling to replace RolledLstmOp with LstmOp, and perform pre-processing
        elif self.unroll_lstm_time_steps:
            OptimizationTranslations.apply_method_to_graph(UNROLL_LSTM_TIME_STEPS, graph, fail_if_no_method=False)
            OptimizationTranslations.apply_method_to_graph(FOLD_RESHAPES, graph, fail_if_no_method=False)

        # Apply time-unrolling to GRU op
        if self.unroll_gru_time_steps:
            OptimizationTranslations.apply_method_to_graph(UNROLL_GRU_TIME_STEPS, graph, fail_if_no_method=False)

        # Apply expansion to GRU op
        if self.expand_gru_op_structure:
            OptimizationTranslations.apply_method_to_graph(EXPAND_GRU_OP_STRUCTURE, graph, fail_if_no_method=False)

        # Convert MergedWeightsGruOp to multi-time step GruOp.
        if self.multi_time_steps_gru:
            log_warning("Multi-timestep Gru Op is not supported by every QNN inference backend.\n" \
                        "Please check the supplemental backend XML for the targeted backend.")
            OptimizationTranslations.apply_method_to_graph(MULTI_TIME_STEPS_GRU, graph, fail_if_no_method=False)

        # Pre-processing of gather indices input
        if self.handle_gather_negative_indices:
            OptimizationTranslations.apply_method_to_all_ops(HANDLE_GATHER_NEGATIVE_INDICES, graph,
                                                             fail_if_no_method=False)

        # TODO Remove optimization once casts are properly removed in optimization stage
        if self.inject_cast_for_gather:
            OptimizationTranslations.apply_method_to_all_ops(INJECT_CAST_FOR_GATHER, graph, fail_if_no_method=False)

        # Prepares inputs in converter IR as parameters, as needed
        if self.prepare_inputs_as_params:
            OptimizationTranslations.apply_method_to_all_ops(PREPARE_INPUTS_AS_PARAMS, graph, fail_if_no_method=False)

        # Apply MaskedSoftmax optimization
        if self.apply_masked_softmax in ["compressed", "uncompressed"]:
            log_warning("MaskedSoftmax Op may not be supported by all the QNN backends.\n" \
                        "Please check the supplemental backend XML for the targeted backend.")
            OptimizationTranslations.apply_method_to_graph(
                MASKED_SOFTMAX,
                graph,
                original_io_layouts,
                self.apply_masked_softmax,
                self.packed_masked_softmax_inputs,
                self.packed_max_seq,
                fail_if_no_method=False,
            )

        # Apply expansion of sparse ops to produce dense output
        if self.expand_sparse_op_structure:
            OptimizationTranslations.apply_method_to_all_ops(EXPAND_SPARSE_OP_STRUCTURE, graph, fail_if_no_method=False)

        if graph.preserve_io_layout_passed:
            if self.dump_ir_graph:
                log_info("Dumping IR graph before applying preserve IO changes as IRGraph_before_preserveIO.json")
                graph.dump_json("IRGraph_before_preserveIO.json")
            self.preserve_io_layout(graph, original_io_layouts)

        if graph.user_custom_io:
            self.validate_custom_io_config(graph)

            if self.perform_layout_transformation is False:
                if self.dump_ir_graph:
                    log_info("Dumping IR graph before custom IO as IRGraph_before_customIO.json")
                    graph.dump_json("IRGraph_before_customIO.json")
                # Custom_io has been integrated in layout-transform, so we only need to apply
                # custom_io when Axis-Tracking is used
                self.apply_custom_io_change(graph)

            self.populate_custom_io_quantization_params(graph)

        # again, squash op with constant input which may be injectd during optimization
        OptimizationTranslations.apply_method_to_all_ops(SQUASH_CONSTANT_INPUT, graph, fail_if_no_method=False)

        # Remove nodes disconnected from the main graph
        # This function should be in the beginning and the end.
        if not self.keep_disconnected_nodes:
            remove_disconnected_nodes(graph)

        if self.dump_ir_graph:
            log_info("Dumping IR graph after all optimizations as IRGraph_after_optimizations.json")
            graph.dump_json("IRGraph_after_optimizations.json")

        if self.dump_qairt_io_config_yaml:
            graph.dump_qairt_converter_io_config_yaml(self.args)

        graph.validate_trace_info_completeness(graph.fw_op_names, graph.fw_tensor_names, "op_graph_optimization")

        # re-evaluate graph macs and params_count given optimization might have added/removed certain ops
        graph.reeval_macs_params()

        return graph

    def squash_multiple_permute(self, graph):
        def is_valid_sequence(nodes_tuple):
            nonlocal graph
            first_permute, second_permute = nodes_tuple
            first_permute_output_nodes = graph.get_op_output_nodes(first_permute)
            if len(first_permute_output_nodes) != 1 or first_permute_output_nodes[0] != second_permute:
                return False
            return True

        sequence = [
                    ("Transpose",
                        (),
                        ()
                    ),
                    ("Transpose",
                        ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
                        ()
                    )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_sequence)

        for node_tuple in matched_node_list:
            # Multipe consecutive Transpose node sequences can be detected as shown below:
            # (Transpose1, Transpose2) and (Transpose2, Transpose3)
            #     Transpose1
            #         |
            #     Transpose2
            #         |
            #     Transpose3
            # If the first two transpose nodes get squashed then we need to get the newly created
            # Transpose node which will now be the input to the Transpose3.
            first_permute_node, second_permute = node_tuple
            first_permute_output_name = first_permute_node.output_names[0]
            # When multiple Transpose node sequences are detected as enumerated in the comment above,
            # it is possible that after squashing (Tanspose1, Transpose2) the Transpose1 Op so obtained
            # can be squashed into its input if their dimensions match (this condition is checked at the
            # end where the new permute order obtained is same as the order of the input dimension). In
            # that case, Transpose1 output buffer name would have been removed from the graph buffers.
            # So, we need to check for this scenario as well.
            if first_permute_output_name not in graph.buffers:
                continue
            first_permute_buffer = graph.get_buffer(first_permute_output_name)
            first_permute = first_permute_buffer.producer

            # If the producer of Output buffer of the node to be squashed is
            # not a Transpose Op, Then Don't Squash.
            # (Transpose1, Transpose2) and (Transpose2, Transpose3)
            #      Reshape (Any Op Other Than Transpose)
            #         |
            #     Transpose1 (0, 2, 1)
            #         |
            #     Transpose2 (0, 2, 1)
            #         |
            #     Transpose3(x, y ,z)
            # After running the code for matched node (Transpose1, Transpose2),The output
            # will be :-
            #      Reshape (Any Op Other Than Transpose)
            #         |
            #     Transpose3(x, y ,z)
            # This case will cause an error for the matched node_tuple (Transpose2, Transpose3)
            # where first_permute is resolved to Reshape Op in this example.Add a termination
            # condition like skip if first_permute is not resolved to Transpose Op

            if not isinstance(first_permute.op, op_adapter.TransposeOp):
                continue

            first_permute_order = first_permute.op.perm
            second_permute_order = second_permute.op.perm

            new_order = first_permute_order[:]
            for i, val in enumerate(second_permute_order):
                new_order[i] = first_permute_order[val]

            status = graph.squash(second_permute, first_permute.output_names[0], is_data_movement_node=True)
            first_permute.op.perm = new_order
            if new_order == list(range(len(new_order))):
                graph.squash(first_permute, first_permute.input_names[0], is_data_movement_node=True)

            if status:
                log_debug2("Found Multiple Permute: First Permute={}, Second Permute={}, First Order={}, "
                           "Second Order={}, Final Order={}".format(first_permute.op.name,
                                                                    second_permute.op.name,
                                                                    first_permute_order,
                                                                    second_permute_order,
                                                                    new_order))

    @staticmethod
    def get_tensor_type(type_in_str):
        str_to_enum = {'PER_TENSOR': ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                       'PER_CHANNEL': ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
                       'PER_BLOCK': ir_graph.QNN_QUANTIZATION_ENCODING_BLOCK,
                       'LPBQ': ir_graph.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION,
                       'VECTOR': ir_graph.QNN_QUANTIZATION_ENCODING_VECTOR}

        if type_in_str not in str_to_enum:
            raise ValueError('Unknown tensor type ', type_in_str)

        return str_to_enum[type_in_str]

    @staticmethod
    def get_default_enc(name, enc):
        # Everything is optional except bw. Default to 0s and overwrite with actual or calculated values later
        return {"name": name,
                "min": numpy.zeros(len(enc), numpy.float32),
                "max": numpy.zeros(len(enc), numpy.float32),
                "bw": int(enc[0]['bitwidth']),
                "offset": numpy.zeros(len(enc), numpy.int32),
                "scale": numpy.zeros(len(enc), numpy.float32),
                "is_symmetric": enc[0]['is_symmetric'].lower() == "true" if 'is_symmetric' in enc[0] else False,
                "is_fixed_point": enc[0]['dtype'].lower() == "int" if 'dtype' in enc[0] else True,
                "overridden": True}

    @staticmethod
    def get_default_enc_v2():
        return {"name": "",
                "enc_type": 0,
                "min": np.array(list()),
                "max": np.array(list()),
                "bw": 0,
                "offset": np.array(list()),
                "scale": np.array(list()),
                "is_symmetric": False,
                "is_fixed_point": True,
                "overridden": True}

    @staticmethod
    def validate_encoding_v2(enc, name):
        is_symmetric = True if 'is_sym' in enc[0] and enc[0]['is_sym'] else False
        is_fixed_point = False if 'dtype' in enc[0] and enc[0]['dtype'].lower() == "float" else True
        enc_type = enc[0]['enc_type'] if 'enc_type' in enc[0] else "PER_TENSOR"

        if 'bw' not in enc[0]:
            raise AttributeError(name + ': Bitwidth must be present')

        if 'offset' in enc[0] and 'scale' not in enc[0]:
            raise AttributeError(name + ": Offset cannot be present without scale")

        if 'scale' in enc[0] and 'offset' not in enc[0] and not is_symmetric:
            raise AttributeError(name + ": Either scale, offset both should be present or neither should be present in case of "
                             "asymmetric encodings ")

        if enc_type == "LPBQ":
            IROptimizations.validate_encoding_lpbq_v2(enc, name)
        elif enc_type == "PER_BLOCK":
            IROptimizations.validate_encoding_bq_v2(enc, name)
        elif enc_type == "PER_CHANNEL":
            IROptimizations.validate_encoding_per_channel_v2(enc, name)
        elif enc_type == "VECTOR":
            IROptimizations.validate_encoding_vq_v2(enc)

    @staticmethod
    def validate_encoding_per_channel_v2(enc, name):
        is_symmetric = True if 'is_sym' in enc[0] and enc[0]['is_sym'] else False
        if not is_symmetric:
            raise ValueError(name + ": For Per-channel, is_sym must be true")

        if 'offset' in enc[0]:
            if np.all(enc[0]['offset'] != [0] * len(enc[0]['offset'])) and \
               np.all(enc[0]['offset'] != [-2**(enc[0]['bw']-1)] * len(enc[0]['offset'])):
                raise ValueError(name + ": For Per-channel, offset must be 0 or -2^(bw-1)")

    @staticmethod
    def validate_encoding_lpbq_v2(enc, name):
        is_symmetric = True if 'is_sym' in enc[0] and enc[0]['is_sym'] else False
        if not is_symmetric:
            raise ValueError(name + ": For LPBQ, is_sym must be true")

        if 'block_size' not in enc[0] or \
           'compressed_bw' not in enc[0] or \
           'per_block_int_scale' not in enc[0] or \
           'scale' not in enc[0]:
            raise ValueError(name + ": For LPBQ, block_size, compressed_bw, per_block_int_scale, and scale must be present")

        if 'offset' in enc[0]:
            if np.all(enc[0]['offset'] != [0] * len(enc[0]['offset'])) and \
               np.all(enc[0]['offset'] != [-2**(enc[0]['bw']-1)] * len(enc[0]['offset'])) and \
               np.all(enc[0]['offset'] != [-2**(enc[0]['compressed_bw']-1)] * len(enc[0]['offset'])):
                raise ValueError(name + ": For LPBQ, offset must be 0 or -2^(bw-1)")

    @staticmethod
    def validate_encoding_bq_v2(enc, name):
        is_symmetric = True if 'is_sym' in enc[0] and enc[0]['is_sym'] else False
        if not is_symmetric:
            raise ValueError(name + ": For BQ, is_sym must be true")

        if 'block_size' not in enc[0] or \
           'scale' not in enc[0]:
            raise ValueError(name + ": For BQ, block_size, and scale must be present")

        if 'offset' in enc[0]:
            if np.all(enc[0]['offset'] != [0] * len(enc[0]['offset'])) and \
                    np.all(enc[0]['offset'] != [-2**(enc[0]['bw']-1)] * len(enc[0]['offset'])):
                raise ValueError(name + ": For BQ, offset must be 0 or -2^(bw-1)")

    @staticmethod
    def validate_encoding_vq_v2(enc):
        is_symmetric = True if 'is_sym' in enc[0] and enc[0]['is_sym'] else False
        if not is_symmetric:
            raise ValueError("For VQ, is_sym must be true")

        if 'rows_per_block' not in enc[0] or \
           'cols_per_block' not in enc[0] or \
           'vector_dim' not in enc[0] or \
           'vector_stride' not in enc[0] or \
           'index_bw' not in enc[0] or \
           'scale' not in enc[0]:
            raise ValueError("For VQ, rows_per_block, cols_per_block, vector_dimension, vector_stride, scale, "
                             "and index_bw must be present")

        if 'offset' in enc[0]:
            if np.all(enc[0]['offset'] != [0] * len(enc[0]['offset'])) and \
                    np.all(enc[0]['offset'] != [-2**(enc[0]['bw']-1)] * len(enc[0]['offset'])):
                raise ValueError("For VQ, offset must be 0 or -2^(bw-1)")

    # This function converts encodings in user_quantization_overrides format to quantization_params format
    @staticmethod
    def fill_enc_format_v2(enc, new_enc, name):
        IROptimizations.validate_encoding_v2(enc, name)
        new_enc['name'] = name
        new_enc['bw'] = int(enc[0]['bw'])
        new_enc['enc_type'] = enc[0]['enc_type'] if 'enc_type' in enc[0] else "PER_TENSOR"
        new_enc["is_symmetric"] = True if 'is_sym' in enc[0] and enc[0]['is_sym'] else False
        new_enc["is_fixed_point"] = True if 'dtype' in enc[0] and enc[0]['dtype'].lower() == "int" else False
        new_enc['block_size'] = enc[0]['block_size'] if 'block_size' in enc[0] else 0
        new_enc['compressed_bw'] = enc[0]['compressed_bw'] if 'compressed_bw' in enc[0] else 0
        new_enc['per_block_int_scale'] = enc[0]['per_block_int_scale'] if 'per_block_int_scale' in enc[0] else list()
        new_enc['rows_per_block'] = enc[0]['rows_per_block'] if 'rows_per_block' in enc[0] else 0
        new_enc['cols_per_block'] = enc[0]['cols_per_block'] if 'cols_per_block' in enc[0] else 0
        new_enc['vector_dim'] = enc[0]['vector_dim'] if 'vector_dim' in enc[0] else 0
        new_enc['vector_stride'] = enc[0]['vector_stride'] if 'vector_stride' in enc[0] else 1
        new_enc['index_bw'] = enc[0]['index_bw'] if 'index_bw' in enc[0] else 1

        new_enc['enc_type'] = IROptimizations.get_tensor_type(new_enc['enc_type'])

        if 'offset' in enc[0]:
            new_enc['offset'] = np.full(len(enc[0]['scale']), -2**(enc[0]['bw']-1)) if new_enc["is_symmetric"] else enc[0]['offset']
            new_enc['scale'] = enc[0]['scale']
            new_enc['min'] = numpy.zeros(len(enc[0]['scale']), numpy.float32)
            new_enc['max'] = numpy.zeros(len(enc[0]['scale']), numpy.float32)
        elif 'scale' in enc[0]:
            new_enc['scale'] = enc[0]['scale']
            new_enc['offset'] = np.full(len(enc[0]['scale']), -2**(enc[0]['bw']-1))
            new_enc['min'] = numpy.zeros(len(enc[0]['scale']), numpy.float32)
            new_enc['max'] = numpy.zeros(len(enc[0]['scale']), numpy.float32)
        else:
            new_enc['scale'] = numpy.array([0.0])
            new_enc['offset'] = numpy.array([0.0])
            new_enc['min'] = numpy.array([0.0])
            new_enc['max'] = numpy.array([0.0])

        if len(new_enc['scale']) > 1:
            new_enc['axis'] = 3

        if new_enc['block_size'] > 0:
            new_enc['block_axis'] = 2



    @staticmethod
    def extract_encoding_dict(name, enc, version="0.0.6"):

        # Grab a default encoding
        if version == "1.0.0":
            new_enc = IROptimizations.get_default_enc_v2()
            IROptimizations.fill_enc_format_v2(enc, new_enc, name)
        else:
            new_enc = IROptimizations.get_default_enc(name, enc)
            # Loop through each encoding for the tensor. More than one indicates the tensor has
            # per-channel (axis) encodings
            new_enc['enc_type'] = "PER_TENSOR"
            for idx, e in enumerate(enc):
                try:
                    if 'bitwidth' in e:
                        log_assert(e['bitwidth'] == new_enc['bw'],
                                   'Mismatching bitwidths {} and {} for encoding {}',
                                   e['bitwidth'], new_enc['bw'], name)
                    if 'is_symmetric' in e:
                        log_assert((e['is_symmetric'].lower() == "true") == new_enc['is_symmetric'],
                                   'Encodings {} for tensor {} cannot be a mix of symmetric and asymmetric',
                                   enc, name)

                    if 'is_fixed_point' in e:
                        log_assert((e['is_fixed_point'].lower() == "true") == new_enc['is_fixed_point'],
                                   'Encodings {} for tensor {} cannot be a mix of fixed and floating point',
                                    enc, name)
                    # For min/max and scale/offset if either of the pairs is provided both must be or throw
                    if any(k in ['min','max'] for k in e.keys()):
                        new_enc['min'][idx] = float(e["min"])
                        new_enc['max'][idx] = float(e["max"])
                    if any(k in ['scale','offset'] for k in e.keys()):
                        new_enc['scale'][idx] = float(e["scale"])
                        new_enc['offset'][idx] = int(-abs(e['offset']))

                    # User quantization overrides may specify only scale/offset/bitwidth and then min/max can be calculated
                    symmetric_max = (2 ** (int(new_enc['bw']) - 1))

                    if all(key not in e for key in ['min', 'max']) \
                            and all(key in e for key in ['scale']):
                        if new_enc['is_symmetric']:
                            new_enc['min'][idx] = (-symmetric_max + 1) * e['scale']
                            new_enc['max'][idx] = (symmetric_max - 1) * e['scale']
                        else:
                            new_enc['min'][idx] = new_enc['offset'][idx] * e['scale']
                            new_enc['max'][idx] = (((2 ** e['bitwidth']) - 1) + new_enc['offset'][idx]) * e['scale']

                    # Symmetric weights should have 0 offset overridden with -symmetric_max, or already be equal to -symmetric_max or -symmetric_max+1
                    if new_enc['is_symmetric']:
                        if new_enc['offset'][idx] == 0:
                            new_enc['offset'][idx] = -symmetric_max
                        else:
                            if new_enc['offset'][idx] != -symmetric_max and new_enc['offset'][idx] != -symmetric_max+1:
                                raise ValueError("Invalid offset overridden for symmetric encodings got {}, expected 0 or {} or {}."
                                                 .format(new_enc['offset'], -symmetric_max, -symmetric_max+1))
                except Exception as exc:
                    log_error("Error: {} in tensor {} encoding {}. Min/max or scale/offset pairs must be present together.".format(str(exc), name, enc))
                    raise exc
            # Force the axis to default of 3 for axis quant
            if len(enc) > 1:
                new_enc['enc_type'] = "PER_CHANNEL"
                new_enc['axis'] = 3

            new_enc['enc_type'] = IROptimizations.get_tensor_type(new_enc['enc_type'])

        return new_enc

    def populate_quantization_params(self, ir_graph):

        def _adjust_bias_encoding(ir_graph):
            # The bias encoding in ir_graph.quantization_params corresponds to BiasAdd node as weights, we need to alter the name
            # 'weights' with 'bias' and add it to the params_encodings of the conv, deconv, matmul or fc node prior to the BiasAdd
            # so that the quantizer can get the bias encoding properly.
            for node in ir_graph.list_nodes():
                if node.op.hasattr('bias_op_name'):
                    _bias_op_name = node.op.bias_op_name

                    if _bias_op_name and _bias_op_name in ir_graph.quantization_params:
                        param_encodings = ir_graph.get_layer_quantization_param(_bias_op_name)[op_graph.QuantParams.PARAM_ENCODINGS]
                        if len(param_encodings) > 0:
                           _bias_encoding = param_encodings[0]
                           _bias_encoding['name'] = 'bias' # alter name 'weights' with 'bias'
                           ir_graph.add_quantization_params(node.op.name, param_encodings=_bias_encoding)

        q = ir_graph.user_quantization_overrides
        acts = q['activation_encodings']
        params = q['param_encodings']
        encoding_count = 0
        version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"

        # Graph inputs are special cases because they aren't owned by a node until IR conversion
        inputs = ir_graph.get_input_nodes_to_graph()
        for i in inputs:
            n = i.op.name
            if n in acts:
                encoding_count += 1
                ir_graph.add_quantization_params(n, output_encodings=[IROptimizations.extract_encoding_dict(n, acts[n], version)])

        # Walk through the original source framework op->input mapping to find the weights
        for op_name, op in ir_graph.src_graph_op_info.items():
            param_encs = []

            inputs = op['inputs']
            node = None
            if op_name in ir_graph.nodes_by_name:
                node = ir_graph.nodes_by_name[op_name]
            if inputs:
                for idx, i in enumerate(inputs):
                    if i in params:
                        encoding_count += 1
                        # If this encoding name is bias op name, the name should be set be "bias"
                        if node is not None and node.op.hasattr('bias_op_name') and node.op.bias_op_name == i:
                            param_encs.append(IROptimizations.extract_encoding_dict('bias', params[i], version))
                        else:
                            param_encs.append(IROptimizations.extract_encoding_dict('weights', params[i], version))
                # only add quantization params if param_encs is not empty
                if param_encs:
                    ir_graph.add_quantization_params(op_name, param_encodings=param_encs)

        # adjust the bias encoding for 'fully_connected', 'convolution', 'TransposeConv2d' ops.
        _adjust_bias_encoding(ir_graph)

        # Walk through the activations and lookup in the IR graph since folding, squashing, pruning
        # may have moved the activation names to new ops.
        for act in acts:
            act_encs = []
            if ir_graph.has_buffer(act):
                op = ir_graph.get_producer_op(act)
                encoding_count += 1
                act_encs.append(IROptimizations.extract_encoding_dict(act, acts[act], version))
                ir_graph.add_quantization_params(op.name, output_encodings=act_encs)

        log_info('Processed '+ str(encoding_count)+' quantization encodings')

    def populate_custom_io_quantization_params(self, ir_graph):
        """
        Populates the quantization_params of the ir_graph with the scale and offset provided in the custom IO YAML file.

        :param graph: an IROpgraph object
        """
        def custom_io_to_quant_enc(entry):
            # Populates the 'enc' dictionary with the data from the custom IO YAML file.
            # The format of 'enc' dictionary is similar to the one generated from quantization_overrides json file.
            datatype_to_bw = {
                'float32':32,
                'float16':16,
                'int32':32,
                'uint32':32,
                'int8':8,
                'uint8':8,
                'int16':16,
                'uint16':16
            }
            datatype_to_range = {
                'int8':(-128,127),
                'uint8':(0,255),
                'int16':(-32768,32767),
                'uint16':(0,65535)
            }
            scale = entry['QuantParam']['Scale']
            offset = entry['QuantParam']['Offset']
            # Default datatype for quantized inputs is uint8 in case of custom IO.
            # If 'QuantParam' are provided and no 'Datatype' is provided, it is assumed to be uint8.
            custom_datatype = 'uint8'
            if 'Datatype' in entry:
                custom_datatype = entry['Datatype']
            minVal = scale*(datatype_to_range[custom_datatype][0] + offset) if custom_datatype in datatype_to_range else 0.0
            maxVal = scale*(datatype_to_range[custom_datatype][1] + offset) if custom_datatype in datatype_to_range else 0.0
            isSymmetricType = 'True' if custom_datatype in ['int8','int16'] else 'False'
            enc = {
                'bitwidth':datatype_to_bw[entry['Datatype']],
                'scale':entry['QuantParam']['Scale'],
                'offset':entry['QuantParam']['Offset'],
                'min':minVal,
                'max':maxVal,
                'is_symmetric':isSymmetricType
            }
            return [enc]

        for entry in ir_graph.user_custom_io:
            if "QuantParam" in entry:
                buffer_name = str(entry['IOName'])
                enc = custom_io_to_quant_enc(entry)
                isInput = False

                # Graph inputs are special cases because they aren't owned by a node until IR conversion
                inputs = ir_graph.get_input_nodes_to_graph()
                for i in inputs:
                    n = i.op.name
                    if n == buffer_name:
                        ir_graph.add_quantization_params(n, output_encodings=[IROptimizations.extract_encoding_dict(n, enc)])
                        isInput = True

                # Walk through the activations and lookup in the IR graph since folding, squashing, pruning
                # may have moved the activation names to new ops.
                if not isInput and ir_graph.has_buffer(buffer_name):
                    op = ir_graph.get_producer_op(buffer_name)
                    ir_graph.add_quantization_params(op.name, output_encodings=[IROptimizations.extract_encoding_dict(buffer_name, enc)])

    # A method to get the layout of the inputs and outputs in the graph.
    def get_original_io_layouts(self, graph):
        original_io_layouts = {}
        for node in graph.get_input_nodes_to_graph() + graph.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                original_io_layouts[buffer_name] = graph.buffers[buffer_name].axis_format
        return original_io_layouts

    # A common method to modify the IO layout for the --preserve_io and --custom_io option.
    def modify_io_layout(self, graph, buffer_name, initial_axis_format, final_axis_format, isInput = True):
        if graph.buffers[buffer_name].rank() <= 2:
            return
        if initial_axis_format == final_axis_format:
            return
        permute_order, reverse_order = self.get_io_permute_order(initial_axis_format, final_axis_format)
        log_assert((permute_order is not None and reverse_order is not None),"Invalid layout tranformation for buffer {}."
            .format(buffer_name))
        if isInput:
            # For modifying the layout of input tensors, modify the shape of the input node
            # and insert a permute op to get the data in the layout that the graph expects
            node = graph.buffers[buffer_name].producer
            new_shape = node.op.shape.permute(reverse_order)
            buf = graph.get_buffer(buffer_name)
            buf.shape = new_shape
            buf.axis_format = final_axis_format
            node.op.shape = buf.shape
            # Int64 inputs present a special case while preserving both layout and datatype. In this case, the following
            # sequence of Input(int64) -> Transpose -> Cast(to int32) does not work while quantizing the network as
            # CPU backend does not allow int64 inputs for Transpose Op. Hence, the following sequence is created instead:
            # Input(int64) -> Cast (to int32) -> Transpose
            if (buffer_name in graph.preserve_datatype_tensors and graph.preserve_datatype_tensors[buffer_name] == 'int64') or\
                (graph.keep_int64_inputs and node.op.input_dtype == np.dtype('int64')):
                for consumer in graph.buffers[buffer_name].consumers:
                    if consumer.op.name == buffer_name + '_cast_int32':
                        for output_buffer in consumer.output_names:
                            buffer = graph.buffers[output_buffer]
                            buffer.shape = new_shape
                            buffer.axis_format = final_axis_format
                            consumers = [str(name) for name in graph.buffers[buffer.name].consumers]
                            graph.inject_implicit_permute(buffer.name, initial_axis_format, permute_order, consumers)
            else:
                consumers = [str(name) for name in graph.buffers[buffer_name].consumers]
                graph.inject_implicit_permute(buffer_name, initial_axis_format, permute_order, consumers)
        else:
            # It is possible that output buffer has multiple consumers. The consumers could be present in
            # the original model or introduced by any optimization like axes_to_spatial transformation. If
            # the consumer is a Transpose Op and has the axis_format same as the axis_format to be preserved,
            # then we make sure that the output buffer has the correct name.
            for node in graph.buffers[buffer_name].consumers:
                if node.op.type == op_adapter.TransposeOp.TRANSLATION_KEY and \
                graph.buffers[node.output_names[0]].axis_format == final_axis_format:
                    new_intermediate_buffer_name = graph.get_implicit_permute_node_name(buffer_name, initial_axis_format)
                    graph.change_buffer_name(buffer_name, new_intermediate_buffer_name)
                    graph.change_buffer_name(node.output_names[0], buffer_name)
                    return

            # For modifying the layout of output tensors, inject a permute op after the output and
            # modify the buffer names and connections appropriately so that the output names
            # are same as in the original graph.
            new_output_node = graph.inject_implicit_permute(buffer_name, final_axis_format, reverse_order, [])
            new_buffer_name = graph.get_implicit_permute_node_name(buffer_name, initial_axis_format)
            new_output_buffer = graph.get_buffer(new_output_node.output_names[0])
            del graph.buffers[new_output_node.output_names[0]]
            graph.naming_policy.remove_output_name(new_output_node.output_names[0])
            old_output_buffer = graph.get_buffer(buffer_name)
            original_output_node = old_output_buffer.producer

            # Update the name of the buffer (which was originally the output but now is consumed by the permute node)
            old_output_buffer.name = new_buffer_name
            # Map the buffer to the correct name in the graph.buffers dictionary
            graph.buffers[new_buffer_name] = old_output_buffer
            # Change the name of the new output buffer to the original output buffer name
            new_output_buffer.name = buffer_name
            # Map the buffer to the correct name in the graph.buffers dictionary
            graph.buffers[buffer_name] = new_output_buffer
            # Make appropriate changes in the connections between nodes.
            # Update the consumer nodes.
            for consumer in old_output_buffer.consumers:
                if consumer.op.name == new_output_node.op.name:
                    continue
                in_idx = consumer.input_names.index(buffer_name)
                consumer.input_names[in_idx] = new_buffer_name

            # Update the producer nodes.
            in_idx = new_output_node.input_names.index(buffer_name)
            new_output_node.input_names[in_idx] = new_buffer_name
            new_output_node.output_names[0] = buffer_name
            out_idx = original_output_node.output_names.index(buffer_name)
            original_output_node.output_names[out_idx] = new_buffer_name

            # Update the new buffer name in quantization_params dictionary for the original output node
            if original_output_node.op.name in graph.quantization_params.keys():
                for encoding_dict in graph.quantization_params[original_output_node.op.name]['output_encodings']:
                    if encoding_dict['name'] == buffer_name:
                        encoding_dict['name'] = new_buffer_name

    def populate_preserve_io_layout_tensors(self, graph):
        if graph.preserve_io_layout_passed == 1:
            for arg in graph.preserve_io:
                if arg[0] == 'layout':
                    for buffer_name in arg[1:]:
                        graph.preserve_layout_tensors.add(buffer_name)
        elif graph.preserve_io_layout_passed == 2:
            # If the user has passed just the --preserve_io without listing tensor names then preserve the
            # layout for all IO tensors except those tensors whose layout is set using the --custom_io option
            for node in graph.get_input_nodes_to_graph():
                for buffer_name in node.output_names:
                    graph.preserve_layout_tensors.add(buffer_name)

            for node in graph.get_output_nodes_of_graph():
                for buffer_name in node.output_names:
                    if buffer_name in graph.output_names:
                        # Add to the preserve_io_layout_tensors only if the buffer_name is the graph output.
                        graph.preserve_layout_tensors.add(buffer_name)

        # Skipping those IO tensors whose layout is set using the --custom_io option
        graph.preserve_layout_tensors = graph.preserve_layout_tensors - set(graph.custom_io_axis_formats)

    def preserve_io_layout(self, graph, original_io_layouts):
        input_output_buffer_names = []
        for node in graph.get_input_nodes_to_graph():
            for buffer_name in node.output_names:
                input_output_buffer_names.append(buffer_name)
                if buffer_name in graph.preserve_layout_tensors:
                    current_axis_format = graph.buffers[buffer_name].axis_format
                    pre_optimization_io_layout = original_io_layouts[buffer_name]
                    if (current_axis_format == AxisTracker.AxisFormat.NONTRIVIAL or
                        pre_optimization_io_layout == AxisTracker.AxisFormat.NONTRIVIAL):
                        # Skip preserving layout for tensors whose layouts are NONTRIVIAL
                        # NOTE: If the Op at input enforces an axis format (eg: convolution op) and the user passes
                        # NONTRIVIAL for such an input, then it's layout cannot be preserved using the preserve_io option.
                        continue
                    self.modify_io_layout(graph, buffer_name, current_axis_format, pre_optimization_io_layout, isInput = True)
        for node in graph.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                input_output_buffer_names.append(buffer_name)
                if buffer_name in graph.preserve_layout_tensors:
                    current_axis_format = graph.buffers[buffer_name].axis_format
                    pre_optimization_io_layout = original_io_layouts[buffer_name]
                    if (current_axis_format == AxisTracker.AxisFormat.NONTRIVIAL or
                        pre_optimization_io_layout == AxisTracker.AxisFormat.NONTRIVIAL):
                        # Skip preserving layout for tensors whose layouts are NONTRIVIAL
                        continue
                    self.modify_io_layout(graph, buffer_name, current_axis_format, pre_optimization_io_layout, isInput = False)

        for buffer_name in graph.preserve_layout_tensors:
            if buffer_name not in input_output_buffer_names:
                log_warning("Provided tensor '{}' is not present in the graph for preserving the layout.", buffer_name)

    def get_io_permute_order(self, initial_axis_format, custom_axis_format):
        permute_order = None
        reverse_order = None
        if custom_axis_format == AxisTracker.AxisFormat.NCDHW and initial_axis_format== AxisTracker.AxisFormat.NDHWC :
            permute_order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            reverse_order = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
        elif custom_axis_format == AxisTracker.AxisFormat.NDHWC and initial_axis_format== AxisTracker.AxisFormat.NCDHW :
            permute_order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            reverse_order = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
        elif custom_axis_format == AxisTracker.AxisFormat.NCS and initial_axis_format== AxisTracker.AxisFormat.NSC :
            permute_order = AxisTracker.AxisFormat.NCS_TO_NSC
            reverse_order = AxisTracker.AxisFormat.NSC_TO_NCS
        elif custom_axis_format == AxisTracker.AxisFormat.NSC and initial_axis_format== AxisTracker.AxisFormat.NCS :
            permute_order = AxisTracker.AxisFormat.NSC_TO_NCS
            reverse_order = AxisTracker.AxisFormat.NCS_TO_NSC
        elif custom_axis_format == AxisTracker.AxisFormat.NCF and initial_axis_format== AxisTracker.AxisFormat.NFC :
            permute_order = AxisTracker.AxisFormat.NCF_TO_NFC
            reverse_order = AxisTracker.AxisFormat.NFC_TO_NCF
        elif custom_axis_format == AxisTracker.AxisFormat.NFC and initial_axis_format== AxisTracker.AxisFormat.NCF :
            permute_order = AxisTracker.AxisFormat.NFC_TO_NCF
            reverse_order = AxisTracker.AxisFormat.NCF_TO_NFC
        elif custom_axis_format == AxisTracker.AxisFormat.TNF and initial_axis_format== AxisTracker.AxisFormat.NTF :
            permute_order = AxisTracker.AxisFormat.TNF_TO_NTF
            reverse_order = AxisTracker.AxisFormat.NTF_TO_TNF
        elif custom_axis_format == AxisTracker.AxisFormat.NTF and initial_axis_format== AxisTracker.AxisFormat.TNF :
            permute_order = AxisTracker.AxisFormat.NTF_TO_TNF
            reverse_order = AxisTracker.AxisFormat.TNF_TO_NTF
        elif custom_axis_format == AxisTracker.AxisFormat.HWIO and initial_axis_format== AxisTracker.AxisFormat.OIHW :
            permute_order = AxisTracker.AxisFormat.HWIO_TO_OIHW
            reverse_order = AxisTracker.AxisFormat.OIHW_TO_HWIO
        elif custom_axis_format == AxisTracker.AxisFormat.OIHW and initial_axis_format== AxisTracker.AxisFormat.HWIO :
            permute_order = AxisTracker.AxisFormat.OIHW_TO_HWIO
            reverse_order = AxisTracker.AxisFormat.HWIO_TO_OIHW
        return permute_order, reverse_order

    def apply_custom_io_change(self, graph):
        for entry in graph.user_custom_io:
            buffer_name = str(entry['IOName'])
            log_assert(buffer_name in graph.buffers,"Incorrect IOName provided in custom IO YAML file. Buffer {} not found in graph"
                .format(buffer_name))
            isInput = False
            # Check if the buffer name provided is input buffer
            for node in graph.get_input_nodes_to_graph():
                if buffer_name in node.output_names:
                    isInput = True
                    break
            # Check if the buffer name provided is output buffer
            isOutput = False
            for node in graph.get_output_nodes_of_graph():
                if buffer_name in node.output_names:
                    isOutput = True
                    break
            log_assert((isInput or isOutput),"Custom IOName {} is neither an input nor an output.".format(buffer_name))

            if "Layout" in entry:
                entry['Layout']['Custom'] = InputLayout.get_axis_format(entry['Layout']['Custom'])
                initial_axis_format = graph.buffers[buffer_name].axis_format
                # if input buffer axis format is NONTRIVIAL, it can be overwritten by the valid axis from user.
                if initial_axis_format == AxisTracker.AxisFormat.NONTRIVIAL and entry['Layout']['Model'] in AxisTracker.AxisFormat.get_valid_formats():
                    initial_axis_format = entry['Layout']['Model']
                custom_axis_format = entry['Layout']['Custom']
                if initial_axis_format != custom_axis_format:
                    if isInput:
                        self.modify_io_layout(graph, buffer_name, initial_axis_format, custom_axis_format, isInput = True)

                    if isOutput:
                        self.modify_io_layout(graph, buffer_name, initial_axis_format, custom_axis_format, isInput = False)

    def validate_custom_io_config(self, graph):
        buffers = set()
        for entry in graph.user_custom_io:
            for field in entry.keys():
                if field not in ['IOName', 'Layout', 'Datatype', 'QuantParam']:
                    log_error("Incorrect field %s provided in the custom IO YAML file. Valid fields are: 'IOName', 'Layout', 'Datatype', 'QuantParam'",
                        field)
            log_assert('IOName' in entry,"No IOName provided in custom IO YAML file. IOName is a mandatory field")
            buffer_name = entry['IOName']
            log_assert(buffer_name not in buffers,"Multiple entries provided for buffer {} in custom IO YAML file".format(buffer_name))
            if 'Layout' in entry:
                log_assert(('Custom' in entry['Layout'] and 'Model' in entry['Layout']),
                    "Both Custom layout and Model layout should be provided in the custom IO YAML file.")
            custom_datatype = None
            if 'Datatype' in entry:
                custom_datatype = entry['Datatype']
                log_assert((custom_datatype in ['float32','float16','uint8','uint16','uint32',
                                                'bool','int8','int16','int32','int64']),
                           "Custom datatype {} of buffer {} is not supported.".format(entry['Datatype'], buffer_name))
            if 'QuantParam' in entry:
                quant_param = entry['QuantParam']
                for sub_field in quant_param.keys():
                    if sub_field not in ['Type', 'Scale', 'Offset']:
                        log_error("Incorrect field %s provided in QuantParam of the custom IO YAML file for buffer %s. Valid fields are: 'Type', 'Scale', 'Offset'",
                            field, buffer_name)
                log_assert('Type' in quant_param,"No Type provided for QuantParam in custom IO YAML file for buffer {}."
                    .format(buffer_name))
                log_assert(quant_param['Type'] == 'QNN_DEFINITION_DEFINED',"Type must be set to 'QNN_DEFINITION_DEFINED' if user wants to provide scale and offset.\
                    Invalid Type provided for buffer {}".format(buffer_name))
                log_assert('Scale' in quant_param and 'Offset' in quant_param,"Scale and/or Offset not provided for buffer {} in the custom IO YAML file."
                    .format(buffer_name))
                log_assert(isinstance(quant_param['Scale'], float) or isinstance(quant_param['Scale'], int),"Scale provided for buffer {} in the custom IO YAML file\
                    is of incorrect datatype. It should be a number.".format(buffer_name))
                log_assert(isinstance(quant_param['Offset'], int), "Offset provided for buffer {} in the custom IO YAML file\
                    is of incorrect datatype. It should be an integer.".format(buffer_name))
                if custom_datatype is not None:
                    log_assert(custom_datatype in ['uint8', 'int8','uint16','int16'], "Valid datatypes for quantized input/output are int8, uint8, int16 and uint16\
                                for custom IO. Invalid datatype {} provided for buffer {}.".format(custom_datatype, buffer_name))
            buffers.add(buffer_name)


class LayoutTransformContext():
    """Context manager for Layout Transform.

    Layout Transform is designed to be independent and decoupled with any optimization in order to
    keep its infrastructure clean and neat. However, such separation makes Layout Transform unable
    to painlessly merge into current optimization flow which is highly coupled to Axis Tracking
    mechanism. Therefore, a few optimizations must be performed to replace Axis Tracking with the
    new Layout Transform algorithm, and this context aims to wrap up all mandatory optimizations
    while constraining the effect within Layout Transform.

    This context is designed to serve as a context manager and used through Python `with`
    statement (refer to the example below). Those necessary optimizations for pre- and post-Layout
    Transform are invoked during entering and exiting this context, respectively. Refer to methods
    __enter__ and __exit__ for currently performed optimizations.

    Usage:

        >>> layout_manager = PyIrGraphLayoutManager(graph)
        >>> with LayoutTransformContext(layout_manager):
        ...     OptimizationTranslations.apply_method_to_all_ops(
        ...         TRANSFORM_LAYOUT, graph, layout_manager
        ...    )

    Attributes:
        meta: A dict recording necessary information to pass across this context.
        new_graph: An IROpGraph instance where optimizations perform post-Layout Transform.
        src_graph: An IROpGraph instance where optimizations perform pre-Layout Transform.

    Methods:
        __enter__: Perform pre-Layout Transform optimizations. This method is expected to be used
            through Python `with` statement.
        __exit__: Perform post-Layout Transform optimizations. This method is expected to be used
            through Python `with` statement.
    """

    def __init__(self, layout_manager):
        """Initialize context."""
        self._src_graph = layout_manager.layout_mutator.src_graph
        self._new_graph = layout_manager.layout_mutator.new_graph

        self._meta = {}

    def __enter__(self):
        """Perform pre-Layout Transform optimizations."""
        self._apply_to_op_type(
            self._src_graph, op_adapter.CustomOp, self._replace_customop_infer_shape
        )

        self._apply_to_op_type(self._src_graph, op_adapter.IdentityOp, self._remove_identity)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Perform post-Layout Transform optimizations."""
        if exc_type is not None:
            return False

        # TODO
        # Transpose Hoisting is currently achieved by indiscriminately inserting NCS to NSC
        # Transpose right after Input. Revisit the algorithm once needed in the future.
        # self._hoist_transpose(self._new_graph)

        # Twist back to HTP favorable patterns.
        self._optimize_width_untilable_prelu(self._new_graph)
        self._optimize_depth_broadcast_elementwise(self._new_graph)
        self._optimize_unmatched_matmul(self._new_graph)
        self._optimize_non_inplace_concat(self._new_graph)

        # Squash Constant.
        # Note that squashing for Reshape must be executed first for elementwise-broadcast cases.
        self._squash_constant_reshape(self._new_graph)
        self._apply_to_op_type(
            self._new_graph, op_adapter.TransposeOp, self._squash_constant_transpose
        )

        # Optimize inserted Transpose.
        self._apply_to_op_type(
            self._new_graph, op_adapter.TransposeOp, self._replace_trt_with_reshape
        )
        # TODO
        # Ideally, replacing Transpose to Reshape can trigger potential optimization of folding
        # with Reshape around. However, current experimental results suggest otherwise due to
        # specific optimizations in HTP backend. Therefore, this replacement is disabled for now
        # and may be revisited in the future.
        # self._apply_to_op_type(
        #     self._new_graph, op_adapter.TransposeOp, self._replace_transpose_with_reshape
        # )
        self._fold_reshapes(self._new_graph)

        self._apply_to_op_type(
            self._new_graph, op_adapter.CustomOp, self._recover_customop_infer_shape
        )

        return True

    def _apply_to_op_type(self, graph, op_type, opt_func):
        """Apply optimization on given op type."""
        for node in graph.list_nodes():
            # The dynamic check is mandatory since optimization may remove nodes from graph.
            if node in graph.list_nodes() and isinstance(node.op, op_type):
                opt_func(node, graph)

    ################################################################################################
    # Optimization Functions.
    ################################################################################################

    def _fold_reshapes(self, graph):
        """Fold consecutive Reshape.

        Generally, Reshape is folded before Layout Transform. However, additional Reshape may be
        introduced by this context, and thus this optimization must be executed again.
        """
        OptimizeReshapeTranslation.fold_reshapes(graph)

    def _recover_customop_infer_shape(self, node, _):
        """Recover CustomOp's `infer_shape` method.

        Refer to `replace_customop_infer_shape` for the intuition.
        """
        converter_utils.log_assert(
            node.op.name in self._meta.get('customop_infer_shape', {}),
            f'Fail to recover `infer_shape` method for CustomOp {node.op.name}.'
        )
        node.op.infer_shape = self._meta['customop_infer_shape'].pop(node.op.name)
        if not self._meta['customop_infer_shape']:
            self._meta.pop('customop_infer_shape')

    def _remove_identity(self, node, graph):
        """Remove Identity.

        Consider a pattern illustrated below,

                       |-- Identity -- Conv2d
            Constant --|
                       |-- Identity -- Conv2d

        , where Transpose will be inserted for each Identity, respectively, by Layout Transform.

                       |-- Identity -- Transpose -- Conv2d
            Constant --|
                       |-- Identity -- Transpose -- Conv2d

        Even though these Transpose are identical, they are unsquashable as Constant has multiple
        consumers. Therefore, Identity must be removed before Layout Transform to allow inserted
        Transpose shared across different consumers.

        Note that general REMOVE_IDENTITY optimization should be executed but requires further
        investigation to validate the correctness.
        """
        OptimizeIdentityTranslation.remove_identity(node, graph)

    def _replace_customop_infer_shape(self, node, graph):
        """Replace CustomOp's `infer_shape` method.

        By current definition of CustomOp, its output shapes could not be inferred more than once
        (refer to C++ CustomOp::inferOutputShapes). However, the method `CustomOp.infer_shape`
        would be inevitably invoked second time during adding to the new graph, where the first
        time happens in constructing the source graph. In order to bypass the shape inferrence
        while not impacting others, the method `CustomOp.infer_shape` is saved in meta and replaced
        by a dummy function to directly return original output shapes. The saved method will be
        recovered while exiting the context (refer to `recover_customop_infer_shape`.).
        """
        if 'customop_infer_shape' not in self._meta:
            self._meta['customop_infer_shape'] = {}

        # Save `CustomOp.infer_shape` in meta.
        self._meta['customop_infer_shape'][node.op.name] = node.op.infer_shape

        # Replace with dummy function.
        output_shapes = [graph.get_buffer(name).shape for name in node.output_names]
        node.op.infer_shape = lambda *_: output_shapes

    def _replace_transpose_with_reshape(self, node, graph):
        """Replace Transpose with Reshape."""
        # Note that below is HTP-specific condition to avoid replacing Transpose with Reshape at
        # graph input or output.
        if (
            isinstance(graph.get_producer_op(node.input_names[0]), op_adapter.InputOp)
            or graph.is_output_node(node)
        ):
            return

        input_shape = graph.get_buffer(node.input_names[0]).shape
        output_shape = graph.get_buffer(node.output_names[0]).shape
        if input_shape.is_dynamic():
            return

        fake_input = np.random.randint(256, size=input_shape, dtype=np.uint8)
        transpose_output = np.transpose(fake_input, node.op.perm)
        reshape_output = np.reshape(fake_input, output_shape)
        if np.all(transpose_output == reshape_output):
            graph.replace(node.op, op_adapter.ReshapeOp(node.op.name, shape=output_shape))

    def _replace_trt_with_reshape(self, node, graph):
        """Replace Transpose-Reshape-Transpose pattern with Reshape."""
        # Match Transpose-Reshape-Transpose pattern.
        buf = graph.get_buffer(node.output_names[0])
        if len(buf.consumers) != 1:
            return

        reshape_node = list(buf.consumers)[0]
        if reshape_node.op.type != ir_graph.QNN_OP_RESHAPE:
            return

        buf = graph.get_buffer(reshape_node.output_names[0])
        if len(buf.consumers) != 1:
            return

        transpose_node = list(buf.consumers)[0]
        if transpose_node.op.type != ir_graph.QNN_OP_TRANSPOSE:
            return

        # Check whether functionally identical.
        input_shape = graph.get_buffer(node.input_names[0]).shape
        output_shape = graph.get_buffer(transpose_node.output_names[0]).shape
        if input_shape.is_dynamic():
            return

        fake_input = np.random.randint(256, size=input_shape, dtype=np.uint8)

        expected_output = np.transpose(fake_input, node.op.perm)
        # Since Reshape may have shape attribute continaing 0 which indicates to directly use dim
        # from input shape, adopt output shape from buffer instead to avoid numpy error.
        expected_output = np.reshape(expected_output, graph.get_output_shapes(reshape_node)[0])
        expected_output = np.transpose(expected_output, transpose_node.op.perm)

        output = np.reshape(fake_input, output_shape)
        if not np.all(output == expected_output):
            return

        # Replace TRT with single Reshape.
        graph.squash(transpose_node, transpose_node.input_names[0], is_data_movement_node=True)
        graph.squash(reshape_node, reshape_node.input_names[0], is_data_movement_node=True)

        reshape_op = op_adapter.ReshapeOp(reshape_node.op.name, shape=output_shape)
        graph.replace(node.op, reshape_op)

    def _squash_constant_reshape(self, graph):
        """Squash Reshape into Constant.

        Layout Transform algorithm may insert Reshape to perform explicit broadcast for elementwise
        ops along with Transpose to handle implictly broadcastable cases. Since the broadcasted
        input of elementwise could be Constant, both insertd Reshape and Transpose are expected to
        be squashed. Thus, this optimization aims to squash Reshape first where Transpose is later
        squashed in `squash_constant_transpose`.
        """
        OptimizeReshapeTranslation.squash_reshape(graph)

    def _squash_constant_transpose(self, node, graph):
        """Squash Transpose into Constant.

        Comparing to Axis Tracking directly transposing Constant's tensor, Layout Transform
        algorithm inserts explicit Transpose and expects SQUASH_CONSTANT_INPUT optimization to
        squash Transpose into Constant's tensor later. However, inserted Transpose may break the
        pattern matching if the squash happens after the matching. Therefore, Layout Transform must
        squash those inserted Transpose right after the algorithm.

        Additionally, since Axis Tracking inplace transposes Constant's tensor, the output buffer
        name of Constant remains the same. However, it is not the case with Layout Transform as
        graph squashing modifies the output buffer name of Constant to the one of Transpose. This
        behavior is expected to be safe within the Converter process but it causes error for the
        LoRA usage. LoRA usage consumes a LoRA weight list which specifies the tensor names to be
        marked updatable, which works with Axis Tracking of inplace transpose but not Layout
        Transform if leveraging graph squashing.

        This optimization aims to separately handle squashing Transpose based on consumer types.
        For Conv2d, FullyConnected, etc, Transpose will be squashed into next to avoid modifying
        the output buffer name of Constant. Otherwise, original SQUASH_CONSTANT_INPUT is invoked.
        """
        input_buffer = graph.get_buffer(node.input_names[0])
        if (
            not isinstance(input_buffer.producer.op, op_adapter.ConstantOp)
            or len(input_buffer.consumers) > 1
        ):
            return

        parameterized_ops = (
            op_adapter.Conv2dOp,
            op_adapter.Conv3dOp,
            op_adapter.DepthwiseConv2dOp,
            op_adapter.FullyConnectedOp,
            op_adapter.TransposeConv2dOp,
            op_adapter.TransposeConv3dOp
        )

        output_buffer = graph.get_buffer(node.output_names[0])
        if not any(
            isinstance(consumer.op, parameterized_ops) for consumer in output_buffer.consumers
        ):
            OptimizeTransposeTranslation.squash_constant_input(node, graph)
            return

        tensor = np.ascontiguousarray(np.transpose(input_buffer.producer.op.tensor, node.op.perm))
        input_buffer.producer.op.tensor = tensor
        input_buffer.shape = list(tensor.shape)
        graph.squash(node, input_name=input_buffer.name, squash_into_next=True)

    ################################################################################################
    # Transpose Hoisting.
    ################################################################################################

    def _hoist_transpose(self, graph):
        """Hoist Transpose.

        This optimization aims to perform Transpose hoisting, transforming graph into HTP favorable
        one which implicitly assumes NHWC layout. Instead of attempting to hoist each Transpose
        in an op-by-op manner, this algorithm first groups applicable Transpose with preceding ops
        as far as it could conceptually be hoisted. Each group has one Transpose topologically at
        the end, which can be hoisted to be topologically before all the other ops in the group.
        Therefore, there are some conditions to verify the capability of grouping to ensure the
        target Transpose can be seemlessly hoisted to the front. The grouped ops are then
        transformed to target format (according to Transpose's permutation) after grouping.
        The hoisting algorithm consists of following steps:

            1. Find the groups to perform hoisting on.
            2. Modify the graph within each discovered group.
                2-1. Insert Transpose at the front.
                2-2. Transform layout agnostic ops (e.g., attribute update).
                2-3. Remove original Transpose at the end.

        The grouping in the algorithm is implemented through disjoint set data structure (or
        so-called union-find) with few modifications. Unlike normal disjoint set which does not
        care about the order, the data structure in this algorithm is also used to record the
        visited status and traversal order.

        Note that current implementation restricts the algorithm operating on groups starting from
        Input since HTP shape preference is extremely complicated and accurately adopting those HTP
        rules for intermediate pattern is nearly impossible. Therefore, hoisting is only performed
        on ops between Input to the first encountered NCHW-to-NHWC Transpose. Nevertheless, the
        implementation can be easily extended and covered other groups once HTP preference is
        determined.
        """
        entry_ops = (op_adapter.InputOp, op_adapter.ConstantOp)
        queue = collections.deque(
            node for node in graph.list_nodes() if isinstance(node.op, entry_ops)
        )
        groups = dict(zip(queue, queue))

        def find(node):
            """Find parent of target node."""
            if groups[node] is not node:
                groups[node] = find(groups[node])
            return groups[node]

        def union(node1, node2):
            """Union two nodes into the same group."""
            node1, node2 = find(node1), find(node2)
            if node1 is node2:
                return
            if isinstance(node1.op, entry_ops) or not isinstance(node2.op, entry_ops):
                groups[node2] = node1
            else:
                groups[node1] = node2

        def enroll_consumers(node):
            """Enroll node consumers to groups and queue."""
            for output_name in node.output_names:
                for consumer in graph.get_buffer(output_name).consumers:
                    # Ensure topological order.
                    if any(
                        graph.get_buffer(input_name).producer not in groups
                        for input_name in consumer.input_names
                    ):
                        continue
                    # Avoid duplicate traversal.
                    if consumer not in groups:
                        groups[consumer] = consumer
                        queue.append(consumer)

                    union(node, consumer)

        def withdraw_group(node):
            """Withdraw enitre group that node belongs to."""
            root = find(node)
            for withdrawn_node in [other for other in groups if find(other) is root]:
                groups.pop(withdrawn_node)

        def is_htp_friendly_shape(shape):
            """Check whether given shape is HTP friendly.

            HTP friendly shape is intrinsically hard to explicitly defined. Here the shape is
            merely validated to be "image-like" shape.
            """
            return len(shape) == 4 and shape[1] < shape[2] and shape[1] < shape[3]

        def is_layout_agnostic(node):
            """Check whether given node's op is layout agnostic."""
            inferer = layout_inferer.LayoutInferers.get_layout_inferer(node.op.TRANSLATION_KEY)
            return isinstance(
                inferer, layout_inferer.agnostic_layout_inferers.AgnosticOpLayoutInfererBase
            )

        # There are two data structure used for traversal, including a queue and a disjoint set
        # (essentially a dict) for recording the traversal order and status, respectively. A node
        # is added into the queue and the dict if and only if all preceding nodes are traversed
        # by checking their presence in the dict. Since the traversal is integrated with modified
        # union-find algorithm, there are few things to be noted:
        #
        #     1. A node is possibly added into the queue but later marked invalid by popping from
        #        the dict, and therefore an additional checking is mandatory each time acquiring
        #        a node from the queue.
        #     2. A node is possibly never traversed due to its precedings violating hoisting
        #        constraints, and therefore an additional checking is mandatory for whether to
        #        perform actual hoisting after grouping.

        valid_groups = set()
        while queue:
            node = queue.popleft()

            # Stop traversal if it is withdrawn by others.
            if node not in groups:
                continue

            if isinstance(node.op, entry_ops):
                # Only Input paths starting with HTP friendly shapes are considered.
                if (
                    not isinstance(node.op, op_adapter.InputOp)
                    or is_htp_friendly_shape(graph.get_buffer(node.output_names[0]).shape)
                ):
                    enroll_consumers(node)
                else:
                    groups.pop(node)
                continue

            # Stop traversal and remove the entire paths from candidates if current op is not
            # layout agnostic.
            if not is_layout_agnostic(node):
                withdraw_group(node)
                continue

            # Stop traversal at Transpose.
            if isinstance(node.op, op_adapter.TransposeOp):
                # If Transpose does not implicitly indicate NCHW -> NHWC or is right after its root,
                # remove the entire paths from candidates.
                if node.op.perm != [0, 2, 3, 1]:
                    withdraw_group(node)
                else:
                    valid_groups.add(find(node))
                continue

            # Keep traversal for layout agnostic ops.
            enroll_consumers(node)

        # Grouped nodes are added based on traversal order.
        for node in groups:
            # Skip actual hoisting if no Transpose exists in corresponding group.
            if find(node) not in valid_groups:
                continue

            if isinstance(node.op, entry_ops):
                transpose_input_name = node.output_names[0]

                # Handle broadcast Constant cases.
                if isinstance(node.op, op_adapter.ConstantOp) and node.op.tensor.ndim != 4:
                    const_shape = list(node.op.tensor.shape)
                    if const_shape == [1]:
                        # Skip scalar Constant as already broadcastable.
                        continue

                    broadcast_shape = [1] * (4 - len(const_shape)) + const_shape
                    reshape_name = f'{node.output_names[0]}_hoist_broadcast'
                    reshape_op = op_adapter.ReshapeOp(reshape_name, shape=broadcast_shape)
                    graph.inject(reshape_op, node.output_names[0], reshape_name)

                    transpose_input_name = reshape_name

                # Insert [0,2,3,1] Transpose right after.
                name = f'{node.output_names[0]}_hoist_0231'
                transpose_op = op_adapter.TransposeOp(name, [0, 2, 3, 1])
                graph.inject(transpose_op, transpose_input_name, name)
            elif isinstance(node.op, op_adapter.TransposeOp):
                # Remove final Transpose since hoisted.
                graph.squash(
                    node, node.input_names[0], squash_into_next=True, is_data_movement_node=True
                )
            else:
                # Update attributes for layout agnostic op.
                inferer = layout_inferer.LayoutInferers.get_layout_inferer(node.op.TRANSLATION_KEY)
                attrs = inferer.update_attr_with_target_input_perm_seqs(
                    [(0, 2, 3, 1)] * len(node.input_names), node.op.list_params()
                )
                for attr_key, attr_val in attrs.items():
                    setattr(node.op, attr_key, attr_val)

                # Manually update shape since permutation is restricted to [0,2,3,1].
                for output_name in node.output_names:
                    buffer = graph.get_buffer(output_name)
                    buffer.shape = [
                        buffer.shape[0], buffer.shape[2], buffer.shape[3], buffer.shape[1]
                    ]

    ################################################################################################
    # HTP-specific Optimizations.
    ################################################################################################

    def _optimize_depth_broadcast_elementwise(self, graph):
        """Optimize broadcast-on-depth Elementwise.

        This pattern focuses on ElementwiseBinary with broadcastable input shapes. Layout Transform
        may decide to insert Transpose on the to-be-broadcasted branch and an additional Reshape
        to explicitly align ranks ahead of the inserted Tranpose. Therefore, the dimensions to be
        broadcasted are changed after inserting Reshape and Transpose. However, HTP will not
        perform tiling on broadcast-on-depth cases, where depth dimension refers to the last
        dimension or channel dimension in HTP. Since Layout Transform currently does not consider
        input shapes during determining the target input permute sequence for ElementwiseBinary and
        thus produces broadcast-on-depth patterns, such patterns consequently lead to performance
        drop as no tiling will be applied in HTP.

        This optimization aims to move the inserted Transpose from to-be-broadcasted branch to the
        other one if broadcast-on-depth no longer exists afterwards. It essentially selects the
        alternative permute sequence when infering layout for ElementwiseBinary, and therefore
        the permutation for new Transpose is determined through reversing the existing one. Note
        that the Reshape inserted by Layout Transform for explicit broadcast is as well removed
        since corresponding path is reversed back to source layout. Below provides an example for
        illustration:

            | [1,32,32,3]     |     [1,32,32]           |     [1,32,32,3] | [1,32,32]
            |              Reshape                  Transpose             |
            |                 |     [1,1,32,32]         |     [1,3,32,32] |
            |             Transpose                     |-----------------|
            |                 |     [1,32,32,1]                  |
            |-----------------|                          ElementwiseBinary
                     |                          ==>              |         [1,3,32,32]
             ElementwiseBinary                               Transpose
                     |         [1,32,32,3]                       |         [1,32,32,3]
        """
        # Limit affected pattern to below elementwise binary types.
        eltwise_binary_types = (
            ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD,
            ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE,
            ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
            ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT
        )

        def format_pattern(op_type):
            legacy_op_type = op_adapter.ElementwiseBinaryOp.operation_to_legacy[op_type]
            return [
                (
                    op_adapter.ReshapeOp.TRANSLATION_KEY,
                    (),
                    ('MATCH_BUFS_AT_INDEX', [(op_adapter.TransposeOp.TRANSLATION_KEY, 0)])
                ),
                (
                    op_adapter.TransposeOp.TRANSLATION_KEY,
                    ('MATCH_BUFS_AT_INDEX', [(op_adapter.ReshapeOp.TRANSLATION_KEY, 0)]),
                    ('MATCH_BUFS_AT_INDEX', [(legacy_op_type, 0)])
                ),
                (
                    legacy_op_type,
                    ('MATCH_BUFS_AT_INDEX', [(op_adapter.TransposeOp.TRANSLATION_KEY, 'ANY')]),
                    ()
                )
            ]

        def validate(matched_nodes):
            """Validate matched nodes with expecting shapes and attributes."""
            reshape_node, transpose_node, eltwise_node = matched_nodes

            eltwise_input_shapes = graph.get_input_shapes(eltwise_node)
            if eltwise_node.input_names[0] == transpose_node.output_names[0]:
                broadcast_shape, target_shape = eltwise_input_shapes
            else:
                target_shape, broadcast_shape = eltwise_input_shapes

            # Check whether Reshape is inserted by Layout Transform to perform explicit broadcast.
            # There are two conditions checked:
            #   1. Reshape is on the to-be-broadcasted branch, verified by comparing the ranks.
            #   2. Reshape has input/output shapes only differed in prepended 1s.
            reshape_input_shape = graph.get_buffer(reshape_node.input_names[0]).shape
            reshape_output_shape = graph.get_buffer(reshape_node.output_names[0]).shape
            rank_diff = len(reshape_output_shape) - len(reshape_input_shape)
            if (
                len(reshape_input_shape) >= len(target_shape)
                or reshape_output_shape != [*([1] * rank_diff), *reshape_input_shape]
            ):
                return False

            # Check whether Elementwise already broadcasts on the last dimension.
            if broadcast_shape[-1] == target_shape[-1]:
                return False

            # Check whether reversing Transpose could avoid broadcasting on the last dimension.
            rev_perm = np.argsort(transpose_node.op.perm).tolist()
            return broadcast_shape[rev_perm[-1]] == target_shape[rev_perm[-1]]

        for pattern in map(format_pattern, eltwise_binary_types):
            for matched_nodes in graph.get_matched_nodes_v2(pattern, validator=validate):
                reshape_node, transpose_node, eltwise_node = matched_nodes

                perm = transpose_node.op.perm
                inv_perm = np.argsort(perm).tolist()

                # Insert Transpose on another branch.
                target_input_name = (
                    eltwise_node.input_names[1]
                    if eltwise_node.input_names[0] == transpose_node.output_names[0]
                    else eltwise_node.input_names[0]
                )
                pre_transpose_op = op_adapter.TransposeOp(
                    f'{target_input_name}_{"".join(map(str, inv_perm))}', inv_perm
                )
                graph.inject(pre_transpose_op, target_input_name, pre_transpose_op.name)

                # Remove broadcasting Reshape and Transpose.
                graph.squash(
                    reshape_node,
                    reshape_node.input_names[0],
                    squash_into_next=True,
                    is_data_movement_node=True
                )
                graph.squash(
                    transpose_node,
                    transpose_node.input_names[0],
                    squash_into_next=True,
                    is_data_movement_node=True
                )

                # Update ElementwiseBinary output shape.
                eltwise_output_buffer = graph.get_buffer(eltwise_node.output_names[0])
                eltwise_output_buffer.shape = [
                    eltwise_output_buffer.shape[axis] for axis in inv_perm
                ]

                # Insert Transpose afterwards.
                post_transpose_op = op_adapter.TransposeOp(
                    f'{eltwise_node.output_names[0]}_{"".join(map(str, perm))}', perm
                )
                graph.inject(
                    post_transpose_op, eltwise_node.output_names[0], post_transpose_op.name
                )

    def _optimize_non_inplace_concat(self, graph):
        """Optimize non-inplace Concat.

        This pattern relates to Concat where HTP runs faster if Concat can be "inplace", and the
        non-inplace Concat may further cause following operations transformed into slower ones in
        HTP side. This optimization aims to insert Transpose before and after Concat, transforming
        it into inplace one if possible. HTP constarint for inplace Concat can be simplified to
        that the input shapes on Concat aixs must be divisble by certain multiples. Refer to
        `is_concat_inplace` for currently adopted rules.

        Here provides an example for illustration:

            | [1,2,3,8]     |     [1,8,2,3]         |     [1,2,3,8]        |     [1,8,2,3]
            |           Transpose                   |                  Transpose
            |               |     [1,2,3,8]         |                      |     [1,2,3,8]
            |---------------|                   Transpose              Transpose
                    |                       ==>     |     [1,8,2,3]        |     [1,8,2,3]
                  Concat                            |----------------------|
                    |    [1,2,3,16]                             |
                                                              Concat
                                                                |     [1,16,2,3]
                                                            Transpose
                                                                |     [1,2,3,16]

        Left graph is produced by Layout Transform where Transpose is inserted on right branch due
        to some voting mechanism, resulting in non-inplace Concat (i.e., both input shapes on
        axis=3 not multiple of 32). After the optimization, Transpose are inserted before and after
        Concat to transform into inplace one (i.e., both input shapes on axis=1 are multiple of 8).

        Note that this optimization in fact attempts to re-determine on which branch to insert
        Transpose, and therefore the permutation to transform Concat into inplace is searched by
        reversing existing Transpose instead of trivially enumerating possible ones. In addition,
        the implementation below only considers restricted op sequence to limit the impact. Refer
        to the defined pattern for current matching one.
        """
        pattern = [
            (
                op_adapter.TransposeOp.TRANSLATION_KEY,
                (),
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.ConcatOp.TRANSLATION_KEY, 0)])
            ),
            (
                op_adapter.ConcatOp.TRANSLATION_KEY,
                (
                    'MATCH_BUFS_AT_INDEX',
                    [
                        (op_adapter.TransposeOp.TRANSLATION_KEY, 'ANY'),
                        (op_adapter.ReshapeOp.TRANSLATION_KEY, 'ANY')
                    ]
                ),
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.ReshapeOp.TRANSLATION_KEY, 'ANY')])
            ),
            (
                op_adapter.ReshapeOp.TRANSLATION_KEY,
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.ConcatOp.TRANSLATION_KEY, 0)]),
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.TransposeOp.TRANSLATION_KEY, 0)]),
            )
        ]

        def is_concat_inplace(axis, shapes):
            """Check whether Concat can be inplace with given axis and shapes.

            Note that this is merely a much more simplified version of validating inplace Concat.
            In HTP, quantization parameters must be taken into consideration as well. For example,
            multiplier for width dimension is 8 and 4 for 8-bit and 16-bit, respectively.
            """
            if axis == 0:
                multiplier = 1
            elif axis == len(shapes[0]) - 1:
                multiplier = 32
            else:
                multiplier = 8

            return all(shape[axis] % multiplier == 0 for shape in shapes)

        def validate(matched_nodes):
            """Validate matched nodes by checking whether Concat is already inplace."""
            concat_node = matched_nodes[1]
            input_shapes = [graph.get_buffer(name).shape for name in concat_node.input_names]
            return not is_concat_inplace(concat_node.op.axis, input_shapes)

        def get_inserted_perm(transpose_node, concat_node):
            """Get inserted permutation to make Concat inplace.

            Note that the inserted permutation is determined through reversing the existed
            Tranpose instead of enumerating all possible candidates to keep these inserted
            permutation meaningful.
            """
            input_shapes = [graph.get_buffer(name).shape for name in concat_node.input_names]
            cand_perm = np.argsort(transpose_node.op.perm).tolist()
            cand_shapes = [[shape[axis] for axis in cand_perm] for shape in input_shapes]

            if is_concat_inplace(cand_perm.index(concat_node.op.axis), cand_shapes):
                return cand_perm
            return None

        for matched_nodes in graph.get_matched_nodes_v2(pattern, validator=validate):
            transpose_node, concat_node, _ = matched_nodes

            # Skip if not possible to make Concat inplace.
            inserted_perm = get_inserted_perm(transpose_node, concat_node)
            if inserted_perm is None:
                continue

            # Insert Transpose before Concat.
            for input_name in concat_node.input_names:
                transpose_op = op_adapter.TransposeOp(
                    f'{input_name}_{"".join(map(str, inserted_perm))}', inserted_perm
                )
                graph.inject(transpose_op, input_name, transpose_op.name)

            # Adjust Concat node accordingly.
            concat_node.op.axis = inserted_perm.index(concat_node.op.axis)
            concat_buffer = graph.get_buffer(concat_node.output_names[0])
            concat_buffer.shape = [concat_buffer.shape[axis] for axis in inserted_perm]

            # Insert Transpose after Concat.
            output_name = concat_node.output_names[0]
            recover_perm = np.argsort(inserted_perm).tolist()

            # TODO
            # Originally, Layout Transform aims to reduce number of inserted Transpose, and thus
            # one Transpose should be shared among all Concat consumers. However, inserting
            # individual Tranpose for each consumer may enable further optimizations.

            # TODO
            # Since consumers are currently stored in a set which does not guarantee the order,
            # sorting must be applied to ensure the injected order is not stochastic.
            consumers = list(graph.get_buffer(output_name).consumers)
            consumers.sort(key=lambda node: node.op.name)

            for idx, consumer in enumerate(consumers):
                transpose_op = op_adapter.TransposeOp(
                    f'{output_name}_{idx}_{"".join(map(str, recover_perm))}', recover_perm
                )
                graph.inject(
                    transpose_op,
                    output_name,
                    transpose_op.name,
                    consumer_names=[consumer.op.name]
                )

    def _optimize_unmatched_matmul(self, graph):
        """Optimize unmatched pattern of Matmul.

        HTP defines a specific optimization by matching following pattern:

            Transpose -> Reshape -> MatMul -> Reshape

        , which is commonly observed in graphs transformed from Axis Tracking. However, the behavior
        is different for Layout Transform which results in following pattern instead:

            Reshape -> Transpose -> MatMul -> Reshape

        , and therefore corresponding HTP optimization could not be triggered, causing significant
        performance drop.

        Below is the illustration of pattern to be matched and transformed, including op sequence
        and shapes:

              | [N,D,C]        |     [N,H,W,C]        | [N,D,C]        |     [N,H,W,C]
              |             Reshape                   |            Transpose
              |                |     [N,HW,C]         |                |     [N,C,H,W]
              |            Transpose                  |             Reshape
              |                |     [N,C,HW]         |                |     [N,C,HW]
              |----------------|                      |----------------|
                       |                       ==>             |
                     MatMul                                  MatMul
                       |    [N,D,HW]                           |    [N,D,HW]
                    Reshape                                 Reshape
                       |    [N,D,H,W]                          |    [N,D,H,W]

        , where Tranpose will be hoisted before Reshape to twist this pattern back to HTP favorable
        one.
        """
        pattern = [
            (
                op_adapter.ReshapeOp.TRANSLATION_KEY,
                (),
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.TransposeOp.TRANSLATION_KEY, 0)])
            ),
            (
                op_adapter.TransposeOp.TRANSLATION_KEY,
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.ReshapeOp.TRANSLATION_KEY, 0)]),
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.MatMulOp.TRANSLATION_KEY, 0)]),
            ),
            (
                op_adapter.MatMulOp.TRANSLATION_KEY,
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.TransposeOp.TRANSLATION_KEY, 1)]),
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.ReshapeOp.TRANSLATION_KEY, 0)]),
            ),
            (
                op_adapter.ReshapeOp.TRANSLATION_KEY,
                ('MATCH_BUFS_AT_INDEX', [(op_adapter.MatMulOp.TRANSLATION_KEY, 0)]),
                (),
            )
        ]

        def validate(matched_nodes):
            """Validate matched nodes with expecting shapes and attributes."""
            # Make sure intermediate matched nodes are valid without other consumers.
            if any(
                len(graph.get_output_buffers(node)[0].consumers) > 1 for node in matched_nodes[:-1]
            ):
                return False

            _, transpose_node, matmul_node, reshape_node = matched_nodes

            # MatMul must have zero bias and not transposing neither inputs. Note that current
            # condition checks whether the third input, which is expected to be the bias, should
            # not exist for simplicity.
            if (
                len(matmul_node.input_names) == 3
                or getattr(matmul_node.op, ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0)
                or getattr(matmul_node.op, ir_graph.QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1)
            ):
                return False

            # Transpose must have spatial-first to spatial-last permute order. Note that current
            # condition constrains to 3D case only.
            if transpose_node.op.perm != [0, 2, 1]:
                return False

            matmul_input_shapes = [
                graph.get_buffer(input_name).shape for input_name in matmul_node.input_names
            ]
            matmul_output_shape = graph.get_buffer(matmul_node.output_names[0]).shape
            reshape_output_shape = graph.get_buffer(reshape_node.output_names[0]).shape
            return (
                len(reshape_output_shape) == 4
                and matmul_input_shapes[0][1] == reshape_output_shape[1]
                and matmul_input_shapes[0][0] == matmul_input_shapes[1][0]
                and matmul_input_shapes[1][2] == reshape_output_shape[2] * reshape_output_shape[3]
                and matmul_output_shape[1] < matmul_output_shape[2]
            )

        for matched_nodes in graph.get_matched_nodes_v2(pattern, validator=validate):
            reshape_node, transpose_node = matched_nodes[:2]

            # Prune Transpose node by squashing into next.
            graph.squash(
                transpose_node,
                transpose_node.input_names[0],
                squash_into_next=True,
                is_data_movement_node=True
            )
            # Inject Transpose node before Reshape node.
            transpose_op = op_adapter.TransposeOp(
                f'{reshape_node.input_names[0]}_0312', (0, 3, 1, 2)
            )
            graph.inject(transpose_op, reshape_node.input_names[0], transpose_op.name)

            # Update Reshape node.
            new_shape = [
                reshape_node.op.shape[0], reshape_node.op.shape[2], reshape_node.op.shape[1]
            ]
            graph.get_buffer(reshape_node.output_names[0]).shape = new_shape
            reshape_node.op.shape = ir_graph.IrStaticTensor(
                ir_graph.IR_OP_RESHAPE_PARAM_SHAPE,
                [len(new_shape)],
                np.array(new_shape, dtype=np.int32),
                ir_graph.QNN_DATATYPE_INT_32,
            )

    def _optimize_width_untilable_prelu(self, graph):
        """Optimize width untilable Prelu.

        HTP tiling is especially beneficial for large dimensions but may only be enabled when
        fulfilling certain constraints. However, these HTP restrictions are highly coupled with
        Axis Tracking and may not be applicable for Layout Transform. This optimization aims to
        twist the Prelu pattern back to meet HTP tiling rule. HTP restricts width-tiling to fulfill
        width > 2048 and depth > 16 but a case of width > 2048 and depth <= 16 is encountered, and
        thus no tiling is applied, resulting in significant performance drop. Before HTP loosening
        this constraint, Transpose is inserted to transform back to Axis Tracking-like pattern
        which enables HTP depth-tiling.

        Note that current pattern restricts on Prelu in 3D shapes, with width > 2048, depth <= 16,
        and followed by Reshape to minimize the impact.
        """
        pattern = [(op_adapter.PreluOp.TRANSLATION_KEY, (), ())]

        def validate(matched_nodes):
            """Validate matched nodes with expecting shapes."""
            output_buffer = graph.get_buffer(matched_nodes[0].output_names[0])
            return (
                all(
                    isinstance(consumer.op, op_adapter.ReshapeOp)
                    for consumer in output_buffer.consumers
                )
                and output_buffer.rank() == 3
                and output_buffer.shape[1] > 2048
                and output_buffer.shape[2] <= 16
            )

        for matched_nodes in graph.get_matched_nodes_v2(pattern, validator=validate):
            prelu_node = matched_nodes[0]

            # Inject Transpose ahead.
            pre_transpose = op_adapter.TransposeOp(f'{prelu_node.input_names[0]}_021', (0, 2, 1))
            graph.inject(
                pre_transpose,
                prelu_node.input_names[0],
                pre_transpose.name,
                consumer_names=[prelu_node.op.name]
            )

            # Adjust Prelu output shape.
            output_buffer = graph.get_buffer(prelu_node.output_names[0])
            output_buffer.shape = [
                output_buffer.shape[0], output_buffer.shape[2], output_buffer.shape[1]
            ]

            # Inject Transpose behind.
            post_transpose = op_adapter.TransposeOp(f'{prelu_node.output_names[0]}_021', (0, 2, 1))
            graph.inject(post_transpose, prelu_node.output_names[0], post_transpose.name)


class OptimizationTranslationBase(translation.Translation):
    """
    This class is to be used to perform graph optimizations such as: folding, squashing,pruning, etc. Additionally,
    it is also used to perform axis tracking and by default implements to spatial first order function
    (NCHW to NHWC, or TNF to NTF). Use this base class to get the default function and call register_method to add a new
    optimization. For eg: The OptimizeBatchnormTranslation overloads the axes_to_spatial_first_order to handle weights
    as well as adds a squash_batchnorm function and registers the method in the __init__ function.
    """
    def __init__(self):
        translation.Translation.__init__(self)
        self.register_method(TRANSFORM_LAYOUT, self.transform_layout)
        self.register_method(AXES_TO_SPATIAL_FIRST_ORDER, self.axes_to_spatial_first_order)
        self.register_method(MERGE_LOW_LEVEL_OPS_TO_LAYERS, self.merge_low_level_ops_to_layers)
        self.register_method(REPLACE_6D_OPERATION, self.replace_6d_operation)

    def transform_layout(self, node: op_graph.OpNode, graph: op_graph.IROpGraph, layout_manager):
        layout_manager.apply_transform(node)

    def axes_to_spatial_first_order(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        """
        Performs axis permutations(as needed) to get a spatial first order.

        Note: The eltwise_...() function that gets called re-populates the node's buffer "axis_format" and "shape" from
        source framework to the destination for certain ranks. If an overload of this function is done for a child class
        and this eltwise_...() function is not called make sure to understand and implement these changes to avoid
        conversion errors.

        :param node: an OpNode object to optimize from the IR graph
        :param graph: an IROpgraph object

        returns: True if any changes were done
                 False if no changes required
        """
        if AxisTracker.input_axis_formats_intact(graph, node, input_nontrivial_as_changed=True):
            # No change in input formats, and none of the input formats are NonTrivial
            # Nothing to do in this case
            return False

        input_axis_formats_before = graph.get_input_axis_formats(node)
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_axis_formats_after = graph.get_input_axis_formats(node)
        input_buffers = graph.get_input_buffers(node)
        for i, buf in enumerate(input_buffers):
            if input_axis_formats_before[i] != input_axis_formats_after[i]:
                transpose_node = buf.producer
                graph.update_trace_info(transpose_node, [node])
                graph.update_trace_info(buf, [node])
        return True

    def merge_low_level_ops_to_layers(self, graph):
        """"
        When overloaded in the child class, it is implemented to merge to the low level ops to layers.

        """
        pass

    def replace_6d_operation(self, node, graph):
        # By default, we do not allow 6d inputs / outputs. Ops need to override this functions to support 6d.
        input_bufs = graph.get_input_buffers(node)
        for input_buf in input_bufs:
            if input_buf.rank() >= 6:
                raise ValueError(f"inputs of {node.op.name} are not supported in 6D")
        output_bufs = graph.get_output_buffers(node)
        for output_buf in output_bufs:
            if output_buf.rank() >= 6:
                raise ValueError(f"outputs of {node.op.name} are not supported in 6D")


# ------------------------------------------------------------------------------------------------------------------
#   Graph Optimizations
# ------------------------------------------------------------------------------------------------------------------
def register_graph_optimization(graph_optimization_method):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param graph_optimization_method: a concrete class for a given optimization
    """
    return graph_optimization_method


@register_graph_optimization
def remove_disconnected_nodes(graph):
    """Removes nodes with all its outputs unconsumed from the graph."""
    all_ops = set(graph.nodes_in_order)
    connected_ops = set()
    queue = []
    graph_output_nodes = graph.get_output_nodes_of_graph()

    if graph_output_nodes:
        queue.extend(graph_output_nodes)
        # Find nodes from Output to Input Op
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node and filter out null input
            node_inputs = [node_input for node_input in graph.get_op_input_nodes(node) if node_input]
            new_nodes = [node_ for node_ in node_inputs if (node_ not in connected_ops and node_ not in queue)]
            queue.extend(new_nodes)

    else:
        # Ensure input nodes have consumers before adding them to queue
        input_nodes = graph.get_input_nodes_to_graph()
        input_nodes = [node for node in input_nodes if graph.get_buffer(node.output_names[0]).consumers]
        queue.extend(input_nodes)
        # Find nodes from Input Op to outputs
        while queue:
            node = queue.pop(0)
            connected_ops.add(node)

            # Add input nodes for the node, this will add the Constant input Ops that will be otherwise missed
            node_inputs = [node_input for node_input in graph.get_op_input_nodes(node) if node_input]
            new_nodes = [node for node in node_inputs if node not in connected_ops]
            for new_node in new_nodes:
                queue.insert(0, new_node)

            # Extend the queue with output nodes
            node_outputs = graph.get_op_output_nodes(node)
            new_nodes = [node for node in node_outputs if node not in queue]
            queue.extend(new_nodes)

    disconnected_nodes = all_ops - connected_ops
    prunable_node_names = [node.op.name for node in disconnected_nodes]
    if disconnected_nodes:
        log_debug("Pruning Disconnected nodes {}".format(prunable_node_names))

    for node in disconnected_nodes:
        try:
            graph.prune(node, force_remove=True)
        except Exception as e:
            log_error("Cannot find node {}".format(node.op.name))
            raise e

    if not graph.list_nodes():
        raise ValueError("After pruning disconnected nodes, this model is empty.")

    return graph


# ------------------------------
# Util used for common squashing
# ------------------------------
def squash_node_into_nn_node(graph, matched_node_list):
    """
    Squashes a node into an NN node. This can be done by accounting for the node's operation in arithmetic adjustments
    to the NN node's weights and biases. Intended use is for Elementwise ops that follow an NN op.
    :param graph: The IROpGraph object
    :param matched_node_list: the list of nodes that contain elementwise ops, have a constant input, and are
                              preceded by a node that contains an NN op
    """

    OPS_HAVING_BIAS_SUM = [
        op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]
    ]
    OPS_HAVING_BIAS_SUB = [
        op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT]
    ]
    OPS_HAVING_WEIGHTS_PRODUCT = [
        op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY]
    ]
    OPS_HAVING_WEIGHTS_DIV = [
        op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE]
    ]

    for node_tuple in matched_node_list:
        # collect previous and current op information
        node = node_tuple[0]
        node_type = node.op.type
        nn_buf, nn_op, const_op = None, None, None
        for name in node.input_names:
            input_buf = graph.get_buffer(name)
            input_op = graph.get_producer_op(name)
            if (len(input_buf.producer.input_names) == 3 and
                input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                  op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                  op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                                  op_adapter.MatMulOp.TRANSLATION_KEY]) or \
                    (hasattr(input_op, "weights") or hasattr(input_op, "bias")):
                # if input_op has output_encodings squashing is disabled for better alignment
                if graph.has_quantization_param(input_op.name) and \
                    graph.quantization_params[input_op.name]["output_encodings"]:
                    return
                nn_buf = input_buf
                nn_op = input_op
                if len(nn_buf.producer.input_names) == 3 and \
                        nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                       op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                       op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                                       op_adapter.MatMulOp.TRANSLATION_KEY]:
                    manage_shared_static_input(graph, nn_buf.producer, 1)
                    src_weight = graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.tensor
                    manage_shared_static_input(graph, nn_buf.producer, 2)
                    src_bias = graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor
                else:
                    src_weight = nn_op.weights
                    src_bias = nn_op.bias
            # Without bias case handled here
            elif (len(input_buf.producer.input_names) == 2 and
                input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                  op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                  op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                                  op_adapter.MatMulOp.TRANSLATION_KEY]) or \
                    (hasattr(input_op, "weights")):
                # if input_op has output_encodings squashing is disabled for better alignment
                if graph.has_quantization_param(input_op.name) and \
                        graph.quantization_params[input_op.name]["output_encodings"]:
                    return
                nn_buf = input_buf
                nn_op = input_op
                if len(nn_buf.producer.input_names) == 2 and \
                        nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                       op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                       op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                                       op_adapter.MatMulOp.TRANSLATION_KEY]:
                    manage_shared_static_input(graph, nn_buf.producer, 1)
                    src_weight = graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.tensor
                else:
                    src_weight = nn_op.weights
            elif input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                const_op = input_op

        if nn_op is None:
            raise ValueError("Failed to retrieve NN op to squash {} node {} into.".format(node_type, node.op.name))

        if const_op is None:
            raise ValueError("Failed to retrieve const op to squash {} node {} into.".format(node_type, node.op.name))

        if nn_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("weights")) and len(src_weight.shape) == 5:
                # weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for broadcasting to handle non-square kernel and then revert
                if nn_op.type == op_adapter.Conv3dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv3d_weights_to_ir)
                elif nn_op.type == op_adapter.TranposeConv3dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv3d_weights_to_ir)
            if const_op is not None and len(const_op.shape) == 5:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
        elif nn_buf.axis_format == AxisTracker.AxisFormat.NCS:
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("weights")) and len(src_weight.shape) == 4:
                # weights are not yet transposed as that happens in axes_to_spatial_first later,
                # so we need to transpose for broadcasting to handle non-square kernel and then revert
                if nn_op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv2d_weights_to_ir)
                elif nn_op.type == op_adapter.TransposeConv2dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv2d_weights_to_ir)
            if const_op is not None and len(const_op.shape) == 4:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC)

        # separate conditionals according to which arithmetic operation needs to happen
        if node_type in OPS_HAVING_BIAS_SUM:
            scale_bias = const_op.tensor
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("bias")):
                src_bias = np.atleast_1d((src_bias + scale_bias).squeeze())
            else:
                src_bias = np.atleast_1d((scale_bias).squeeze())
        elif node_type in OPS_HAVING_BIAS_SUB:
            scale_bias = const_op.tensor
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("bias")):
                src_bias = np.atleast_1d((src_bias + scale_bias).squeeze())
            else:
                src_bias = np.atleast_1d((scale_bias).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_PRODUCT:
            scale_weights = const_op.tensor
            src_weight = src_weight * scale_weights
            scale_bias = const_op.tensor
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("bias")):
                src_bias = np.atleast_1d((src_bias * scale_weights).squeeze())
            else:
                src_bias = np.atleast_1d((scale_weights).squeeze())
        elif node_type in OPS_HAVING_WEIGHTS_DIV:
            scale_weights = const_op.tensor
            src_weight = src_weight / scale_weights
            if (len(nn_buf.producer.input_names) == 3 or nn_op.hasattr("bias")):
                src_bias = np.atleast_1d((src_bias / scale_weights).squeeze())
            else:
                src_bias = np.atleast_1d((scale_weights).squeeze())
        else:
            raise ValueError("Squashing {} node {} into {} node {} unsupported.".format(node_type, node.op.name,
                                                                                        nn_op.type, nn_op.name))

        # For Instancenorm the weight and bias should be of rank 1
        # After optimization if the weights and bias becomes rank>1 then it will fail in op validation
        if nn_op.type==op_adapter.InstanceNormOp.TRANSLATION_KEY and any([len(src_weight.shape)>1, len(src_bias.shape)>1]):
            return

        if nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY]:
            src_weight= np.atleast_1d(src_weight.squeeze())

        if nn_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            if (len(input_buf.producer.input_names) == 3 or nn_op.hasattr("weights")) and len(src_weight.shape) == 5:
                if nn_op.type == op_adapter.Conv3dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv3d_weights_from_ir)
                elif nn_op.type == op_adapter.TransposeConv3dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv3d_weights_from_ir)
            if const_op is not None and len(const_op.shape) == 5:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NDHWC_TO_NCDHW)
        elif nn_buf.axis_format == AxisTracker.AxisFormat.NCS:
            if (len(input_buf.producer.input_names) == 3 or nn_op.hasattr("weights")) and len(src_weight.shape) == 4:
                if nn_op.type in [op_adapter.Conv2dOp.TRANSLATION_KEY,
                                  op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY]:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_conv2d_weights_from_ir)
                elif nn_op.type == op_adapter.TransposeConv2dOp.TRANSLATION_KEY:
                    src_weight = np.transpose(src_weight, graph.src_axis_order.permute_deconv2d_weights_from_ir)
            if const_op is not None and len(const_op.shape) == 4:
                const_op.tensor = np.transpose(const_op.tensor, AxisTracker.AxisFormat.NSC_TO_NCS)

        if len(nn_buf.producer.input_names) == 3 and \
                nn_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                               op_adapter.InstanceNormOp.TRANSLATION_KEY,
                               op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                               op_adapter.MatMulOp.TRANSLATION_KEY]:
            origin_bias_tensor = graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor
            graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.tensor = src_weight
            graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.tensor = src_bias
        else:
            origin_bias_tensor = const_op.tensor
            nn_op.weights = src_weight
            nn_op.bias = const_op.tensor
        log_debug2(code_to_message.get_debugging_message("DEBUG_SQUASH_INTO_NN_NODE")
                   (node_type, node.op.name, nn_op.type, nn_op.name))

        # for nn_node without bias, graph will prepare bias for it (zero bias),
        # however its true bias name is stored in eltwise node. for better user experience,
        # we should use eltwise bias name instead of prepared one
        eltwise_bias_name = const_op.name
        eltwise_bias_buffer = graph.get_buffer(eltwise_bias_name)
        if np.allclose(origin_bias_tensor, 0) and node_type in OPS_HAVING_BIAS_SUM:
            # since we will update eltwise.bias, so its should have only one consumer
            if len(eltwise_bias_buffer.consumers) == 1:
                bias_buffer = graph.get_buffer(nn_buf.producer.input_names[2])

                # remove nn_node from nn_node.bias's consumers
                bias_buffer.consumers.remove(nn_buf.producer)
                if len(bias_buffer.consumers)==0:
                    graph.prune(bias_buffer.producer)

                # change nn_node input_names[2] to eltwise_node.bias
                nn_buf.producer.input_names[2] = eltwise_bias_name
                eltwise_bias_buffer.consumers.add(nn_buf.producer)
                eltwise_bias_buffer.axis_format = bias_buffer.axis_format

                # update eltwise_bias_node with bias tensor
                eltwise_bias_node = eltwise_bias_buffer.producer
                eltwise_bias_node.op.tensor = src_bias
                eltwise_bias_buffer.shape = list(eltwise_bias_node.op.shape)
                eltwise_bias_buffer.axis_format = bias_buffer.axis_format

                # weight, nn_node, bias => weight, bias, nn_node
                idx_nn = graph.nodes_in_order.index(nn_buf.producer)
                idx_bias = graph.nodes_in_order.index(eltwise_bias_buffer.producer)
                if idx_nn < idx_bias:
                    graph.nodes_in_order[idx_nn] = eltwise_bias_buffer.producer
                    graph.nodes_in_order[idx_bias] = nn_buf.producer
            elif graph.has_quantization_param(node.op.name):
                # add quantization params for nn.bias
                param_encodings = graph.get_layer_quantization_param(node.op.name)[op_graph.QuantParams.PARAM_ENCODINGS]
                if len(param_encodings) > 0:
                    bias_encoding = param_encodings[0].copy()
                    bias_producer = graph.get_buffer(nn_buf.producer.input_names[2]).producer
                    bias_encoding['name'] = bias_producer.output_names[0]
                    graph.add_quantization_params(bias_producer.op.name, output_encodings=bias_encoding)

        nn_node = nn_buf.producer
        graph.squash(node, input_name=nn_buf.name)
        # update weight's and bias's op trace data of new nn_node
        if len(nn_node.input_names) >= 2 and (node_type in OPS_HAVING_WEIGHTS_PRODUCT or node_type in OPS_HAVING_WEIGHTS_DIV):
            weight_buffer = graph.get_buffer(nn_node.input_names[1])
            graph.update_trace_info(weight_buffer.producer, [weight_buffer.producer, eltwise_bias_buffer.producer])
            graph.update_trace_info(weight_buffer, [weight_buffer, eltwise_bias_buffer])
        if len(nn_node.input_names) == 3:
            bias_buffer = graph.get_buffer(nn_node.input_names[2])
            graph.update_trace_info(bias_buffer.producer, [bias_buffer.producer, eltwise_bias_buffer.producer])
            graph.update_trace_info(bias_buffer, [bias_buffer, eltwise_bias_buffer])


def validate_eltwise_pattern(graph, nodes_tuple, mode):
    """
    Common function to validate if pattern is squashable
    :param graph: the IROpGraph
    :param nodes_tuple: the matched list of nodes
    :param mode: either bias or weight. Use to determine if squashing is
                 eltwise[add|sub] or eltwise[prod|div] respectively.
    :return:
    """

    OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS = [
        op_adapter.Conv2dOp.TRANSLATION_KEY,
        op_adapter.TransposeConv2dOp.TRANSLATION_KEY,
        op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY
    ]

    node = nodes_tuple[0]
    nn_buf, nn_op, const_op = None, None, None
    is_batchnorm_input = False
    is_fully_connected_input = False
    for name in node.input_names:
        input_node = graph.get_buffer(name).producer
        input_op = graph.get_buffer(name).producer.op
        # verify that one of the inputs is constant and the other input is produced by nn_type op(BN, FC, Conv/Deconv)
        if input_op.type in OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS:
            # Squashing elementwise operations into these nodes is handled in their respective optimizations classes
            return False
        elif input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op
        elif (mode == "weights" and hasattr(input_op, "weights") and hasattr(input_op, "bias")) or \
                (mode == "bias" and hasattr(input_op, "bias")) or \
                    (len(input_node.input_names) == 3 and
                     input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY,
                                       op_adapter.InstanceNormOp.TRANSLATION_KEY,
                                       op_adapter.FullyConnectedOp.TRANSLATION_KEY,
                                       op_adapter.MatMulOp.TRANSLATION_KEY
                                       ]):
            if len(graph.get_buffer(name).consumers) != 1:
                # Unable to squash into nn_op which has more than one consumer
                return False
            nn_op = input_op
            nn_buf = graph.get_buffer(name)
            if input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY]:
                is_batchnorm_input = True
            if input_op.type == op_adapter.FullyConnectedOp.TRANSLATION_KEY or op_adapter.MatMulOp.TRANSLATION_KEY:
                is_fully_connected_input = True
        elif (mode == "bias") and (len(input_node.input_names) == 2 and
                 input_op.type in [op_adapter.FullyConnectedOp.TRANSLATION_KEY
                                   ]):
            if len(graph.get_buffer(name).consumers) != 1:
                # Unable to squash into nn_op which has more than one consumer
                return False
            nn_op = input_op
            nn_buf = graph.get_buffer(name)
            if input_op.type == op_adapter.FullyConnectedOp.TRANSLATION_KEY:
                is_fully_connected_input = True
    # For mode:weights
    #      Only valid to squash if the nn_op has act output that are broadcastable with the scale weights AND
    #      the scale weights are same rank with nn_op bias and broadcastable
    # For mode:bias
    #      Only valid if the nn_op has a bias with the same rank as const_op and broadcastable
    if nn_op is not None and const_op is not None:
        const_shape = const_op.shape
        const_shape_squeezed = np.atleast_1d(const_op.tensor.squeeze()).shape
        if is_batchnorm_input:
            bias_shape = graph.get_buffer(nn_buf.producer.input_names[1]).producer.op.shape
        elif is_fully_connected_input and len(nn_buf.producer.input_names) == 3:
            bias_shape = graph.get_buffer(nn_buf.producer.input_names[2]).producer.op.shape
        elif len(nn_buf.producer.input_names) == 3:
            bias_shape = nn_op.bias.shape
        if mode == 'bias' and len(nn_buf.producer.input_names) == 3:
            if len(const_shape_squeezed) == len(bias_shape) and \
                    translation_utils.broadcastable(bias_shape, const_shape_squeezed):
                return True
        if mode == 'bias' and len(nn_buf.producer.input_names) == 2:
                return True
        elif mode == 'weights' and len(nn_buf.producer.input_names) == 3:
            nn_buf_shape = nn_buf.get_buf_dims()
            axis_order = graph.src_axis_order
            input_ir_shapes = [axis_order.permute_shape_to_ir(nn_buf_shape),
                               axis_order.permute_shape_to_ir(const_shape)]
            # Note: verify with the ir shapes for inputs since this is done pre axis-tracking
            if translation_utils.broadcastable(*input_ir_shapes) and \
                    (len(const_shape_squeezed) == len(bias_shape) and
                     translation_utils.broadcastable(bias_shape, const_shape_squeezed)):
                return True
        elif mode == 'weights' and len(nn_buf.producer.input_names) == 2:
            nn_buf_shape = nn_buf.get_buf_dims()
            axis_order = graph.src_axis_order
            input_ir_shapes = [axis_order.permute_shape_to_ir(nn_buf_shape),
                               axis_order.permute_shape_to_ir(const_shape)]
            return True
    return False


def add_or_broadcast_bias(node, graph, output_channel):
    weights_buffer = graph.get_buffer(node.input_names[1])
    if len(node.input_names) < 3:
        bias_tensor = np.zeros([output_channel], dtype=np.float32)
        bias_op_name = node.op.name + "_bias"
        new_bias_op_name = bias_op_name
        while new_bias_op_name in graph.nodes_by_name:
            new_bias_op_name = bias_op_name + "__" + str(random.randint(100000, 999999))
            if new_bias_op_name not in graph.nodes_by_name:
                break
        bias_op_name = new_bias_op_name
        bias_op = op_adapter.ConstantOp(bias_op_name, tensor=bias_tensor.copy())
        conv_idx = graph.list_nodes().index(node)
        bias_node = graph.add(bias_op, [], [bias_op_name], axis_formats=[AxisTracker.AxisFormat.ANY], idx=conv_idx)
        graph.update_trace_info(bias_node, node)
        graph.update_trace_info(graph.get_buffer(bias_op_name), node)
        graph.get_buffer(bias_op_name).consumers.add(node)
        node.input_names.append(bias_op_name)
    else:
        bias_buffer = graph.get_buffer(node.input_names[2])
        # Represents case where broadcasting biases is required
        if bias_buffer.shape[0] < output_channel:
            bias_const_node = bias_buffer.producer
            if len(bias_const_node.op.tensor) != 1:
                raise ValueError("Unable to broadcast bias tensor for node {}".format(node.op.name))
            bias_const_node.op.tensor = np.repeat(bias_const_node.op.tensor, weights_buffer.shape[3])
            bias_buffer.shape = list(bias_const_node.op.shape)


def squash_eltwise_into_conv(graph, conv_node):
    conv_output_buffer = graph.get_buffer(conv_node.output_names[0])
    eltwise_node = list(conv_output_buffer.consumers)[0]
    # Find and assign the const_op from eltwise_node's input_names
    const_op = None
    for name in eltwise_node.input_names:
        input_op = graph.get_producer_op(name)
        if input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op

    # Ensure the constant operation has the proper squash shape based on source axis order
    const_tensor = const_op.tensor
    if conv_output_buffer.axis_format == AxisTracker.AxisFormat.NCDHW and len(const_op.shape) == 5:
        const_tensor = np.transpose(const_tensor, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
    elif conv_output_buffer.axis_format == AxisTracker.AxisFormat.NCS and len(const_op.shape) == 4:
        const_tensor = np.transpose(const_tensor, AxisTracker.AxisFormat.NCS_TO_NSC)

    manage_shared_static_input(graph, conv_node, 2)
    bias_buffer = graph.get_buffer(conv_node.input_names[2])
    bias_producer = bias_buffer.producer
    bias_tensor = bias_producer.op.tensor
    origin_bias_tensor = bias_tensor

    # Apply the const_node's tensor to the conv_node's bias according to type of elementwise operation
    if eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
        bias_tensor = np.atleast_1d((bias_tensor + const_tensor).squeeze())
    elif eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT]:
        bias_tensor = np.atleast_1d((bias_tensor - const_tensor).squeeze())
    else:
        # Only ElementwiseProduct/DivOp require static weights, so extract the static weights only in these cases
        manage_shared_static_input(graph, conv_node, 1)
        weights_producer = graph.get_buffer(conv_node.input_names[1]).producer
        weights_buffer = graph.get_buffer(conv_node.input_names[1])
        weights_tensor = weights_producer.op.tensor
        if eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY]:
            if weights_buffer.axis_format == AxisTracker.AxisFormat.OIHW:
                weights_tensor = np.transpose(weights_tensor, graph.src_axis_order.permute_conv2d_weights_to_ir)
                weights_tensor = weights_tensor * const_tensor
                weights_tensor = np.transpose(weights_tensor, graph.src_axis_order.permute_conv2d_weights_from_ir)
            else:
                weights_tensor = weights_tensor * const_tensor
            bias_tensor = np.atleast_1d((bias_tensor * const_tensor).squeeze())
        elif eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE]:
            if weights_buffer.axis_format == AxisTracker.AxisFormat.OIHW:
                weights_tensor = np.transpose(weights_tensor, graph.src_axis_order.permute_conv2d_weights_to_ir)
                weights_tensor = weights_tensor / const_tensor
                weights_tensor = np.transpose(weights_tensor, graph.src_axis_order.permute_conv2d_weights_from_ir)
            else:
                weights_tensor = weights_tensor / const_tensor
            bias_tensor = np.atleast_1d((bias_tensor / const_tensor).squeeze())
        weights_producer.op.tensor = weights_tensor

    # Reincorporate the new bias and squash the elementwise operation
    bias_producer.op.tensor = bias_tensor
    log_debug2(code_to_message.get_debugging_message("DEBUG_SQUASH_INTO_NN_NODE")
               (eltwise_node, eltwise_node.op.name, conv_node.op.type, conv_node.op.name))

    # for conv_node without bias, graph will prepare bias for it (zero bias),
    # however its true bias name is stored in eltwise node. for better user experience,
    # we should use eltwise bias name instead of prepared one
    eltwise_bias_name = eltwise_node.input_names[1]
    eltwise_bias_buffer = graph.get_buffer(eltwise_bias_name)
    if np.allclose(origin_bias_tensor, 0) and \
        eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
        # since we will update eltwise.bias, so its should have only one consumer
        if len(eltwise_bias_buffer.consumers) == 1:

            # change conv_node input_names[2] to eltwise_node.bias
            conv_node.input_names[2] = eltwise_bias_name
            eltwise_bias_buffer.consumers.add(conv_node)

            # remove conv_node from conv.bias's consumers
            bias_buffer.consumers.remove(conv_node)
            if len(bias_buffer.consumers)==0:
                graph.prune(bias_producer)

            # update eltwise_bias_node with bias tensor
            eltwise_bias_node = eltwise_bias_buffer.producer
            eltwise_bias_node.op.tensor = bias_tensor
            eltwise_bias_buffer.shape = list(eltwise_bias_node.op.shape)
            eltwise_bias_buffer.axis_format = bias_buffer.axis_format

            # weight, conv, bias => weight, bias, conv
            idx_conv = graph.nodes_in_order.index(conv_node)
            idx_bias = graph.nodes_in_order.index(eltwise_bias_node)
            if idx_conv < idx_bias:
                graph.nodes_in_order[idx_conv] = eltwise_bias_node
                graph.nodes_in_order[idx_bias] = conv_node
        elif graph.has_quantization_param(eltwise_node.op.name):
            # add quantization params for conv.bias
            param_encodings = graph.get_layer_quantization_param(eltwise_node.op.name)[op_graph.QuantParams.PARAM_ENCODINGS]
            if len(param_encodings) > 0:
                bias_encoding = param_encodings[0].copy()
                bias_encoding['name'] = bias_producer.output_names[0]
                graph.add_quantization_params(bias_producer.op.name, output_encodings=bias_encoding)

    graph.squash(eltwise_node, input_name=conv_output_buffer.name)
    # update weight's and bias's op trace data of new conv_node
    if len(conv_node.input_names) >= 2 and eltwise_node.op.type in [op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY], \
                                                                    op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE]]:
        weight_buffer = graph.get_buffer(conv_node.input_names[1])
        graph.update_trace_info(weight_buffer.producer, [weight_buffer.producer, eltwise_bias_buffer.producer])
        graph.update_trace_info(weight_buffer, [weight_buffer, eltwise_bias_buffer])
    if len(conv_node.input_names) == 3 and eltwise_node.op.type in [op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD], \
                                                                    op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT]]:
        bias_buffer = graph.get_buffer(conv_node.input_names[2])
        graph.update_trace_info(bias_buffer.producer, [bias_buffer.producer, eltwise_bias_buffer.producer])
        graph.update_trace_info(bias_buffer, [bias_buffer, eltwise_bias_buffer])


def validate_conv_eltwise_pattern(graph, conv_node, eltwise_type):
    conv_node_output_buffer = graph.get_buffer(conv_node.output_names[0])
    if len(conv_node_output_buffer.consumers) != 1 or \
            list(conv_node_output_buffer.consumers)[0].op.type != eltwise_type:
        return False

    eltwise_node = list(conv_node_output_buffer.consumers)[0]

    # Find the constant op from input_names of the eltwise_node
    const_op = None
    for name in eltwise_node.input_names:
        input_op = graph.get_producer_op(name)
        if input_op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            const_op = input_op

    # Constant op was not found, so we cannot squash this elementwise operation
    if const_op is None:
        return False

    # Scalar products are able to be squashed into convolution weights
    if eltwise_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY]:
        return len(const_op.shape) == 1

    const_shape_squeezed = np.atleast_1d(const_op.tensor.squeeze()).shape
    bias_shape = graph.get_buffer(conv_node.input_names[2]).shape
    # Const shape and bias shape should have the same rank and be broadcastable
    return len(const_shape_squeezed) == len(bias_shape) and \
        translation_utils.broadcastable(bias_shape, const_shape_squeezed)


def prepare_conv_inputs_as_params(graph, conv_node):
    weights_buffer = graph.get_buffer(conv_node.input_names[1])
    weights_node = weights_buffer.producer
    bias_buffer = graph.get_buffer(conv_node.input_names[2])
    bias_node = bias_buffer.producer
    if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
            bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
        conv_node.op.weights = weights_node.op.tensor
        conv_node.op.bias = bias_node.op.tensor
        # Remove the weights/bias inputs from the IR graph
        graph.remove_node_as_consumer(conv_node, weights_buffer.name)
        graph.remove_node_as_consumer(conv_node, bias_buffer.name)
        conv_node.input_names = [conv_node.input_names[0]]
        graph.update_trace_info(conv_node, [weights_node, weights_buffer, bias_node, bias_buffer, conv_node])


def manage_shared_static_input(graph, node, idx):
    """
    Create a copy of node's input[idx] if input[idx] has more than one consumer and
    assigns the copy as node's input[idx].
    """
    if not len(graph.get_buffer(node.input_names[idx]).consumers) > 1 :
        return
    input_buffer = graph.get_buffer(node.input_names[idx])
    producer_op = input_buffer.producer.op
    weight_tensor = producer_op.tensor
    weight_tensor_copy = np.copy(weight_tensor)
    if idx == 1:
        name = node.op.name + "_kernel_weight"
    else:
        name = node.op.name + "_kernel_bias"
    const_op = op_adapter.ConstantOp(name, tensor=weight_tensor_copy)
    producer_idx = graph.list_nodes().index(input_buffer.producer)
    const_node = graph.add(const_op, [], [name], axis_formats=[input_buffer.axis_format], idx=producer_idx+1)
    # Update trace info to new const node and the output buffer
    graph.update_trace_info(const_node, [input_buffer.producer])
    graph.update_trace_info(graph.get_output_buffers(const_node)[0], [input_buffer])

    graph.get_buffer(name).consumers.add(node)
    input_buffer.consumers.remove(node)
    node.input_names[idx] = name
    # copy quant overrides to new const buffer only if it exists in original static input
    if(graph.has_quantization_param(input_buffer.name)):
        quant_param = graph.get_layer_quantization_param(input_buffer.name)
        graph.add_quantization_params(name,
                                      bn_params=quant_param['bn_params'],
                                      output_encodings=quant_param['output_encodings'],
                                      param_encodings=quant_param['param_encodings'])


# -----------------------------------------------------------------------------------------------------
# Util used for replace_6d_operation
# -----------------------------------------------------------------------------------------------------

def check_if_6d_supportable(node, graph):
    # We insert pre-reshape after input buffer / before the op to reduce the rank
    # We insert post-reshape before output buffer / after the op to recover the shape
    # These ops_handle_6D_tensor override replace_6d_operation function to insert reshapes
    ops_handle_6D_tensor = {
        op_adapter.ElementwiseUnaryOp,
        op_adapter.ElementwiseBinaryOp,
        op_adapter.TransposeOp,
        op_adapter.ReduceOp,
        op_adapter.GatherOp,
        op_adapter.TileOp,
        op_adapter.SoftmaxOp,
    }
    # However, ReshapeOp does not support 6D inputs / outputs as well.
    # 6d inputs should be produced by the cases so that the inserted Reshape can be squashed
    # Case 1: ConstantOp
    # Case 2: Another ReshapeOp
    input_bufs = graph.get_input_buffers(node)
    for input_buf in input_bufs:
        # this optimization adds reshape, and reshape is not supported for dynamic shaped inputs
        if input_buf.shape.is_dynamic():
            raise ValueError(f"inputs of {node.op.name} are dynamic and not supported in 6D")
        if input_buf.rank() >= 6 and not isinstance(
            input_buf.producer.op, (op_adapter.ConstantOp, op_adapter.ReshapeOp)
        ):
            raise ValueError(f"inputs of {node.op.name} are not supported in 6D")
    # 6d outputs should be consumed by the cases so that the inserted Reshape can be squashed
    # Case 1: Another ReshapeOp
    # Case 2: ops_handle_6D_tensor (It will insert ReshapeOp)
    output_buf = graph.get_output_buffers(node)[0]
    for consumer in output_buf.consumers:
        if output_buf.rank() >= 6 and not isinstance(
            consumer.op, (*ops_handle_6D_tensor, op_adapter.ReshapeOp)
        ):
            raise ValueError(f"outputs of {node.op.name} are not supported in 6D")

def post_reshape_insertion(node, graph, new_out_shapes, orig_out_shapes):
    output_bufs = graph.get_output_buffers(node)
    assert len(output_bufs) == len(new_out_shapes)
    assert len(output_bufs) == len(orig_out_shapes)
    # We need to prepare post reshape for each consumer of each output buffer
    for out_buf, new_out_shape, orig_out_shape in zip(output_bufs, new_out_shapes, orig_out_shapes):
        out_buf.set_buf_dims(new_out_shape)
        consumers = out_buf.consumers.copy()
        for idx, consumer in enumerate(consumers):
            post_reshape_op_name = node.op.name + f'_{out_buf.name}_6d_post_reshape_{idx}'
            post_reshape_op = op_adapter.ReshapeOp(
                name=post_reshape_op_name, shape=orig_out_shape
            )
            graph.inject(
                post_reshape_op, input_name=out_buf.name,
                output_name=post_reshape_op_name, consumer_names=[consumer.op.name]
            )


# ------------------------------------------------------------------------------------------------------------------
#   Translations
#   Note: each Optimization Concrete class has at a minimum 1 optimize function. i.e axes_to_spatial_first_order(..)
#         if more is needed for a given op, it needs to register that method_key and implement a function for it.
# ------------------------------------------------------------------------------------------------------------------
def register_layer_optimization(layer_translation):
    """
    For anything decorated with register in this module, the class along with its op_type is registered in
    a TranslationBank
    :param layer_translation: a concrete class for a given optimization
    """
    OptimizationTranslations.register_translation(layer_translation(), layer_translation().op_type)
    return layer_translation


@register_layer_optimization
class OptimizeInputTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InputOp.TRANSLATION_KEY
        self.register_method(EXTRACT_COLOR_TRANSFROM, self.extract_color_transform)

    @staticmethod
    def extract_color_transform(graph):
        """
        Optional Optimization to create separate Op to handle color transformation pre-processing for network
        inputs
        """
        def validate_transformation(nodes_tuple):
            node_ = nodes_tuple[0]
            if node_.op.input_encoding_in != node_.op.input_encoding_out and \
                    node_.op.input_encoding_in not in [InputEncodings.TIME_SERIES, InputEncodings.OTHER]:
                return True
            return False

        sequence = [("input", (), ())]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_transformation)

        for node_tuple in matched_node_list:
            input_node = node_tuple[0]
            # adjust shape for input as that will be the expected shape after transformation
            color_transform_name = input_node.output_names[0] + "_post_transform"
            color_transform_output_shape = input_node.op.shape

            input_buf = graph.get_buffer(input_node.output_names[0])
            old_input_format = input_buf.axis_format
            b, h, w, c = graph.src_axis_order.extract_2d_spatial_dims(input_node.op.shape)
            if input_node.op.input_encoding_in in (InputEncodings.NV21, InputEncodings.NV12):
                # determine expected shape for yuv_(nv21|nv12)(width * height * 3 / 2)
                shape = int(h * w * (3 / 2))
                input_node.op.shape = [input_node.op.shape[0], shape]
                input_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                b, h, w, c = graph.src_axis_order.extract_2d_spatial_dims(input_node.op.shape)
                input_node.op.shape = graph.src_axis_order.format_2d_spatial_output_shape(b, h, w, 4)
            input_buf.set_buf_dims(input_node.op.shape)

            color_transform_op = op_adapter.ColorTransformOp(color_transform_name,
                                                             color_transform_output_shape,
                                                             input_encoding_in=input_node.op.input_encoding_in,
                                                             input_encoding_out=input_node.op.input_encoding_out)
            graph.inject(color_transform_op,
                         input_name=input_node.output_names[0],
                         output_name=color_transform_name,
                         axis_format=old_input_format)
            log_debug2(code_to_message.get_debugging_message("DEBUG_COLOR_TRANSFORM_EXTRACTION")
                       (input_node.op.name, input_node.op.shape, input_node.op.input_encoding_in))

    def axes_to_spatial_first_order(self, node, graph):
        if graph.preserve_io_layout_passed and node.op.name in graph.preserve_layout_tensors:
            return False
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
            buf.axis_format = AxisTracker.AxisFormat.NDHWC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.NCF:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCF_TO_NFC)
            buf.axis_format = AxisTracker.AxisFormat.NFC
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.TNF:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            buf.axis_format = AxisTracker.AxisFormat.NTF
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.OIDHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.OIDHW_TO_DHWIO)
            buf.axis_format = AxisTracker.AxisFormat.DHWIO
            node.op.shape = buf.shape
        elif buf.axis_format == AxisTracker.AxisFormat.OIHW:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.OIHW_TO_HWIO)
            buf.axis_format = AxisTracker.AxisFormat.HWIO
            node.op.shape = buf.shape
        return True


class Optimize1DNNTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.nn_2d_op = None
        self.idx_to_insert = 0

    @staticmethod
    def add_reshape_op(graph, reshape_op_name, output_shape, input_names, output_names, axis_formats=None, idx=-1):
        node = graph.add(op_adapter.ReshapeOp(reshape_op_name, shape=output_shape), input_names, output_names,
                         axis_formats=axis_formats, idx=idx)
        input_buffers = graph.get_input_buffers(node)
        node.op.data_axis_formats = [in_buf.axis_format for in_buf in input_buffers]
        return node

    def setup_for_1d_to_2d_nn_replacement(self, node, graph):
        # compute the correct idx to insert new nodes
        self.idx_to_insert = 0
        for input_name in node.input_names:
            buf = graph.get_buffer(input_name)
            cur_idx = graph.nodes_in_order.index(buf.producer)
            if self.idx_to_insert < cur_idx:
                self.idx_to_insert = cur_idx
        self.idx_to_insert = self.idx_to_insert + 1

    def reshape_to_2d_spatial(self, input_name, node, graph):
        reshape_2d_op_name = node.op.name + '_reshape_to_2d'
        buffer = graph.get_buffer(input_name)
        batch, feature, channel = graph.src_axis_order.extract_1d_spatial_dims(buffer.shape)
        output_shape = graph.src_axis_order.format_2d_spatial_output_shape(batch, 1, feature, channel)
        # add Reshape to transform NN Op input from 1D to 2D spatial dimension
        reshape_node = self.add_reshape_op(graph, reshape_2d_op_name, output_shape, [input_name],
                                           [reshape_2d_op_name], axis_formats=[graph.src_axis_order.get_axis_format(len(output_shape))],
                                           idx=self.idx_to_insert)
        graph.update_trace_info(reshape_node, [node])

        # increment idx_to_insert
        self.idx_to_insert = self.idx_to_insert + 1
        return reshape_2d_op_name

    def reshape_weights(self, input_name, node, graph):
        weight_buffer = graph.get_buffer(input_name)
        if len(weight_buffer.shape) == 3:
            feature, in_channels, out_channels = graph.src_axis_order.extract_conv1d_weights_dims(weight_buffer.shape)
            output_shape = graph.src_axis_order.format_conv2d_weights_output_shape(1, feature, in_channels, out_channels)
            if weight_buffer.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                weight_buffer.producer.op.tensor = weight_buffer.producer.op.tensor.reshape(output_shape)
                weight_buffer.shape = output_shape
                if node.op.type == op_adapter.TransposeConv1dOp.TRANSLATION_KEY:
                    weight_buffer.axis_format = graph.src_axis_order.deconv2d_weights_format
                else:
                    weight_buffer.axis_format = graph.src_axis_order.conv2d_weights_format
                return input_name
            else:
                # add Reshape to transform weight input from 1D to 2D spatial dimension
                weights_reshape_op_name = node.op.name + '_reshaped_weights'
                if node.op.type == op_adapter.TransposeConv1dOp.TRANSLATION_KEY:
                    axis_formats = [graph.src_axis_order.deconv2d_weights_format]
                else:
                    axis_formats = [graph.src_axis_order.conv2d_weights_format]
                reshape_node = self.add_reshape_op(graph, weights_reshape_op_name, output_shape,
                                                   [input_name], [weights_reshape_op_name], axis_formats=axis_formats,
                                                   idx=self.idx_to_insert)
                graph.update_trace_info(reshape_node, [node])
                # increment idx_to_insert
                self.idx_to_insert = self.idx_to_insert + 1
                return weights_reshape_op_name
        # no reshape needed return original input name
        return input_name

    def add_nn_2d_node(self, input_names, node, graph):
        nn_2d_output_name = node.op.name + '_intermediate'
        # add 2D NN Op to the graph
        nn_2d_node = graph.add(self.nn_2d_op, input_names, [nn_2d_output_name], idx=self.idx_to_insert)
        graph.update_trace_info(nn_2d_node, [node])
        # increment idx_to_insert
        self.idx_to_insert = self.idx_to_insert + 1
        # Transfer activation encoding of 1d node to 2d node.
        if graph.has_quantization_param(node.op.name):
            output_encodings = graph.get_layer_quantization_param(node.op.name)[op_graph.QuantParams.OUTPUT_ENCODINGS]
            if len(output_encodings) > 0:
                output_encoding = output_encodings[0].copy()
                output_encoding['name'] = nn_2d_output_name
                graph.add_quantization_params(self.nn_2d_op.name, output_encodings=output_encoding)
        return nn_2d_output_name

    def reshape_to_1d_spatial(self, nn_2d_output_name, output_names, nn1d_consumers, node, graph, pos=None):
        nn_2d_output_buffer = graph.get_buffer(nn_2d_output_name)
        batch, height, width, channel = graph.src_axis_order.extract_2d_spatial_dims(nn_2d_output_buffer.shape)
        output_shape = graph.src_axis_order.format_1d_spatial_output_shape(batch, width, channel)
        # add Reshape to transform NN Op output from 2D back to 1D spatial dimension
        reshape_node = self.add_reshape_op(graph, nn_2d_output_name, output_shape, [nn_2d_output_name],
                                           [output_names[0]], axis_formats=[graph.src_axis_order.get_axis_format(len(output_shape))],
                                           idx=self.idx_to_insert)
        graph.update_trace_info(reshape_node, [node])

        # add back 1D NN node's consumers to reshape's output buffer
        output_buffer = graph.get_buffer(output_names[0])
        for i, c in enumerate(nn1d_consumers):
            if pos == None:
                c.input_names.insert(0, output_names[0])
            else:
                c.input_names.insert(pos[i], output_names[0])
            output_buffer.consumers.add(c)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        pos = []
        nn_2d_input_names = node.input_names[:]
        nn1d_consumers = graph.get_buffer(node.output_names[0]).consumers

        self.setup_for_1d_to_2d_nn_replacement(node, graph)

        # reshape nn_1d inputs and update input_names for nn_2d op
        nn_2d_input_names[0] = self.reshape_to_2d_spatial(node.input_names[0], node, graph)
        if len(node.input_names) > 1:
            # reshape weights if applicable
            nn_2d_input_names[1] = self.reshape_weights(node.input_names[1], node, graph)

        # add 2d variant for nn_op as intermediate op
        nn_2d_output_name = self.add_nn_2d_node(nn_2d_input_names, node, graph)
        for _ , c in enumerate(nn1d_consumers):
            pos.append(c.input_names.index(node.output_names[0]))
        # prune the 1D NN node
        graph.prune(node, force_remove=True)

        # post reshape to mimic nn_1d output shape
        self.reshape_to_1d_spatial(nn_2d_output_name, node.output_names, nn1d_consumers, node, graph, pos)


@register_layer_optimization
class OptimizeArgOpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ArgOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            axis_map = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Optimize special case that channel dimension is removed
                if node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                input_axis_formats_before = graph.get_input_axis_formats(node)
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                input_axis_formats_after = graph.get_input_axis_formats(node)
                input_buffers = graph.get_input_buffers(node)
                for i, buf in enumerate(input_buffers):
                    if input_axis_formats_before[i] != input_axis_formats_after[i]:
                        transpose_node = buf.producer
                        graph.update_trace_info(transpose_node, [node])
                        graph.update_trace_info(buf, [node])
                node.op.axis = axis_map[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            axis_map = AxisTracker.AxisFormat.NSC_TO_NCS
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Optimize special case that channel dimension is removed
                if node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                input_axis_formats_before = graph.get_input_axis_formats(node)
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                input_axis_formats_after = graph.get_input_axis_formats(node)
                input_buffers = graph.get_input_buffers(node)
                for i, buf in enumerate(input_buffers):
                    if input_axis_formats_before[i] != input_axis_formats_after[i]:
                        transpose_node = buf.producer
                        graph.update_trace_info(transpose_node, [node])
                        graph.update_trace_info(buf, [node])
                node.op.axis = axis_map[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            axis_map = AxisTracker.AxisFormat.NFC_TO_NCF
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Channel dimension is removed
                if node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NF
                # Feature dimension is removed
                elif node.op.axis == 2:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NC
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                                  AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                input_axis_formats_before = graph.get_input_axis_formats(node)
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                input_axis_formats_after = graph.get_input_axis_formats(node)
                input_buffers = graph.get_input_buffers(node)
                for i, buf in enumerate(input_buffers):
                    if input_axis_formats_before[i] != input_axis_formats_after[i]:
                        transpose_node = buf.producer
                        graph.update_trace_info(transpose_node, [node])
                        graph.update_trace_info(buf, [node])
                node.op.axis = axis_map[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            axis_map = AxisTracker.AxisFormat.NTF_TO_TNF
            # If keep dims is False we must permute as it will remove dimensions
            if not node.op.keep_dims:
                # Time dimension is removed
                if node.op.axis == 0:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NF
                # Batch dimension is removed
                elif node.op.axis == 1:
                    node.op.axis = axis_map[node.op.axis]
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                                  AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
            else:
                input_axis_formats_before = graph.get_input_axis_formats(node)
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                input_axis_formats_after = graph.get_input_axis_formats(node)
                input_buffers = graph.get_input_buffers(node)
                for i, buf in enumerate(input_buffers):
                    if input_axis_formats_before[i] != input_axis_formats_after[i]:
                        transpose_node = buf.producer
                        graph.update_trace_info(transpose_node, [node])
                        graph.update_trace_info(buf, [node])
                node.op.axis = axis_map[node.op.axis]
        else:
            # Add warning message for other axis formats
            log_warning("No need to handle other axis formats now, but might optimize them in the future.")
        return True


@register_layer_optimization
class OptimizeAxisAlignedBboxTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.AxisAlignedBboxTransformOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeBatchnormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchnormOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if 1 < input_buf.rank() <= 5:
            input_axis_formats_before = graph.get_input_axis_formats(node)
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            input_axis_formats_after = graph.get_input_axis_formats(node)
            input_buffers = graph.get_input_buffers(node)
            for i, buf in enumerate(input_buffers):
                if input_axis_formats_before[i] != input_axis_formats_after[i]:
                    transpose_node = buf.producer
                    graph.update_trace_info(transpose_node, [node])
                    graph.update_trace_info(buf, [node])
            output_buffer = graph.get_output_buffers(node)[0]
            # (image/feature)_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
            # Enforce the output format here to be NDHWC/NSC/NFC
            output_buffer.axis_format = AxisOrder().get_axis_format(len(output_buffer.shape))
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_BATCHNORM_DIM_UNSUPPORTED")(input_buf.rank(),
                                                                                                  node.op.name))
        return True

    def merge_low_level_ops_to_layers(self, graph):
        def validate(nodes_tuple_):
            OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS = [
                op_adapter.Conv2dOp.TRANSLATION_KEY,
                op_adapter.TransposeConv2dOp.TRANSLATION_KEY,
                op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY,
                op_adapter.FullyConnectedOp.TRANSLATION_KEY
            ]
            prod_node = nodes_tuple_[1]
            prod_node_input_op = graph.get_producer_op(prod_node.input_names[0])
            # previous must not be a Batchnorm and previous node must be a nn_node for sequence to match batchnorm
            if prod_node_input_op.type in [op_adapter.BatchnormOp.TRANSLATION_KEY, op_adapter.InstanceNormOp.TRANSLATION_KEY] or \
                    prod_node_input_op.type not in OPS_HAVING_WEIGHTS_AND_BIASES_AS_INPUTS:
                return False

            mul_const_ip_node_ = nodes_tuple_[0]
            add_const_ip_node_ = nodes_tuple_[2]
            # batchnorm nodes require 1D weights/biases
            mul_const_ip_ = np.atleast_1d(mul_const_ip_node_.op.tensor.squeeze())
            add_const_ip_ = np.atleast_1d(add_const_ip_node_.op.tensor.squeeze())
            if len(mul_const_ip_.shape) != 1 or len(add_const_ip_.shape) != 1:
                return False

            # if Mul has output_encodings squashing is disabled for better alignment
            # with the expected/simulated accuracy
            # Note: This is not necessarily better accuracy
            if graph.has_quantization_param(prod_node.op.name) and \
                graph.quantization_params[prod_node.op.name]["output_encodings"]:
                return False
            return True

        sequence = [
                    ("constant", (), ()),
                    ("elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
                        ()),
                    ("constant", (), ()),
                    ("elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0),
                                                 ("constant", 1)]), ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            mul_const_ip_node = nodes_tuple[0]
            mul_node = nodes_tuple[1]
            add_const_ip_node = nodes_tuple[2]
            add_node = nodes_tuple[3]

            # batchnorm nodes require 1D weights/biases
            mul_const_ip = np.atleast_1d(mul_const_ip_node.op.tensor.squeeze())
            add_const_ip = np.atleast_1d(add_const_ip_node.op.tensor.squeeze())

            # Squashes the add node
            add_const_trace_info = graph.get_trace_info(add_const_ip_node)
            add_const_output_trace_data = graph.get_trace_info(graph.get_buffer(add_const_ip_node.output_names[0]))
            add_input_buffer = graph.get_input_buffers(add_node)[0]
            graph.squash(add_node, input_name=add_input_buffer.name)

            # Remove mul_node as consumer of const node's buffer
            graph.get_buffer(mul_const_ip_node.output_names[0]).consumers.remove(mul_node)
            # Remove const node from mul_node's input names
            mul_node.input_names.remove(mul_const_ip_node.output_names[0])
            # Change the weight/bias to a constant node
            weights_name = mul_node.op.name + "_bn_w"
            bias_name = mul_node.op.name + "_bn_b"
            weights_constant_op = op_adapter.ConstantOp(weights_name, tensor=mul_const_ip)
            bias_constant_op = op_adapter.ConstantOp(bias_name, tensor=add_const_ip)
            weights_constant_node = graph.add(weights_constant_op, [], [weights_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.update_trace_info(weights_constant_node, [mul_const_ip_node, graph.get_buffer(mul_const_ip_node.output_names[0])])
            bias_constant_node = graph.add(bias_constant_op, [], [bias_name], axis_formats=[AxisTracker.AxisFormat.ANY])
            graph.set_trace_info(bias_constant_node, add_const_trace_info)
            graph.set_trace_info(bias_constant_node, add_const_output_trace_data)
            # Replace the mul node to an batchnorm node

            batchnorm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.BatchnormOp.TRANSLATION_KEY,
                                                                        op_adapter.BatchnormOp.LEGACY_TRANSLATION_KEY,
                                                                        folded_op=True)
            batchnorm_op = op_adapter.BatchnormOp(batchnorm_op_name)

            graph.replace(mul_node.op, batchnorm_op)
            batchnorm_node = graph.nodes_by_name[batchnorm_op.name]
            batchnorm_node.input_names.append(weights_name)
            batchnorm_node.input_names.append(bias_name)
            graph.get_buffer(weights_name).consumers.add(batchnorm_node)
            graph.get_buffer(bias_name).consumers.add(batchnorm_node)

    @staticmethod
    def squash_batchnorm(graph):
        def validate(nodes_tuple):
            if graph.is_output_node(nodes_tuple[0]):
                # This means we reached end of graph but the pattern to
                # be matched is still has some nodes.
                log_debug2("This node {} is an output node of the graph. Skipping...", nodes_tuple[0])
                return False
            bn_node_ = next(iter(graph.get_output_buffers(nodes_tuple[0])[0].consumers))
            bn_input_buffer_ = graph.get_input_buffers(bn_node_)[0]
            return bn_node_.op.type == op_adapter.BatchnormOp.TRANSLATION_KEY and bn_input_buffer_.rank() >= 4

        sequences = [[("Conv2d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))],
                     [("DepthWiseConv2d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))],
                     [("TransposeConv2d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))],
                     [("TransposeConv3d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))],
                     [("Conv3d",
                       ("MATCH_BUFS_AT_INDEX", [("constant", 1),
                                                ("constant", 2)]),
                       ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")]))]
                     ]

        for sequence in sequences:
            for node_tuple in graph.get_matched_nodes(sequence, validator=validate):
                # sanity check
                log_assert(len(node_tuple) == len(sequence),
                           "Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                           len(node_tuple), len(sequence))

                conv_node = node_tuple[0]
                bn_node = next(iter(graph.get_output_buffers(conv_node)[0].consumers))
                bn_input_buffer = graph.get_input_buffers(bn_node)[0]
                bn_node_weights = graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
                bn_node_bias = graph.get_buffer(bn_node.input_names[2]).producer.op.tensor

                manage_shared_static_input(graph, conv_node, 1)
                conv_node_weights_buffer = graph.get_buffer(conv_node.input_names[1])
                conv_node_weights_op = conv_node_weights_buffer.producer.op
                conv_node_weights = conv_node_weights_op.tensor

                # Extract bias from ConstantOp
                manage_shared_static_input(graph, conv_node, 2)
                conv_node_bias_op = graph.get_buffer(conv_node.input_names[2]).producer.op
                conv_node_bias = conv_node_bias_op.tensor

                if conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.OIDHW:
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_conv3d_weights_to_ir)
                    weights = weights * bn_node_weights
                    weights = np.transpose(weights, graph.src_axis_order.permute_conv3d_weights_from_ir)
                elif conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.OIHW:
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_conv2d_weights_to_ir)
                    weights = weights * bn_node_weights
                    weights = np.transpose(weights, graph.src_axis_order.permute_conv2d_weights_from_ir)
                elif conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.IOHW:
                    #         - bn_weight_shape (bn_shape = G * O): the size of batchnorm op's weight
                    #         - conv_group (G): the conv op's num_group
                    #
                    #   weights: (H, W, I, O)          ---(reshape)-->    weights_:                      (H, W, G, I/G, O)
                    #                                                                                           |    |  |
                    #   bn_node_weights: (bn_shape)    ---(reshape)-->    bn_node_weights_:              (      G,   1, O)
                    #                                                                                           |    |  |
                    #   weights(new): (H, W, I, O)     <--(reshape)---    weights_ * bn_node_weights_:   (H, W, G, I/G, O)
                    bn_weight_shape = bn_node_weights.shape[0]
                    conv_group = conv_node.op.group
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_deconv2d_weights_to_ir)
                    height, width, input_channel, output_channel = weights.shape
                    weights_ = weights.reshape(height, width, conv_group, input_channel//conv_group, output_channel)
                    bn_node_weights_ = bn_node_weights.reshape(conv_group, 1, bn_weight_shape//conv_group)
                    weights_ = weights_ * bn_node_weights_
                    weights = weights_.reshape(height, width, input_channel, output_channel)
                    weights = np.transpose(weights, graph.src_axis_order.permute_deconv2d_weights_from_ir)
                elif conv_node_weights_buffer.axis_format == AxisTracker.AxisFormat.IODHW:
                    #         - bn_weight_shape (bn_shape = G * O): the size of batchnorm op's weight
                    #         - conv_group (G): the conv op's num_group
                    #
                    #   weights: (D, H, W, I, O)          ---(reshape)-->    weights_:                      (D, H, W, G, I/G, O)
                    #                                                                                                 |    |  |
                    #   bn_node_weights: (bn_shape)       ---(reshape)-->    bn_node_weights_:              (         G,   1, O)
                    #                                                                                                 |    |  |
                    #   weights(new): (D, H, W, I, O)     <--(reshape)---    weights_ * bn_node_weights_:   (D, H, W, G, I/G, O)
                    bn_weight_shape = bn_node_weights.shape[0]
                    conv_group = conv_node.op.group
                    weights = np.transpose(conv_node_weights, graph.src_axis_order.permute_deconv3d_weights_to_ir)
                    depth, height, width, input_channel, output_channel = weights.shape
                    weights_ = weights.reshape(depth, height, width, conv_group, input_channel//conv_group, output_channel)
                    bn_node_weights_ = bn_node_weights.reshape(conv_group, 1, bn_weight_shape//conv_group)
                    weights_ = weights_ * bn_node_weights_
                    weights = weights_.reshape(depth, height, width, input_channel, output_channel)
                    weights = np.transpose(weights, graph.src_axis_order.permute_deconv3d_weights_from_ir)
                else:
                    weights = conv_node_weights * bn_node_weights

                conv_node_weights_op.tensor = weights
                conv_node_bias = np.atleast_1d(
                    (conv_node_bias * bn_node_weights + bn_node_bias).squeeze())
                conv_node_bias_op.tensor = conv_node_bias.copy()

                # add cached bn parameters as conv node supplemental attributes before squashing
                if graph.has_quantization_param(bn_node.op.name) and \
                        (conv_node.op.type == op_adapter.Conv2dOp.TRANSLATION_KEY or \
                            conv_node.op.type == op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY):
                    gamma_data = graph.quantization_params[bn_node.op.name]["bn_params"]["gamma"]
                    beta_data = graph.quantization_params[bn_node.op.name]["bn_params"]["beta"]
                    gamma = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_PARAM_BN_GAMMA,
                                                    gamma_data.shape,
                                                    gamma_data,
                                                    ir_graph.QNN_DATATYPE_FLOAT_32)
                    beta = ir_graph.IrStaticTensor(ir_graph.IR_OP_CONV_PARAM_BN_BETA,
                                                   beta_data.shape,
                                                   beta_data,
                                                   ir_graph.QNN_DATATYPE_FLOAT_32)
                    attrs = conv_node.op.c_op.attrs
                    attrs.add(ir_graph.IR_OP_CONV_PARAM_BN_GAMMA, gamma, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
                    attrs.add(ir_graph.IR_OP_CONV_PARAM_BN_BETA, beta, ir_graph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)

                graph.squash(bn_node, input_name=bn_input_buffer.name)
                log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                           conv_node.op.type,
                                                                                           conv_node.op.name))

    def prepare_inputs_as_params(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        bias_buffer = graph.get_buffer(node.input_names[2])
        bias_node = bias_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
            node.op.bias = bias_node.op.tensor
            # Remove the weights/bias inputs from the IR graph
            graph.remove_node_as_consumer(node, weights_buffer.name)
            graph.remove_node_as_consumer(node, bias_buffer.name)
            node.input_names = [node.input_names[0]]


@register_layer_optimization
class OptimizeBoxWithNmsLimitTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BoxWithNmsLimitOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeBatchPermutationTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BatchPermutationOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeCastTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CastOp.TRANSLATION_KEY
        self.register_method(REMOVE_CAST_IDENTITY, self.remove_identity)
        self.register_method(FOLD_CAST, self.fold_cast)

    @staticmethod
    def remove_identity(node, graph, force_prune=True):
        # TODO Properly identify and remove casts once datatypes are trackable in IR
        if node.op.from_type == node.op.to_type or force_prune:
            graph.squash_identity(node, is_data_movement_node=True)

    @staticmethod
    def fold_cast(graph):
        # scenario : one cast, one back to back cast type consumer.
        # in_tensor -> cast_0 -> cast_1 -> out_tensor
        # scenario_2 : one cast, two or more back to back cast type consumers with the same output dtype.
        # in_tensor -> cast_0 -> cast_1 -> out_tensor_1
        #                     -> cast_2 -> out_tensor_2
        sequence = [
                ("cast",
                    (),
                    ()
                ),
                ("cast",
                    ("MATCH_NUM_BUFS", [("cast", "ALL")]),
                    ()
                )
                ]

        matched_node_list = graph.get_matched_nodes(sequence)
        for node_tuple in matched_node_list:
            cast_node, _ = node_tuple
            cast_node_output_buf = graph.get_output_buffers(cast_node)[0]
            cast_op_name = cast_node.op.name
            cast_from_dtype = cast_node.op.from_type
            cast_node_input_names = cast_node.input_names

            # transform set to list
            consumers = list(cast_node_output_buf.consumers)
            if len(consumers) >= 1 and \
                all([consumer.op.type == op_adapter.CastOp.TRANSLATION_KEY for consumer in consumers]):
                # check if cast_node has override encodings
                if graph.has_quantization_param(cast_op_name):
                    continue
                # check if all the consumers' op type is cast then squash the first cast node in each node_tuple.
                status = graph.squash(cast_node, cast_node_input_names[0])
                # change the from_dtype of all the Casts in consumer list to the from_dtype of the squash Cast.
                for i in range(len(consumers)):
                    consumers[i].op.from_type = cast_from_dtype
                if status:
                    log_debug2("Squash Cast: {}".format(cast_op_name))


@register_layer_optimization
class OptimizeChannelShuffleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ChannelShuffleOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        ret = super().axes_to_spatial_first_order(node, graph)
        if not ret:
            return ret
        else:
            buf = graph.get_buffer(node.input_names[0])
            data_axis_format = node.op.data_axis_formats[0]
            if buf.axis_format == AxisTracker.AxisFormat.NSC and \
               data_axis_format == AxisTracker.AxisFormat.NCS and \
               node.op.axis == 1:
                node.op.axis = 3
            elif buf.axis_format == AxisTracker.AxisFormat.NCS and \
                data_axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.axis == 3:
                node.op.axis = 1
            else:
                raise ValueError("input axis format={}, data_axis_format={} and axis = {} " \
                    "of channelshuffle node are not valid".format(buf.axis_format,
                                                                  data_axis_format,
                                                                  node.op.axis))
            return True


@register_layer_optimization
class OptimizeCollectRpnProposalsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CollectRpnProposalsOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeColorTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ColorTransformOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        buf = graph.get_buffer(node.output_names[0])
        if buf.axis_format == AxisTracker.AxisFormat.NCS:
            buf.shape = AxisTracker.permute_shape(buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            buf.axis_format = AxisTracker.AxisFormat.NSC
            return True
        return False


@register_layer_optimization
class OptimizeConvolutionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Conv2dOp.TRANSLATION_KEY
        self.register_method(PREPARE_BIASES, self.prepare_biases)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_conv2d_weights_dims(weights_buffer.shape.dims)[-1]
        add_or_broadcast_bias(node, graph, output_channel)

    def prepare_inputs_as_params(self, node, graph):
        prepare_conv_inputs_as_params(graph, node)

    def axes_to_spatial_first_order(self, node, graph):
        if isinstance(graph.src_axis_order, (CaffeAxisOrder, SpatialLastAxisOrder)):
            input_buffers = graph.get_input_buffers(node)
            input_axis_formats = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NDHWC, transpose it to OIDHW by using a transpose to NCDHW. Then, to DHWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NDHWC,
                                         AxisTracker.AxisFormat.OIDHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 5):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCDHW, which is actually taking us back to OIDHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.OIDHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to DHWIO from OIDHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.DHWIO,
                                              graph.src_axis_order.permute_conv3d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.DHWIO

                # Update input_buffers and input_axis_formats after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_axis_formats = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NSC, transpose it to OIHW by using a transpose to NCS. Then, to HWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.OIHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 4):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCS, which is actually taking us back to OIHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_buffers[1].axis_format = AxisTracker.AxisFormat.OIHW

                # Inject an implicit permute to HWIO from OIHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.HWIO,
                                              graph.src_axis_order.permute_conv2d_weights_to_ir, [node.op.name])

                # Update input_buffers and input_axis_formats after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_buffers[1].axis_format = AxisTracker.AxisFormat.HWIO
                input_axis_formats = [buf.axis_format for buf in input_buffers]

            if any(axis_format in input_axis_formats for axis_format in [AxisTracker.AxisFormat.NDHWC,
                                                                         AxisTracker.AxisFormat.DHWIO,
                                                                         AxisTracker.AxisFormat.NSC,
                                                                         AxisTracker.AxisFormat.HWIO,
                                                                         AxisTracker.AxisFormat.ANY,
                                                                         AxisTracker.AxisFormat.NONTRIVIAL]):
                AxisTracker.image_to_channel_last_order(node, graph)
                output_buffer = graph.get_output_buffers(node)[0]
                # image_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
                # Enforce the output format here according to output buffer's rank
                output_buffer.axis_format = AxisOrder().get_axis_format(output_buffer.rank())
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_CONVOLUTION_UNEXPECTED_INPUT_ORDER")
                                 (input_axis_formats))
            return True


@register_layer_optimization
class OptimizeConvolution1DTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.Conv1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        conv_op_name = node.op.name + "_2d"
        self.nn_2d_op = op_adapter.Conv2dOp(conv_op_name,
                                            bias_op_name=node.op.bias_op_name,
                                            padx_before=node.op.pad_amount[0],
                                            padx_after=node.op.pad_amount[1],
                                            pady_before=0,
                                            pady_after=0,
                                            padding_size_strategy=node.op.padding_size_strategy,
                                            stridey=1,
                                            stridex=node.op.stride[0],
                                            dilationy=1,
                                            dilationx=node.op.dilation[0],
                                            groups=node.op.group,
                                            data_layout=AxisTracker.AxisFormat.NCS)
        super().expand_1d_spatial_nn_nodes(node, graph)


@register_layer_optimization
class OptimizeConvolution3DTranslation(OptimizeConvolutionTranslation):
    def __init__(self):
        OptimizeConvolutionTranslation.__init__(self)
        self.op_type = op_adapter.Conv3dOp.TRANSLATION_KEY

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_conv3d_weights_dims(weights_buffer.shape)[-1]
        add_or_broadcast_bias(node, graph, output_channel)


@register_layer_optimization
class OptimizeConcatTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConcatOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)

    def axes_to_spatial_first_order(self, node, graph):
        self.axes_to_spatial_first_order_for_aisw(node, graph)

    def axes_to_spatial_first_order_for_aic(self, node, graph):
        if AxisTracker.AxisFormat.NONTRIVIAL not in graph.get_input_axis_formats(node):
            ret = super().axes_to_spatial_first_order(node, graph)
            if not ret:
                # If ret is False, no change happened in super(), no futher action is needed, so return
                return ret
        buf = graph.get_buffer(node.output_names[0])

        # permute axis if input is permuted
        input_axis_formats = graph.get_input_axis_formats(node)
        # assert that axis formats of all inputs match
        first_in_format = input_axis_formats[0]
        if not all([in_format == first_in_format for in_format in input_axis_formats]):
            input_bufs = graph.get_input_buffers(node)
            input_ranks = [input_buf.rank() for input_buf in input_bufs]
            first_input_rank = input_ranks[0]
            if not all([input_rank == first_input_rank for input_rank in input_ranks]):
                raise ValueError("ranks of all inputs are not matched: {}".format(input_ranks))
            elif AxisTracker.AxisFormat.NONTRIVIAL not in input_axis_formats:
                raise ValueError("axis formats of all inputs are not matched: {}".format(input_axis_formats))
            else:
                for i in range(len(node.input_names)):
                    input_name = node.input_names[i]
                    input_buf = graph.get_buffer(input_name)
                    if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
                        graph.inject_implicit_permute(
                            input_buf.name,
                            AxisTracker.AxisFormat.NONTRIVIAL,
                            spatial_first_format_to_channel_first_permute_order[input_buf.axis_format],
                            consumers=[node.op.name]
                        )
                    else:
                        # for NONTRIVIAL, ANY, NC, NF, and channel first orders
                        # we should directly pass without modification
                        pass
                # after aligning all input axis formats, refresh the input_axis_formats variable
                input_axis_formats = graph.get_input_axis_formats(node)
                # set output buffer's axis format to NONTRIVIAL
                output_buf = graph.get_output_buffers(node)[0]
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        # check whether the concat needs to be transformed into IR format
        need_transform_axis = False
        for data_axis_format, input_axis_format in zip(node.op.data_axis_formats, input_axis_formats):
            # if input buffer is NONTRIVIAL, we don't need to transform axis
            # if the axis_format of input buffer equals data_axis_format, we also don't need to transform axis
            if input_axis_format != AxisTracker.AxisFormat.NONTRIVIAL and data_axis_format != input_axis_format:
                need_transform_axis = True
                # stop for loop once we found input axis format changed
                break

        spatial_last_axis_formats = [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF, AxisTracker.AxisFormat.TNF]
        spatial_first_axis_formats = [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC, AxisTracker.AxisFormat.NTF]
        if need_transform_axis:
            if (data_axis_format, input_axis_format) in zip(spatial_last_axis_formats[:2], spatial_first_axis_formats[:2]):
                axis_map = SpatialLastAxisOrder().permute_sequence_from_ir[buf.rank() - 1]
            elif (data_axis_format, input_axis_format) in (spatial_last_axis_formats[-1], spatial_first_axis_formats[-1]):
                axis_map = SpatialLastAxisOrder().permute_time_series_from_ir
            elif (data_axis_format, input_axis_format) in (spatial_first_axis_formats[:2], spatial_last_axis_formats[:2]):
                axis_map = AxisOrder().permute_sequence_from_ir[buf.rank() - 1]
            elif (data_axis_format, input_axis_format) in (spatial_first_axis_formats[-1], spatial_last_axis_formats[-1]):
                axis_map = AxisOrder().permute_time_series_from_ir
            else:
                if buf.axis_format == AxisTracker.AxisFormat.NTF:
                    axis_map = graph.src_axis_order.permute_time_series_from_ir
                else:
                    axis_map = graph.src_axis_order.permute_sequence_from_ir[buf.rank() - 1]

            node.op.axis = axis_map[node.op.axis]

        # shape assertion
        ref_inp_shape = graph.get_buffer(node.input_names[0]).shape
        for input_buf in graph.get_input_buffers(node)[1:]:
            inp_shape = input_buf.shape
            for i in range(len(ref_inp_shape)):
                if i != node.op.axis and ref_inp_shape[i] != inp_shape[i]:
                    raise ValueError("input shapes of concat op {} not aligned: {}, {} while axis = {}"
                                     .format(node.op.name, ref_inp_shape, inp_shape, node.op.axis))
        return True

    def axes_to_spatial_first_order_for_aisw(self, node, graph):
        # Scenario_1 : Two inputs are Channel_first and Channel_last in src model, e.g data_axis_formats = [NCF, NFC]
        # Solution : Revert current input_axis_formats to data_axis_formats to avoid the shape mismatch.
        # Scenario_2 : Data_axis_formats are the same and 'Nontrivial' is not included in current input_axis_formats,
        # Solution : Alert the input_axis_formats to IR Order and change axis accordingly.
        # Scenario_3 : Data_axis_formats are the same and at least one of current input_axis_formats is 'Nontrivial',
        # Solution : Revert other input_axis_formats to data_axis_formats to follow 'Nontrivial' input.
        if AxisTracker.input_axis_formats_intact(graph, node):
            # Nothing to do in this case
            return False

        has_changed = False
        # Incase the input_axis_formats were already different in translation stage which means data_axis_formats are different
        # Revert to original axis_formats for each input
        # Ignore 'ANY' to compare other layouts
        checked_axis_formats = copy.deepcopy(node.op.data_axis_formats)
        data_axis_formats = node.op.data_axis_formats
        if AxisTracker.AxisFormat.ANY in checked_axis_formats:
            checked_axis_formats.remove(AxisTracker.AxisFormat.ANY)
        if len(checked_axis_formats) > 1:
            first_data_format = checked_axis_formats[0]
            if not all([data_format == first_data_format for data_format in checked_axis_formats]):
                input_bufs = graph.get_input_buffers(node)
                for idx, buf in enumerate(input_bufs):
                    # fetch input buffers one by one to revert to original axis format.
                    # keep the input shapes and node.op.axis
                    ret = AxisTracker.revert_input_axis_format(graph,
                                                               node,
                                                               buf.name,
                                                               buf.axis_format,
                                                               data_axis_formats[idx])
                    has_changed = has_changed or ret
                return has_changed

        # Normal case
        pre_input_formats = graph.get_input_axis_formats(node)
        if AxisTracker.AxisFormat.NONTRIVIAL not in pre_input_formats:
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            input_axis_formats = graph.get_input_axis_formats(node)
            if pre_input_formats != input_axis_formats:
                has_changed = True

            # check whether the concat has transformed into IR format
            need_transform_axis = False
            for data_axis_format, input_axis_format in zip(data_axis_formats, input_axis_formats):
                # if the axis_format of input buffer equals data_axis_format, we also don't need to transform axis
                if data_axis_format != input_axis_format:
                    need_transform_axis = True
                    # stop for loop once we found input axis format changed
                    break

            buf = graph.get_buffer(node.output_names[0])
            spatial_last_axis_formats = [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF, AxisTracker.AxisFormat.TNF]
            spatial_first_axis_formats = [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC, AxisTracker.AxisFormat.NTF]
            if need_transform_axis:
                if (data_axis_format, input_axis_format) in zip(spatial_last_axis_formats[:3], spatial_first_axis_formats[:3]):
                    axis_map = SpatialLastAxisOrder().permute_sequence_from_ir[buf.rank() - 1]
                elif (data_axis_format, input_axis_format) == (spatial_last_axis_formats[-1], spatial_first_axis_formats[-1]):
                    axis_map = SpatialLastAxisOrder().permute_time_series_from_ir
                elif (data_axis_format, input_axis_format) in zip(spatial_first_axis_formats[:3], spatial_last_axis_formats[:3]):
                    axis_map = AxisOrder().permute_sequence_from_ir[buf.rank() - 1]
                elif (data_axis_format, input_axis_format) == (spatial_first_axis_formats[-1], spatial_last_axis_formats[-1]):
                    axis_map = AxisOrder().permute_time_series_from_ir
                else:
                    if buf.axis_format == AxisTracker.AxisFormat.NTF:
                        axis_map = graph.src_axis_order.permute_time_series_from_ir
                    else:
                        axis_map = graph.src_axis_order.permute_sequence_from_ir[buf.rank() - 1]

                node.op.axis = axis_map[node.op.axis]

        else:
            # 'NONTRIVIAL' in input_buf_formats, regard 'NONTRIVIAL' the same as data_axis_format
            # revert other format to data_axis_formats
            input_bufs = graph.get_input_buffers(node)
            for idx, buf in enumerate(input_bufs):
                # fetch input buffers one by one to revert to original axis format.
                # keep the input shapes and node.op.axis
                ret = AxisTracker.revert_input_axis_format(graph,
                                                           node,
                                                           buf.name,
                                                           buf.axis_format,
                                                           data_axis_formats[idx])
                has_changed = has_changed or ret

        # shape assertion
        ref_inp_shape = graph.get_buffer(node.input_names[0]).shape
        for input_buf in graph.get_input_buffers(node)[1:]:
            inp_shape = input_buf.shape
            for i in range(len(ref_inp_shape)):
                if i != node.op.axis and not ref_inp_shape.is_equal_at(i, inp_shape, i):
                    raise ValueError("Input shapes of concat op {} not aligned: {}, {} while axis = {}"
                                    .format(node.op.name, ref_inp_shape, inp_shape, node.op.axis))

        return has_changed

    @staticmethod
    def fold_concats(graph):
        def validate_concat_axis(nodes_tuple):
            concat_node_ = nodes_tuple[0]
            concat_node_input_bufs_ = graph.get_input_buffers(concat_node_)
            for buf_ in concat_node_input_bufs_:
                if buf_.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_node_ = buf_.producer
                    # only fold concats with same axis
                    if prev_concat_node_.op.axis != concat_node_.op.axis:
                        log_debug2("Found concat node({}) with a concat input, but axis does not match for input ({}), "
                                   "{} != {} ", concat_node_.op.name, prev_concat_node_.op.name,
                                   prev_concat_node_.op.axis, concat_node_.op.axis)
                        return False
            # If concat node output is also a graph output, don't do folding
            if concat_node_.output_names[0] in graph.output_names:
                return False

            return True

        sequence = [
                    ("Concat",
                     ("FLEXIBLE_NUM_BUFS", [("Concat", "ANY")]),
                     ()
                     )
                   ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_concat_axis)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_node_input_bufs = graph.get_input_buffers(concat_node)

            for buf in concat_node_input_bufs:
                if buf.producer.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    prev_concat_buf = buf  # for readability
                    prev_concat_node = prev_concat_buf.producer

                    # remove prev concat as input from current concat and replace with prev concat's input names
                    prev_concat_inputs = prev_concat_node.input_names
                    idx = concat_node.input_names.index(prev_concat_buf.name)
                    concat_node.input_names.remove(prev_concat_buf.name)
                    # extend the inputs in the same index as prev concat
                    concat_node.input_names[idx:idx] = prev_concat_inputs
                    # update the concat.op.data_axis_formats since inputs are updated
                    concat_node.op.data_axis_formats.pop(idx)
                    concat_node.op.data_axis_formats[idx:idx] = prev_concat_node.op.data_axis_formats

                    if concat_node in prev_concat_buf.consumers:
                        prev_concat_buf.consumers.remove(concat_node)

                    # we can prune the prev concat node if the current concat was the only consumer.
                    if len(prev_concat_buf.consumers) == 0 and graph.has_node(prev_concat_node.op.name):
                        graph.prune(prev_concat_node, True)

                    # remove prev concat as consumer for prev concat's input bufs and replace with current concat
                    for input_name in prev_concat_inputs:
                        input_buf = graph.get_buffer(input_name)
                        input_buf.consumers.add(concat_node)

                    log_debug2(code_to_message.get_debugging_message("DEBUG_CONCAT_FOLD")(prev_concat_node.op.name,
                                                                                          concat_node.op.name))

            concat_node_input_bufs = graph.get_input_buffers(concat_node)
            concat_node.op.data_axis_formats = [in_buf.axis_format for in_buf in concat_node_input_bufs]


@register_layer_optimization
class OptimizeConstantTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConstantOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)
        self.register_method(CAST_FP16_TO_FP32, self.cast_fp16_to_fp32)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_buffer(node.output_names[0])
        output_rank = output_buf.rank()

        # TODO Remove this code once limitations of AxisTracking are resolved
        # If the consumer of this buffer has another input with NSC format, and this buffer is 3D, it needs to be
        # padded with a 1 and have its constant operation permuted
        consumers = list(output_buf.consumers)
        if len(consumers) and output_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            consumer_has_dimension_mismatch = [False] * len(consumers)
            for i, consumer in enumerate(consumers):
                if not isinstance(consumer.op, op_adapter.ElementwiseBinaryOp):
                    continue
                for input_buffer in graph.get_input_buffers(consumer):
                    if input_buffer.axis_format == AxisTracker.AxisFormat.NDHWC and output_rank == 4:
                        consumer_has_dimension_mismatch[i] = True
                        break
                    elif input_buffer.axis_format == AxisTracker.AxisFormat.NSC and output_rank == 3:
                        consumer_has_dimension_mismatch[i] = True
                        break

            if all(consumer_has_dimension_mismatch):
                log_debug("All consumers of {} node {} have {}D-{}D rank mismatch in inputs. Updating buffer {}.".format(
                    node.op.type, node.op.name, output_rank+1, output_rank, output_buf.name))
                # Capture tensor and prepare for placement in graph
                const_tensor = output_buf.producer.op.tensor
                const_tensor_shape = [1, *list(const_tensor.shape)]
                const_tensor = np.reshape(const_tensor, const_tensor_shape)
                # Modify the graph according to updated shape
                output_buf.producer.op.tensor = const_tensor
                output_buf.shape = const_tensor_shape
                output_buf.axis_format = graph.src_axis_order.get_axis_format(output_rank+1)
            elif any(consumer_has_dimension_mismatch):
                # Remove consumers that need to be updated from current graph
                consumers_to_update = [consumer for i, consumer in output_buf.consumers if
                                       consumer_has_dimension_mismatch[i]]
                for consumer in consumers_to_update:
                    consumer.input_names.remove(output_buf.name)
                    output_buf.remove(consumer)
                # Create the new constant tensor
                const_tensor = output_buf.producer.op.tensor
                const_tensor_shape = [1, *list(const_tensor.shape)]
                const_tensor = np.reshape(const_tensor, const_tensor_shape)
                # Create the new N+1D constant operation
                const_op_name = output_buf.name + "_{}d".format(output_rank+1)
                const_op = op_adapter.ConstantOp(const_op_name, const_tensor,
                                                 quantizable=output_buf.producer.op.quantizable)
                # Place the new N+1D constant operation in graph
                log_debug("At least one, but not all consumers of buffer {} have {}D-{}D dimension mismatch. Creating "
                          "a new constant {}D constant operation named {}."
                          .format(output_buf.name, output_rank+1, output_rank, output_rank+1, const_op_name))
                const_node = graph.add(const_op, [], [const_op_name], axis_formats=[graph.src_axis_order.get_axis_format(output_rank+1)])
                graph.update_trace_info(const_node, output_buf)
                graph.update_trace_info(graph.get_buffer(const_op_name), output_buf)
                graph.get_buffer(const_op_name).consumers = consumers_to_update
                for consumer in consumers_to_update:
                    consumer.input_names.add(const_op_name)

        # Permute the constant data if necessary
        if output_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.NCDHW_TO_NDHWC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
            output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
        elif output_buf.axis_format == AxisTracker.AxisFormat.NCS:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.NCS_TO_NSC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
            output_buf.axis_format = AxisTracker.AxisFormat.NSC
        elif output_buf.axis_format == AxisTracker.AxisFormat.NCF:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.NCF_TO_NFC))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCF_TO_NFC)
            output_buf.axis_format = AxisTracker.AxisFormat.NFC
        elif output_buf.axis_format == AxisTracker.AxisFormat.TNF:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.TNF_TO_NTF))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            output_buf.axis_format = AxisTracker.AxisFormat.NTF
        elif output_buf.axis_format == AxisTracker.AxisFormat.OIDHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.OIDHW_TO_DHWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.OIDHW_TO_DHWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.DHWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.IODHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.IODHW_TO_DHWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.IODHW_TO_DHWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.DHWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.OIHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.OIHW_TO_HWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.OIHW_TO_HWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.HWIO
        elif output_buf.axis_format == AxisTracker.AxisFormat.IOHW:
            node.op.tensor = np.ascontiguousarray(np.transpose(node.op.tensor, AxisTracker.AxisFormat.IOHW_TO_HWIO))
            output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.IOHW_TO_HWIO)
            output_buf.axis_format = AxisTracker.AxisFormat.HWIO

        return True

    @staticmethod
    def remove_identity(node, graph):
        # Prune this node if it's an input to a weight layer and was used internally
        if getattr(graph, "weights", None) and getattr(graph.weights, "consumed", None) \
                and graph.weights.consumed(node.output_names[0]):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONSTANT_PRUNED")(node.output_names[0]))
            graph.prune(node)

    @staticmethod
    def cast_fp16_to_fp32(node, graph):
        if node.op.dtype == np.float16:
            node.op.tensor = node.op.tensor.astype(np.float32)

    def replace_6d_operation(self, node, graph):
        # skip optimization for ConstantOp
        pass

@register_layer_optimization
class OptimizeConvertTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ConvertOp.TRANSLATION_KEY

@register_layer_optimization
class OptimizeCreateSparseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CreateSparseOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        output_buffer = graph.get_buffer(node.output_names[0])
        output_buffer.set_buf_dims(graph.src_axis_order.extract_3d_spatial_dims(output_buffer.shape))
        output_buffer.axis_format = AxisTracker.AxisFormat.NDHWC
        return True


@register_layer_optimization
class OptimizeCumSumTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CumSumOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeCumSumTranslation, self).axes_to_spatial_first_order(node, graph):
            return False

        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            node.op.axis = AxisTracker.AxisFormat.NDHWC_TO_NCDHW[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCDHW and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC:
            node.op.axis = AxisTracker.AxisFormat.NCDHW_TO_NDHWC[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            node.op.axis = AxisTracker.AxisFormat.NSC_TO_NCS[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCS and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NSC:
            node.op.axis = AxisTracker.AxisFormat.NCS_TO_NSC[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            node.op.axis = AxisTracker.AxisFormat.NFC_TO_NCF[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NFC:
            node.op.axis = AxisTracker.AxisFormat.NCF_TO_NFC[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            node.op.axis = AxisTracker.AxisFormat.NTF_TO_TNF[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.TNF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NTF:
            node.op.axis = AxisTracker.AxisFormat.TNF_TO_NTF[node.op.axis]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NONTRIVIAL:
            pass
        else:
            raise ValueError("Unexpected input buffer axis format: {}, for {} Op".format(input_buf.axis_format, node.op.name))

        return True


@register_layer_optimization
class OptimizeCustomOpTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CustomOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Todo: revisit and modify when the layout support for CustomOps is added [AISW-55482].
        ret = super(OptimizeCustomOpTranslation, self).axes_to_spatial_first_order(node, graph)

        return ret


@register_layer_optimization
class OptimizeTransposeConv1dTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.TransposeConv1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        conv_op_name = node.op.name + "_2d"
        self.nn_2d_op = op_adapter.TransposeConv2dOp(conv_op_name,
                                                     bias_op_name=node.op.bias_op_name,
                                                     stridey=1,
                                                     stridex=node.op.stride[0],
                                                     pady_before=0,
                                                     pady_after=0,
                                                     padx_before=node.op.pad_amount[0],
                                                     padx_after=node.op.pad_amount[1],
                                                     output_paddingy=0,
                                                     output_paddingx=node.op.output_padding[0],
                                                     padding_size_strategy=node.op.padding_size_strategy,
                                                     output_height=1,
                                                     output_width=node.op.output_size[0],
                                                     groups=node.op.group)
        super().expand_1d_spatial_nn_nodes(node, graph)

@register_layer_optimization
class OptimizeTransposeConv2dTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TransposeConv2dOp.TRANSLATION_KEY
        self.register_method(PREPARE_BIASES, self.prepare_biases)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        if isinstance(graph.src_axis_order, (CaffeAxisOrder, SpatialLastAxisOrder)):
            input_buffers = graph.get_input_buffers(node)
            input_axis_formats = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NDHWC, transpose it to IODHW by using a transpose to NCDHW. Then, to DHWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NDHWC,
                                         AxisTracker.AxisFormat.IODHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 5):
                if node.op.data_axis_formats[1] != input_axis_formats[1]:
                    # Inject an implicit permute to NCDHW, which is actually taking us back to IODHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.IODHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to DHWIO from IODHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.DHWIO,
                                              graph.src_axis_order.permute_deconv3d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.DHWIO

                # Update input_buffers and input_orders after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_orders = [buf.axis_format for buf in input_buffers]

            # If the weights input is already NSC, transpose it to IOHW by using a transpose to NCS. Then, to HWIO.
            if input_axis_formats[1] in [AxisTracker.AxisFormat.NSC,
                                         AxisTracker.AxisFormat.IOHW] or \
                    (input_axis_formats[1] in [AxisTracker.AxisFormat.NONTRIVIAL] and \
                     input_buffers[1].rank() == 4):
                if node.op.data_axis_formats[1] != input_axis_formats[1] and input_axis_formats[1] in [AxisTracker.AxisFormat.NSC]:
                    # Inject an implicit permute to NCS, which is actually taking us back to IOHW
                    graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.IOHW

                # Must update input_buffers after first injection of implicit permute
                input_buffers = graph.get_input_buffers(node)

                # Inject an implicit permute to HWIO from IOHW
                graph.inject_implicit_permute(input_buffers[1].name, AxisTracker.AxisFormat.HWIO,
                                              graph.src_axis_order.permute_deconv2d_weights_to_ir, [node.op.name])
                input_buffers[1].axis_format = AxisTracker.AxisFormat.HWIO

                # Update input_buffers and input_orders after second injection of implicit permute
                input_buffers = graph.get_input_buffers(node)
                input_orders = [buf.axis_format for buf in input_buffers]

            if any(format in input_axis_formats for format in [AxisTracker.AxisFormat.NDHWC,
                                                               AxisTracker.AxisFormat.DHWIO,
                                                               AxisTracker.AxisFormat.NSC,
                                                               AxisTracker.AxisFormat.HWIO,
                                                               AxisTracker.AxisFormat.ANY,
                                                               AxisTracker.AxisFormat.NONTRIVIAL]):
                AxisTracker.image_to_channel_last_order(node, graph)
                output_buffer = graph.get_output_buffers(node)[0]
                # image_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
                # Enforce the output format here according to output buffer's rank
                output_buffer.axis_format = AxisOrder().get_axis_format(output_buffer.rank())
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_TRANPOSE_CONV_UNEXPECTED_INPUT_ORDER")
                                 (input_orders))

            return True

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_deconv2d_weights_dims(weights_buffer.shape)[-1] * node.op.group
        add_or_broadcast_bias(node, graph, output_channel)

    def prepare_inputs_as_params(self, node, graph):
        prepare_conv_inputs_as_params(graph, node)


@register_layer_optimization
class OptimizeTransposeConv3dTranslation(OptimizeTransposeConv2dTranslation):
    def __init__(self):
        OptimizeTransposeConv2dTranslation.__init__(self)
        self.op_type = op_adapter.TransposeConv3dOp.TRANSLATION_KEY

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_deconv3d_weights_dims(weights_buffer.shape)[-1] * node.op.group
        add_or_broadcast_bias(node, graph, output_channel)


@register_layer_optimization
class OptimizeDepthwiseConvolution1DTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.DepthwiseConv1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        conv_op_name = node.op.name + "_2d"
        self.nn_2d_op = op_adapter.DepthwiseConv2dOp(conv_op_name,
                                                    bias_op_name=node.op.bias_op_name,
                                                    padx_before=node.op.pad_amount[0],
                                                    padx_after=node.op.pad_amount[1],
                                                    pady_before=0,
                                                    pady_after=0,
                                                    padding_size_strategy=node.op.padding_size_strategy,
                                                    stridey=1,
                                                    stridex=node.op.stride[0],
                                                    dilationy=1,
                                                    dilationx=node.op.dilation[0],
                                                    data_layout=AxisTracker.AxisFormat.NCS)
        super().expand_1d_spatial_nn_nodes(node, graph)


@register_layer_optimization
class OptimizeDepthwiseConvolutionTranslation(OptimizeConvolutionTranslation):
    def __init__(self):
        OptimizeConvolutionTranslation.__init__(self)
        self.op_type = op_adapter.DepthwiseConv2dOp.TRANSLATION_KEY

    def prepare_biases(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        output_channel = graph.src_axis_order.extract_conv2d_weights_dims(weights_buffer.shape)[-1]
        add_or_broadcast_bias(node, graph, output_channel)


@register_layer_optimization
class OptimizeDetectionOutTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DetectionOutputOp.TRANSLATION_KEY
        self.register_method(FOLD_CONCATS, self.fold_concats)
        self.register_method(MATCH_CAFFE_SSD_TO_TF, self.caffe_ssd_to_tf)

    @staticmethod
    def fold_concats(graph):
        def process_ssd_priorbox_concat_layer(input_buffers_):
            concatenated_priorbox_data = []
            concatenated_priorbox_cz_data = []
            concatenated_priorbox_variance = []
            scale_factors_ = input_buffers_[0].producer.op.scale_factors
            for input_buffer in input_buffers_:
                priorbox_op = input_buffer.producer.op
                concatenated_priorbox_data.extend(priorbox_op.priorbox_box_output[0])
                concatenated_priorbox_variance.extend(priorbox_op.priorbox_box_output[1])
                concatenated_priorbox_cz_data.extend(priorbox_op.priorbox_box_cz_output)
                if scale_factors_ != priorbox_op.scale_factors:
                    # Currently only support 1 set of scale factor for priorboxes.
                    raise ValueError(code_to_message.get_error_message("ERROR_INVALID_PRIORBOX_VARIANCES")
                                     (scale_factors_, input_buffers_[0].producer.op.name,
                                      priorbox_op.scale_factors, priorbox_op.name))

            return concatenated_priorbox_data + concatenated_priorbox_variance, concatenated_priorbox_cz_data, \
                   scale_factors_

        sequence = [
            ("Concat",
                ("FLEXIBLE_NUM_BUFS", [(op_adapter.IdentityOp.TRANSLATION_KEY, "ALL")]),  # identity here since all priorboxes are mapped to IdentityOp
                ("MATCH_NUM_BUFS", [("DetectionOutput", "ALL")])
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            concat_node = node_tuple[0]
            concat_input_buffers = graph.get_input_buffers(concat_node)
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            detection_out_node = concat_output_buffer.consumers.pop()
            priorbox_data, priorbox_cz_data, scale_factors = process_ssd_priorbox_concat_layer(concat_input_buffers)
            detection_out_node.op.priorbox_data = priorbox_data
            detection_out_node.op.priorbox_center_size_data = priorbox_cz_data
            # order determined per caffe/util/bbox_util.cpp
            delta_scaling_factors = np.array([
                scale_factors[0],
                scale_factors[1],
                scale_factors[2],
                scale_factors[3]
            ], dtype=np.float32)
            detection_out_node.op.delta_scaling_factors = delta_scaling_factors

            # remove concat node.
            detection_out_node.input_names.remove(concat_output_buffer.name)
            graph.prune(concat_node)

            # remove priorboxes
            for buf in concat_input_buffers:
                graph.prune(buf.producer)

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_FOLDING")(concat_node.op.name,
                                                                                           detection_out_node.op.name))

    @staticmethod
    def caffe_ssd_to_tf(graph):
        sequence = [
            ("DetectionOutput",
                ("MATCH_NUM_BUFS", [("Reshape", "ANY"), ("Concat", "ANY")]),  # flattened scores and boxes
                ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence)

        for node_tuple in matched_node_list:
            detection_out_node = node_tuple[0]
            for input_name in detection_out_node.input_names:
                node = graph.get_producer_node(input_name)
                if node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                    reshape_node = node
                elif node.op.type == op_adapter.ConcatOp.TRANSLATION_KEY:
                    concat_node = node
                else:
                    raise ValueError(code_to_message.get_error_message("ERROR_DETECTIONOUT_UNKNOWN_INPUTS")
                                     (node.op.type))

            # 0. Verify valid anchors/priorboxes
            log_assert(detection_out_node.op.code_type == op_adapter.DetectionOutputOp.PriorBoxType.CENTER_SIZE,
                       "DetectionOut Op only supports center size code type. Got {}".
                       format(detection_out_node.op.code_type))

            # 1. Pre-process steps
            # Caffe score input is flattened, remove reshape to match shape [batch, num_anchors, num_classes]
            reshape_output_buffer = graph.get_output_buffers(reshape_node)[0]
            detection_out_node.input_names.remove(reshape_output_buffer.name)
            detection_out_node.input_names.insert(0, reshape_node.input_names[0])
            graph.get_buffer(reshape_node.input_names[0]).consumers.add(detection_out_node)

            reshape_output_buffer.consumers.remove(detection_out_node)
            # remove reshape node if applicable.
            if len(reshape_output_buffer.consumers) == 0:
                graph.prune(reshape_node)

            # Caffe boxes(location) data is also flattened. Reshape to [batch, num_boxes, 4]
            concat_output_buffer = graph.get_output_buffers(concat_node)[0]
            concat_buf_shape = concat_output_buffer.shape
            # add reshape node
            reshape_name = concat_node.op.name + "_preprocess_reshape"
            reshape_op = op_adapter.ReshapeOp(reshape_name, shape=[concat_buf_shape[0],
                                                                   int(concat_buf_shape[1] / 4),
                                                                   4])
            graph.inject(reshape_op, input_name=concat_node.output_names[0], output_name=reshape_name,
                         consumer_names=detection_out_node.output_names)

            # DetectionOut in IR has priorboxes as param, need to add those to input instead
            detection_out_name = detection_out_node.op.name
            detection_out_node_idx = graph.nodes_in_order.index(detection_out_node)
            prior_box_name = detection_out_name + "_anchors"
            pbox_data = np.asarray(detection_out_node.op.priorbox_center_size_data, dtype=np.float32)\
                        .reshape(int(len(detection_out_node.op.priorbox_center_size_data)/4), 4)
            prior_box_op = op_adapter.ConstantOp(name=prior_box_name, tensor=pbox_data)
            prior_box_node = graph.add(prior_box_op, input_names=[], output_names=[prior_box_name], idx=detection_out_node_idx-1)
            detection_out_node.input_names.append(prior_box_name)
            # Add op trace data for prior_box_node and it's output tensor
            graph.update_trace_info(prior_box_node, detection_out_node)
            graph.update_trace_info(graph.get_buffer(prior_box_name), detection_out_node)

            # Caffe Ssd scales is the reciprocal compared to TF scales
            detection_out_node.op.delta_scaling_factors = np.array([
                1 / detection_out_node.op.delta_scaling_factors[0],
                1 / detection_out_node.op.delta_scaling_factors[1],
                1 / detection_out_node.op.delta_scaling_factors[2],
                1 / detection_out_node.op.delta_scaling_factors[3],
            ], dtype=np.float32)
            # 2. Change DetectionOut's single output to multiple. Outputs:
            #    Expected: scores[1, max_num_det], boxes[1, max_num_det, 4], classes[1, max_num_det], num_det[batch],
            #    Caffe Style: 1 output of shape [1, 1, max_num_det, 7]
            #                   7(last dim above): [image_batch, label, confidence, x_min, y_min, x_max, y_max]
            detection_out_buf = graph.get_buffer(detection_out_node.output_names[0])
            boxes_shape = [detection_out_buf.shape[0], detection_out_node.op.keep_top_k, 4]  # [batch, max_num_detections, 4)
            boxes_name = detection_out_name + "_boxes"
            boxes_buf = op_graph.Buffer(boxes_name, boxes_shape, detection_out_node)
            graph.buffers[boxes_name] = boxes_buf

            scores_name = detection_out_name + "_scores"
            scores_buf = op_graph.Buffer(scores_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[scores_name] = scores_buf

            classes_name = detection_out_name + "_classes"
            classes_buf = op_graph.Buffer(classes_name, boxes_shape[:-1], detection_out_node)
            graph.buffers[classes_name] = classes_buf

            num_det_name = detection_out_name + "_num_detections"
            num_det_buf = op_graph.Buffer(num_det_name, [boxes_shape[0]], detection_out_node)
            graph.buffers[num_det_name] = num_det_buf

            del graph.buffers[detection_out_node.output_names[0]]
            detection_out_node.output_names = [scores_name, boxes_name, classes_name, num_det_name]

            log_debug2(code_to_message.get_debugging_message("DEBUG_DETECTIONOUT_CAFFE_TO_TF_STYLE")
                       (detection_out_node.op.name))


@register_layer_optimization
class OptimizeDequantizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DequantizeOp.TRANSLATION_KEY
        self.register_method(REMOVE_QUANT_NODES, self.remove_quant_nodes)

    @staticmethod
    def remove_quant_nodes(node, graph):
        if graph.has_buffer(node.input_names[0]) and \
            isinstance(graph.get_producer_op(node.input_names[0]), op_adapter.InputOp):
           graph.get_producer_op(node.input_names[0]).input_dtype =  np.dtype("float32")
        graph.squash(node, input_name=str(node.input_names[0]))
        log_debug("Remove dequantize op {}".format(node.op.name))


@register_layer_optimization
class OptimizeDistributeFpnProposalsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DistributeFpnProposalsOp.TRANSLATION_KEY


class OptimizeElementwiseUnaryTranslation(OptimizationTranslationBase):
    def __init__(self, op_type):
        super().__init__()
        self.op_type = op_type

    def replace_6d_operation(self, node, graph):
        """
        replace 6D ElementUnary by inserting reshapes around the op
        """
        in_shapes = graph.get_input_shapes(node)
        out_shapes = graph.get_output_shapes(node)
        rank = len(in_shapes[0]) # only single input
        if rank <= 5:
            return
        check_if_6d_supportable(node, graph)
        # insert pre-reshape before ElementwiseUnaryOp
        new_input_shape = [np.prod(in_shapes[0])] # Just flatten input
        pre_reshape_op_name = node.op.name + '_6d_pre_reshape'
        pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_op_name, shape=new_input_shape)
        graph.inject(
            pre_reshape_op, input_name=node.input_names[0],
            output_name=pre_reshape_op_name, consumer_names=[node.op.name]
        )
        # ElementwiseUnary has new_out_shape same as new_input_shape.
        new_out_shape = new_input_shape
        # insert post-reshape.
        post_reshape_insertion(
            node, graph, new_out_shapes=[new_out_shape], orig_out_shapes=out_shapes
        )

class OptimizeElementwiseTranslation(OptimizationTranslationBase):
    def __init__(self, op_type):
        super().__init__()
        self.op_type = op_type
        self.register_method(SINK_TRANSPOSE_BELOW_SUM, self.sink_transpose_below_sum)

    def replace_6d_operation(self, node, graph):
        """
        replace 6D ElementBinary by inserting reshapes around the op
        """
        input_shapes = graph.get_input_shapes(node)
        output_shapes = graph.get_output_shapes(node)
        input_ranks = [len(input_shape) for input_shape in input_shapes]
        max_rank = max(input_ranks)
        if max_rank <= 5:
            return
        check_if_6d_supportable(node, graph)
        log_assert(
            len(node.input_names) == 2,
            f"Currently we do not support 6d inputs/outputs for ElementwiseTernaryOp"
        )
        # We attempt to flatten contiguous dimensions of A and B from right to left to reduce ranks
        # (A[i], A[i+1]) and (B[i], B[i+1]) can be flattened into (A[i]* A[i+1]) and (B[i]* B[i+1]) if
        # 1. A[i]==B[i] and A[i+1]==B[i+1]
        # 2. A[i]*A[i+1]==1 or B[i]*B[i+1]==1

        # Example: Given A with shape [4,3,1,1,5,5] and B with shape [7,6,5,1]
        # step 1. broadcast inputs to same rank
        #   A [4,3,1,1,5,5] B [1,1,7,6,5,1]                                       (expand to same rank)
        # step 2. check if last two dimensions can be flattened
        #   A [4,3,1,1,5] B [1,1,7,6,5] new_input_shape1 [5] new_input_shape2 [1] (cannot flatten)
        #   A [4,3,1,1] B [1,1,7,6] new_input_shape1 [5,5] new_input_shape2 [5,1] (cannot flatten)
        #   A [4,3,1] B [1,1,42] new_input_shape1 [5,5] new_input_shape2 [5,1]    (flatten last two 1,1 in A and last two 7,6 in B)
        #   A [4,3] B [1,1] new_input_shape1 [1,5,5] new_input_shape2 [42,5,1]    (cannot flatten)
        #   A [12] B [1] new_input_shape1 [1,5,5] new_input_shape2 [42,5,1]       (flatten last two 4,3 in A and last two 1,1 in B)
        #   A [] B [] new_input_shape1 [12,1,5,5] new_input_shape2 [1,42,5,1]     (Put the result into new_input_shape)
        # Thus, A and B can be flattened into [12,1,5,5] and [1,42,5,1] respectively

        broadcast_shape1 = [1]*(max_rank-input_ranks[0]) + input_shapes[0].dims
        broadcast_shape2 = [1]*(max_rank-input_ranks[1]) + input_shapes[1].dims
        new_input_shape1, new_input_shape2 = [], []
        while len(broadcast_shape1):
            assert len(broadcast_shape1) == len(broadcast_shape2)
            if len(broadcast_shape1) >= 2:
                prod1 = broadcast_shape1[-2]* broadcast_shape1[-1]
                prod2 = broadcast_shape2[-2]* broadcast_shape2[-1]
                can_be_flattened = broadcast_shape1[-2:] == broadcast_shape2[-2:] or prod1 == 1 or prod2 == 1
            else:
                can_be_flattened = False
            if can_be_flattened:
                broadcast_shape1 = broadcast_shape1[:-2] + [prod1]
                broadcast_shape2 = broadcast_shape2[:-2] + [prod2]
            else:
                new_input_shape1.insert(0, broadcast_shape1.pop())
                new_input_shape2.insert(0, broadcast_shape2.pop())
        # there are some corner cases that cannot be handled...
        # e.g., input1: [3,3,3,3,3,3], input2: [3,1,3,1,3,1]
        log_assert(
            len(new_input_shape1) <= 5,
            f"inputs of {node.op.name} are not supported 6D inputs and cannot use reshapes to support the op"
        )

        # insert pre-reshape to reduce the rank of output
        pre_reshape_op_name1 = node.op.name + '_input1_6d_pre_reshape'
        pre_reshape_op1 = op_adapter.ReshapeOp(name=pre_reshape_op_name1, shape=new_input_shape1)
        graph.inject(
            pre_reshape_op1, input_name=node.input_names[0],
            output_name=pre_reshape_op_name1, consumer_names=[node.op.name]
        )
        pre_reshape_op_name2 = node.op.name + '_input2_6d_pre_reshape'
        pre_reshape_op2 = op_adapter.ReshapeOp(name=pre_reshape_op_name2, shape=new_input_shape2)
        graph.inject(
            pre_reshape_op2, input_name=node.input_names[1],
            output_name=pre_reshape_op_name2, consumer_names=[node.op.name]
        )

        # insert post-reshape to recover the rank (shape) of output
        post_reshape_insertion(
            node, graph,
            new_out_shapes=[list(np.broadcast_shapes(new_input_shape1, new_input_shape2))],
            orig_out_shapes=output_shapes
        )

    # Conditions:
    #   1. if all inputs have matching ranks, or one of them is 1-element tensor,
    #      then their axis_format can be changed to spatial first order.
    #      e.g. input0.shape: [1,3,255,255]; input1.shape: [1,3,255,255] or
    #           input0.shape: [1]          ; input1.shape: [1,3,255,255]
    #
    #   2. Otherwise, axis_format of all inputs should keep source format.
    #      e.g. input0.shape: [1,3,255,255]; input1.shape: [255,255]
    def broadcastable_in_spatial_first_order(self, input_buffers):
        rank_list = [i.rank() for i in input_buffers]
        max_rank = max(rank_list)
        if all(map(lambda buf: (buf.rank() == max_rank and buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL) or
                                (buf.rank() == 1 and buf.shape.dims[0] == 1 and not buf.shape.is_dynamic()),
                    input_buffers)):
            return True
        else:
            return False

    def axes_to_spatial_first_order(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        self.axes_to_spatial_first_order_for_aisw(node, graph)

    def axes_to_spatial_first_order_for_aic(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        input_buffers = graph.get_input_buffers(node)
        if self.broadcastable_in_spatial_first_order(input_buffers):
            # super() function will enforce spatial-first order
            return super().axes_to_spatial_first_order(node, graph)
        else:
            # Ensure format stays in source format to ensure broadcastability
            for i, buf in enumerate(input_buffers):
                # Only need to insert permute for rank > 2, since ANY & NF will not change
                if node.op.data_axis_formats[i] != buf.axis_format and \
                        buf.axis_format in AxisOrder().axis_formats and buf.rank() > 2:
                    # Transpose to maintain src format
                    graph.inject_implicit_permute(
                        buf.name,
                        spatial_first_format_to_channel_first_format[buf.axis_format],
                        spatial_first_format_to_channel_first_permute_order[buf.axis_format],
                        [node.op.name]
                    )
            return False

    def axes_to_spatial_first_order_for_aisw(self, node: op_graph.OpNode, graph: op_graph.IROpGraph):
        # Scenario_1 : Two inputs are Channel_first and Channel_last in src model, e.g data_axis_formats = [NCF, NFC]
        # Solution : Revert current input_axis_formats to data_axis_formats to avoid the shape mismatch.
        # Scenario_2 : Data_axis_formats are the same, 'Nontrivial' is not in input_axis_formats and shapes are broadcastable,
        # Solution : Alert the input_axis_formats to IR Order and change axis accordingly.
        # Scenario_3 : Data_axis_formats are the same, 'Nontrivial' is in input_axis_formats or shapes are not broadcastable,
        # Solution : Revert other input_axis_formats to data_axis_formats to follow src models' layout.

        if AxisTracker.input_axis_formats_intact(graph, node):
            # Nothing to do in this case
            return False

        has_changed = False
        # Incase the input_axis_formats were already different in translation stage which means data_axis_formats are different
        # Revert to original axis_formats for each input
        input_buffers = graph.get_input_buffers(node)
        # Ignore 'ANY' to compare other layouts
        checked_axis_formats = copy.deepcopy(node.op.data_axis_formats)
        data_axis_formats = node.op.data_axis_formats
        if AxisTracker.AxisFormat.ANY in checked_axis_formats:
            checked_axis_formats.remove(AxisTracker.AxisFormat.ANY)
        if len(checked_axis_formats) > 1:
            first_data_format = checked_axis_formats[0]
            if not all([data_format == first_data_format for data_format in checked_axis_formats]):
                for idx, buf in enumerate(input_buffers):
                    # fetch input buffers one by one to revert to original axis format.
                    # keep the input shapes
                    # keep the output axis format as 'NONTRIVIAL'
                    ret = AxisTracker.revert_input_axis_format(graph,
                                                               node,
                                                               buf.name,
                                                               buf.axis_format,
                                                               data_axis_formats[idx])
                    has_changed = has_changed or ret
                return has_changed

        pre_input_formats = graph.get_input_axis_formats(node)
        if AxisTracker.AxisFormat.NONTRIVIAL not in pre_input_formats and \
            self.broadcastable_in_spatial_first_order(input_buffers):
            # Enforce spatial-first order
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            input_axis_formats = graph.get_input_axis_formats(node)
            if pre_input_formats != input_axis_formats:
                return True
            else:
                return False
        else:
            # Ensure format stays in source format to ensure broadcastability
            # Ensure format stays in source format because NT is not traceable
            for idx, buf in enumerate(input_buffers):
                # Only need to insert permute for rank > 2, since ANY & NF will not change
                # revert to maintain src format.
                ret = AxisTracker.revert_input_axis_format(graph,
                                                           node,
                                                           buf.name,
                                                           buf.axis_format,
                                                           data_axis_formats[idx])
                has_changed = has_changed or ret
            return has_changed

    def sink_transpose_below_sum(self, graph):

        def validate_permute_node(node_tuple):

            input_nodes = graph.get_op_input_nodes(node_tuple[0])

            node_io_shapes= []
            # check if transpose has other outputs than elementwise_sum
            for input_node in input_nodes:
                if len(graph.get_op_output_nodes(input_node)) > 1 or input_node.op.type != 'Transpose' \
                        or graph.is_output_node(input_node):
                    return False
                node_io_shapes.append([graph.get_input_buffers(input_node)[0].shape,
                                       graph.get_output_buffers(input_node)[0].shape])

            # verifying the input and output shape of two transpose is equal, this will work with input ranks being equal
            # TODO: support when the input shapes are broadcastable with transpose having equal permutation
            shape_info = node_io_shapes[0]
            for i in range(1, len(node_io_shapes)):
                if shape_info != node_io_shapes[i]:
                    return False
            return True



        sequence = [
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("Transpose", "ANY"), ("Transpose", "ANY")]),
             ()
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence,
                                                    validator=validate_permute_node,
                                                    ignore_constants=True)

        for matched_node in matched_node_list:
            node = matched_node[0]
            transpose_1, transpose_2 = graph.get_op_input_nodes(node)

            transpose_perm = transpose_1.op.c_op.perm
            output_buffer = graph.get_output_buffers(node)[0]

            # Pruning extra Transposes
            graph.squash(transpose_1, transpose_1.input_names[0], squash_into_next=True)
            graph.squash(transpose_2, transpose_2.input_names[0], squash_into_next=True)

            # Updating shape and axis_format of output buffer of Elementwise_sum
            input_buffer = graph.get_input_buffers(node)[0]
            output_buffer.shape = input_buffer.shape
            output_buffer.axis_format = input_buffer.axis_format

            node.op.populate_data_axis_formats(graph, graph.get_input_buffers(node))

            # Adding Transpose op after Elementwise Sum
            idx = graph.nodes_in_order.index(matched_node[0])
            transpose_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.TransposeOp.TRANSLATION_KEY,
                                                                        op_adapter.TransposeOp.LEGACY_TRANSLATION_KEY)

            transpose_op = op_adapter.TransposeOp(transpose_op_name, transpose_perm)
            graph.inject(transpose_op, input_name=node.output_names[0], output_name="{}_permute".format(node.output_names[0]))


@register_layer_optimization
class OptimizeElementwiseAndTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND])


@register_layer_optimization
class OptimizeElementwiseDivTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE])
        self.register_method(SQUASH_DIV, self.squash_div)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    @staticmethod
    def squash_div(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0],
                op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])

    @staticmethod
    def remove_identity(node, graph):
        divisor_op = graph.get_buffer(node.input_names[1]).producer.op
        # squash the op if the divisor is a tensor of all ones
        if divisor_op.type == "constant" and np.all(divisor_op.tensor == 1):
            try:
                graph.squash(node, node.input_names[0])
            except RuntimeError as e:
                log_debug("Squash elementwise div op {} due to identity not possible ".format(node.op.name))

@register_layer_optimization
class OptimizeElementwiseEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL])


@register_layer_optimization
class OptimizeElementwiseFloorDivTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV])


@register_layer_optimization
class OptimizeElementwiseFmodTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD])


@register_layer_optimization
class OptimizeElementwiseGreaterTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER])


@register_layer_optimization
class OptimizeElementwiseGreaterEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL])


@register_layer_optimization
class OptimizeElementwiseLessTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS])


@register_layer_optimization
class OptimizeElementwiseLessEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL])


@register_layer_optimization
class OptimizeElementwiseMaxTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM])


@register_layer_optimization
class OptimizeElementwiseMinTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM])


@register_layer_optimization
class OptimizeElementwiseModTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD])


@register_layer_optimization
class OptimizeElementwiseNotEqualTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL])


@register_layer_optimization
class OptimizeElementwisePowerTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER])
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def prepare_inputs_as_params(self, node, graph):
        exponent_buffer = graph.get_buffer(node.input_names[1])
        exponent_node = exponent_buffer.producer
        if exponent_node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY:
            raise ValueError("Dynamic exponents on node {} are not supported in this backend.".format(node.op.name))
        node.op.power = exponent_node.op.tensor
        # merge the constant op source to the node since the constant are merged into parameter
        graph.update_trace_info(node, [exponent_node, exponent_buffer])
        graph.remove_node_as_consumer(node, exponent_buffer.name)


@register_layer_optimization
class OptimizeElementwiseProductTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
                                                op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY])
        self.register_method(SQUASH_PROD, self.squash_prod)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    @staticmethod
    def remove_identity(node,graph):
        input_buff0 = graph.get_buffer(node.input_names[0])
        input_buff1 = graph.get_buffer(node.input_names[1])
        # do not squash the op if the node has multiple consumers
        if len(graph.get_buffer(node.output_names[0]).consumers) > 1 or (input_buff0.shape.dims != input_buff1.shape.dims):
            return
        # squash the mul op if one of input having a tensor of all ones
        if (input_buff0.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY \
                and (not input_buff0.producer.op.constant_tensor.is_lazy_data()) \
                and (np.all(input_buff0.producer.op.tensor == 1))):
            try:
                graph.squash(node, input_name=node.input_names[1], squash_into_next=True)
            except RuntimeError as e:
                log_debug("Squash elementwise product op {} due to identity not possible ".format(node.op.name))
        elif (input_buff1.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY \
              and (not input_buff1.producer.op.constant_tensor.is_lazy_data()) \
              and (np.all(input_buff1.producer.op.tensor == 1))):
            try:
                graph.squash(node, input_name=node.input_names[0], squash_into_next=True)
            except RuntimeError as e:
                log_debug("Squash elementwise product op {} due to identity not possible ".format(node.op.name))


    @staticmethod
    def squash_prod(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "weights")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseSelectTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseTernaryOp.ir_to_legacy_type[ir_graph.QNN_OP_ELEMENT_WISE_SELECT])


@register_layer_optimization
class OptimizeElementwiseSubTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT])
        self.register_method(SQUASH_SUB, self.squash_sub)

    @staticmethod
    def squash_sub(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0], op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])


@register_layer_optimization
class OptimizeElementwiseSumTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD])
        self.register_method(SQUASH_SUM, self.squash_sum)
        self.register_method(EXPAND_SPARSE_OP_STRUCTURE, self.expand_sparse_op_structure)

    @staticmethod
    def squash_sum(graph):
        def validate_node(nodes_tuple):
            return validate_eltwise_pattern(graph, nodes_tuple, "bias")

        sequence = [
            (op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD], (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        squash_node_into_nn_node(graph, matched_node_list)

        def validate_conv_sequences(nodes_tuple):
            return validate_conv_eltwise_pattern(graph, nodes_tuple[0],
                                                 op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD])

        sequences = [
            [("Conv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))],
            [("DepthWiseConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))],
            [("TransposeConv2d",
              ("MATCH_BUFS_AT_INDEX", [("constant", 2)]),
              ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]))]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence,
                                                        validator=validate_conv_sequences,
                                                        ignore_constants=True)
            for match in matched_node_list:
                squash_eltwise_into_conv(graph, match[0])

    @staticmethod
    def expand_sparse_op_structure(node, graph):

        def validate_conv_submanifold(nodes_tuple):
            first_conv_node = nodes_tuple[1]
            first_conv_subm = first_conv_node.op.__getattr__(ir_graph.QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES)
            second_conv_node = nodes_tuple[-2]
            second_conv_subm = second_conv_node.op.__getattr__(ir_graph.QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES)
            create_sparse_node = nodes_tuple[0]

            if first_conv_subm != 1 or second_conv_subm != 1:
                return False

            if create_sparse_node.output_names[0] != first_conv_node.input_names[0]:
                return False

            return True

        sequence = [
            ("CreateSparse",
             ("MATCH_BUFS_AT_INDEX", [("GetSparseIndices", 0), (ir_graph.QNN_OP_ELEMENT_WISE_NEURON, 1)]),
             ("MATCH_NUM_BUFS", [("Conv3d", "ANY"), ("elementwise_sum", "ANY")])
             ),
            ("Conv3d",
             ("MATCH_BUFS_AT_INDEX", [("CreateSparse", 0)]),
             ("MATCH_NUM_BUFS", [("GetSparseIndices", "ANY"), ("GetSparseValues", "ANY")])
             ),
            ("GetSparseValues",
             ("MATCH_NUM_BUFS", [("Conv3d", "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")])
             ),
            ("GetSparseIndices",
             ("MATCH_NUM_BUFS", [("Conv3d", "ALL")]),
             ("MATCH_NUM_BUFS", [("CreateSparse", "ALL")])
             ),
            (ir_graph.QNN_OP_ELEMENT_WISE_NEURON,
             ("MATCH_NUM_BUFS", [("GetSparseValues", "ALL")]),
             ("MATCH_NUM_BUFS", [("CreateSparse", "ALL")])
             ),
            ("CreateSparse",
             ("MATCH_BUFS_AT_INDEX", [("GetSparseIndices", 0), (ir_graph.QNN_OP_ELEMENT_WISE_NEURON, 1)]),
             ("MATCH_NUM_BUFS", [("Conv3d", "ALL")])
             ),
            ("Conv3d",
             ("MATCH_BUFS_AT_INDEX", [("CreateSparse", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("CreateSparse", "ANY"), ("Conv3d", "ANY")]),
             ()
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_conv_submanifold, ignore_constants=True)

        for node_tuple in matched_node_list:
            add_op_output = graph.get_output_buffers(node)[0]
            add_op_consumers = add_op_output.consumers
            add_op_output_name = node.op.name + '_addOut'
            add_op_inputs = node.input_names
            input_names = graph.get_input_buffers(node)
            post_expansion_idx = graph.nodes_in_order.index(node)
            post_expansion_op = node.op
            create_sparse_op_name = node.op.name + '_createSparse'
            create_sparse_output_name = node.op.name + '_createSparseOut'
            create_sparse_output_shape = add_op_output.get_buf_dims()
            list_ids = []

            for consumer in add_op_consumers:
                list_ids.append(consumer.input_names.index(str(add_op_output)))

            sparse_indices_op_output_name = None
            sparse_params = None

            for buf in input_names:
                input_name = buf.name
                input_buf = graph.get_buffer(input_name)
                sparse_params = input_buf.get_sparse_params()
                if sparse_params.layout != ir_graph.QNN_SPARSE_LAYOUT_UNDEFINED:
                    #first prune and add back to adjust its input
                    graph.prune(node, force_remove=True)
                    sparse_indices_op_name = post_expansion_op.name + "_" + input_name + '_sparseIndices'
                    sparse_values_op_name = post_expansion_op.name + "_" + input_name + '_sparseValues'
                    sparse_indices_op_output_name = sparse_indices_op_name + '_out'
                    sparse_values_op_output_name = sparse_values_op_name + '_out'
                    sparse_indices_op = op_adapter.GetSparseIndicesOp(sparse_indices_op_name,
                                                                      num_specified_elements=sparse_params.cooInfo.numSpecifiedElements)
                    sparse_values_op = op_adapter.GetSparseValuesOp(sparse_values_op_name,
                                                                    num_specified_elements=sparse_params.cooInfo.numSpecifiedElements)
                    add_op_inputs.remove(input_name)
                    add_op_inputs.append(sparse_values_op_output_name)
                    graph.add(sparse_values_op, [input_name], [sparse_values_op_output_name], idx=post_expansion_idx)
                    node = graph.add(sparse_indices_op, [input_name], [sparse_indices_op_output_name], idx=post_expansion_idx+1)
                    post_expansion_idx = post_expansion_idx + 1

            if sparse_indices_op_output_name is not None:
                add_op = op_adapter.ElementwiseBinaryOp(name=post_expansion_op.name,
                                                        operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
                graph.add(add_op, add_op_inputs, [add_op_output_name], idx=post_expansion_idx+1)
                create_sparse_op = op_adapter.CreateSparseOp(create_sparse_op_name, create_sparse_output_shape)
                graph.add(create_sparse_op, [sparse_indices_op_output_name, add_op_output_name],
                          [create_sparse_output_name], idx=post_expansion_idx+2, sparse_params=sparse_params)
                for i, consumer in enumerate(add_op_consumers):
                    graph.get_buffer(create_sparse_output_name).consumers.add(consumer)
                    consumer.input_names.insert(list_ids[i], create_sparse_output_name)


@register_layer_optimization
class OptimizeElementwiseOrTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR])


@register_layer_optimization
class OptimizeElementwiseXorTranslation(OptimizeElementwiseTranslation):
    def __init__(self):
        OptimizeElementwiseTranslation.__init__(self,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR])


@register_layer_optimization
class OptimizeElementwiseUnaryAbsTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryAsinTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryAtanTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryCeilTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryCosTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryExpTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryFloorTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryLogTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryNegTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG]
        )

    @staticmethod
    def optimize_negation(graph):
        def validate_neg(nodes_tuple):
            for input_name_ in nodes_tuple[0].input_names:
                node_ = graph.get_producer_node(input_name_)
                if node_.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                        all(val == -1 for val in np.array(node_.op.tensor).flatten()):
                    return True

            return False

        # Optimization: -1 * A => Neg(A)
        sequences = [
            [
                ("elementwise_product",
                 ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_neg)
            for node_tuple in matched_node_list:
                prod_node = node_tuple[0]
                non_neg_input_node = None
                neg_const_input_node = None
                for input_name in prod_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    if input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                            all(val == -1 for val in np.array(input_node.op.tensor).flatten()):
                        neg_const_input_node = input_node
                    else:
                        non_neg_input_node = input_node
                neg_const_input_buf = graph.get_buffer(neg_const_input_node.output_names[0])
                non_neg_input_buf = graph.get_buffer(non_neg_input_node.output_names[0])

                if non_neg_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    input_tensor = non_neg_input_node.op.tensor
                    output_tensor = np.negative(input_tensor).astype(input_tensor.dtype)
                    # remove all input of prod to replace with constant node
                    prod_node.input_names = []
                    neg_const_input_buf.consumers.remove(prod_node)
                    non_neg_input_buf.consumers.remove(prod_node)

                    # replace prod with const
                    const_op = op_adapter.ConstantOp(prod_node.op.name, tensor=output_tensor)
                    graph.replace(prod_node.op, const_op)
                    log_debug2("Optimization of -1 * const(A) => Const(B)  complete. Op {} replaced with ConstOp"
                               .format(prod_node.op.name))
                else:
                    # remove const as input to prod, the prod node will then be replaced as Neg
                    neg_const_input_buf.consumers.remove(prod_node)
                    prod_node.input_names.remove(neg_const_input_node.output_names[0])

                    neg_op_name = graph.naming_policy.get_op_name_by_type(ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG,
                                                                          op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG],
                                                                          folded_op=True)
                    neg_op = op_adapter.ElementwiseUnaryOp(neg_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG)
                    graph.replace(prod_node.op, neg_op)
                    log_debug2("Optimization of -1 * A => Neg(A) complete. Op {} replaced with NegOp"
                               .format(prod_node.op.name))

                if len(neg_const_input_buf.consumers) == 0:
                    graph.prune(neg_const_input_node)
                if len(non_neg_input_buf.consumers) == 0:
                    graph.prune(non_neg_input_node)

        # Optimization: A + Neg(B) => A - B
        #               Neg(A) + B => B - A
        #               Neg(A) + Neg(B) => Neg(A) - B
        sequences = [
            [
                ("elementwise_sum",
                 ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_neg", "ANY")]),
                 ())
            ]
        ]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence)
            for node_tuple in matched_node_list:
                sum_node = node_tuple[0]
                neg_node_to_prune = None
                for input_name in sum_node.input_names:
                    input_node = graph.get_producer_node(input_name)
                    input_buf = graph.get_buffer(input_name)
                    if input_node.op.type == op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG]:
                        # if more than consumer of NegOp then we cant remove it hence optimization
                        # is not really relevant.
                        if len(input_buf.consumers) == 1:
                            neg_node_to_prune = input_node

                if neg_node_to_prune is not None:
                    # Update the input and consumer list and remove NegOp from graph
                    neg_idx = sum_node.input_names.index(neg_node_to_prune.output_names[0])
                    sum_input_names = sum_node.input_names[:]
                    neg_input_name = neg_node_to_prune.input_names[0]
                    neg_input_buf = graph.get_buffer(neg_input_name)
                    graph.prune(neg_node_to_prune, force_remove=True)
                    if neg_idx == 0:
                        # got Neg(A) + B, need B - A
                        sum_input_names[0] = sum_input_names[1]
                        sum_input_names[1] = neg_input_name
                    else:
                        # Neg(A) + Neg(B) or A + Neg(B)
                        sum_input_names[neg_idx] = neg_input_name
                    neg_input_buf.consumers.add(sum_node)
                    sum_node.input_names = sum_input_names

                    op_type = ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT
                    legacy_op_type = op_adapter.ElementwiseBinaryOp.operation_to_legacy[op_type]
                    sub_op_name = graph.naming_policy.get_op_name_by_type(op_type, legacy_op_type, folded_op=True)
                    sub_op = op_adapter.ElementwiseBinaryOp(sub_op_name,
                                                            operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT)

                    graph.replace(sum_node.op, sub_op)
                    log_debug2("Optimization of addition to a negative of an op (e.g: A + Neg(B) => A - B) complete. "
                               "Op {} replaced with SubOp"
                               .format(sum_node.op.name))


@register_layer_optimization
class OptimizeElementwiseUnaryNotTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryRoundTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND]
        )


@register_layer_optimization
class OptimizeElementwiseUnaryRsqrtTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT]
        )


@register_layer_optimization
class OptimizeElementwiseUnarySignTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN]
        )


@register_layer_optimization
class OptimizeElementwiseUnarySinTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN]
        )


@register_layer_optimization
class OptimizeElementwiseUnarySqrtTranslation(OptimizeElementwiseUnaryTranslation):
    def __init__(self):
        super().__init__(
            op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT]
        )


@register_layer_optimization
class OptimizeErfTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ErfOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExpandTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExpandOp.TRANSLATION_KEY
        self.register_method(EXPAND_TO_TILE, self.expand_to_tile)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    @staticmethod
    def expand_to_tile(node, graph):
        input_shape = graph.get_input_shapes(node)[0]
        output_shape = graph.get_output_shapes(node)[0]
        # If input rank and output rank matches, but replace expand with tile
        if len(input_shape) == len(output_shape):
            repeats = []
            for in_dim, out_dim in zip(input_shape, output_shape):
                if in_dim == 1:
                    repeats.append(out_dim)
                elif in_dim == out_dim:
                    repeats.append(1)
                else:
                    raise ValueError("Input shape is not compatible with output shape for broadcasting")

            tile_op = op_adapter.TileOp(node.op.name+"_tile", multiples=repeats)

            graph.replace(node.op, tile_op)

    @staticmethod
    def remove_identity(node, graph):
        input_shape = graph.get_input_shapes(node)[0]
        output_shape = graph.get_output_shapes(node)[0]
        if input_shape == output_shape:
            graph.squash(node, input_name=node.input_names[0], is_data_movement_node=True)


@register_layer_optimization
class OptimizeExpandDimsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExpandDimsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if AxisTracker.input_axis_formats_intact(graph, node) and \
                input_buf.axis_format in AxisTracker.AxisFormat.get_valid_formats():
            return False

        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.TransposeOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NC or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.TNF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

            return True


@register_layer_optimization
class OptimizeFullyConnectedTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.FullyConnectedOp.TRANSLATION_KEY
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)
        self.register_method(SQUASH_SUM, self.squash_sum)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.log_axes_transformation(node, graph)
        input_buf = graph.get_input_buffers(node)[0]
        if input_buf.rank() == 4:
            if input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
                AxisTracker.enforce_input_axis_format(graph, input_buf.name, AxisTracker.AxisFormat.NSC,
                                                      AxisTracker.AxisFormat.NCS_TO_NSC)

                # weights axis_format will be set to NONTRIVIAL after transpose
                # to avoid transposing shared weights multiple times
                weights_buf = graph.get_buffer(node.input_names[1])
                # weights expect NCHW order, need to permute
                input_buf = graph.get_input_buffers(node)[0]
                batch, height, width, channel = input_buf.shape
                if weights_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and len(weights_buf.consumers) == 1 and len(weights_buf.consumers) == 1 and not isinstance(graph.src_axis_order, RelayAxisOrder):
                    weights = weights_buf.producer.op.tensor

                    # Assuming FC: W^Tx + b and weights have shape (input_size, output_size)
                    input_size = weights.shape[0]
                    output_size = weights.shape[1]
                    log_assert(input_size == channel * height * width,
                               code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                              (input_size, output_size),
                                                                                              (batch,  height, width, channel)))

                    weights.shape = (channel, height, width, output_size)
                    weights = np.transpose(weights, (3, 1, 2, 0))
                    weights = np.ascontiguousarray(weights, dtype=np.float32)
                    weights.shape = (output_size, input_size)
                    weights_buf.producer.op.tensor = weights
                    weights_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                    weights_buf.shape = list(weights.shape)
                elif not isinstance(graph.src_axis_order, RelayAxisOrder):
                    # Add a reshape op, a transpose op and a reshape op after the weights ops
                    n_size = weights_buf.shape[0]
                    m_size = weights_buf.shape[1]
                    log_assert(n_size == channel * height * width,
                               code_to_message.get_error_message("ERROR_FC_WRONG_INPUT_SIZE")(node.op.name,
                                                                                              (n_size, m_size),
                                                                                              (batch,  height, width, channel)))
                    input_name = node.input_names[1]

                    post_reshape_input_name = input_name + '_post_reshape'
                    if not graph.has_buffer(post_reshape_input_name):
                        post_reshape_op = op_adapter.ReshapeOp(name=post_reshape_input_name, shape=[channel, height, width, m_size])
                        cur_idx = graph.nodes_in_order.index(node)
                        post_reshape_node = graph.add(post_reshape_op, [input_name], [post_reshape_input_name], idx=cur_idx)
                        graph.add_src_op_info(post_reshape_input_name, [input_name], [post_reshape_input_name])
                        # Add op trace data for new created post reshape op and it's output tensor
                        graph.update_trace_info(post_reshape_node, node)
                        graph.update_trace_info(graph.get_buffer(post_reshape_input_name), node)

                    target_format = 'NCS'
                    permute_order = [3, 1, 2, 0]
                    permute_name = graph.get_implicit_permute_node_name(post_reshape_input_name, target_format)
                    if not graph.has_buffer(permute_name):
                        implicit_permute = op_adapter.TransposeOp(permute_name, permute_order)
                        cur_idx = graph.nodes_in_order.index(node)
                        implicit_permute_node = graph.add(implicit_permute, [post_reshape_input_name], [permute_name], idx=cur_idx)
                        graph.add_src_op_info(permute_name, [post_reshape_input_name], [permute_name])
                        # Add op trace data of new created implicit permute node and it's output tensor
                        graph.update_trace_info(implicit_permute_node, node)
                        graph.update_trace_info(graph.get_buffer(permute_name), node)

                    post_reshape_permute_name = permute_name + '_post_reshape'
                    if not graph.has_buffer(post_reshape_permute_name):
                        permute_post_reshape_op = op_adapter.ReshapeOp(name=post_reshape_permute_name, shape=[m_size, n_size])
                        cur_idx = graph.nodes_in_order.index(node)
                        permute_post_reshape_node = graph.add(permute_post_reshape_op, [permute_name], [post_reshape_permute_name], idx=cur_idx)
                        graph.add_src_op_info(post_reshape_permute_name, [permute_name], [post_reshape_permute_name])
                        # Add op trace data of new created permute post reshape node and it's output tensor
                        graph.update_trace_info(permute_post_reshape_node, node)
                        graph.update_trace_info(graph.get_buffer(post_reshape_permute_name), node)

                    # Delete the input buffer's consumers
                    input_buf = graph.get_buffer(input_name)
                    input_consumers = input_buf.consumers
                    if node in input_consumers:
                        input_buf.consumers.remove(node)

                    # Add new buffer's consumers
                    graph.get_buffer(post_reshape_permute_name).consumers.add(node)

                    # Change current node's input
                    node.input_names[1] = post_reshape_permute_name
        else:
            # weights axis_format will be set to NONTRIVIAL after transpose
            weights_buf = graph.get_buffer(node.input_names[1])
            # Modify the weights tensor if the weights is constant and only has one consumer
            if weights_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and len(weights_buf.consumers) == 1 and len(weights_buf.consumers) == 1 and not isinstance(graph.src_axis_order, RelayAxisOrder):
                # again, need to transpose weights for spatial_first order
                weights = weights_buf.producer.op.tensor
                weights = np.ascontiguousarray(np.transpose(weights, (1, 0)))
                weights_buf.producer.op.tensor = weights
                weights_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                weights_buf.shape = list(weights.shape)
            elif not isinstance(graph.src_axis_order, RelayAxisOrder):
                # Add a transpose op after the other weights ops
                input_name = node.input_names[1]
                target_format = 'FN'
                permute_order = [1, 0]
                permute_name = graph.get_implicit_permute_node_name(input_name, target_format)
                if not graph.has_buffer(permute_name):
                    implicit_permute = op_adapter.TransposeOp(permute_name, permute_order)

                    cur_idx = graph.nodes_in_order.index(node)
                    implicit_permute_node = graph.add(implicit_permute, [input_name], [permute_name], idx=cur_idx)
                    graph.add_src_op_info(permute_name, [input_name], [permute_name])
                    # Add op trace data of new created implicit permute node and it's output tensor
                    graph.update_trace_info(implicit_permute_node, node)
                    graph.update_trace_info(graph.get_buffer(permute_name), node)

                # Delete the input buffer's consumers
                input_buf = graph.get_buffer(input_name)
                input_consumers = input_buf.consumers
                if node in input_consumers:
                    input_buf.consumers.remove(node)

                # Add new buffer's consumers
                graph.get_buffer(permute_name).consumers.add(node)

                # Change current node's input
                node.input_names[1] = permute_name

        # since weights are getting transposed here, the info should be updated in transpose_b
        # showing no further transpose is required
        node.op.transpose_b = False

        return True

    @staticmethod
    def squash_batchnorm(graph):

        def validate_squash(nodes_tuple):
            fc_node = nodes_tuple[0]
            # if FC has output_encodings squashing is disabled for better alignment
            # with the expected/simulated accuracy
            # Note: This is not necessarily better accuracy
            if graph.has_quantization_param(fc_node.op.name) and \
                graph.quantization_params[fc_node.op.name]["output_encodings"]:
                return False

            if len(nodes_tuple) == 1:
                bn_node = next(iter(graph.get_output_buffers(fc_node)[0].consumers))
            if len(nodes_tuple) == 2:
                reshape_node = nodes_tuple[1]
                bn_node = next(iter(graph.get_output_buffers(reshape_node)[0].consumers))

            # check if the shapes of weights and biases of FC, BN are broadcastable
            bn_node_weights =  graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
            bn_node_bias =  graph.get_buffer(bn_node.input_names[2]).producer.op.tensor
            weights = graph.get_buffer(fc_node.input_names[1]).producer.op.tensor
            if(len(fc_node.input_names) >= 3):
                bias = graph.get_buffer(fc_node.input_names[2]).producer.op.tensor
            broadcasted_tensor = np.zeros(len(bn_node_weights), dtype=np.float32)
            if not fc_node.op.transpose_b:
                weight_tensor = np.transpose(weights, (1, 0)).copy()
            else:
                weight_tensor = weights.copy()
            if len(fc_node.input_names) >= 3 and \
                translation_utils.broadcastable(weight_tensor.shape, broadcasted_tensor.shape) and \
                translation_utils.broadcastable(bias.shape, bn_node_bias.shape):
                return True
            elif len(fc_node.input_names) == 2 and \
                translation_utils.broadcastable(weight_tensor.shape, broadcasted_tensor.shape):
                return True
            else:
                return False

        sequence1 = [
            ("FullyConnected",
                ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
                ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
             )
        ]

        sequence2 = [
            ("FullyConnected",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
             ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
             )
        ]
        sequence3 = [
            ("FullyConnected",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
               ("MATCH_NUM_BUFS", [("Reshape", "ANY")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("FullyConnected", "ANY")]),
               ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
            )
        ]

        sequences = [sequence1, sequence2, sequence3]
        for idx, sequence in enumerate(sequences):

            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_squash)

            for node_tuple in matched_node_list:
                # sanity check
                log_assert(len(node_tuple) == len(sequence),
                        "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                        len(node_tuple), len(sequence))

                fc_node = node_tuple[0]
                if(idx != 2):
                    bn_node = next(iter(graph.get_output_buffers(fc_node)[0].consumers))
                else:
                    reshape_node = node_tuple[1]
                    bn_node = next(iter(graph.get_output_buffers(reshape_node)[0].consumers))
                fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                bn_node_weights =  graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
                bn_node_bias =  graph.get_buffer(bn_node.input_names[2]).producer.op.tensor
                bn_input_buffer = graph.get_input_buffers(bn_node)[0]
                bn_output_buffer = graph.get_output_buffers(bn_node)[0]
                manage_shared_static_input(graph, fc_node, 1)
                weights = graph.get_buffer(fc_node.input_names[1]).producer.op.tensor

                if(len(fc_node.input_names) >= 3):
                    manage_shared_static_input(graph, fc_node, 2)
                    bias = graph.get_buffer(fc_node.input_names[2]).producer.op.tensor
                broadcasted_tensor = np.zeros(len(bn_node_weights), dtype=np.float32)
                if not fc_node.op.transpose_b:
                    weight_tensor = np.transpose(weights, (1, 0)).copy()
                else:
                    weight_tensor = weights.copy()
                broadcasted_tensor = broadcasted_tensor + weight_tensor
                broadcasted_tensor = broadcasted_tensor * bn_node_weights
                if not fc_node.op.transpose_b:
                    broadcasted_transpose = np.transpose(broadcasted_tensor, (1, 0)).copy()
                else:
                    broadcasted_transpose = broadcasted_tensor.copy()

                graph.get_buffer(fc_node.input_names[1]).producer.op.tensor = broadcasted_transpose
                if(len(fc_node.input_names) >= 3):
                    graph.get_buffer(fc_node.input_names[2]).producer.op.tensor = bias * bn_node_weights + bn_node_bias
                graph.squash(bn_node, input_name=bn_input_buffer.name)
                log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                        fc_node.op.type,
                                                                                        fc_node.op.name))
                # Transferring activation encoding of BN to fullyconnected.
                q = graph.user_quantization_overrides
                if q and 'activation_encodings' in q and bn_output_buffer.name in q['activation_encodings']:
                    activations = q['activation_encodings']
                    version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                    act_encs = [IROptimizations.extract_encoding_dict(fc_node_output_buffer.name, activations[bn_output_buffer.name], version)]
                    graph.add_quantization_params(fc_node.op.name, output_encodings=act_encs)

    def prepare_inputs_as_params(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        bias_buffer = graph.get_buffer(node.input_names[2])
        bias_node = bias_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
            node.op.bias = bias_node.op.tensor
            # Remove the weights/bias inputs from the IR graph
            graph.remove_node_as_consumer(node, weights_buffer.name)
            graph.remove_node_as_consumer(node, bias_buffer.name)
            node.input_names = [node.input_names[0]]

    @staticmethod
    def squash_sum(graph):
        def validate_dims(nodes_tuple):
            fc_node = nodes_tuple[0]
            post_reshape_node = nodes_tuple[1]
            elementwise_sum_node = nodes_tuple[2]
            if(len(fc_node.input_names) >= 3):
                fc_bias_buffer = graph.get_input_buffers(fc_node)[2]
            elementwise_sum_input_buffers = graph.get_input_buffers(elementwise_sum_node)
            elementwise_sum_constant_buffer = elementwise_sum_input_buffers[0]
            if elementwise_sum_input_buffers[0].producer.op.TRANSLATION_KEY != 'constant':
                elementwise_sum_constant_buffer = elementwise_sum_input_buffers[1]
            fc_input_buffer_0 = graph.get_input_buffers(fc_node)[0]
            fc_input_buffer_1 = graph.get_input_buffers(fc_node)[1]
            q = graph.user_quantization_overrides
            # For simplified logic, currently we only support this merge when the bias input to FC is zeros.
            if ((len(fc_node.input_names) >= 3) and not np.all(fc_bias_buffer.producer.op.tensor == 0)):
                return False
            if q and 'activation_encodings' in q and elementwise_sum_constant_buffer.name in q['activation_encodings']:
                version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                encoding = IROptimizations.extract_encoding_dict(elementwise_sum_constant_buffer.name, q['activation_encodings'][elementwise_sum_constant_buffer.name], version)
                # After this optimization, the constant input to the elementwise sum will be the bias for the FC.
                # External overrides (for the constant input) can have bitwidth that is not equal to 8 or 32.
                # Since the bitwidth for the bias can only be 8 or 32 bits, we cannot do this optimization.
                if encoding['bw'] != 8 and encoding['bw'] != 32:
                    return False
            if len(fc_node.input_names) >= 3 and fc_bias_buffer.shape == elementwise_sum_constant_buffer.shape \
                    and fc_node.output_names[0] in post_reshape_node.input_names \
                    and post_reshape_node.output_names[0] in elementwise_sum_node.input_names \
                    and fc_input_buffer_0.rank() == 2 \
                    and fc_input_buffer_1.rank() == 2 \
                    and fc_bias_buffer.rank() == 1 :
                return True
            elif len(fc_node.input_names) == 2 and fc_node.output_names[0] in post_reshape_node.input_names \
                        and post_reshape_node.output_names[0] in elementwise_sum_node.input_names \
                        and fc_input_buffer_0.rank() == 2 \
                        and fc_input_buffer_1.rank() == 2 :
                return True
            return False

        sequence = [
            ("FullyConnected",
             ("MATCH_BUFS_AT_INDEX", [("Reshape", 0), ("constant", 1), ("constant", 2)]), ("MATCH_NUM_BUFS", [("Reshape", "ANY")]),
             ),
            ("FullyConnected",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("FullyConnected", "ANY")]), ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("constant", "ANY"), ("Reshape", "ANY")]), ()
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_dims, ignore_constants=True)
        for node_tuple in matched_node_list:
            fc_node = node_tuple[0]
            reshape_node = node_tuple[1]
            elementwise_sum_node = node_tuple[2]
            fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
            if(len(fc_node.input_names) >= 3):
                fc_bias_node = graph.get_op_input_nodes(fc_node)[2]
            elementwise_sum_input_buffers = graph.get_input_buffers(elementwise_sum_node)
            elementwise_sum_constant_buffer = elementwise_sum_input_buffers[0]
            if elementwise_sum_input_buffers[0].producer.op.TRANSLATION_KEY != 'constant':
                elementwise_sum_constant_buffer = elementwise_sum_input_buffers[1]
            elementwise_sum_constant_node = [node for node in graph.get_op_input_nodes(elementwise_sum_node) if node.op.TRANSLATION_KEY == 'constant'][0]
            elementwise_sum_output_buffer = graph.get_output_buffers(elementwise_sum_node)[0]
            if(len(fc_node.input_names) >= 3):
                fc_bias_buffer = graph.get_output_buffers(fc_bias_node)[0]

                # Replacing Bias node with elementwise sum constant input node
                fc_bias_buffer.consumers.clear()
            fc_node.input_names[2] = elementwise_sum_constant_buffer.name
            elementwise_sum_constant_buffer.consumers.add(fc_node)
            graph.remove_node_as_consumer(elementwise_sum_node, elementwise_sum_constant_buffer.name)
            graph.squash(elementwise_sum_node, graph.get_output_buffers(reshape_node)[0].name)

            # update order from [fully_connected, reshape, bias] to [bias, fully_connected, reshape]
            idx_bias = graph.nodes_in_order.index(elementwise_sum_constant_node)
            idx_fc = graph.nodes_in_order.index(fc_node)
            if idx_bias > idx_fc:
                graph.nodes_in_order.pop(idx_bias)
                graph.nodes_in_order.insert(idx_fc, elementwise_sum_constant_node)

            # Transferring activation encoding of elementwise sum to fullyconnected.
            q = graph.user_quantization_overrides
            if q and 'activation_encodings' in q and elementwise_sum_output_buffer.name in q['activation_encodings']:
                activations = q['activation_encodings']
                version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                act_encs = [IROptimizations.extract_encoding_dict(fc_node_output_buffer.name, activations[elementwise_sum_output_buffer.name], version)]
                graph.add_quantization_params(fc_node.op.name, output_encodings=act_encs)

@register_layer_optimization
class OptimizeGatherTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherOp.TRANSLATION_KEY
        self.register_method(INJECT_CAST_FOR_GATHER, self.inject_cast_for_gather)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)
        self.register_method(HANDLE_GATHER_NEGATIVE_INDICES, self.handle_gather_negative_indices)

    def replace_6d_operation(self, node, graph):
        """
        - GatherOp has data and indices, do the following steps to support 6D operaiton:
            - Insert ReshapeOp to flatten data's dimension except axis
            - Keep indices as the same

        - Example:
            before optimization:
                 data [2,3,5,7] \
                indices [2,3,4] -> Gather(axis=1) -> [2,2,3,4,5,7]

            after optimization:
                 data [2,3,5,7] -> pre-reshape -> [2,3,35]  \
                                           indices [2,3,4] -> Gather(axis=1) -> [2,2,3,4,35] -> post-reshape -> [2,2,3,4,5,7]
        """
        data_shape, indice_shape = graph.get_input_shapes(node)
        output_shapes = graph.get_output_shapes(node)
        data_rank = len(data_shape)
        out_rank = len(output_shapes[0])
        if data_rank <= 5 and out_rank <= 5:
            return
        check_if_6d_supportable(node, graph)

        # new_input_shape
        left_shape = [] if node.op.axis == 0 else [np.prod(data_shape[:node.op.axis])]
        right_shape = [] if node.op.axis == (data_rank-1) else [np.prod(data_shape[node.op.axis+1:])]
        new_input_shape = left_shape + [data_shape[node.op.axis]] + right_shape

        log_assert(
            len(left_shape + indice_shape.dims + right_shape) <= 5,
            f"{node.op.name} are not supported in 6D"
        )

        # insert pre-reshape before GatherOp's data
        pre_reshape_op_name = node.op.name + '_6d_pre_reshape'
        pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_op_name, shape=new_input_shape)
        graph.inject(
            pre_reshape_op, input_name=node.input_names[0],
            output_name=pre_reshape_op_name, consumer_names=[node.op.name]
        )

        # replace GatherOp
        new_gather_op = op_adapter.GatherOp(
            name=node.op.name,
            axis=0 if node.op.axis == 0 else 1
        )
        new_gather_op_output_shape = left_shape + indice_shape.dims + right_shape
        graph.replace(node.op, new_gather_op)

        # insert post-reshape after GatherOp to back to original output_shapes
        post_reshape_insertion(
            node, graph,
            new_out_shapes=[new_gather_op_output_shape],
            orig_out_shapes=output_shapes
        )

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis and if needed permute it for NSC
        # In addition, output buffer axis tracking stays the same as input so long
        # as the rank of indices == 1. Otherwise it's non trivial as the rank will change
        input_name = node.input_names[0]
        indices_name = node.input_names[1]
        input_buf = graph.get_input_buffers(node)[0]
        indices_buf = graph.get_input_buffers(node)[1]
        output_buf = graph.get_output_buffers(node)[0]
        if node.op.axis < 0:
            node.op.axis = node.op.axis+input_buf.rank()

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                log_debug1("{} axis is already in spatial first order {}, no need to reorder.",
                           buf_name, buf_axis_format)
                return
            elif buf_axis_format in [AxisTracker.AxisFormat.NDHWC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NSC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NFC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NTF,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if indices_buf.rank() > 1:
            set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
            set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])
        else:
            if (input_buf.axis_format == AxisTracker.AxisFormat.NDHWC or \
                input_buf.axis_format == AxisTracker.AxisFormat.NSC or \
                input_buf.axis_format == AxisTracker.AxisFormat.NFC) and \
                    node.op.data_axis_formats[0] != input_buf.axis_format:
                if input_buf.rank() == 5:
                    order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
                elif input_buf.rank() == 4:
                    order = AxisTracker.AxisFormat.NCS_TO_NSC
                elif input_buf.rank() == 3:
                    order = AxisTracker.AxisFormat.NCF_TO_NFC
                else:
                    raise ValueError('Unsupported input rank, expected 3/4/5, but got {}.'
                                     .format(input_buf.rank()))
                axis_map = graph.src_axis_order.permute_sequence_from_ir[input_buf.rank() - 1]
                node.op.axis = axis_map[node.op.axis]
                output_buf.axis_format = input_buf.axis_format
                output_buf.shape = AxisTracker.permute_shape(output_buf.shape, order)
            else:
                output_buf.axis_format = input_buf.axis_format

        return True

    def handle_gather_negative_indices(self, node, graph):
        indices_name = node.input_names[1]
        if isinstance(graph.get_producer_op(indices_name), op_adapter.ConstantOp):
            indices_buffer = graph.get_buffer(indices_name)
            indices_buffer_tensor = indices_buffer.producer.op.tensor
            # return if there is no negative indices in the buffer
            if np.all(indices_buffer_tensor >= 0):
                return
            # Don't modify the original indices buffer if there are multiple consumer
            # add a new constant op and modify it.
            if len(indices_buffer.consumers) > 1:
                const_op_name = node.op.name + "_indices"
                const_op = op_adapter.ConstantOp(const_op_name, tensor=indices_buffer_tensor.copy())
                producer_idx = graph.list_nodes().index(indices_buffer.producer)
                const_node = graph.add(const_op, [], [const_op_name], axis_formats=[indices_buffer.axis_format], idx=producer_idx+1)
                # Add op trace data of new created const node and it's output tensor
                graph.update_trace_info(const_node, node)
                graph.update_trace_info(graph.get_buffer(const_op_name), node)

                graph.get_buffer(const_op_name).consumers.add(node)
                indices_buffer.consumers.remove(node)
                node.input_names[1] = const_op_name

            const_op = graph.get_producer_op(node.input_names[1])
            input_data_shape = graph.get_buffer(node.input_names[0]).shape.dims
            with np.nditer(const_op.tensor, op_flags=['readwrite']) as it:
                for index in it:
                    if index < 0:
                        index += input_data_shape[node.op.axis]

    # TODO Remove this optimization once casts are properly optimized out in IR
    def inject_cast_for_gather(self, node, graph):
        cast_node_name = node.input_names[1] + "_cast"
        cast_op = op_adapter.CastOp(name=cast_node_name, to_type="int32")
        # check and reuse existing CastOp if already added
        if graph.has_buffer(cast_node_name):
            cast_buffer = graph.buffers[cast_node_name]
            cast_buffer.consumers.add(node)
            input_buffer = graph.buffers[node.input_names[1]]
            input_buffer.consumers.remove(node)
            node.input_names[1] = cast_node_name
        else:
            log_debug("Injecting cast op {} for node {}'s indices input.".format(cast_node_name, node.op.name))
            graph.inject(cast_op, input_name=node.input_names[1], output_name=cast_node_name, consumer_names=[node.op.name])

    @staticmethod
    def remove_identity(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        indices_buffer = graph.get_input_buffers(node)[1]
        output_buffer_shape = graph.get_output_buffers(node)[0].shape
        if input_buffer.shape.is_dynamic() or indices_buffer.shape.is_dynamic():
            return
        if input_buffer.shape == output_buffer_shape and len(input_buffer.consumers) == 1 and \
                indices_buffer.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # examine cases that indices make output exactly the same as input
            # i.e. indices = [0,1,...,n-1] along the gather axis w/ dim=n
            if all(indices_buffer.producer.op.tensor == list(range(input_buffer.shape.dims[node.op.axis]))):
                # remove current gather op from indices op's consumers
                if node in indices_buffer.consumers:
                    indices_buffer.consumers.remove(node)
                    node.input_names.remove(indices_buffer.name)
                # this gather has no effect, remove indices first
                if len(indices_buffer.consumers) == 0:
                    indices_node = indices_buffer.producer
                    graph.prune(indices_node, force_remove=True)
                # then remove gather
                ret = graph.squash(node, input_name=input_buffer.name, is_data_movement_node=True)
                if ret:
                    log_debug("Squash Gather op {} due to IdentityOp. "
                              "Input shape {}".format(node.op.name,
                                                      input_buffer.shape))


@register_layer_optimization
class OptimizeGatherElementsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherElementsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Remap the axis if < 0 to the real axis
        input_name, indices_name = node.input_names
        input_buf, indices_buf = graph.get_input_buffers(node)
        if node.op.axis < 0:
            node.op.axis = node.op.axis+input_buf.rank()

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                log_debug1("{} axis is already in spatial first order {}, no need to reorder.", buf_name, buf_axis_format)
                return
            elif buf_axis_format in [AxisTracker.AxisFormat.NDHWC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NSC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NFC,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format in [AxisTracker.AxisFormat.NTF,
                                     AxisTracker.AxisFormat.NONTRIVIAL] and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])

        return True

@register_layer_optimization
class OptimizeGatherNDTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GatherNDOp.TRANSLATION_KEY
        self.register_method(MATCH_GATHERND, self.match_gathernd)

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name = node.input_names
        input_buf, indices_buf = graph.get_input_buffers(node)

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                log_debug1("{} axis is already in spatial first order {}, no need to reorder.", buf_name, buf_axis_format)
                return
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if AxisTracker.input_axis_formats_intact(graph, node):
            return False

        # Check and reorder buffers axis to spatial first.
        # All inputs need to be in source framework order.
        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])

        return True

    @staticmethod
    def match_gathernd(graph):
        sequence = [
            (ir_graph.QNN_OP_TRANSPOSE,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_TRANSPOSE, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            (ir_graph.QNN_OP_GATHER,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ANY"), ("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_TRANSPOSE, "ALL")])
            ),
            (ir_graph.QNN_OP_TRANSPOSE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_TRANSPOSE, "ALL")]),
             ()
            )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)
        for node_tuple in matched_node_list:
            node_types_seq = [node.op.type for node in node_tuple]
            gather_node_idx = node_types_seq.index(ir_graph.QNN_OP_GATHER)
            gather_node = node_tuple[gather_node_idx]

            # Turn Add(Mul(X, c)+Y) into Pack([X, Y], axis=2)
            multi_node = node_tuple[2]
            graph.squash(multi_node, multi_node.input_names[0])

            add_node = node_tuple[3]
            add_node_consumers = graph.get_output_buffers(add_node)[0].consumers

            pack_op = op_adapter.PackOp(add_node.op.name + '_pack', axis=2)
            graph.replace(add_node.op, pack_op)

            # Keep first Transpose and Gather, then squash the rest of nodes in sequence
            graph.squash(node_tuple[1], node_tuple[1].input_names[0], squash_into_next=True)
            for node in node_tuple[1:]:
                if node != gather_node and node in graph.list_nodes():
                    graph.squash(node, node.input_names[0])

            # Replace Gather by GatherND
            data_shape = graph.get_input_shapes(gather_node)[0]
            indices_shape = graph.get_input_shapes(gather_node)[1]

            gathernd_op = op_adapter.GatherNDOp(name=gather_node.op.name, batch_dims=0)
            graph.replace(gather_node.op, gathernd_op)

            permute_op_name = gather_node.op.name + '_permute'
            permute_op = op_adapter.TransposeOp(permute_op_name, perm=[2,3,0,1])
            permute_node = graph.inject(permute_op, input_name=gather_node.output_names[0], output_name=permute_op_name)


@register_layer_optimization
class OptimizeGenerateProposalsOp(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GenerateProposalsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # feature input
        feature_name = node.input_names[0]
        feature_buf = graph.get_buffer(feature_name)
        if feature_buf.axis_format != node.op.data_axis_formats[0]:
            AxisTracker.enforce_input_axis_format(graph, feature_name, AxisTracker.AxisFormat.NSC,
                                                  AxisTracker.AxisFormat.NCS_TO_NSC)

        # transform input
        transform_name = node.input_names[1]
        transform_buf = graph.get_buffer(transform_name)
        if transform_buf.axis_format != node.op.data_axis_formats[1]:
            AxisTracker.enforce_input_axis_format(graph, transform_name, AxisTracker.AxisFormat.NSC,
                                                  AxisTracker.AxisFormat.NCS_TO_NSC)

        return True


@register_layer_optimization
class OptimizeGridSampleTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GridSampleOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        log_assert(len(input_bufs[0].shape) in [4, 5],
                   "GridSample op only support 4D or 5D input shape, got input shape {} for op {}.".format(input_bufs[0].shape, node.op.name))

        has_changed = True
        if input_bufs[0].axis_format == node.op.data_axis_formats[0] and \
                input_bufs[0].axis_format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC]:
            # No change
            has_changed = False
        else:
            # 1.Change the input_buf[0]'s layout as Spacial First layout
            if input_bufs[0].axis_format == AxisTracker.AxisFormat.NCS:
                graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NSC,
                                            AxisTracker.AxisFormat.NCS_TO_NSC, [node.op.name])
            elif input_bufs[0].axis_format == AxisTracker.AxisFormat.NCDHW:
                graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NDHWC,
                                            AxisTracker.AxisFormat.NCDHW_TO_NDHWC, [node.op.name])
            elif input_bufs[0].axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                if len(input_bufs[0].shape) == 4:
                    graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NSC,
                                                AxisTracker.AxisFormat.NCS_TO_NSC, [node.op.name])
                elif len(input_bufs[0].shape) == 5:
                    graph.inject_implicit_permute(input_bufs[0].name, AxisTracker.AxisFormat.NDHWC,
                                                AxisTracker.AxisFormat.NCDHW_TO_NDHWC, [node.op.name])

            # 2.Change the output_buf[0]'s layout as Spacial First layout
            # Current input layout is already Spacial First layout
            if node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS]:
                output_buf = graph.get_output_buffers(node)[0]
                if output_buf.rank() == 4:
                    output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCS_TO_NSC)
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    node.op.output_shape = output_buf.shape
                elif output_buf.rank() == 5:
                    output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    node.op.output_shape = output_buf.shape

        # We don't want to change the shape and axis for the second input, so revert it.
        ret = AxisTracker.revert_input_axis_format(graph, node, input_bufs[1].name, input_bufs[1].axis_format, node.op.data_axis_formats[1])
        has_changed = has_changed or ret

        return has_changed


@register_layer_optimization
class OptimizeGroupNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GroupNormOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # channel dimension should be last dimension of the input for groupnorm as
        # per QNN opdef.
        input_axis_formats_before = graph.get_input_axis_formats(node)
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_axis_formats_after = graph.get_input_axis_formats(node)
        input_buffers = graph.get_input_buffers(node)
        for i, buf in enumerate(input_buffers):
            if input_axis_formats_before[i] != input_axis_formats_after[i]:
                transpose_node = buf.producer
                graph.update_trace_info(transpose_node, [node])
                graph.update_trace_info(buf, [node])
        output_buffer = graph.get_output_buffers(node)[0]
        # (image/feature)_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
        # Enforce the output format here to be NDHWC/NSC/NFC
        output_buffer.axis_format = AxisOrder().get_axis_format(len(output_buffer.shape))

    def merge_low_level_ops_to_layers(self, graph):
        # ---------------- Utilities for Matching Strategies -------------------
        def validate(node_tuple):
            def validate_reshape_shape(node_tuple):
                first_reshape_node = node_tuple[0]
                second_reshape_node = node_tuple[2]
                input_name = first_reshape_node.input_names[0]
                if graph.has_buffer(input_name):
                    first_shape = graph.get_buffer(input_name).shape
                    second_shape = second_reshape_node.op.shape.tolist()
                    if len(first_shape) < 2:
                        return False
                    if len(first_shape) == len(second_shape):
                        if first_shape != second_shape:
                            return False
                    elif len(first_shape) > len(second_shape):
                        # For cases where length of first shape is greater than second shape eg.
                        # first_shape = [1, 512, 128, 85] and second_shape = [1, 512, 10880]
                        # Check if all the values except the last one for second_shape is the same as first_shape
                        last_index = len(second_shape) - 1
                        if first_shape[:last_index] != second_shape[:last_index]:
                            return False
                        # Check if the last value in the second_shape is the combination of the rest of the values
                        # in first_shape
                        if second_shape[-1] != math.prod(first_shape[last_index:]):
                            return False
                else:
                    raise ValueError('There is no such buffer {} in the graph.'.format(input_name))
                return True

            # Check if the gamma and beta has the same shape of Channel
            def validate_gamma_beta_shape(node_tuple):
                channel = get_channel(node_tuple)
                if not channel:
                    return False
                # Check gamma's shape
                if len(node_tuple) > 3:
                    mul_node = node_tuple[3]
                    previous_node = node_tuple[2]
                    out_buff = previous_node.output_names
                    for input_name in mul_node.input_names:
                        if input_name not in out_buff and graph.has_buffer(input_name):
                            gamma_buff = graph.get_buffer(input_name)
                            tmp_tensor = np.zeros(gamma_buff.shape)
                            gamma_tensor = np.atleast_1d(np.squeeze(tmp_tensor))
                            if list(gamma_tensor.shape) != [channel]:
                                return False
                # Check beta's shape
                if len(node_tuple) > 4:
                    add_node = node_tuple[-1]
                    mul_out_buff = mul_node.output_names
                    for input_name in add_node.input_names:
                        if input_name not in mul_out_buff and graph.has_buffer(input_name):
                            beta_buff = graph.get_buffer(input_name)
                            tmp_tensor = np.zeros(beta_buff.shape)
                            beta_tensor = np.atleast_1d(np.squeeze(tmp_tensor))
                            if list(beta_tensor.shape) != [channel]:
                                return False
                return True

            # Check if the sequences can be merge without missing override encodings
            def validate_not_override(node_tuple):
                if graph.has_quantization_param(node_tuple[-1].op.name):
                    output_encodings = graph.get_layer_quantization_param(node_tuple[-1].op.name)[op_graph.QuantParams.OUTPUT_ENCODINGS]
                    if len(output_encodings) > 0:
                        return True
                # Check if there's any override encodings for intermediate tensors
                for node in node_tuple[:-1]:
                    if graph.has_quantization_param(node.op.name):
                        return False
                return True

            def validate_following_op(node_tuple):
                if len(node_tuple) in [3, 4]:
                    last_node = node_tuple[-1]
                    last_buffer = graph.get_output_buffers(last_node)[0]
                    consumers = last_buffer.consumers
                    target_op_type = "elementwise_product" if len(node_tuple) == 3 else "elementwise_sum"
                    for consumer in consumers:
                        if consumer.op.type == target_op_type:
                            return False
                return True

            def validate_group(node_tuple):
                group = get_group(node_tuple)
                if not graph:
                    return False
                instancenorm_node = node_tuple[1]
                instancenorm_group = graph.get_input_buffers(instancenorm_node)[1].shape[0]
                if group == instancenorm_group:
                    return True
                return False

            if (validate_reshape_shape(node_tuple) and \
                validate_gamma_beta_shape(node_tuple) and \
                validate_not_override(node_tuple) and \
                validate_following_op(node_tuple) and \
                validate_group(node_tuple)):
                return True
            else:
                return False

        def get_epsilon(instancenorm_node):
            epsilon = instancenorm_node.op.epsilon
            return epsilon

        # Group is the first reshape node's C dim of shape param
        def get_group(node_tuple):
            reshape_node = node_tuple[0]
            instance_norm_node = node_tuple[1]
            shape = list(reshape_node.op.shape)
            instance_norm_input_buf = graph.get_input_buffers(instance_norm_node)[0]
            # For InstanceNorm it is an axis layout sensitive op. The 1st reshape output axis layout will be reset same
            # as the InstanceNorm input axis layout after translation. So here we followed the InstanceNorm input axis
            # format to fetch the group information
            if instance_norm_input_buf.axis_format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC]:
                group = shape[-1]
                return group
            elif instance_norm_input_buf.axis_format in [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF]:
                group = shape[1]
                return group
            else:
                # cannot fetch valid group, return None for sequence match validation to handle
                log_warning("Fetch group for GroupNorm failed since only '{}' are expected for the 1st reshape output buffer but got '{}'"
                            .format("[NDHWC, NCDHW, NHWC, NCHW, NFC, NCF]", instance_norm_input_buf.axis_format))
                return None

        def get_channel(node_tuple):
            reshape_node = node_tuple[0]
            instance_norm_node = node_tuple[1]
            input_buf = graph.get_input_buffers(reshape_node)[0]
            input_shape = input_buf.shape
            instance_norm_input_buf = graph.get_input_buffers(instance_norm_node)[0]
            # For InstanceNorm it is an axis layout sensitive op. The 1st reshape output axis layout will be reset same
            # as the InstanceNorm input axis layout after translation. So here we followed the InstanceNorm input axis
            # format to fetch the channel information
            if instance_norm_input_buf.axis_format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC]:
                channel = input_shape[-1]
                return channel
            elif instance_norm_input_buf.axis_format in [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF]:
                channel = input_shape[1]
                return channel
            else:
                # cannot fetch valid channel, return None for sequence match validation to handle
                log_warning("Fetch channel for GroupNorm failed since only '{}' are expected for the 1st reshape output buffer but got '{}'"
                            .format("[NDHWC, NCDHW, NHWC, NCHW, NFC, NCF]", instance_norm_input_buf.axis_format))
                return None

        def get_gamma(node_tuple):
            # match_base_groupnorm and match_no_beta
            if len(node_tuple) > 3:
                mul_node = node_tuple[3]
                previous_node = node_tuple[2]
                out_buff = previous_node.output_names
                for input_name in mul_node.input_names:
                    if input_name not in out_buff:
                        if graph.has_buffer(input_name):
                            # For example: change the weight shape from [C,1,1] to [C]
                            weight_buff = graph.get_buffer(input_name)
                            weights_node = weight_buff.producer
                            if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and len(weight_buff.consumers) <= 1:
                                tensor = weights_node.op.tensor
                                weights_node.op.tensor = np.atleast_1d(np.squeeze(tensor))
                                weight_buff.shape = list(weights_node.op.shape)
                                # gamma must be 1D tensor so directly update the axis format to ANY
                                # after modify the tensor, to avoid layout mismatch
                                weight_buff.axis_format = AxisTracker.AxisFormat.ANY
                                return input_name
                            else:
                                weight_reshape_name = input_name + "_squeeze"
                                if graph.has_buffer(weight_reshape_name):
                                    return weight_reshape_name
                                weight_reshape_op = op_adapter.ReshapeOp(name=weight_reshape_name,
                                                                         shape=[-1])
                                weight_reshape_op_node = graph.add(weight_reshape_op,
                                          input_names=[input_name],
                                          output_names=[weight_reshape_name],
                                          idx=graph.nodes_in_order.index(weights_node)+1)
                                # update trace info for new create op
                                graph.update_trace_info(weight_reshape_op_node, [node_tuple])
                                return weight_reshape_name

                raise ValueError('Couldn\'t get the gamma input for GroupNorm node.')
            # match_no_affine_transformation
            elif len(node_tuple) == 3:
                return None

        def get_beta(node_tuple):
            # match_base_groupnorm
            if len(node_tuple) > 4:
                add_node = node_tuple[-1]
                previous_node = node_tuple[-2]
                out_buff = previous_node.output_names
                for input_name in add_node.input_names:
                    if input_name not in out_buff:
                        if graph.has_buffer(input_name):
                            # Change the bias shape from [C,1,1] to [C]
                            bias_buff = graph.get_buffer(input_name)
                            bias_node = bias_buff.producer
                            if bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and len(bias_buff.consumers) <= 1:
                                tensor = bias_node.op.tensor
                                bias_node.op.tensor = np.atleast_1d(np.squeeze(tensor))
                                bias_buff.shape = list(bias_node.op.shape)
                                # beta must be 1D tensor so directly update the axis format to ANY
                                # after modify the tensor, to avoid layout mismatch
                                bias_buff.axis_format = AxisTracker.AxisFormat.ANY
                                return input_name
                            else:
                                bias_reshape_name = input_name + "_squeeze"
                                if graph.has_buffer(bias_reshape_name):
                                    return bias_reshape_name
                                bias_reshape_op = op_adapter.ReshapeOp(name=bias_reshape_name,
                                                                       shape=[-1])
                                bias_reshape_node = graph.add(bias_reshape_op,
                                          input_names=[input_name],
                                          output_names=[bias_reshape_name],
                                          idx=graph.nodes_in_order.index(bias_node)+1)
                                graph.update_trace_info(bias_reshape_node, [node_tuple])
                            return bias_reshape_name
                raise ValueError('Couldn\'t get the beta input for GroupNorm node.')
            # match_no_affine_transformation and match_no_beta
            else:
                return None

        def make_groupnorm_op(node_tuple):
            first_op = node_tuple[0].op
            first_op_name = first_op.name
            group_norm_op_name = first_op_name + '_GroupNorm'
            group_norm_op = op_adapter.GroupNormOp(name = group_norm_op_name,
                                                   epsilon = get_epsilon(node_tuple[1]),
                                                   group = get_group(node_tuple))
            return group_norm_op

        def get_groupnorm_input_names(node_tuple):
            input_buffers = graph.get_input_buffers(node_tuple[0])
            input_names = [buf.name for buf in input_buffers]
            # Add weight and bias inputs
            gamma_input = get_gamma(node_tuple)
            beta_input = get_beta(node_tuple)
            if gamma_input:
                input_names.append(gamma_input)
            if beta_input:
                input_names.append(beta_input)
            return input_names

        def get_groupnorm_output_names_and_idx(node_tuple):
            output_buffers = graph.get_output_buffers(node_tuple[-1])
            output_names = [buf.name for buf in output_buffers]

            # Get previous input_name idx for output_buffers[0]
            output_consumers = output_buffers[0].consumers
            output_consumers_dict = {}
            for consumer in output_consumers:
                buf_idx = consumer.input_names.index(output_names[0])
                output_consumers_dict[consumer] = buf_idx

            # return output_names and the idx_dict
            return output_names, output_consumers_dict

        def update_groupnorm_consumers_inputs(output_names, output_consumers_dict):
            # Restore the input_name for following nodes
            for consumer in output_consumers_dict.keys():
                buf_idx = output_consumers_dict[consumer]
                consumer.input_names[buf_idx] = output_names[0]

        # set the previous last_node_consumers as groupnorm_node's outputs consumers
        def update_groupnorm_output_buffer(groupnorm_node, last_node_consumers):
            for output_name in groupnorm_node.output_names:
                output_buf = graph.get_buffer(output_name)
                output_buf.consumers = last_node_consumers

        def save_old_encodings(last_node):
            if graph.has_quantization_param(last_node.op.name):
                return graph.quantization_params[last_node.op.name]
            else:
                return None

        def update_groupnorm_output_encodings(last_node, old_encodings):
            if old_encodings:
                output_encodings = old_encodings[op_graph.QuantParams.OUTPUT_ENCODINGS]
                if len(output_encodings) > 0:
                    output_encoding = output_encodings[0].copy()
                    output_encoding['name'] = groupnorm_node.output_names[0]
                    graph.add_quantization_params(groupnorm_node.op.name, output_encodings=output_encoding)

                param_encodings = old_encodings[op_graph.QuantParams.PARAM_ENCODINGS]
                if len(param_encodings) > 0:
                    param_encoding = param_encodings[0].copy()
                    if last_node.op.type == 'elementwise_product':
                        param_encoding['name'] = 'weight'
                    elif last_node.op.type ==  'elementwise_sum':
                        param_encoding['name'] = 'bias'
                    graph.add_quantization_params(groupnorm_node.op.name, param_encodings=param_encoding)

        def squash_node_tuple(node_tuple):
            for node in node_tuple[:-1]:
                input_names = node.input_names[:]
                # pick squashable input based on whether it produced by constantOp
                input_name = [name for name in input_names if (
                            not isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                input_names.remove(input_name)
                for input_name_ in input_names:
                    # disconnect rest of inputs from node
                    # skip if current input_name_ equal to the input_name to squash
                    if input_name_ == input_name:
                        continue
                    input_buf_ = graph.get_buffer(input_name_)
                    input_buf_.consumers.remove(node)
                    node.input_names.remove(input_name_)
                graph.squash(node, input_name=input_name, squash_into_next=True)

        def check_shape_mismatch(node_tuple):
            first_reshape_node = node_tuple[0]
            second_reshape_node = node_tuple[2]
            input_name = first_reshape_node.input_names[0]
            first_shape = graph.get_buffer(input_name).shape
            second_shape = second_reshape_node.op.shape.tolist()

            return len(first_shape) != len(second_shape)


        # ---------------------------- Main Function ---------------------------
        sequence1 = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("InstanceNorm", "ALL")])
             ),
            ("InstanceNorm",
             ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("InstanceNorm", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("Reshape", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ())
        ]

        sequence2 = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("InstanceNorm", "ALL")])
             ),
            ("InstanceNorm",
             ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("InstanceNorm", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("Reshape", "ANY")]),
             ()
             )
        ]

        sequence3 = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("InstanceNorm", "ALL")])
             ),
            ("InstanceNorm",
             ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("InstanceNorm", "ALL")]),
             ()
             )
        ]

        sequences = [sequence1, sequence2, sequence3]

        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate, ignore_constants=True)

            for node_tuple in matched_node_list:
                # Add reshape node before the first reshape in the sequence if the input shape is not equal to the
                # parameter shape of second reshape node eg. input shape = [1, 512, 128, 85] param shape for second
                # reshape node = [1, 512, 10880]
                if check_shape_mismatch(node_tuple):
                    reshape_op = op_adapter.ReshapeOp(name=node_tuple[0].op.name + '_preReshape',
                                                      shape=node_tuple[2].op.shape)
                    output_name = reshape_op.name + '_output'

                    graph.inject(reshape_op, input_name=node_tuple[0].input_names[0], output_name=output_name,
                                 consumer_names=[node_tuple[0].op.name])

                # Get groupnorm_node's input_names and output_names
                input_names = get_groupnorm_input_names(node_tuple)
                output_names, output_consumers_dict = get_groupnorm_output_names_and_idx(node_tuple)

                # Create groupnorm node
                groupnorm_op = make_groupnorm_op(node_tuple)
                old_encodings = save_old_encodings(node_tuple[-1])

                # Squash all nodes except the last node then replace it by new node
                squash_node_tuple(node_tuple)
                graph.replace(node_tuple[-1].op, groupnorm_op)

                # update the input names of the groupnorm node
                groupnorm_node = graph.nodes_by_name[groupnorm_op.name]
                groupnorm_node.input_names = input_names
                groupnorm_op.data_axis_formats = []
                for name in input_names:
                    # Append null axis format if the input is null
                    groupnorm_op.data_axis_formats.append(graph.get_buffer(name).axis_format if name else graph.null_buffer.axis_format)
                    graph.get_buffer(name).consumers.add(groupnorm_node)
                # Restore the input_name for following nodes
                update_groupnorm_consumers_inputs(output_names, output_consumers_dict)

                # Add consumer for output_buffers[0]
                last_node = node_tuple[-1]
                last_node_consumers = set(graph.get_op_output_nodes(last_node))
                update_groupnorm_output_buffer(groupnorm_node, last_node_consumers)

                # Update groupnorm output encodings
                update_groupnorm_output_encodings(last_node, old_encodings)


@register_layer_optimization
class OptimizeGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.GruOp.TRANSLATION_KEY
        self.register_method(EXPAND_GRU_OP_STRUCTURE, self.expand_gru_op_structure)

    def get_dims(self, buffer_shape, time_major_param=False):
        if time_major_param:
            return [buffer_shape[1], buffer_shape[0], buffer_shape[2]]
        else:
            return buffer_shape

    def align_to_source_output_names(self, graph, current_output_names, source_output_names, align_only_in_src_output=True):
        # If the current_output names are in graph.output_names, then we need to align
        if align_only_in_src_output:
            new_current, new_src = [], []
            for idx, name in enumerate(source_output_names):
                if name in graph.output_names:
                    new_current.append(current_output_names[idx])
                    new_src.append(name)
            current_output_names, source_output_names = new_current, new_src

        # Replace current name with source name for alignment
        for current_name, source_name in zip(current_output_names, source_output_names):
            # override encoding info update
            pre_node = graph.get_producer_node(current_name)
            if graph.has_quantization_param(pre_node.op.name):
                src_encodings = graph.quantization_params[pre_node.op.name]['output_encodings']
                for i in range(len(src_encodings)):
                    if (src_encodings[i]["name"] == current_name):
                        src_encodings[i]["name"] = source_name
                        break

            buf = graph.get_buffer(current_name)
            if source_name in graph.buffers:
                raise ValueError("Buffer {} already exists in graph, duplicate buffer name when replacing buffer {} with it".format(
                        source_name, current_name))

            # Update consumers input name
            for consumer in list(buf.consumers):
                # The consumer may have the same buffer as input twice
                consumer.input_names = [source_name if name == current_name else name for name in consumer.input_names]

            # Update producer output name
            producer_node = graph.get_producer_node(current_name)
            idx = producer_node.output_names.index(current_name)
            producer_node.output_names[idx] = source_name

            # Update buffer in graph
            buf.name = source_name
            graph.buffers[source_name] = graph.buffers.pop(current_name)

    # (1) - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    def expand_gru_update_gate(self, graph, gru_node, Xt, Ht_1):
        gru_node_name = gru_node.op.name

        # update gate (z)
        Wz_name = gru_node.input_names[1]
        Rz_name = gru_node.input_names[4]
        WBz_name = gru_node.input_names[7]
        RBz_name = gru_node.input_names[10]
        Wbz_size = graph.get_buffer(WBz_name).shape[-1]

        xt_wz_matmul_op_name = gru_node_name + "_xt_wz_matmul_op"
        Wbz = np.zeros(Wbz_size, dtype=np.float32)
        xt_wz_matmul_op = op_adapter.MatMulOp(name=xt_wz_matmul_op_name,
                                                bias=Wbz,
                                                transpose_in0=False,
                                                transpose_in1=True)
        xt_wz_matmul_node = graph.add(xt_wz_matmul_op, input_names=[Xt, Wz_name],
                                      output_names=[xt_wz_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created xt_wz_matmul_node and it's output tensor
        graph.update_trace_info(xt_wz_matmul_node, gru_node)
        graph.update_trace_info(graph.get_buffer(xt_wz_matmul_op_name), gru_node)

        ht_1_rz_matmul_op_name = gru_node_name + "_ht_1_rz_matmul_op_name"
        Wbh = np.zeros(Wbz_size, dtype=np.float32)
        ht_1_rz_matmul_op = op_adapter.MatMulOp(name=ht_1_rz_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        ht_1_rz_matmul_node = graph.add(ht_1_rz_matmul_op, input_names=[Ht_1, Rz_name],
                                        output_names=[ht_1_rz_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created ht_1_rz_matmul_node and it's output tensor
        graph.update_trace_info(ht_1_rz_matmul_node, gru_node)
        graph.update_trace_info(graph.get_buffer(ht_1_rz_matmul_op_name), gru_node)

        elementsum_of_term1_part1_op_name = gru_node_name + "_elementsum_of_term1_part1_op"
        elementsum_of_term1_part1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_part1_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        elementsum_of_term1_part1_node = graph.add(elementsum_of_term1_part1_op, input_names=[xt_wz_matmul_op_name, ht_1_rz_matmul_op_name],
                                                   output_names=[elementsum_of_term1_part1_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created elementsum_of_term1_part1_node and it's output tensor
        graph.update_trace_info(elementsum_of_term1_part1_node, gru_node)
        graph.update_trace_info(graph.get_buffer(elementsum_of_term1_part1_op_name), gru_node)

        control_gate_b_name = gru_node_name + '_control_gate_b'
        control_gate_b_op = op_adapter.ElementwiseBinaryOp(name=control_gate_b_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        control_gate_b_node = graph.add(control_gate_b_op, input_names=[WBz_name, RBz_name],
                                        output_names=[control_gate_b_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created control_gate_b_node and it's output tensor
        graph.update_trace_info(control_gate_b_node, gru_node)
        graph.update_trace_info(graph.get_buffer(control_gate_b_name), gru_node)

        elementsum_of_term1_bias_op_name = gru_node_name + "_elementsum_of_term1_bias_op"
        elementsum_of_term1_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term1_bias_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        elementsum_of_term1_bias_node = graph.add(elementsum_of_term1_bias_op, input_names=[elementsum_of_term1_part1_op_name, control_gate_b_name],
                                                  output_names=[elementsum_of_term1_bias_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created elementsum_of_term1_bias_node and it's output tensor
        graph.update_trace_info(elementsum_of_term1_bias_node, gru_node)
        graph.update_trace_info(graph.get_buffer(elementsum_of_term1_bias_op_name), gru_node)

        activation_update_op_name = gru_node_name + "_activation_update_op"
        activation_update_op = op_adapter.ElementwiseNeuronOp(name=activation_update_op_name, operation=op_adapter.ElementwiseNeuronOp.neuron_to_operation[gru_node.op.activation])
        activation_update_node = graph.add(activation_update_op, input_names=[elementsum_of_term1_bias_op_name],
                                 output_names=[activation_update_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created activation_update_node and it's output tensor
        graph.update_trace_info(activation_update_node, gru_node)
        graph.update_trace_info(graph.get_buffer(activation_update_op_name), gru_node)

        return activation_update_op_name

    # (2) - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    def expand_gru_reset_gate(self, graph, gru_node, Xt, Ht_1):
        gru_node_name = gru_node.op.name

        # reset gate (r)
        Wr_name = gru_node.input_names[2]
        Rr_name = gru_node.input_names[5]
        WBr_name = gru_node.input_names[8]
        RBr_name = gru_node.input_names[11]
        Wbr_size = graph.get_buffer(WBr_name).shape[-1]

        xt_wr_matmul_op_name = gru_node_name + "_xt_wr_matmul_op"
        Wbr = np.zeros(Wbr_size, dtype=np.float32)
        xt_wr_matmul_op = op_adapter.MatMulOp(name=xt_wr_matmul_op_name,
                                                bias=Wbr,
                                                transpose_in0=False,
                                                transpose_in1=True)
        xt_wr_matmul_node = graph.add(xt_wr_matmul_op, input_names=[Xt, Wr_name],
                                      output_names=[xt_wr_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created xt_wr_matmul_node and it's output tensor
        graph.update_trace_info(xt_wr_matmul_node, gru_node)
        graph.update_trace_info(graph.get_buffer(xt_wr_matmul_op_name), gru_node)

        ht_1_rr_matmul_op_name = gru_node_name + "_ht_1_rr_matmul_op_name"
        Wbh = np.zeros(Wbr_size, dtype=np.float32)
        ht_1_rr_matmul_op = op_adapter.MatMulOp(name=ht_1_rr_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        ht_1_rr_matmul_node = graph.add(ht_1_rr_matmul_op, input_names=[Ht_1, Rr_name],
                                        output_names=[ht_1_rr_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created ht_1_rr_matmul_node and it's output tensor
        graph.update_trace_info(ht_1_rr_matmul_node, gru_node)
        graph.update_trace_info(graph.get_buffer(ht_1_rr_matmul_op_name), gru_node)

        elementsum_of_term2_part1_op_name = gru_node_name + "_elementsum_of_term2_part1_op"
        elementsum_of_term2_part1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_part1_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        elementsum_of_term2_part1_node = graph.add(elementsum_of_term2_part1_op, input_names=[xt_wr_matmul_op_name, ht_1_rr_matmul_op_name],
                                                   output_names=[elementsum_of_term2_part1_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created elementsum_of_term2_part1_node and it's output tensor
        graph.update_trace_info(elementsum_of_term2_part1_node, gru_node)
        graph.update_trace_info(graph.get_buffer(elementsum_of_term2_part1_op_name), gru_node)

        reset_gate_b_name = gru_node_name + '_reset_gate_b'
        reset_gate_b_op = op_adapter.ElementwiseBinaryOp(name=reset_gate_b_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        reset_gate_b_node = graph.add(reset_gate_b_op, input_names=[WBr_name, RBr_name],
                                      output_names=[reset_gate_b_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created reset_gate_b_node and it's output tensor
        graph.update_trace_info(reset_gate_b_node, gru_node)
        graph.update_trace_info(graph.get_buffer(reset_gate_b_name), gru_node)

        elementsum_of_term2_bias_op_name = gru_node_name + "_elementsum_of_term2_bias_op"
        elementsum_of_term2_bias_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term2_bias_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        elementsum_of_term2_bias_node = graph.add(elementsum_of_term2_bias_op, input_names=[elementsum_of_term2_part1_op_name, reset_gate_b_name],
                                                  output_names=[elementsum_of_term2_bias_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created elementsum_of_term2_bias_node and it's output tensor
        graph.update_trace_info(elementsum_of_term2_bias_node, gru_node)
        graph.update_trace_info(graph.get_buffer(elementsum_of_term2_bias_op_name), gru_node)

        activation_reset_op_name = gru_node_name + "_activation_reset_op"
        activation_reset_op = op_adapter.ElementwiseNeuronOp(name=activation_reset_op_name, operation=op_adapter.ElementwiseNeuronOp.neuron_to_operation[gru_node.op.gate_activation])
        activation_reset_node = graph.add(activation_reset_op, input_names=[elementsum_of_term2_bias_op_name],
                                          output_names=[activation_reset_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created activation_reset_node and it's output tensor
        graph.update_trace_info(activation_reset_node, gru_node)
        graph.update_trace_info(graph.get_buffer(activation_reset_op_name), gru_node)

        return activation_reset_op_name

    # (3.1) - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) if linear_before_reset == 0 (default)
    # (3.2) - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) if linear_before_reset != 0
    def expand_gru_hidden_gate(self, graph, gru_node, Xt, Ht_1, rt, linear_before_reset = 0):
        gru_node_name = gru_node.op.name

        # hidden gate(h)
        Wh_name = gru_node.input_names[3]
        Rh_name = gru_node.input_names[6]
        WBh_name = gru_node.input_names[9]
        RBh_name = gru_node.input_names[12]
        Wbh_size = graph.get_buffer(WBh_name).shape[-1]

        xt_wh_matmul_op_name = gru_node_name + "_xt_wh_matmul_op"
        Wbh = np.zeros(Wbh_size, dtype=np.float32)
        xt_wh_matmul_op = op_adapter.MatMulOp(name=xt_wh_matmul_op_name,
                                                bias=Wbh,
                                                transpose_in0=False,
                                                transpose_in1=True)
        xt_wh_matmul_node = graph.add(xt_wh_matmul_op, input_names=[Xt, Wh_name],
                                      output_names=[xt_wh_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created rt_dot_ht_1_rh_rbh_matmul_bias_node and it's output tensor
        graph.update_trace_info(xt_wh_matmul_node, gru_node)
        graph.update_trace_info(graph.get_buffer(xt_wh_matmul_op_name), gru_node)

        xt_wh_wbh_bias_op_name = gru_node_name + "_xt_wh_wbh_bias_op"
        xt_wh_wbh_bias_op = op_adapter.ElementwiseBinaryOp(name=xt_wh_wbh_bias_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        xt_wh_wbh_bias_node = graph.add(xt_wh_wbh_bias_op, input_names=[xt_wh_matmul_op_name, WBh_name],
                                        output_names=[xt_wh_wbh_bias_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created rt_dot_ht_1_rh_rbh_matmul_bias_node and it's output tensor
        graph.update_trace_info(xt_wh_wbh_bias_node, gru_node)
        graph.update_trace_info(graph.get_buffer(xt_wh_wbh_bias_op_name), gru_node)

        activation_rec_gate_name = ''
        if linear_before_reset == 0:
            rt_dot_ht_1_op_name = gru_node_name + "_rt_dot_ht_1_op"
            rt_dot_ht_1_op = op_adapter.ElementwiseBinaryOp(name=rt_dot_ht_1_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
            rt_dot_ht_1_node = graph.add(rt_dot_ht_1_op, input_names=[rt,  Ht_1],
                                         output_names=[rt_dot_ht_1_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created rt_dot_ht_1_rh_rbh_matmul_bias_node and it's output tensor
            graph.update_trace_info(rt_dot_ht_1_node, gru_node)
            graph.update_trace_info(graph.get_buffer(rt_dot_ht_1_op_name), gru_node)

            rt_dot_ht_1_rh_matmul_op_name = gru_node_name + "_rt_dot_ht_1_rh_matmul_op"
            Rbh = np.zeros(Wbh_size, dtype=np.float32)
            rt_dot_ht_1_rh_matmul_op = op_adapter.MatMulOp(name=rt_dot_ht_1_rh_matmul_op_name,
                                                                bias=Rbh,
                                                                transpose_in0=False,
                                                                transpose_in1=True)
            rt_dot_ht_1_rh_matmul_node = graph.add(rt_dot_ht_1_rh_matmul_op, input_names=[rt_dot_ht_1_op_name, Rh_name],
                                                   output_names=[rt_dot_ht_1_rh_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created rt_dot_ht_1_rh_rbh_matmul_bias_node and it's output tensor
            graph.update_trace_info(rt_dot_ht_1_rh_matmul_node, gru_node)
            graph.update_trace_info(graph.get_buffer(rt_dot_ht_1_rh_matmul_op_name), gru_node)

            rt_dot_ht_1_rh_rbh_matmul_bias_op_name = gru_node_name + "_rt_dot_ht_1_rh_rbh_matmul_bias_op"
            rt_dot_ht_1_rh_rbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=rt_dot_ht_1_rh_rbh_matmul_bias_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            rt_dot_ht_1_rh_rbh_matmul_bias_node = graph.add(rt_dot_ht_1_rh_rbh_matmul_bias_op, input_names=[rt_dot_ht_1_rh_matmul_op_name, RBh_name],
                                                  output_names=[rt_dot_ht_1_rh_rbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created rt_dot_ht_1_rh_rbh_matmul_bias_node and it's output tensor
            graph.update_trace_info(rt_dot_ht_1_rh_rbh_matmul_bias_node, gru_node)
            graph.update_trace_info(graph.get_buffer(rt_dot_ht_1_rh_rbh_matmul_bias_op_name), gru_node)

            elementsum_of_term3p1_op_name = gru_node_name + "_elementsum_of_term3p1_op"
            elementsum_of_term3p1_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term3p1_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            elementsum_of_term3p1_node = graph.add(elementsum_of_term3p1_op, input_names=[xt_wh_wbh_bias_op_name, rt_dot_ht_1_rh_rbh_matmul_bias_op_name],
                                                   output_names=[elementsum_of_term3p1_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created elementsum_of_term3p1_node and it's output tensor
            graph.update_trace_info(elementsum_of_term3p1_node, gru_node)
            graph.update_trace_info(graph.get_buffer(elementsum_of_term3p1_op_name), gru_node)

            activation_rec_gate_name = gru_node_name + "_activation_rec_gate_term3p1_op"
            activation_rec_gate_op = op_adapter.ElementwiseNeuronOp(name=activation_rec_gate_name, operation=op_adapter.ElementwiseNeuronOp.neuron_to_operation[gru_node.op.rec_gate_activation])
            activation_rec_gate_node = graph.add(activation_rec_gate_op, input_names=[elementsum_of_term3p1_op_name],
                                                 output_names=[activation_rec_gate_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created activation_rec_gate_node and it's output tensor
            graph.update_trace_info(activation_rec_gate_node, gru_node)
            graph.update_trace_info(graph.get_buffer(activation_rec_gate_name), gru_node)
        else:
            ht_1_rh_matmul_op_name = gru_node_name + "_ht_1_rh_matmul_op"
            Rbh = np.zeros(Wbh_size, dtype=np.float32)
            ht_1_rh_matmul_op = op_adapter.MatMulOp(name=ht_1_rh_matmul_op_name,
                                                        bias=Rbh,
                                                        transpose_in0=False,
                                                        transpose_in1=True)
            ht_1_rh_matmul_node = graph.add(ht_1_rh_matmul_op, input_names=[Ht_1, Rh_name],
                                            output_names=[ht_1_rh_matmul_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created ht_1_rh_matmul_node and it's output tensor
            graph.update_trace_info(ht_1_rh_matmul_node, gru_node)
            graph.update_trace_info(graph.get_buffer(ht_1_rh_matmul_op_name), gru_node)

            ht_1_rh_rbh_matmul_bias_op_name = gru_node_name + "_ht_1_rh_rbh_matmul_bias_op"
            ht_1_rh_rbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=ht_1_rh_rbh_matmul_bias_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            ht_1_rh_rbh_matmul_bias_node = graph.add(ht_1_rh_rbh_matmul_bias_op, input_names=[ht_1_rh_matmul_op_name, RBh_name],
                                                     output_names=[ht_1_rh_rbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created ht_1_rh_rbh_matmul_bias_node and it's output tensor
            graph.update_trace_info(ht_1_rh_rbh_matmul_bias_node, gru_node)
            graph.update_trace_info(graph.get_buffer(ht_1_rh_rbh_matmul_bias_op_name), gru_node)

            rt_dot_ht_1_rh_rbh_matmul_bias_op_name = gru_node_name + "_rt_dot_ht_1_rh_rbh_matmul_bias_op"
            rt_dot_ht_1_rh_rbh_matmul_bias_op = op_adapter.ElementwiseBinaryOp(name=rt_dot_ht_1_rh_rbh_matmul_bias_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
            rt_dot_ht_1_rh_rbh_matmul_bias_node = graph.add(rt_dot_ht_1_rh_rbh_matmul_bias_op, input_names=[rt, ht_1_rh_rbh_matmul_bias_op_name],
                                                            output_names=[rt_dot_ht_1_rh_rbh_matmul_bias_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created rt_dot_ht_1_rh_rbh_matmul_bias_node and it's output tensor
            graph.update_trace_info(rt_dot_ht_1_rh_rbh_matmul_bias_node, gru_node)
            graph.update_trace_info(graph.get_buffer(rt_dot_ht_1_rh_rbh_matmul_bias_op_name), gru_node)

            elementsum_of_term3p2_op_name = gru_node_name + "_elementsum_of_term3p2_op"
            elementsum_of_term3p2_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term3p2_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            elementsum_of_term3p2_node = graph.add(elementsum_of_term3p2_op, input_names=[xt_wh_wbh_bias_op_name, rt_dot_ht_1_rh_rbh_matmul_bias_op_name],
                                                   output_names=[elementsum_of_term3p2_op_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created elementsum_of_term3p2_node and it's output tensor
            graph.update_trace_info(elementsum_of_term3p2_node, gru_node)
            graph.update_trace_info(graph.get_buffer(elementsum_of_term3p2_op_name), gru_node)

            activation_rec_gate_name = gru_node_name + "_activation_rec_gate_term3p2_op"
            activation_rec_gate_op = op_adapter.ElementwiseNeuronOp(name=activation_rec_gate_name, operation=op_adapter.ElementwiseNeuronOp.neuron_to_operation[gru_node.op.rec_gate_activation])
            activation_rec_gate_node = graph.add(activation_rec_gate_op, input_names=[elementsum_of_term3p2_op_name],
                                                 output_names=[activation_rec_gate_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created activation_rec_gate_node and it's output tensor
            graph.update_trace_info(activation_rec_gate_node, gru_node)
            graph.update_trace_info(graph.get_buffer(activation_rec_gate_name), gru_node)

        return activation_rec_gate_name

    # (4) - Ht = (1 - zt) (.) ht + zt (.) Ht-1
    def update_gru_hidden_state(self, graph, gru_node, Ht_1, zt, ht):
        gru_node_name = gru_node.op.name
        gru_node_idx = graph.nodes_in_order.index(gru_node)

        initial_ones_op_name = gru_node_name + '_initial_ones_op'
        if not graph.has_buffer(initial_ones_op_name):
            initial_ones_tensor = np.ones(graph.get_buffer(zt).shape, dtype=np.float32)
            initial_ones_op = op_adapter.ConstantOp(name=initial_ones_op_name, tensor=initial_ones_tensor)
            initial_ones_node = graph.add(initial_ones_op, input_names=[], output_names=[initial_ones_op_name], idx=gru_node_idx-1)
            # Add op trace data of new created initial_ones_node and it's output tensor
            graph.update_trace_info(initial_ones_node, gru_node)
            graph.update_trace_info(graph.get_buffer(initial_ones_op_name), gru_node)

        one_minus_zt_op_name = gru_node_name + "_one_minus_zt_op"
        one_minus_zt_op = op_adapter.ElementwiseBinaryOp(name=one_minus_zt_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT)
        one_minus_zt_node = graph.add(one_minus_zt_op, input_names=[initial_ones_op_name, zt],
                                      output_names=[one_minus_zt_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created one_minus_zt_node and it's output tensor
        graph.update_trace_info(one_minus_zt_node, gru_node)
        graph.update_trace_info(graph.get_buffer(one_minus_zt_op_name), gru_node)

        one_minus_zt_dot_ht_op_name = gru_node_name + '_one_minus_zt_dot_ht_op'
        one_minus_zt_dot_ht_op = op_adapter.ElementwiseBinaryOp(name=one_minus_zt_dot_ht_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
        one_minus_zt_dot_ht_node = graph.add(one_minus_zt_dot_ht_op, input_names=[one_minus_zt_op_name, ht],
                                             output_names=[one_minus_zt_dot_ht_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created one_minus_zt_dot_ht_node and it's output tensor
        graph.update_trace_info(one_minus_zt_dot_ht_node, gru_node)
        graph.update_trace_info(graph.get_buffer(one_minus_zt_dot_ht_op_name), gru_node)

        zt_dot_ht_1_op_name = gru_node_name + '_zt_dot_ht_1_op'
        zt_dot_ht_1_op = op_adapter.ElementwiseBinaryOp(name=zt_dot_ht_1_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
        zt_dot_ht_1_node = graph.add(zt_dot_ht_1_op, input_names=[zt, Ht_1],
                                     output_names=[zt_dot_ht_1_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created zt_dot_ht_1_node and it's output tensor
        graph.update_trace_info(zt_dot_ht_1_node, gru_node)
        graph.update_trace_info(graph.get_buffer(zt_dot_ht_1_op_name), gru_node)

        elementsum_of_term4_op_name = gru_node_name + "_elementsum_of_term4_op"
        elementsum_of_term4_op = op_adapter.ElementwiseBinaryOp(name=elementsum_of_term4_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
        elementsum_of_term4_node = graph.add(elementsum_of_term4_op, input_names=[one_minus_zt_dot_ht_op_name, zt_dot_ht_1_op_name],
                                             output_names=[elementsum_of_term4_op_name], idx=graph.nodes_in_order.index(gru_node))
        # Add op trace data of new created elementsum_of_term4_node and it's output tensor
        graph.update_trace_info(elementsum_of_term4_node, gru_node)
        graph.update_trace_info(graph.get_buffer(elementsum_of_term4_op_name), gru_node)

        return elementsum_of_term4_op_name

    def expand_gru_op_structure(self, graph):
        sequence = [
            (op_adapter.GruOp.TRANSLATION_KEY, (), ())
        ]
        matched_node_list = graph.get_matched_nodes(sequence)
        for nodes_tuple in matched_node_list:
            gru_node = nodes_tuple[0]
            time_major_param = gru_node.op.time_major
            gru_node_name = gru_node.op.name
            gru_node_idx = graph.nodes_in_order.index(gru_node)
            number_of_inputs = len(gru_node.input_names)
            number_of_outputs = len(gru_node.output_names)

            input_buffer_shape = self.get_dims(graph.get_buffer(gru_node.input_names[0]).shape, time_major_param)
            batch_size, seq_length, input_size = input_buffer_shape[:]

            output_size = graph.get_buffer(gru_node.output_names[0]).shape[-1]

            # Check that extracted sequence length is 1
            if seq_length != 1:
                raise ValueError('Unsupported sequence length for GRU node {}, expected 1, got {}.'.format(
                        gru_node_name, seq_length))

            # Only 1 or 2 outputs are supported for this optimization
            if number_of_outputs != 1 and number_of_outputs != 2:
                raise ValueError("Unsupported number of outputs for GRU node {}, expected 1 or 2, got {}.".format(
                    gru_node_name, number_of_outputs))

            Xt = gru_node.input_names[0]
            if len(graph.get_buffer(gru_node.input_names[0]).shape) == 3:
                Xt = gru_node_name + "_" + gru_node.input_names[0] + "_reshape"
                input_x_reshape_output_shape = [batch_size, input_size]
                input_x_reshape_op = op_adapter.ReshapeOp(name=Xt,
                                                          shape=input_x_reshape_output_shape)
                input_x_reshape_op_node = graph.add(input_x_reshape_op, input_names=[gru_node.input_names[0]],
                             output_names=[Xt], idx=graph.nodes_in_order.index(gru_node))
                # Update trace info for new created node
                graph.update_trace_info(input_x_reshape_op_node, [gru_node])

            Ht_1 = gru_node.input_names[-1]
            if len(graph.get_buffer(gru_node.input_names[-1]).shape) == 3:
                Ht_1 = gru_node_name + "_" + gru_node.input_names[-1] + "_reshape"
                input_h_reshape_output_shape = [batch_size, output_size]
                input_h_reshape_op = op_adapter.ReshapeOp(name=Ht_1,
                                                          shape=input_h_reshape_output_shape)
                input_h_reshape_op_node = graph.add(input_h_reshape_op, input_names=[gru_node.input_names[-1]],
                             output_names=[Ht_1], idx=graph.nodes_in_order.index(gru_node))
                # Update trace info for new created node
                graph.update_trace_info(input_h_reshape_op_node, [gru_node])

            # expand gru op structure
            # zt = f(Xt*(Wz^T) + Ht_1*(Rz^T) + Wbz + Rbz) ........(1)
            # rt = f(Xt*(Wr^T) + Ht_1*(Rr^T) + Wbr + Rbr) ........(2)
            # ht = g(Xt*(Wh^T) + (rt (.) Ht_1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0 ........(3.1)
            # ht = g(Xt*(Wh^T) + (rt (.) (Ht_1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0 ........(3.2)
            # Ht = (1 - zt) (.) ht + zt (.) Ht_1 ........(4)

            zt = self.expand_gru_update_gate(graph, gru_node, Xt, Ht_1) # ........(1)
            rt = self.expand_gru_reset_gate(graph, gru_node, Xt, Ht_1) # ........(2)
            ht = self.expand_gru_hidden_gate(graph, gru_node, Xt, Ht_1, rt, gru_node.op.linear_before_reset) # ........(3)
            Ht = self.update_gru_hidden_state(graph, gru_node, Ht_1, zt, ht) # ........(4)

            output_all_hiddens_reshape_node_name = gru_node.output_names[0] + "_reshape"
            output_all_hiddens_output_shape = graph.get_buffer(gru_node.output_names[0]).shape
            output_all_hiddens_reshape_op = op_adapter.ReshapeOp(name=output_all_hiddens_reshape_node_name,
                                                                 shape=output_all_hiddens_output_shape)
            output_all_hiddens_reshape_node = graph.add(output_all_hiddens_reshape_op, input_names=[Ht],
                                                        output_names=[output_all_hiddens_reshape_node_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created output_all_hiddens_reshape_node and it's output tensor
            graph.update_trace_info(output_all_hiddens_reshape_node, gru_node)
            graph.update_trace_info(graph.get_buffer(output_all_hiddens_reshape_node_name), gru_node)

            output_hidden_reshape_node_name = gru_node.output_names[1] + "_reshape"
            output_hidden_output_shape = graph.get_buffer(gru_node.output_names[1]).shape
            output_all_hiddens_reshape_op = op_adapter.ReshapeOp(name=output_hidden_reshape_node_name,
                                                                 shape=output_hidden_output_shape)
            output_all_hiddens_reshape_node = graph.add(output_all_hiddens_reshape_op, input_names=[Ht],
                                                        output_names=[output_hidden_reshape_node_name], idx=graph.nodes_in_order.index(gru_node))
            # Add op trace data of new created output_all_hiddens_reshape_node and it's output tensor
            graph.update_trace_info(output_all_hiddens_reshape_node, gru_node)

            # Adjust gru output consumers
            # Replace all the consumers' input names coming from previous gru node with the new corresponding output buffer names
            # Delete the consumers for previous gru node's output buffers
            for consumer in list(graph.get_buffer(gru_node.output_names[0]).consumers).copy():
                output_all_hiddens_buffer = graph.get_buffer(output_all_hiddens_reshape_node_name)
                output_all_hiddens_buffer.consumers.add(consumer)
                h_all_idx = consumer.input_names.index(gru_node.output_names[0])
                consumer.input_names[h_all_idx] = output_all_hiddens_reshape_node_name
                graph.get_buffer(gru_node.output_names[0]).consumers.remove(consumer)

            for consumer in list(graph.get_buffer(gru_node.output_names[1]).consumers).copy():
                output_hidden_reshape_buffer = graph.get_buffer(output_hidden_reshape_node_name)
                output_hidden_reshape_buffer.consumers.add(consumer)
                h_idx = consumer.input_names.index(gru_node.output_names[1])
                consumer.input_names[h_idx] = output_hidden_reshape_node_name
                graph.get_buffer(gru_node.output_names[1]).consumers.remove(consumer)

            # prune original gru_node
            source_output_names = gru_node.output_names
            graph.prune(gru_node, force_remove=True)

            # If the output name of the gru node is in the output name of the graph,
            # record the current output name and source (graph) output name for alignment.
            current_output_names = [output_all_hiddens_reshape_node_name, output_hidden_reshape_node_name]
            # At this point, current output names are not aligned to source output names,
            # we need to restore the source output names from graph.
            self.align_to_source_output_names(graph, current_output_names, source_output_names)


@register_layer_optimization
class OptimizeIdentityTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.IdentityOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    def axes_to_spatial_first_order(self, node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf.shape = input_buf.shape
        output_buf.axis_format = input_buf.axis_format
        return True

    @staticmethod
    def remove_identity(node, graph):
        try:
            graph.squash_identity(node, is_data_movement_node=True)
        except:
            # Replace IdentityOp with ReshapeOp to avoid implementation issue when it can't be squashed
            input_buf = graph.get_input_buffers(node)[0]
            input_shape = input_buf.shape
            graph.replace(node.op, op_adapter.ReshapeOp(str(node.op.name), shape=input_shape))


@register_layer_optimization
class OptimizeInstanceNormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InstanceNormOp.TRANSLATION_KEY
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        if 1 < input_buf.rank() <= 5:
            input_axis_formats_before = graph.get_input_axis_formats(node)
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            input_axis_formats_after = graph.get_input_axis_formats(node)
            input_buffers = graph.get_input_buffers(node)
            for i, buf in enumerate(input_buffers):
                if input_axis_formats_before[i] != input_axis_formats_after[i]:
                    transpose_node = buf.producer
                    graph.update_trace_info(transpose_node, [node])
                    graph.update_trace_info(buf, [node])
            output_buffer = graph.get_output_buffers(node)[0]
            # (image/feature)_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
            # Enforce the output format here to be NDHWC/NSC/NFC
            output_buffer.axis_format = AxisOrder().get_axis_format(len(output_buffer.shape))
        else:
            raise ValueError(code_to_message.get_error_message("ERROR_INSTANCE_NORM_DIM_UNSUPPORTED")(input_buf.rank(),
                                                                                                      node.op.name))
        return True

    def prepare_inputs_as_params(self, node, graph):
        weights_buffer = graph.get_buffer(node.input_names[1])
        weights_node = weights_buffer.producer
        bias_buffer = graph.get_buffer(node.input_names[2])
        bias_node = bias_buffer.producer
        if weights_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and \
                bias_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.weights = weights_node.op.tensor
            node.op.bias = bias_node.op.tensor
            graph.update_trace_info(node, [weights_node, weights_buffer, bias_node, bias_buffer])
            # Remove the weights/bias inputs from the IR graph
            graph.remove_node_as_consumer(node, weights_buffer.name)
            graph.remove_node_as_consumer(node, bias_buffer.name)
            node.input_names = [node.input_names[0]]


@register_layer_optimization
class OptimizeInverseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.InverseOp.TRANSLATION_KEY
        self.register_method(EXPAND_INVERSE, self.expand_inverse)
        self.idx = 0
        self.input_shape = []

    def expand_inverse(self, node, graph):
        # Inverse of a square matrix is defined as:
        #
        #  inverse(X) = (1/determinant(X))*adjoint(X)

        sequence = [(op_adapter.InverseOp.TRANSLATION_KEY, (), ())]
        matched_node_list = graph.get_matched_nodes(sequence)

        for matched_node in matched_node_list:
            input_names = matched_node[0].input_names
            output_names = matched_node[0].output_names
            inverse_ir_node = matched_node[0]
            inverse_ir_op_name = inverse_ir_node.op.name

            # Store index of InverseOp
            self.idx = graph.list_nodes().index(inverse_ir_node)

            # Store input shape
            self.input_shape = graph.get_buffer(inverse_ir_node.input_names[0]).shape.dims

            # Store consumers of InverseOp
            consumers = graph.get_buffer(output_names[0]).consumers.copy()

            # Dictionary {"consumer_op_name": [indices of consumer input names with inverse op output]
            consumers_dict = {}
            for c in consumers:
                indices = [i for i in range(len(c.input_names)) if c.input_names[i] == output_names[0]]
                consumers_dict[c.op.name] = indices

            # Prune InverseOp
            graph.prune(inverse_ir_node, force_remove=True)

            # Apply adjoint operation
            adjoint_op_last_node = self.add_adjoint_op(inverse_ir_node, graph)

            # Apply inverse operation
            inverse_op_last_node = self.apply_inverse_operation(inverse_ir_node, adjoint_op_last_node, graph)

            # Update the consumers of the last added node
            consumers = list(consumers)
            if len(consumers) > 0:
                for i in range(len(consumers)):
                    indices = consumers_dict[consumers[i].op.name]
                    for idx in indices:
                        consumers[i].input_names.insert(idx, inverse_op_last_node.output_names[0])
                graph.get_buffer(inverse_op_last_node.output_names[0]).consumers = consumers

    def add_adjoint_op(self, inverse_ir_node, graph):
        # For a 2x2 matrix, the adjoint operation can be expanded as follows:
        #
        #       X       P = [[0, 1],
        #        \     /     [1, 0]]
        #         \   /
        #          \ /
        #       +--------+
        #       | MatMul |
        #       +--------+
        #           |
        #     +-----------+
        #     | Transpose |
        #     +-----------+
        #           | +-------- P
        #           | |
        #       +--------+
        #       | MatMul |
        #       +--------+
        #           |
        #           | +--------Q = [[1, -1],
        #           | |             [-1, 1]]
        #       +-------+
        #       |  Mul  |
        #       +-------+
        #           |
        #          Adj(X)

        flipped_identity_matrix = np.array([[0, 1],
                                            [1, 0]])
        sign_change_matrix = np.array([[1, -1],
                                       [-1, 1]])

        # ConstantOp corresponding to flipped_identity_matrix
        flipped_identity_op = op_adapter.ConstantOp(name="flipped_identity",
                                                    tensor=flipped_identity_matrix.astype('float32'))
        flipped_identity_node = graph.add(flipped_identity_op, [], flipped_identity_op.name, idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(flipped_identity_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # MatMulOp with inputs: [inverse_ir_node.input_names[0], flipped_identity_op.name]
        matmul_op_1 = op_adapter.MatMulOp(name=inverse_ir_node.op.name + '_adj_matmul_1')
        matmul_node = graph.add(matmul_op_1,
                                [inverse_ir_node.input_names[0], flipped_identity_op.name],
                                [matmul_op_1.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(matmul_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # TransposeOp with inputs: [matmul_node.output_names[0]]
        perm = self.calc_perm(self.input_shape)
        transpose_op = op_adapter.TransposeOp(name=inverse_ir_node.op.name + '_adj_transpose',
                                              perm=perm)
        transpose_node = graph.add(transpose_op,
                                   [matmul_node.output_names[0]],
                                   [transpose_op.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(transpose_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # MatMulOp with inputs: [transpose_node.output_names[0], flipped_identity_op.name]
        matmul_op_2 = op_adapter.MatMulOp(name=inverse_ir_node.op.name + '_adj_matmul_2')
        matmul_node_2 = graph.add(matmul_op_2,
                                  [transpose_node.output_names[0], flipped_identity_op.name],
                                  [matmul_op_2.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(matmul_node_2, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # ConstantOp corresponding to sign change matrix
        sign_change_op = op_adapter.ConstantOp(name="sign_change_matrix",
                                               tensor=sign_change_matrix.astype('float32'))
        sign_change_node = graph.add(sign_change_op, [], [sign_change_op.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(sign_change_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # MulOp inputs: [matmul_node_2.output_names[0], sign_change_op.name]
        mul_op = op_adapter.ElementwiseBinaryOp(name=inverse_ir_node.op.name + '_adj_sign_change',
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
        mul_node = graph.add(mul_op,
                             [matmul_node_2.output_names[0], sign_change_op.name],
                             [mul_op.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(mul_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # Return last added node
        return mul_node

    def apply_inverse_operation(self, inverse_ir_node, adjoint_op_last_node, graph):
        # Inverse of matrix can be decomposed into the following sequence of operations
        #
        #         +---------------+
        #   X ----|    Adjoint    |----- adj(X)
        #    \    +---------------+     /
        #     \                        /
        #      \                      /
        #       \                    /
        #        \                  /
        #         +----------------+
        #         |     MatMul     |
        #         +----------------+
        #                 |
        #                 |
        #           +-----------+
        #           | ReduceSum |
        #           +-----------+
        #                 |
        #          +--------------+
        #          |  ReduceMean  |
        #          +--------------+
        #                 |
        #  adj(X)------+  |
        #              |  |
        #             +-------+
        #             |  Div  |
        #             +-------+
        #                 |
        #                 Y
        #

        # MatMulOp with inputs: [inverse_ir_node.input_names[0], adjoint_op_last_node.output_names[0]]
        matmul_op = op_adapter.MatMulOp(name=inverse_ir_node.op.name + '_inv_matmul')
        matmul_node = graph.add(matmul_op,
                                [inverse_ir_node.input_names[0],
                                 adjoint_op_last_node.output_names[0]],
                                [matmul_op.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(matmul_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # ReduceSumOp with inputs: [matmul_node.output_names[0]]
        # Attrs: keepdims=True, axes=[-1]
        axes = [len(self.input_shape) - 1]
        reduce_sum_op = op_adapter.ReduceOp(name=inverse_ir_node.op.name + '_inv_reduce_sum',
                                            reduce_type="ReduceSum", keep_dims=True, axes=axes)
        reduce_sum_node = graph.add(reduce_sum_op,
                                    [matmul_node.output_names[0]],
                                    [reduce_sum_op.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(reduce_sum_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # ReduceMeanOp with inputs: [reduce_sum_op.output_names[0]]
        # Attrs: keepdims=True, axes=[-2]
        axes = [len(self.input_shape) - 2]
        reduce_mean_op = op_adapter.ReduceOp(name=inverse_ir_node.op.name + '_inv_reduce_mean',
                                             reduce_type="ReduceMean", keep_dims=True, axes=axes)
        reduce_mean_node = graph.add(reduce_mean_op,
                                     [reduce_sum_node.output_names[0]],
                                     [reduce_mean_op.name], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(reduce_mean_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # DivOp with inputs: [adjoint_op_last_node.output_names[0], reduce_mean_node.output_names[0]]
        div_op = op_adapter.ElementwiseBinaryOp(name=inverse_ir_node.op.name + '_inv_div',
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE)
        div_node = graph.add(div_op,
                             [adjoint_op_last_node.output_names[0],
                              reduce_mean_node.output_names[0]],
                             [inverse_ir_node.output_names[0]], idx=self.idx)
        if graph.enable_trace:
            graph.set_trace_info(div_node, (inverse_ir_node.op.name, op_graph.TraceType.OP))
        self.idx += 1

        # Return last added node
        return div_node

    def calc_perm(self, input_shape):
        res = list(range(len(input_shape)))
        temp = res[-1]
        res[-1] = res[-2]
        res[-2] = temp
        return res


@register_layer_optimization
class OptimizeL2NormTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.L2NormOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_axis_formats = graph.get_input_axis_formats(node)

        if not super(OptimizeL2NormTranslation, self).axes_to_spatial_first_order(node, graph):
            # No change in input formats
            return False

        # transform axis to the correct index, also ensures axis is always positive
        input_buf = graph.get_input_buffers(node)[0]
        if (input_axis_formats[0] == AxisTracker.AxisFormat.NDHWC and \
            node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW) or \
            (input_axis_formats[0] == AxisTracker.AxisFormat.NSC and \
             node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS) or \
            (input_axis_formats[0] == AxisTracker.AxisFormat.NFC and \
             node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF):
            axis_map = graph.src_axis_order.permute_sequence_from_ir[input_buf.rank() - 1]
            if node.op.hasattr(ir_graph.QNN_OP_L2_NORM_PARAM_AXES):
                node.op.axes = [axis_map[axis] for axis in node.op.axes]
            else:
                node.op.axis = axis_map[node.op.axis]

        return True

    def merge_low_level_ops_to_layers(self, graph):
        def validate(nodes_tuple):
            # Reduce_l2 can be matched to L2Norm only if input to ReduceL2 is also one of the inputs to Div
            reduce_l2_node = nodes_tuple[0]
            div_node = nodes_tuple[-1]
            reduce_l2_input_name = reduce_l2_node.input_names[0]
            if not reduce_l2_input_name in div_node.input_names:
                return False
            # reduce_l2 is reduced to l2_norm only if keep_dims is True
            if not reduce_l2_node.op.keep_dims:
                return False
            return True

        sequence1 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_l2", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_EXPAND, "ALL")])
             ),
            (ir_graph.IR_OP_EXPAND,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [(ir_graph.IR_OP_EXPAND, 1)]), ())
        ]

        sequence2 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_l2", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 1)]), ())
        ]

        sequence3 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [("reduce_l2", 1)]), ())
        ]

        sequence4 = [
            ("reduce_l2",
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")])
             ),
            (ir_graph.QNN_OP_ELEMENT_WISE_NEURON, # clip
             ("MATCH_NUM_BUFS", [("reduce_l2", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_EXPAND, "ALL")])
             ),
            (ir_graph.IR_OP_EXPAND,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div", ("MATCH_BUFS_AT_INDEX", [(ir_graph.IR_OP_EXPAND, 1)]), ())
        ]

        sequences = [sequence1, sequence2, sequence3, sequence4]

        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate, ignore_constants=True)

            for node_tuple in matched_node_list:
                # L2 Norm lower sequence found
                # Check for either sequence 1 or sequence 2 found
                if len(node_tuple) > 3:
                    reduce_l2_node, epsilon_node, _, div_node = node_tuple
                elif len(node_tuple) > 2:
                    reduce_l2_node, epsilon_node, div_node = node_tuple
                else:
                    reduce_l2_node, div_node = node_tuple
                    epsilon_node = None

                l2norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.L2NormOp.TRANSLATION_KEY,
                                                                         op_adapter.L2NormOp.LEGACY_TRANSLATION_KEY,
                                                                         folded_op=True)
                if epsilon_node:
                    if isinstance(epsilon_node.op, op_adapter.ElementwiseNeuronOp):
                        # get epsilon from clip min
                        elementwise_sum_constant_tensor = epsilon_node.op.min_value
                    else:
                        # get epsilon from elementwise_sum constant op and assign epsilon to new L2normOp
                        elementwise_sum_constant_tensor = graph.get_producer_op(epsilon_node.input_names[1]).tensor
                    l2norm_op = op_adapter.L2NormOp(l2norm_op_name, axes=reduce_l2_node.op.axes, epsilon=elementwise_sum_constant_tensor)
                else:
                    # epsilon_node is not present
                    l2norm_op = op_adapter.L2NormOp(l2norm_op_name, axes=reduce_l2_node.op.axes)

                # Prune all matched nodes except Last node
                for node in node_tuple[:-1]:
                    graph.prune(node, force_remove=True, merge_trace_info_to_next=True)

                # Replace Last op with L2norm. No need to connect input of first node in pattern to the last_node
                # since Div node already has that as the other input
                last_op = node_tuple[-1].op
                graph.replace(last_op, l2norm_op)


class OptimizeNormTranslationBase(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.register_method(ADJUST_NORM_OP_BUFFERS, self.adjust_norm_op_buffers)

    # ---------------- Utilities for Matching Strategies -------------------
    def get_affine_scalar_input(self, node, node_tuple, graph):
        for input_name in node.input_names[:]:
            input_node = graph.get_producer_node(input_name)
            if input_node not in node_tuple and input_node.op.type != op_adapter.InputOp.TRANSLATION_KEY:
                return input_name
        raise ValueError("Gamma or beta scalar input is missing for {type} node.".format(type=node.op.type))

    def get_constant_input_node(self, node, graph):
        for input_name in node.input_names[:]:
            input_node = graph.get_producer_node(input_name)
            if isinstance(input_node.op, op_adapter.ConstantOp):
                return input_node
        raise ValueError("Expected node to have at least 1 constant input.")

    def get_axes(self, node_tuple):
        for node in node_tuple:
            # infer axes parameter of LayerNorm/RMSNorm from ReduceMean Op
            if node.op.type == op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN]:
                axes = node.op.axes
                return axes
        return [0]

    def get_weight_bias_dimensions(self, node_tuple, graph):
        norm_input_shape = graph.get_input_shapes(node_tuple[0])[0]
        dimensions = [1 for _ in range(len(norm_input_shape))]
        axes = self.get_axes(node_tuple)
        for axis in axes:
            if axis in norm_input_shape.dynamic_axes:
                norm_input_name = node_tuple[0].op.name
                raise ValueError("RMS norm Input {} with shape {} is dynamic on axes {}"
                                 .format(norm_input_name, norm_input_shape, axes))
            else:
                dimensions[axis] = norm_input_shape.dims[axis]
        return dimensions

    def prune_nodes(self, node_tuple, graph):
        last_node = node_tuple[-1]
        last_node_buf = graph.get_output_buffers(last_node)[0]
        last_node_consumers = last_node_buf.consumers
        # maps consumers of last node buffer with corresponding input_names
        last_node_consumers_input_names = {}
        for consumer in last_node_consumers:
            last_node_consumers_input_names[consumer] = copy.deepcopy(consumer.input_names)

        # Prune all matched nodes in reverse order
        for node in node_tuple[::-1]:
            if graph.has_node(node.op.name):
                graph.prune(node, force_remove=True)

        # reassign input_names to consumers of last node buffer post pruning of the nodes
        for consumer in last_node_consumers:
            consumer.input_names = last_node_consumers_input_names[consumer]

    def get_norm_op_insertion_index(self, norm_input_names, graph):
        insertion_index = 0
        for input_name in norm_input_names:
            buf = graph.get_buffer(input_name)
            curr_idx = graph.nodes_in_order.index(buf.producer)
            insertion_index = max(insertion_index, curr_idx)
        return insertion_index + 1

    def is_already_squeezed(self, input_name, graph, node_tuple):
        buffer = graph.get_buffer(input_name)
        main_input_buffer = graph.get_input_buffers(node_tuple[0])[0]
        input_rank = main_input_buffer.rank()
        return buffer.rank() < input_rank

    def squeeze_axes(self, input_name, graph, node_tuple):
        """
        Squeeze the dimensions of the beta/gamma inputs if they are one
        extended to match the dimensions of the main input along the
        axes to be normalized.

        Example) Given input with shape [1,2,5,8], axes parameter [1,3], and gamma
        input shape [1,2,1,8], the gamma tensor is squeezed to have shape [2,8].
        """

        axes_to_keep = self.get_axes(node_tuple)
        input_op = graph.get_producer_op(input_name)
        squeezed_shape = [input_op.shape[axis] for axis in axes_to_keep]
        reshape_name = input_name + "_reshape"

        reshape_op = op_adapter.ReshapeOp(reshape_name, shape=squeezed_shape)

        # In this case the reshape injected is because of the input buffer producer
        # Hence setting set_consumers_trace_info to False
        graph.inject(reshape_op, input_name, reshape_name, set_consumers_trace_info=False)

        buf = graph.get_buffer(reshape_name)
        buf.set_axis_format(AxisTracker.AxisFormat.ANY)
        return reshape_name

    def broadcast_axes(self, input_name, idx, input_shapes_axes, graph, input_buf, op_name, node_tuple):
        """
        Broadcast the dimensions of the beta/gamma inputs to match the
        dimensions of the main input along the axes to be normalized.

        Example) Given input with shape [1,2,5,8], axes
        parameter [1,3], and gamma input shape [8], the gamma tensor
        is broadcasted to have shape [2,8].
        """

        if not self.is_already_squeezed(input_name, graph, node_tuple):
            input_name = self.squeeze_axes(input_name, graph, node_tuple)

        buf = graph.get_buffer(input_name)
        op = buf.producer.op
        append_string = "_gamma"

        if idx == 0:
            append_string = "_beta"

        if buf.shape != input_shapes_axes:
            if not translation_utils.unidirectional_broadcastable(input_shapes_axes, buf.shape):
                raise ValueError("Input {} with shape {} is not broadcastable to shape {}"
                                 .format(input_name, buf.shape, input_shapes_axes))
            if not isinstance(op, op_adapter.ConstantOp):
                raise ValueError("Input {} is not a constant op"
                                 .format(input_name))
            input_tensor = op.tensor
            input_tensor_broadcasted = np.full(input_shapes_axes, input_tensor)
            if len(input_buf.consumers) > 1:
                new_input_op = op_adapter.ConstantOp(op_name + append_string,
                                                     input_tensor_broadcasted)
                input_name = op_name + append_string
                input_op_idx = graph.get_node_idx_in_order(input_buf.producer)
                new_input_node = graph.add(new_input_op, [], input_name, idx=input_op_idx)
                graph.update_trace_info(new_input_node, buf)
                graph.update_trace_info(graph.get_buffer(input_name), buf)
            else:
                op.tensor = input_tensor_broadcasted
                input_buf.shape = input_tensor_broadcasted.shape

        return input_name

    def update_norm_op_output_buffer(self, norm_op_node, last_node_consumers, graph):
        for output_name in norm_op_node.output_names:
            output_buf = graph.get_buffer(output_name)
            output_buf.consumers = last_node_consumers

    def update_consumer_data_axis_formats(self, graph, norm_op_node):
        for consumer in graph.get_op_output_nodes(norm_op_node):
            consumer_input_buffers = graph.get_input_buffers(consumer)
            consumer.op.populate_data_axis_formats(graph, consumer_input_buffers)

    def validate_reduce_mean_ops(self, node_tuple):
        for node in node_tuple:
            if isinstance(node.op, op_adapter.ReduceOp):
                if node.op.keep_dims == False:
                    return False
        return True

    def adjust_norm_op_buffers(self, graph):
        def get_default_permute_order(node):
            input_buffer = graph.get_input_buffers(node)[0]
            return list(range(input_buffer.rank()))

        def get_input_buffer_permute_order(node):
            input_buffer = graph.get_input_buffers(node)[0]
            axes_to_normalize = list(node.op.axes)
            non_normalized_axes = [i for i in range(input_buffer.rank()) if i not in axes_to_normalize]
            permute_order = non_normalized_axes + axes_to_normalize
            return permute_order

        def move_normalized_axes_last(node):
            """
            Makes the "axes" parameter correspond to the last dimensions of the
            input.

            E.g. given initial axes=[1,2] with a 4D input, the "axes" will
            become [2,3].

            E.g. given initial axes=[1] with a 3D input, the "axes" will become
            [2].
            """
            main_input_buffer = graph.get_input_buffers(node)[0]
            input_rank = main_input_buffer.rank()
            num_axes = len(node.op.axes)
            new_axes = list(range(input_rank-num_axes, input_rank))
            node.op.axes = new_axes

        def permute_output_buffer(node):
            """
            Changes the rmsnorm output buffer to match its input buffer. Then
            injects a transpose op between the rmsnorm's output and its
            consumers so that the consumers maintain the same input axis format
            and shape.

            For example, given a rmsnorm node with:
                input axis format: NHWC
                input shape: 1,2,3,4

                output axis format: NCHW
                output shape: 1,4,2,3

            The output axis format and shape will become NHWC and (1,2,3,4) and
            a transpose op will be injected between the rmsnorm and its
            consumers so that their inputs are still NCHW and (1,4,2,3).
            """
            def get_output_permute_order(input_axis_format, output_axis_format):
                if AxisTracker.AxisFormat.NONTRIVIAL in [input_axis_format, output_axis_format]:
                    original_input_order = get_default_permute_order(node)
                    new_input_order = get_input_buffer_permute_order(node)
                    new_to_original_permute_order = AxisTracker.compute_permute_order(new_input_order, original_input_order)
                else:
                    new_to_original_permute_order = AxisTracker.compute_permute_order(input_axis_format, output_axis_format)

                return new_to_original_permute_order

            def match_output_buffer_to_input_buffer(input_buffer, output_buffer):
                input_shape = input_buffer.get_buf_dims()
                input_axis_format = input_buffer.get_axis_format()
                output_buffer.set_buf_dims(input_shape)
                output_buffer.set_axis_format(input_axis_format)

            input_axis_format = graph.get_input_axis_formats(node)[0]
            output_axis_format = graph.get_output_axis_formats(node)[0]
            permute_order = get_output_permute_order(input_axis_format, output_axis_format)

            input_buffer = graph.get_input_buffers(node)[0]
            output_buffer = graph.get_output_buffers(node)[0]
            match_output_buffer_to_input_buffer(input_buffer, output_buffer)
            output_consumers = [node.op.name for node in list(output_buffer.consumers)]

            graph.inject_implicit_permute(output_buffer.name,
                                          output_axis_format,
                                          permute_order,
                                          output_consumers)

        def permute_input_buffer(node):
            """
            Inject a transpose op between the rmsnorm op and its main input so
            that the axes being normalized are last.

            For example, given a rmsnorm node that normalizes along the "HW"
            dimensions of a "NHWC" input, this function injects a transpose
            between the input and layernorm so that the input axis format to the
            rmsnorm is now "NCHW".
            """
            def get_normalization_axes(node):
                if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NONTRIVIAL:
                    return AxisTracker.AxisFormat.NONTRIVIAL
                normalization_axes = "".join([node.op.data_axis_formats[0][axis] for axis in node.op.axes])
                return normalization_axes

            input_buffer = graph.get_input_buffers(node)[0]
            normalization_axes = get_normalization_axes(node)

            if normalization_axes == "HW":
                AxisTracker.enforce_spatial_last_input(input_buffer, node, graph)
            elif normalization_axes == "C":
                AxisTracker.enforce_channel_last_input(input_buffer, node, graph)
            elif normalization_axes == "F":
                AxisTracker.enforce_feature_last_input(input_buffer, node, graph)
            else:
                permute_order = get_input_buffer_permute_order(node)
                input_consumers = [node.op.name for node in list(input_buffer.consumers)]
                graph.inject_implicit_permute(input_buffer.name,
                                              AxisTracker.AxisFormat.NONTRIVIAL,
                                              permute_order,
                                              input_consumers)
            # Reset data_axis_formats
            node.op.data_axis_formats[0] = graph.get_input_buffers(node)[0].axis_format

        def requires_input_buffer_permute(node):
            def is_input_2d(node):
                axis_format = graph.get_input_axis_formats(node)[0]
                return axis_format in [AxisTracker.AxisFormat.NF,
                                       AxisTracker.AxisFormat.NC]

            def is_normalized_axes_last(node):
                permute_order = get_input_buffer_permute_order(node)
                default_order = get_default_permute_order(node)
                return permute_order == default_order

            return not any([is_input_2d(node),
                            is_normalized_axes_last(node)])

        norm_op_nodes = [node for node in graph.list_nodes() if isinstance(node.op, op_adapter.RMSNormOp)
                         or isinstance(node.op, op_adapter.LayerNormOp)]
        for node in norm_op_nodes:
            if requires_input_buffer_permute(node):
                permute_input_buffer(node)
                permute_output_buffer(node)
                move_normalized_axes_last(node)

    def axes_to_spatial_first_order(self, node, graph):
        def revert_input_axis_format(graph, node, buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                return False
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC and \
                    data_axis_format == AxisTracker.AxisFormat.NCDHW:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC and \
                    data_axis_format == AxisTracker.AxisFormat.NCS:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC and \
                    data_axis_format == AxisTracker.AxisFormat.NCF:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF and \
                    data_axis_format == AxisTracker.AxisFormat.TNF:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NCDHW and \
                    data_axis_format == AxisTracker.AxisFormat.NDHWC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NDHWC,
                                              AxisTracker.AxisFormat.NCDHW_TO_NDHWC, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NCS and \
                    data_axis_format == AxisTracker.AxisFormat.NSC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NSC,
                                              AxisTracker.AxisFormat.NCS_TO_NSC, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NCF and \
                    data_axis_format == AxisTracker.AxisFormat.NFC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NFC,
                                              AxisTracker.AxisFormat.NCF_TO_NFC, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.TNF and \
                    data_axis_format == AxisTracker.AxisFormat.NTF:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NTF,
                                              AxisTracker.AxisFormat.TNF_TO_NTF, [node.op.name])
            else:
                # going to nontrivial, do nothing.
                log_warning("Op {} with input_buf axis format {} and data_axis_format {} is no need to revert.".format(node.op.name,
                                                                                                                       buf_axis_format,
                                                                                                                       data_axis_format))
                return False
            return True

        input_bufs = graph.get_input_buffers(node)
        return revert_input_axis_format(graph, node, input_bufs[0].name, input_bufs[0].axis_format, node.op.data_axis_formats[0])


@register_layer_optimization
class OptimizeLayerNormTranslation(OptimizeNormTranslationBase):
    def __init__(self):
        OptimizeNormTranslationBase.__init__(self)
        self.op_type = op_adapter.LayerNormOp.TRANSLATION_KEY
        self.register_method(MATCH_LAYERNORM, self.match_layer_norm)

    def match_layer_norm(self, graph):
        # --------------------- Sequences to be matched -----------------------
        sequence1 = [
            ("reduce_mean",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0), ("elementwise_sub", 1)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequence2 = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("Reshape", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0), ("elementwise_sub", 1)]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("Reshape", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequence3 = [
            ("reduce_mean",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0), ("elementwise_sub", 1)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
        ]

        sequence4 = [
            ("reduce_mean",
             (),
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sub", "ANY")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0), ("elementwise_sub", 1)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
             ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
        ]

        sequence5 = [
            ("reduce_mean",
             (),
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sub", "ANY")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0), ("elementwise_sub", 1)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
             ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequence6 = [
            ("reduce_mean",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("cast", "ANY"), ("elementwise_div", "ANY")])
             ),
            ("cast",
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")]),
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("cast", 0), ("cast", 1)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
             ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence7 = [
            ("reduce_mean",
             (),
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sub", "ANY")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),("elementwise_div", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0), ("elementwise_sub", 0)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_sub", "ANY")]),
             (),
             ),
        ]

        sequence8 = [
            ("reduce_mean",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")])
             ),
            ("elementwise_sub",
             ("FLEXIBLE_NUM_BUFS", [("reduce_mean", "ANY")]),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
             ),
             ("Transpose",
              ("MATCH_NUM_BUFS", [("elementwise_sub", "ALL")]),
              ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"), ("elementwise_div", "ANY")]),
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("Transpose", 0), ("Transpose", 1)]),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("Transpose", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]

        def get_node_consumer_names(node):
            node_consumers = graph.get_op_output_nodes(node)
            node_consumers_names = [consumer.op.name for consumer in node_consumers]
            return node_consumers_names

        def get_epsilon(node_tuple):
            for node in node_tuple:
                if node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
                    constant_node = self.get_constant_input_node(node, graph)
                    if constant_node.op.tensor.size == 1:
                        epsilon = constant_node.op.tensor[0]
                        return epsilon
            return op_adapter.LayerNormOp.EPSILON

        def make_layernorm_op(node_tuple):
            layer_norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.LayerNormOp.TRANSLATION_KEY,
                                                                         op_adapter.LayerNormOp.LEGACY_TRANSLATION_KEY,
                                                                         folded_op=True)
            axes = self.get_axes(node_tuple)
            epsilon = get_epsilon(node_tuple)
            layer_norm_op = op_adapter.LayerNormOp(name=layer_norm_op_name,
                                                   axes=axes,
                                                   epsilon=epsilon)
            return layer_norm_op

        def get_layernorm_input_names(node_tuple):
            first_node = node_tuple[0]
            sum_node = node_tuple[-1]
            mul_node = node_tuple[-2]

            main_input_name = first_node.input_names[0]
            beta_input_name = self.get_affine_scalar_input(sum_node, node_tuple, graph)
            gamma_input_name = self.get_affine_scalar_input(mul_node, node_tuple, graph)

            layer_norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.LayerNormOp.TRANSLATION_KEY,
                                                                         op_adapter.LayerNormOp.LEGACY_TRANSLATION_KEY,
                                                                         folded_op=True)

            input_buf = graph.get_input_buffers(node_tuple[0])[0]
            axes = self.get_axes(node_tuple)
            input_shapes_axes = [input_buf.shape.dims[i] for i in axes]

            beta_input_name = self.broadcast_axes(beta_input_name, 0, input_shapes_axes, graph, input_buf,
                                                  layer_norm_op_name, node_tuple)
            gamma_input_name = self.broadcast_axes(gamma_input_name, 1, input_shapes_axes, graph, input_buf,
                                                   layer_norm_op_name, node_tuple)

            layer_norm_input_names = [main_input_name, gamma_input_name, beta_input_name]
            return layer_norm_input_names

        def get_layernorm_output_names(node_tuple):
            last_node = node_tuple[-1]
            return last_node.output_names

        # ------------------------- Matching Strategies ------------------------
        def match_base_layernorm(node_tuple, origin_node_tuple=None):
            '''
            Matches layernorm patterns equivalent to:
            y = (x - E[x]) / sqrt(Var[x] + epsilon) * gamma + beta
            '''

            last_node = node_tuple[-1]
            last_node_consumers = set(graph.get_op_output_nodes(last_node))

            layer_norm_op = make_layernorm_op(node_tuple)
            layer_norm_input_names = get_layernorm_input_names(node_tuple)
            layer_norm_output_names = get_layernorm_output_names(node_tuple)

            # Record trace info before prune nodes
            if not origin_node_tuple:
                origin_node_tuple = node_tuple
            node_tuple_trace_info = graph.get_trace_info_sub_graph(origin_node_tuple,
                                                                   graph.get_output_buffers(last_node))
            output_trace_info = graph.get_trace_info(graph.get_buffer(layer_norm_output_names[0]))

            self.prune_nodes(node_tuple, graph)
            insertion_index = self.get_norm_op_insertion_index(layer_norm_input_names, graph)

            layer_norm_node = graph.add(op=layer_norm_op,
                                        input_names=layer_norm_input_names,
                                        output_names=layer_norm_output_names,
                                        idx=insertion_index)
            # use the orig trace info of these pruned ops set layernorm op's trace info
            graph.set_trace_info(layer_norm_node, node_tuple_trace_info)
            graph.set_trace_info(graph.get_buffer(layer_norm_output_names[0]), output_trace_info)

            self.update_norm_op_output_buffer(layer_norm_node, last_node_consumers, graph)
            graph.replace_quantization_param(last_node.op.name, layer_norm_node.op.name)
            self.update_consumer_data_axis_formats(graph, layer_norm_node)

            return layer_norm_node

        def match_layernorm_with_transpose(node_tuple):
            '''
            Matches layernorm patterns equivalent to:
            y = (x - E[x]) / sqrt(Var[x] + epsilon) * gamma + beta
            '''

            # Populate info required for injecting Transpose at input
            first_node = node_tuple[0]
            transpose_node = [node for node in list(node_tuple)
                              if node.op.type == op_adapter.TransposeOp.TRANSLATION_KEY][0]
            perm = transpose_node.op.perm
            axis_format = graph.get_output_buffers(transpose_node)[0].axis_format
            node_tuple = list(node_tuple)
            consumers = [node.op.name for node in list(graph.get_input_buffers(first_node)[0].consumers)
                         if node in node_tuple]
            # Inject Transpose Node at the start of Op sequence
            graph.inject_implicit_permute(graph.get_input_buffers(first_node)[0].name,
                                          axis_format,
                                          perm,
                                          consumers)

            # Update the axes of first Reduce because of change in axis format
            reduce_node = [node for node in list(node_tuple)
                           if node.op.type == op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN]][0]
            reduce_node.op.axes = [perm.index(reduce_node.op.axes)]

            # Remove original Transpose node
            graph.prune(transpose_node, force_remove=True)
            del node_tuple[node_tuple.index(transpose_node)]

            # Create LayerNorm with rest of the node pattern
            return match_base_layernorm(tuple(node_tuple))

        def match_no_beta(node_tuple, origin_node_tuple=None):
            '''
            Matches layernorm patterns equivalent to:
            y = (x - E[x]) / sqrt(Var[x] + epsilon) * gamma
            '''
            mul_node = node_tuple[-1]
            gamma_buffer_name = self.get_affine_scalar_input(mul_node, node_tuple, graph)
            gamma_buffer = graph.get_buffer(gamma_buffer_name)
            gamma_node = gamma_buffer.producer

            if graph.has_quantization_param(gamma_buffer_name):
                #if gamma buffer encoding is not symmteric for bw=16 then skip matching this sequence
                gamma_encoding = graph.quantization_params[gamma_buffer_name]['output_encodings'][0]
                if not gamma_encoding['is_symmetric'] and gamma_encoding['bw']==16:
                    return

            bias_name = mul_node.op.name + "_bias"
            tensor_dimensions = self.get_weight_bias_dimensions(node_tuple, graph)
            bias_tensor = np.zeros(tensor_dimensions)
            bias_op = op_adapter.ConstantOp(bias_name, tensor=bias_tensor)
            bias_index = graph.list_nodes().index(gamma_node) + 1
            bias_node = graph.add(bias_op,
                                  [],
                                  [bias_name],
                                  axis_formats=[AxisTracker.AxisFormat.ANY],
                                  idx=bias_index)
            # update trace info of new added bias op and output
            if not origin_node_tuple:
                origin_node_tuple = node_tuple
            graph.set_trace_info([bias_node, graph.get_buffer(bias_name)],
                                 graph.get_trace_info_sub_graph(origin_node_tuple))

            add_name = mul_node.op.name + "_add"
            add_op = op_adapter.ElementwiseBinaryOp(name=add_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            add_input_names = bias_node.output_names
            add_output_names = mul_node.output_names
            dummy_add_node = op_graph.OpNode(add_op, add_input_names, add_output_names)

            new_tuple = node_tuple + (dummy_add_node,)
            return match_base_layernorm(new_tuple, origin_node_tuple)

        def match_no_affine_transformation(node_tuple):
            '''
            Matches layernorm patterns equivalent to:
            y = (x - E[x]) / sqrt(Var[x] + epsilon)
            '''
            last_node = node_tuple[-1]

            weight_name = last_node.op.name + "_weight"
            tensor_dimensions = self.get_weight_bias_dimensions(node_tuple, graph)
            weight_tensor = np.ones(tensor_dimensions)
            weight_op = op_adapter.ConstantOp(weight_name, tensor=weight_tensor)
            weight_index = graph.list_nodes().index(last_node) + 1
            weight_node = graph.add(weight_op,
                                    [],
                                    [weight_name],
                                    axis_formats=[AxisTracker.AxisFormat.ANY],
                                    idx=weight_index)
            # update trace info of new added weight op and output
            graph.set_trace_info([weight_node, graph.get_buffer(weight_name)],
                                 graph.get_trace_info_sub_graph(node_tuple))

            mul_output_name = last_node.op.name + "_mul"
            mul_op = op_adapter.ElementwiseBinaryOp(name=mul_output_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
            mul_input_names = weight_node.output_names
            mul_output_names = last_node.output_names
            dummy_mul_node = op_graph.OpNode(mul_op, mul_input_names, mul_output_names)

            new_tuple = node_tuple + (dummy_mul_node,)
            return match_no_beta(new_tuple, node_tuple)

        def match_first_node_reshape(node_tuple):
            '''
            Matches layernorm equivalent patterns that follow a reshape op
            '''
            reshape_node = node_tuple[0]
            reshape_node_input_bufs = graph.get_input_buffers(reshape_node)

            layer_norm_node = match_base_layernorm(node_tuple[1:])
            graph.change_buffer_name(layer_norm_node.output_names[0], layer_norm_node.op.name)

            # add reshape node after layer_norm
            last_node = node_tuple[-1]
            last_node_consumers_names = get_node_consumer_names(layer_norm_node)
            post_reshape_name = layer_norm_node.op.name + "_postprocess_reshape"
            post_reshape_op = op_adapter.ReshapeOp(post_reshape_name, shape=reshape_node_input_bufs[0].shape)
            post_reshape_node = graph.inject(post_reshape_op,
                                             input_name=layer_norm_node.output_names[0],
                                             output_name=post_reshape_name,
                                             consumer_names=last_node_consumers_names if last_node_consumers_names else None)
            graph.replace_quantization_param(last_node.op.name, post_reshape_name)
            return layer_norm_node

        # ---------------------------- Main Function ---------------------------
        matching_strategy_map = {
            match_base_layernorm: [sequence1, sequence5, sequence6],
            match_no_beta: [],
            match_no_affine_transformation: [sequence3, sequence4, sequence7],
            match_first_node_reshape : [sequence2],
            match_layernorm_with_transpose : [sequence8],
        }

        for matching_function, sequences in matching_strategy_map.items():
            sequences_in_descending_length = sorted(sequences,
                                                    key=len,
                                                    reverse=True)
            for sequence in sequences_in_descending_length:
                matched_node_list = graph.get_matched_nodes(sequence,
                                                            validator=self.validate_reduce_mean_ops,
                                                            ignore_constants=True,
                                                            use_dfs=True)
                for node_tuple in matched_node_list:
                    matching_function(node_tuple)


@register_layer_optimization
class OptimizeLogSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LogSoftmaxOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeLogSoftmaxTranslation, self).axes_to_spatial_first_order(node, graph):
            return False

        # Ensure we're using the correct input buffer as a permute might have been inserted above
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            axis_map = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            log_debug('Mapping axis from {} to {}: '.format(node.op.axis, axis_map[node.op.axis]))
            node.op.axis = axis_map[node.op.axis]
        return True


@register_layer_optimization
class OptimizeLrnTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LrnOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMaskedSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MaskedSoftmaxOp.TRANSLATION_KEY
        self.register_method(MASKED_SOFTMAX, self.match_masked_softmax)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.rank() != 4:
            raise ValueError(
                "Backend only support MaskedSoftmax with rank 4, but got an input rank with {}.".format(
                    input_buf.rank()
                )
            )
        # To ensure the MaskedSoftmax's input and output format as NSC
        AxisTracker.image_to_channel_last_order(node, graph)
        return True

    @staticmethod
    def check_conditions(input_ids_buffer, attention_mask_buffer):
        """
        Check buffer shapes for masked softmax operation.
        For Compressed variant:
            input_ids: [B, X, Y, D], B==1 and Y==D, datatype==float32
            attention_mask: [B, S], B==1, datatype==int64, S is the total number of sequences packed in single batch.
            result: [B, X, Y, D], B==1 and Y==D, datatype==float32

        For Uncompressed variant:
            input_ids: [B, X, Y, D], B==1 and Y==D, datatype==float32
            attention_mask: [B, Y], B==1, datatype==float32
            result: [B, X, Y, D], B==1 and Y==D, datatype==float32
        """
        if (input_ids_buffer.shape[0] != 1) or (attention_mask_buffer.shape[0] != 1):
            log_warning(
                f"MaskedSoftmax Optimization only supports batch=1, but got {input_ids_buffer.shape[0]} for tensor {input_ids_buffer.name} and got {attention_mask_buffer.shape[0]} for tensor {attention_mask_buffer.name}."
            )
            return False

        # attention_mask for compressed shall be of shape [batch, sequence] or [batch, 1, sequence].
        if (input_ids_buffer.rank() != 4) or (
            attention_mask_buffer.rank() not in [2, 3]
        ):
            log_warning(
                f"MaskedSoftmax Optimization only supports 4D input_ids and 2D/3D attention_mask inputs, but got rank as {input_ids_buffer.rank()} for input_ids buffer {input_ids_buffer.name} and {attention_mask_buffer.rank()} for attention_mask buffer {attention_mask_buffer.name}."
            )
            return False

        if input_ids_buffer.shape[3] != attention_mask_buffer.shape[1]:
            log_warning(
                f"For MaskedSoftmax Optimization, the input_ids buffer's shape at 4th index and attention_mask buffer's shape at 2nd index should be same. Instead got {input_ids_buffer.shape} for input_ids buffer {input_ids_buffer.name} and {attention_mask_buffer.shape} for attention_mask buffer {attention_mask_buffer.name}."
            )
            return False

        if input_ids_buffer.shape[2] != input_ids_buffer.shape[3]:
            log_warning(
                f"For MaskedSoftmax Optimization, the input_ids buffer's shape at last 2 indices should match. Instead got {input_ids_buffer.shape} for input_ids buffer {input_ids_buffer.name}."
            )
            return False
        return True

    @staticmethod
    def get_reachable_inputs(graph, node):
        """
        Get the list of inputs the given node is connected with.
        """
        stack = graph.get_parent_nodes(node)
        reachable_input_nodes = []
        while len(stack) != 0:
            _node = stack.pop()
            if isinstance(_node.op, op_adapter.InputOp):
                reachable_input_nodes.append(_node)
            else:
                _par_nodes = graph.get_parent_nodes(_node)
                stack.extend(_par_nodes)
        return reachable_input_nodes

    @staticmethod
    def get_embedding_node(graph, input_tensor_name):
        """
        Get the positional embedding node from the graph based on given input_tensor_name.
        """
        sequences = [
            # pattern-1, albert-base-v2
            {
                "sequence": [
                    (
                        "Gather",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("Gather", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("elementwise_sum", "ANY"), ("constant", "ANY")],
                        ),
                        ("MATCH_BUFS_AT_INDEX", [("LayerNorm", "ANY")]),
                    ),
                ],
                "check_embedding_node": -1,
            },
            # pattern-2, 3, distil-bert-uncased, distil-gpt
            {
                "sequence": [
                    (
                        "Gather",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("Gather", "ANY"), ("constant", "ANY")],
                        ),
                        ("MATCH_BUFS_AT_INDEX", [("LayerNorm", "ANY")]),
                    ),
                ],
                "check_embedding_node": -1,
            },
            # pattern-6, opennmt-encoder
            {
                "sequence": [
                    (
                        "Gather",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("Gather", "ANY"), ("constant", "ANY")],
                        ),
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                    ),
                ],
                "check_embedding_node": -1,
            },
            # pattern-8, bart
            # pattern-9, trocr-decoder
            {
                "sequence": [
                    (
                        "Gather",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0)]),
                    ),
                    (
                        "elementwise_product",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("Gather", "ANY"), ("constant", "ANY")],
                        ),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("elementwise_product", "ANY"), ("constant", "ANY")],
                        ),
                        ("MATCH_BUFS_AT_INDEX", [("LayerNorm", "ANY")]),
                    ),
                ],
                "check_embedding_node": -1,
            },
            # pattern-x, bert-large-mlcommons
            {
                "sequence": [
                    (
                        "Gather",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("Gather", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("elementwise_sum", "ANY"), ("Gather", "ANY")],
                        ),
                        ("MATCH_BUFS_AT_INDEX", [("LayerNorm", 0)]),
                    ),
                ],
                "check_embedding_node": -2,
            },
        ]

        embedding_nodes = []
        for sequence_dict in sequences:
            pattern_sequence = sequence_dict["sequence"]
            check_embedding_node = sequence_dict["check_embedding_node"]
            matched_nodes_list = graph.get_matched_nodes_v2(
                pattern_sequence, ignore_constants=True
            )
            if len(matched_nodes_list) == 0:
                continue

            filtered_matched_nodes_list = []
            for matched_node_list in matched_nodes_list:
                start_node = matched_node_list[0]
                reachable_input_nodes = (
                    OptimizeMaskedSoftmaxTranslation.get_reachable_inputs(
                        graph, start_node
                    )
                )

                for r_inp_node in reachable_input_nodes:
                    if input_tensor_name in r_inp_node.output_names:
                        filtered_matched_nodes_list.append(matched_node_list)
                        break

            if len(filtered_matched_nodes_list) != 1:
                log_warning(
                    f"Only one sequence of nodes should be identified for embedding node identification. But following sequences are identified: {matched_nodes_list}"
                )
                return False, None

            matched_nodes = filtered_matched_nodes_list[0]
            embedding_nodes.append(matched_nodes[check_embedding_node])

        if len(embedding_nodes) != 1:
            log_warning(
                f"Only one embedding node should be identified but following nodes are identified: {embedding_nodes}"
            )
            return False, None
        return True, embedding_nodes[0]

    @staticmethod
    def get_constant_node(graph, node):
        """
        Get the constant input node and its input index for a given node.
        """
        par_nodes = graph.get_parent_nodes(node)
        total_constant_node = sum(
            [isinstance(node.op, op_adapter.ConstantOp) for node in par_nodes]
        )
        if total_constant_node != 1:
            log_warning(
                f"One input of {node} shall be a constant input to fetch embedding tensor."
            )
            return False, None
        constant_input_idx, constant_node = [
            (i, node)
            for i, node in enumerate(par_nodes)
            if isinstance(node.op, op_adapter.ConstantOp)
        ][0]
        return True, [constant_input_idx, constant_node]

    @staticmethod
    def add_position_ids(graph, original_io_layouts, packed_masked_softmax_inputs=[]):
        """
        This function will identify constant positional embedding tensor and based on that
        it will create a new input 'positional_ids' followed by Gather and its output
        will be fed to the node where earlier constant embedding tensor was added.
        See below snippet for more details.

        Original model:
            input_ids -> Gather(token_embeddings) -> Add(token_type_embeddings) -> Add(positional_embeddings) -> LayerNorm

        Compressed Packed model:
            input_ids -> Gather(token_embeddings) -> Add(token_type_embeddings) -> Add -> LayerNorm
                                                                                    /
            position_ids -> Gather(positional_embeddings) -------------------------/
        """
        for tensor_to_pack in packed_masked_softmax_inputs:
            try:
                tensor = graph.get_buffer(tensor_to_pack)
            except:
                log_warning(
                    f"Provided tensor name for --packed_masked_softmax_inputs is not present as input in graph. Provided value: {packed_masked_softmax_inputs}"
                )
                return False
            if not isinstance(tensor.producer.op, op_adapter.InputOp):
                log_warning(
                    f"--packed_masked_softmax_inputs flag should contain graph's inputs only. Got {tensor_to_pack} which is not graph input."
                )
                return False

            node_to_pack = tensor.producer
            (
                status,
                embedding_node,
            ) = OptimizeMaskedSoftmaxTranslation.get_embedding_node(
                graph, tensor_to_pack
            )
            if not status:
                return False

            embedding_node_idx = graph.get_node_idx_in_order(embedding_node)
            status, (
                constant_input_idx,
                constant_input_node,
            ) = OptimizeMaskedSoftmaxTranslation.get_constant_node(
                graph, embedding_node
            )
            if not status:
                return False

            embedding_tensor = constant_input_node.op.tensor
            if len(embedding_tensor.shape) != 3:
                log_warning(
                    f"Embedding tensor shape shall be of 3d: [batch, sequence, embedding_dim]. But the shape of embedding tensor is: {embedding_tensor.shape}"
                )
                return False

            # Converting 3d embedding tensor into 2d to use it inside gather node.
            embedding_tensor = np.squeeze(embedding_tensor)

            position_ids_dtype = np.int32
            position_ids_shape = tensor.shape
            if len(position_ids_shape) == 3:
                if position_ids_shape[-1] != 1:
                    log_warning(
                        f"Can't create position_ids input as the shape is not compatible. Identified shape : {position_ids_shape}."
                    )
                    return False
                position_ids_shape = position_ids_shape[:2]

            position_ids_name = tensor_to_pack + "_position_ids"
            insert_idx = embedding_node_idx
            position_ids_op = op_adapter.InputOp(
                position_ids_name,
                position_ids_shape,
                input_encoding_in=node_to_pack.op.input_encoding_in,
                input_encoding_out=node_to_pack.op.input_encoding_out,
                input_type=node_to_pack.op.input_type,
                input_dtype=position_ids_dtype,
            )
            position_ids_op_node = graph.add(
                position_ids_op,
                input_names=[],
                output_names=[position_ids_name],
                idx=insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(position_ids_op_node, [embedding_node])
            original_io_layouts[position_ids_name] = graph.buffers[
                tensor_to_pack
            ].axis_format
            insert_idx += 1

            constant_op_name = tensor_to_pack + "_position_ids_embeddings"
            constant_op = op_adapter.ConstantOp(
                name=constant_op_name, tensor=embedding_tensor
            )
            constant_op_node = graph.add(
                constant_op,
                input_names=[],
                output_names=[constant_op_name],
                idx=insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(constant_op_node, [embedding_node])
            insert_idx += 1

            gather_op_name = tensor_to_pack + "_position_ids_gather"
            gather_op = op_adapter.GatherOp(gather_op_name)
            gather_op_node = graph.add(
                gather_op,
                input_names=[constant_op_name, position_ids_name],
                output_names=[gather_op_name],
                idx=insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(gather_op_node, [embedding_node])
            insert_idx += 1

            # Replace the constant embedding tensor input with gather's output.
            embedding_node.input_names[constant_input_idx] = gather_op_name

            # Clear the consumers of old constant embedding tensor.
            constant_input_node_output_buffer = graph.get_output_buffers(
                constant_input_node
            )[0]
            constant_input_node_output_buffer.consumers.clear()

            # Append the 'Add' node into the consumers of the Gather node.
            gather_output_buffer = graph.get_buffer(gather_op_name)
            gather_output_buffer.consumers.add(embedding_node)
        return True

    @staticmethod
    def get_masked_softmax_input_buffers(
        graph, matched_node_list, check_input_ids_node
    ):
        """
        This function will identify the inputs to the masked softmax node. The
        input_ids input of masked softmax will be identified using check_input_ids_node
        value, whereas the attention_mask input of masked softmax will be identified
        from the 1st node of matched_node_list.
        """
        start_node = matched_node_list[0]

        # Get the input_ids input to Masked Softmax node by checking the parent of
        # check_node which is not a part of identified pattern nodes.
        check_node = matched_node_list[check_input_ids_node]
        matched_parent_of_check_node = matched_node_list[check_input_ids_node - 1]
        parents_of_check_node = [
            graph.get_producer_node(buf.name)
            for buf in graph.get_input_buffers(check_node)
        ]
        node_to_check = [
            node
            for node in parents_of_check_node
            if (node != matched_parent_of_check_node)
            and (node.op.type != op_adapter.ConstantOp.TRANSLATION_KEY)
        ]
        if len(node_to_check) != 1:
            log_warning(
                f"Only one parent should be identified for node {check_node} which is not part of identified pattern."
            )
            return False, []
        input_ids_buffer = graph.get_output_buffers(node_to_check[0])[0]
        attention_mask_buffer = graph.get_input_buffers(start_node)[0]
        return True, [input_ids_buffer, attention_mask_buffer]

    @staticmethod
    def add_compressed_atten_mask_nodes(
        graph, attention_mask_buffer, original_io_layouts, packed_max_seq
    ):
        """
        It will create a new compressed attention_mask input based on packed_max_seq value.
        This new input will be added to original_io_layouts.
        """
        if graph.has_user_quantization_overrides():
            log_warning(
                "Compressed Masked Softmax model can't be generated for models with quantization encodings."
            )
            return False, None

        orig_atten_mask_node = None
        comp_atten_mask_dtype = None
        if isinstance(attention_mask_buffer.producer.op, op_adapter.InputOp):
            # No cast is added after attention_mask buffer and attention_mask is the graph input.
            orig_atten_mask_node = attention_mask_buffer.producer
            comp_atten_mask_dtype = orig_atten_mask_node.op.input_dtype
        elif isinstance(attention_mask_buffer.producer.op, op_adapter.CastOp):
            par_node = graph.get_producer_node(
                attention_mask_buffer.producer.input_names[0]
            )
            if isinstance(par_node.op, op_adapter.InputOp):
                orig_atten_mask_node = par_node
                comp_atten_mask_dtype = attention_mask_buffer.producer.op.to_type
            else:
                log_warning(
                    f"Compressed Masked Softmax model can't be generated as the attention_mask input '{attention_mask_buffer.name}' is not compatible."
                )
                return False, None
        else:
            log_warning(
                f"Compressed Masked Softmax model can't be generated as the attention_mask input '{attention_mask_buffer.name}' is not compatible."
            )
            return False, None

        if comp_atten_mask_dtype == "bool":
            comp_atten_mask_dtype = "int32"

        comp_atten_mask_op_insert_idx = graph.get_node_idx_in_order(
            orig_atten_mask_node
        )

        # Assuming that the original model don't have any node with this name.
        comp_atten_mask_op_name = orig_atten_mask_node.op.name + "_compressed"

        if graph.get_node_by_name(comp_atten_mask_op_name) is not None:
            # This will indicate that the compressed_attention_mask input is already created.
            # So reuse the same.
            return True, comp_atten_mask_op_name
        else:
            batch_size_value = orig_atten_mask_node.op.shape[0]
            comp_atten_mask_input_shape = [batch_size_value, packed_max_seq]
            comp_atten_mask_op = op_adapter.InputOp(
                comp_atten_mask_op_name,
                comp_atten_mask_input_shape,
                input_encoding_in=orig_atten_mask_node.op.input_encoding_in,
                input_encoding_out=orig_atten_mask_node.op.input_encoding_out,
                input_type=orig_atten_mask_node.op.input_type,
                input_dtype=comp_atten_mask_dtype,
            )
            comp_atten_mask_op_node = graph.add(
                comp_atten_mask_op,
                input_names=[],
                output_names=[comp_atten_mask_op_name],
                idx=comp_atten_mask_op_insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(comp_atten_mask_op_node, [attention_mask_buffer])

            log_info(
                f"For Compressed Masked Softmax model, new input '{comp_atten_mask_op_name}' is added in the graph with shape '{comp_atten_mask_input_shape}' and dtype '{comp_atten_mask_dtype}'"
            )

            # Add the newly created input op in the original_io_layouts.
            original_io_layouts[comp_atten_mask_op_name] = graph.buffers[
                comp_atten_mask_op_name
            ].axis_format

            return True, comp_atten_mask_op_name

    @staticmethod
    def identify_mask_inversion(graph, matched_node_list):
        """
        Identify whether the attention mask input is inverted by sub node or equal node.
        """
        invert_mask = False
        for node in matched_node_list:
            if node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL]:
                for p_node in graph.get_parent_nodes(node):
                    if isinstance(p_node.op, op_adapter.ConstantOp):
                        const_operand = p_node.op.tensor
                        if (len(const_operand.flatten()) == 1) and (const_operand[0] == 0):
                            return True
            elif node.op.type == op_adapter.ElementwiseUnaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT]:
                return True
            elif node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT]:
                if isinstance(graph.get_parent_nodes(node)[0].op, op_adapter.ConstantOp):
                    const_operand = graph.get_parent_nodes(node)[0].op.tensor
                    if (len(const_operand.flatten()) == 1) and (const_operand[0] == 1):
                        return True
        return invert_mask

    @staticmethod
    def add_uncompressed_atten_mask_nodes(graph, attention_mask_buffer, invert_mask):
        """
        Modify the graph for uncompressed attention mask by adding relevant nodes
        to make the attention mask buffer compatible for Uncompressed MaskedSoftmax operator.
        """
        # The validation for attention_mask shape either 2 or 3 is already done during check_conditions call.
        # If the attention_mask buffer is of 3d shape [batch, 1, seq], then add a squeeze node.
        squeeze_mask = False
        if (len(attention_mask_buffer.shape) == 3) and (
            attention_mask_buffer.shape[1:].count(1) >= 1
        ):
            squeeze_mask = True

        if squeeze_mask:
            reshape_op_name = graph.naming_policy.get_op_name_by_type(
                op_adapter.ReshapeOp.TRANSLATION_KEY,
                op_adapter.ReshapeOp.LEGACY_TRANSLATION_KEY,
            )

            attention_mask_shape = attention_mask_buffer.shape
            indices = [
                i
                for i in range(1, len(attention_mask_shape))
                if attention_mask_shape[i] == 1
            ]
            if len(indices) > 1:
                # In case of more than one axis with shape 1 then consider 1st instance.
                ignore_index = indices[0]
            elif len(indices) == 1:
                ignore_index = indices[0]
            else:
                log_warning(
                    f"Can't determine new shape for reshaping the attention mask tensor. Shape of attention mask found as: {attention_mask_shape}."
                )
                return False, None

            new_shape = [
                s
                for i, s in enumerate(attention_mask_buffer.shape)
                if i != ignore_index
            ]

            reshape_op = op_adapter.ReshapeOp(reshape_op_name, shape=new_shape)
            reshape_insert_idx = (
                graph.get_node_idx_in_order(
                    graph.get_producer_node(attention_mask_buffer.name)
                )
                + 1
            )
            reshape_op_node = graph.add(
                reshape_op,
                input_names=[attention_mask_buffer.name],
                output_names=[reshape_op_name],
                idx=reshape_insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(reshape_op_node, [attention_mask_buffer])

        # Add a cast node to convert the attention mask into float32.
        cast_op_name = graph.naming_policy.get_op_name_by_type(
            op_adapter.CastOp.TRANSLATION_KEY, op_adapter.CastOp.LEGACY_TRANSLATION_KEY
        )

        cast_op = op_adapter.CastOp(cast_op_name, to_type="float32")
        cast_inputs = [reshape_op_name if squeeze_mask else attention_mask_buffer.name]
        cast_insert_idx = (
            graph.get_node_idx_in_order(graph.get_producer_node(cast_inputs[0])) + 1
        )
        cast_op_node = graph.add(
            cast_op,
            input_names=cast_inputs,
            output_names=[cast_op_name],
            idx=cast_insert_idx,
        )
        # Update trace info for new created node
        graph.update_trace_info(cast_op_node, [attention_mask_buffer])

        # The typical attention_mask path converts [1,1,1,1,0,0] input into [0,0,0,0,-inf,-inf].
        # For uncompressed mask, we will need to pass this inverted mask for such model since
        # uncompressed mask takes [0,0,0,0,-inf,-inf] mask. 0s for valid token and large non-zero
        # negative number for ignore tokens.
        # But for models which directly take the inverted attention mask input, i.e. [0,0,0,1,1]
        # 0s for valid token and 1s for invalid tokens, we don't need to have nodes which invert the mask.
        # However, the multiplication with -10000 or some large negative value is still required and
        # will be performed after the inversion of the mask.
        if invert_mask:
            ones_tensor = np.ones([1], dtype=np.float32)
            ones_op_name = graph.naming_policy.get_op_name_by_type(
                op_adapter.ConstantOp.TRANSLATION_KEY,
                op_adapter.ConstantOp.LEGACY_TRANSLATION_KEY,
            )
            ones_op = op_adapter.ConstantOp(name=ones_op_name, tensor=ones_tensor)
            add_insert_idx = cast_insert_idx + 1
            ones_op_node = graph.add(
                ones_op,
                input_names=[],
                output_names=[ones_op_name],
                idx=add_insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(ones_op_node, [attention_mask_buffer])

            # Sub node will subtract attention_mask input from 1 to invert the mask values.
            sub_op_name = graph.naming_policy.get_op_name_by_type(
                ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT,
                op_adapter.ElementwiseBinaryOp.operation_to_legacy[
                    ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT
                ],
            )
            sub_op = op_adapter.ElementwiseBinaryOp(
                name=sub_op_name,
                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT,
            )
            sub_insert_idx = add_insert_idx + 1
            sub_op_node = graph.add(
                sub_op,
                input_names=[ones_op_name, cast_op_name],
                output_names=[sub_op_name],
                idx=sub_insert_idx,
            )
            # Update trace info for new created node
            graph.update_trace_info(sub_op_node, [attention_mask_buffer])

        # Large non-zero negative number will work.
        neg_tensor = np.array([-10000], dtype=np.float32)
        neg_op_name = graph.naming_policy.get_op_name_by_type(
            op_adapter.ConstantOp.TRANSLATION_KEY,
            op_adapter.ConstantOp.LEGACY_TRANSLATION_KEY,
        )
        neg_op = op_adapter.ConstantOp(name=neg_op_name, tensor=neg_tensor)
        neg_insert_idx = (sub_insert_idx + 1) if invert_mask else (cast_insert_idx + 1)
        neg_op_node = graph.add(
            neg_op,
            input_names=[],
            output_names=[neg_op_name],
            idx=neg_insert_idx,
        )
        # Update trace info for new created node
        graph.update_trace_info(neg_op_node, [attention_mask_buffer])

        # Mul node will multiply the output of sub node with -10000 to make the ignore locations negative.
        # These negative locations will be ignored in the implementation.
        mul_op_name = graph.naming_policy.get_op_name_by_type(
            ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
            op_adapter.ElementwiseBinaryOp.operation_to_legacy[
                ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY
            ],
        )
        mul_op = op_adapter.ElementwiseBinaryOp(
            name=mul_op_name,
            operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY,
        )

        if invert_mask:
            mul_input_names = [neg_op_name, sub_op_name]
        else:
            mul_input_names = [neg_op_name, cast_op_name]

        mul_insert_idx = neg_insert_idx + 1
        mul_op_node = graph.add(
            mul_op,
            input_names=mul_input_names,
            output_names=[mul_op_name],
            idx=mul_insert_idx,
        )
        # Update trace info for new created node
        graph.update_trace_info(mul_op_node, [attention_mask_buffer])

        return True, mul_op_name

    @staticmethod
    def add_masked_softmax_node(
        graph, last_matched_node, masked_softmax_inputs, is_compressed
    ):
        """
        Add the masked softmax node and remove the irrelevant connection and
        add the new connection between the nodes.
        """
        masked_softmax_insert_idx = (
            max(
                [
                    graph.get_node_idx_in_order(graph.get_producer_node(inp_name))
                    for inp_name in masked_softmax_inputs
                ]
            )
            + 1
        )
        masked_softmax_op_name = graph.naming_policy.get_op_name_by_type(
            op_adapter.MaskedSoftmaxOp.TRANSLATION_KEY,
            op_adapter.MaskedSoftmaxOp.LEGACY_TRANSLATION_KEY,
            folded_op=True
        )
        masked_softmax_op = op_adapter.MaskedSoftmaxOp(
            masked_softmax_op_name, mode=is_compressed
        )
        masked_softmax_op_node = graph.add(
            masked_softmax_op,
            input_names=masked_softmax_inputs,
            output_names=[masked_softmax_op_name],
            idx=masked_softmax_insert_idx,
        )
        # Update trace info for new created node
        graph.update_trace_info(masked_softmax_op_node, [last_matched_node])
        masked_softmax_op_buffer = graph.get_buffer(masked_softmax_op_name)
        masked_softmax_output_shape = masked_softmax_op_buffer.shape

        last_node_buffer = graph.get_output_buffers(last_matched_node)[0]

        apply_squeeze = False
        if len(last_node_buffer.shape) == 3:
            # This means that the original model has 3d output from softmax node.
            # Add a reshape node to squeeze the 4d output from MaskedSoftmax node into 3d output.
            squeeze_op_name = graph.naming_policy.get_op_name_by_type(
                op_adapter.ReshapeOp.TRANSLATION_KEY,
                op_adapter.ReshapeOp.LEGACY_TRANSLATION_KEY,
            )

            # Assuming the batch=1 limitation of Masked Softmax.
            if masked_softmax_output_shape[0] != 1:
                log_warning("Masked Softmax node's output shape shall have batch=1.")
                return False

            new_shape = masked_softmax_output_shape[1:]
            squeeze_op = op_adapter.ReshapeOp(squeeze_op_name, shape=new_shape)
            squeeze_op_node = graph.add(
                squeeze_op,
                input_names=[masked_softmax_op_name],
                output_names=[squeeze_op_name],
                idx=masked_softmax_insert_idx + 1,
            )
            # Update trace info for new created node
            graph.update_trace_info(squeeze_op_node, [last_matched_node])
            squeeze_op_buffer = graph.get_buffer(squeeze_op_name)
            apply_squeeze = True

        # Update the child_node's input which was earlier last_node_buffer.name to masked_softmax_op_name
        child_node = graph.get_consumer_nodes(last_node_buffer.name)
        if len(child_node) != 1:
            log_warning(
                "MaskedSoftmax pattern's output shall be consumed by single node only."
            )
            return False

        # Clear the consumers of the last detected node in the pattern. This dangling node will be removed during remove_disconnected_nodes call.
        last_node_buffer.consumers.clear()

        for i, input_name in enumerate(child_node[0].input_names):
            if input_name == last_node_buffer.name:
                child_node[0].input_names[i] = (
                    squeeze_op_name if apply_squeeze else masked_softmax_op_name
                )

        # Add the child_node as a consumer of last node in the sequence of newly added nodes.
        if apply_squeeze:
            squeeze_op_buffer.consumers.add(child_node[0])
        else:
            masked_softmax_op_buffer.consumers.add(child_node[0])
        return True

    @staticmethod
    def apply_masked_softmax(
        graph,
        matched_nodes_list,
        original_io_layouts,
        check_input_ids_node=-2,
        is_compressed=True,
        packed_max_seq=1,
    ):
        """
        This function will apply modifications to generate Compressed or Uncompressed
        variant of Masked Softmax model.

        Original Model:
            input_ids -> ....................... -> Q<matmul>K_T -> Div
                                                                       \
                                                                        \
            attention_mask -> Cast -> Reshape -> Cast -> Sub -> Mul -> Add -> Softmax

        Uncompressed Model:
            input_ids -> ......... -> Q<matmul>K_T -> Div
                                                       \
                                                        \
            attention_mask -> Cast -> Sub -> Mul -> MaskedSoftmax(mode=0)

        Compressed Model:
            input_ids -> ......... -> Q<matmul>K_T -> Div
                                                       \
                                                        \
            attention_mask ------------------------> MaskedSoftmax(mode=0)
        """
        masked_softmax_opt_applied = False
        for matched_node_list in matched_nodes_list:
            is_quant_info_available = any([graph.has_quantization_param(node) for node in matched_node_list])
            if is_quant_info_available:
                log_debug(f"Nodes: {matched_node_list} have quantization parameter. Hence masked softmax optimization will be skipped.")
                continue

            status, [
                input_ids_buffer,
                attention_mask_buffer,
            ] = OptimizeMaskedSoftmaxTranslation.get_masked_softmax_input_buffers(
                graph, matched_node_list, check_input_ids_node
            )
            if not status:
                continue
            status = OptimizeMaskedSoftmaxTranslation.check_conditions(
                input_ids_buffer, attention_mask_buffer
            )
            if not status:
                log_warning(
                    f"For matched nodes: {matched_node_list}, the input shapes are not compatible. So MaskedSoftmax optimization won't be applied for these nodes."
                )
                continue

            masked_softmax_inputs = [input_ids_buffer.name]
            if is_compressed:
                (
                    status,
                    comp_atten_mask_op_name,
                ) = OptimizeMaskedSoftmaxTranslation.add_compressed_atten_mask_nodes(
                    graph, attention_mask_buffer, original_io_layouts, packed_max_seq
                )
                if not status:
                    continue
                masked_softmax_inputs.append(comp_atten_mask_op_name)
            else:
                invert_mask = OptimizeMaskedSoftmaxTranslation.identify_mask_inversion(graph, matched_node_list)
                (
                    status,
                    mul_op_name,
                ) = OptimizeMaskedSoftmaxTranslation.add_uncompressed_atten_mask_nodes(
                    graph, attention_mask_buffer, invert_mask
                )
                masked_softmax_inputs.append(mul_op_name)
                if not status:
                    continue

            status = OptimizeMaskedSoftmaxTranslation.add_masked_softmax_node(
                graph, matched_node_list[-1], masked_softmax_inputs, is_compressed
            )
            if not status:
                continue
            masked_softmax_opt_applied = True
        return masked_softmax_opt_applied

    @staticmethod
    def match_masked_softmax(
        graph,
        original_io_layouts,
        masked_softmax_level="uncompressed",
        packed_masked_softmax_inputs=[],
        packed_max_seq=1,
    ):
        """
        Apply masked softmax optimization by identifying relevant pattern in the
        graph and replacing those nodes with new set of node including MaskedSoftmax
        node.
        """
        # "sequence": Represents node sequence.
        # "check_input_ids_node": This is to identify the input ids tensor for Masked Softmax node. This is used for uncompressed and compressed variants both.
        sequences = [
            # pattern-1, albert-base-v2
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0)])
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0)]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0)]),
                    ),
                    (
                        "elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 1)]),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-2, distilbert-base
            {
                "sequence": [
                    (
                        "elementwise_equal",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                    ),
                    (
                        "Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_equal", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("Expand", 0)]),
                    ),
                    (
                        "Expand",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", 0)]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("Expand", 0), ("MatMul", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Softmax", 0)]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_select", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-3, distilgpt2
            # pattern-4, codegen-350m
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0)])
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0)]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0)]),
                    ),
                    (
                        "elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("elementwise_select", 0), ("elementwise_product", 1)],
                        ),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-3, distilgpt2 (attention_mask in float32)
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0)])
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0)]),
                    ),
                    (
                        "elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("elementwise_select", 0), ("elementwise_product", 1)],
                        ),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-5, t5-small-encoder
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0)])
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", 0)]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", 0)]),
                    ),
                    (
                        "elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        (
                            "MATCH_BUFS_AT_INDEX",
                            [("MatMul", 0), ("elementwise_sum", 1)],
                        ),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-6, opennmt_encoder,
            # pattern-7-b, opennmt_decoder
            {
                "sequence": [
                    (
                        "Transpose",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)])
                    ),
                    (
                        "Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", 0)]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0), ("MatMul", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Softmax", 0)]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_select", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-7-a, opennmt_decoder
            {
                "sequence": [
                    (
                        "Transpose",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0)])
                    ),
                    (
                        "Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                    ),
                    (
                        "Transpose",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Tile", "ANY")]),
                    ),
                    (
                        "Tile",
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                    ),
                    (
                        "Transpose",
                        ("MATCH_BUFS_AT_INDEX", [("Tile", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                    ),
                    (
                        "Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("Transpose", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                    ),
                    (
                        "Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", "ANY")]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", 0), ("MatMul", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Softmax", 0)]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_select", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -2,
            },
            # pattern-8-a, bart (encoder - self attention)
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("Expand", "ANY")])
                    ),
                    (
                        "Expand",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("Expand", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", "ANY")]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0), ("elementwise_sub", 2)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_NUM_BUFS", [("Reshape", 0), ("elementwise_select", 1)]),
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                    ),
                    (
                        "Reshape",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")]),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -3,
            },
            # pattern-8-b, bart (decoder - cross attention - not supported)
            # pattern-8-c, bart (decoder - self attention - supported)
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")])
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", "ANY")]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0), ("elementwise_sub", 2)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", 0)]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_NUM_BUFS", [("Reshape", 0), ("elementwise_select", 1)]),
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                    ),
                    (
                        "Reshape",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")]),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -3,
            },
            # pattern-9, trocr-decoder
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("Expand", "ANY")])
                    ),
                    (
                        "Expand",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("Expand", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", "ANY")]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0), ("elementwise_sub", 2)]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
                    ),
                    (
                        "elementwise_sum",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                    ),
                    (
                        "Reshape",
                        ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")]),
                        ("MATCH_NUM_BUFS", [("Softmax", "ALL")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ALL")]),
                    ),
                ],
                "check_input_ids_node": -3,
            },
            # pattern-10, deberta-base
            {
                "sequence": [
                    (
                        "Reshape",
                        (),
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")])
                    ),
                    (
                        "Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
                    ),
                    (
                        "elementwise_product",
                        ("MATCH_BUFS_AT_INDEX", [("Reshape", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                    ),
                    (
                        "elementwise_sub",
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("cast", "ANY")]),
                    ),
                    (
                        "cast",
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_sub", "ANY")]),
                        ("MATCH_BUFS_AT_INDEX", [("elementwise_select", "ANY")]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0), ("elementwise_sum", 2)]),
                        ("MATCH_BUFS_AT_INDEX", [("Softmax", "ANY")]),
                    ),
                    (
                        "Softmax",
                        ("MATCH_NUM_BUFS", [("elementwise_select", "ANY")]),
                        ("MATCH_NUM_BUFS", [("elementwise_select", "ANY")]),
                    ),
                    (
                        "elementwise_select",
                        ("MATCH_BUFS_AT_INDEX", [("cast", 0), ("Softmax", 2)]),
                        ("MATCH_NUM_BUFS", [("MatMul", "ANY")]),
                    ),
                ],
                "check_input_ids_node": -3,
            },
        ]

        if (masked_softmax_level == "uncompressed") and (
            (packed_max_seq != 1) or (len(packed_masked_softmax_inputs) != 0)
        ):
            log_warning(
                "--apply_masked_softmax 'uncompressed' flag is used with either --packed_max_seq or --packed_masked_softmax_inputs. These 2 flags will be ignored since for uncompressed these flags are redundant."
            )

        is_compressed = True if masked_softmax_level == "compressed" else False

        masked_softmax_applied = False
        for sequence_dict in sequences:
            pattern_sequence = sequence_dict["sequence"]
            check_input_ids_node = sequence_dict["check_input_ids_node"]
            matched_nodes_list = graph.get_matched_nodes_v2(
                pattern_sequence, ignore_constants=True
            )

            status = (
                OptimizeMaskedSoftmaxTranslation.apply_masked_softmax(
                    graph,
                    matched_nodes_list,
                    original_io_layouts,
                    check_input_ids_node=check_input_ids_node,
                    is_compressed=is_compressed,
                    packed_max_seq=packed_max_seq,
                )
            )
            masked_softmax_applied = status or masked_softmax_applied

        if not masked_softmax_applied:
            log_debug("Masked Softmax optimization is not applied. This could " \
                "be due to failure to identify patterns or failure to satisfy " \
                "conditions pertaining to Masked Softmax Optimization.")

        if is_compressed and masked_softmax_applied:
            if (packed_max_seq == 1) and (len(packed_masked_softmax_inputs) != 0):
                log_warning(
                    f"For compressed masked softmax model generation, the --packed_masked_softmax_inputs flag is provided with --packed_max_seq=1, which is redundant."
                )
            elif (packed_max_seq != 1) and len(packed_masked_softmax_inputs) == 0:
                raise RuntimeError(
                    f"For compresed masked softmax model generation, to pack {packed_max_seq} sequences, the input_ids tensor name shall be provided via --packed_masked_softmax_inputs flag."
                )
            elif (packed_max_seq != 1) and len(packed_masked_softmax_inputs) != 0:
                # FIXME: If user has provided quantization_encodings, then shall we create this input?
                status = OptimizeMaskedSoftmaxTranslation.add_position_ids(
                    graph,
                    original_io_layouts,
                    packed_masked_softmax_inputs=packed_masked_softmax_inputs,
                )
                if not status:
                    raise RuntimeError(
                        "For compressed masked softmax model, position ids are not added correctly."
                    )


@register_layer_optimization
class OptimizeMatmulTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MatMulOp.TRANSLATION_KEY
        self.register_method(ALIGN_MATMUL_RANKS, self.align_matmul_input_ranks)
        self.register_method(MATMUL_ADD_FUSION, self.matmul_add_fusion)
        self.register_method(MATMUL_TO_FC, self.matmul_to_fc)
        self.register_method(SQUASH_BATCHNORM, self.squash_batchnorm)

    @staticmethod
    def align_matmul_input_ranks(node, graph):
        inputs = graph.get_input_buffers(node)
        output = graph.get_output_buffers(node)[0]
        log_debug1("Running matmal optimization for {}".format(node.op.name))
        if inputs[0].rank() != inputs[1].rank():
            log_debug1("Matmul {} input {} rank {} != input2 {} rank {}".format(node.op.name, inputs[0].name, inputs[0].rank(), inputs[1].name, inputs[1].rank()))
            lower_rank_input_buf, larger_rank_input_buf = (inputs[0], inputs[1]) \
                            if inputs[0].rank() < inputs[1].rank() else (inputs[1], inputs[0])

            # Adding reshape nodes to expand rank to match other input
            producer = lower_rank_input_buf.producer.op
            new_shape = translation_utils.expand_to_rank(lower_rank_input_buf.shape, len(larger_rank_input_buf.shape))
            log_debug1("This matmul impl requires identical rank, reshaping {} to {}".format(lower_rank_input_buf.shape, new_shape))
            if producer.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                producer.tensor = producer.tensor.reshape(new_shape)
                lower_rank_input_buf.shape = new_shape
                lower_rank_input_buf.axis_format = larger_rank_input_buf.axis_format
            else:
                reshape_node_name = output.name + "_" + lower_rank_input_buf.name + "_reshape"
                reshape_op = op_adapter.ReshapeOp(name=reshape_node_name,
                                                  shape=new_shape)
                graph.inject(reshape_op, input_name=lower_rank_input_buf.name,
                             output_name=reshape_node_name, consumer_names=[node.op.name],
                             axis_format=larger_rank_input_buf.axis_format)
        # Reevaluate input buffers since reshape may have been added
        inputs = graph.get_input_buffers(node)
        node.op.populate_data_axis_formats(graph, inputs)

    def axes_to_spatial_first_order(self, node, graph):
        # matmul is always performed in Src Framework order,
        # because only then the last 2 dimensions will align
        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        for i, input_name in enumerate(node.input_names):
            input_buf = graph.get_buffer(input_name)
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NCDHW:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NCS:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NCF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[i] == AxisTracker.AxisFormat.NSC:
                pass
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NC or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_MATMUL_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = graph.get_input_buffers(node)[0].axis_format

        return True

    ## Do not perform this optimization if input tensor has dynamic dimensions as we don't support dynamic axes for FC Op.
    def validate_dims(self, matmul_input_0_buffer, matmul_input_1_buffer, matmul_op):
        ## Remove the check once dynamic axes support is provided for MatMul/FC.
        if matmul_input_0_buffer.shape.is_dynamic() or matmul_input_1_buffer.shape.is_dynamic():
            log_warning("MatMul/FC Op does not have dynamic axes support. " \
                        "MatMul-Add Fusion / Matmul to FC optimization will not be performed.")
            return False

        # Let matmul_input_0_buffer shape = [m0, n0] and matmul_input_1_buffer shape = [m1, n1]
        # For matrix multiplication, the input shapes must align as given below:
        # If transpose_in0 = False and transpose_in1 = True then n1 = n0
        # If transpose_in0 = False and transpose_in1 = False then m1 = n0
        if matmul_input_1_buffer.rank() == 2 and matmul_op.transpose_in0 == False:
            dim_0_to_match = matmul_input_0_buffer.shape[-1]
            if matmul_op.transpose_in1 == True:
                dim_1_to_match = matmul_input_1_buffer.shape[-1]
            else:
                dim_1_to_match = matmul_input_1_buffer.shape[-2]
            return dim_0_to_match == dim_1_to_match
        return False

    def validate_bias(self, matmul_input_1_buffer, bias_buffer, matmul_op):
        ## Remove the check once dynamic axes support is provided for MatMul/FC.
        if matmul_input_1_buffer.shape.is_dynamic() or bias_buffer.shape.is_dynamic():
            log_warning("MatMul/FC Op does not have dynamic axes support. " \
                        "MatMul-Add Fusion / Matmul to FC optimization will not be performed.")
            return False

        # Let matmul_input_1_buffer shape = [m1, n1] and bias_buffer shape = [n2]
        # For matrix multiplication, the input shapes must align as given below:
        # If transpose_in1 = True then n2 = m1
        # If transpose_in1 = False then n2 = n1
        if matmul_op.transpose_in1 == True:
            dim_1_to_match = matmul_input_1_buffer.shape[-2]
        else:
            dim_1_to_match = matmul_input_1_buffer.shape[-1]
        if np.prod(bias_buffer.shape) == bias_buffer.shape[-1] and \
            dim_1_to_match == bias_buffer.shape[-1]:
            return True
        return False

    def matmul_add_fusion(self, graph):
        def validate_sequence(nodes_tuple):
            matmul_node = nodes_tuple[0]
            elementwise_sum_node = nodes_tuple[1]
            elementwise_sum_input_nodes = graph.get_op_input_nodes(elementwise_sum_node)
            constant_bias_node = [node for node in elementwise_sum_input_nodes if node.op.TRANSLATION_KEY == 'constant'][0]
            matmul_input_buffers = graph.get_input_buffers(matmul_node)
            matmul_input_0_buffer = matmul_input_buffers[0]
            matmul_input_1_buffer = matmul_input_buffers[1]
            bias_buffer = graph.get_output_buffers(constant_bias_node)[0]
            matmul_op = matmul_node.op

            if matmul_node.output_names[0] in elementwise_sum_node.input_names and \
                    self.validate_dims(matmul_input_0_buffer, matmul_input_1_buffer, matmul_op) and \
                    self.validate_bias(matmul_input_1_buffer, bias_buffer, matmul_op):
                return True
            return False

        sequence = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]), ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("constant", "ANY"), ("MatMul", "ANY")]), ()
             )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_sequence, ignore_constants=True)
        for node_tuple in matched_node_list:
            matmul_node = node_tuple[0]
            elementwise_sum_node = node_tuple[1]
            elementwise_input_nodes = graph.get_op_input_nodes(elementwise_sum_node)

            # Get the Constant bias node which is the input to the Add node
            constant_bias_node = [node for node in elementwise_input_nodes if node.op.TRANSLATION_KEY == 'constant'][0]
            bias_name = constant_bias_node.op.name
            bias_buffer = graph.get_buffer(bias_name)

            # Update the bias_buffer consumers.
            bias_buffer.consumers.add(matmul_node)
            matmul_node.input_names.append(bias_name)
            bias_buffer.consumers.remove(elementwise_sum_node)
            elementwise_sum_node.input_names.remove(bias_name)

            # update order from [input, weight, matmul, bias] to [input, weight, bias, matmul]
            idx_bias = graph.nodes_in_order.index(constant_bias_node)
            idx_matmul = graph.nodes_in_order.index(matmul_node)
            if idx_bias > idx_matmul:
                graph.nodes_in_order[idx_matmul] = constant_bias_node
                graph.nodes_in_order[idx_bias] = matmul_node
            graph.squash(elementwise_sum_node, graph.get_output_buffers(matmul_node)[0].name)

            elementwise_sum_output_buffer = graph.get_output_buffers(elementwise_sum_node)[0]
            # Refetch matmul_node after squash
            matmul_node = graph.get_node_by_name(matmul_node.op.name)

            # Transferring activation encoding of elementwise sum to MatMul.
            q = graph.user_quantization_overrides
            if q and 'activation_encodings' in q and elementwise_sum_output_buffer.name in q['activation_encodings']:
                activations = q['activation_encodings']
                version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                act_encs = [IROptimizations.extract_encoding_dict(graph.get_output_buffers(matmul_node)[0].name,
                                                                    activations[elementwise_sum_output_buffer.name], version)]
                graph.add_quantization_params(matmul_node.op.name, output_encodings=act_encs)

            # adds the quantization params to the graph if present
            q = graph.user_quantization_overrides
            if q and 'param_encodings' in q and bias_name in q['param_encodings']:
                params = q['param_encodings']
                version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                param_encs = [IROptimizations.extract_encoding_dict('bias', params[bias_name], version)]
                graph.add_quantization_params(matmul_node.op.name, param_encodings=param_encs)

    def matmul_to_fc(self, graph):
        def validate_sequence1(nodes_tuple):
            matmul_node = nodes_tuple[0]
            elementwise_sum_node = nodes_tuple[1]
            elementwise_sum_input_nodes = graph.get_op_input_nodes(elementwise_sum_node)
            constant_bias_node = [node for node in elementwise_sum_input_nodes if node.op.TRANSLATION_KEY == 'constant'][0]
            matmul_input_buffers = graph.get_input_buffers(matmul_node)
            matmul_input_0_buffer = matmul_input_buffers[0]
            matmul_input_1_buffer = matmul_input_buffers[1]
            bias_buffer = graph.get_output_buffers(constant_bias_node)[0]
            matmul_op = matmul_node.op

            if matmul_node.output_names[0] in elementwise_sum_node.input_names and \
                    self.validate_dims(matmul_input_0_buffer, matmul_input_1_buffer, matmul_op) and \
                    self.validate_bias(matmul_input_1_buffer, bias_buffer, matmul_op):
                return True
            return False

        def validate_sequence2(nodes_tuple):
            matmul_node = nodes_tuple[0]
            matmul_input_buffers = graph.get_input_buffers(matmul_node)
            matmul_input_0_buffer = matmul_input_buffers[0]
            matmul_input_1_buffer = matmul_input_buffers[1]
            matmul_op = matmul_node.op
            return self.validate_dims(matmul_input_0_buffer, matmul_input_1_buffer, matmul_op)

        def transpose_weight(weights_buf):
            weights = weights_buf.producer.op.tensor
            weights = np.ascontiguousarray(np.transpose(weights, (1, 0)))
            weights_buf.producer.op.tensor = weights
            weights_buf.shape = list(weights.shape)

        sequence1 = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]), ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("constant", "ANY"), ("MatMul", "ANY")]), ()
             )
        ]

        sequence2 = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
             ()
             )
        ]
        sequences = [sequence1, sequence2]
        for idx, sequence in enumerate(sequences):
            if idx == 0:
                matched_node_list = graph.get_matched_nodes(sequence, validator=validate_sequence1, ignore_constants=True)
            else:
                matched_node_list = graph.get_matched_nodes(sequence, validator=validate_sequence2)
            for node_tuple in matched_node_list:
                matmul_node = node_tuple[0]
                constant_weights_node = graph.get_op_input_nodes(matmul_node)[1]
                if idx == 0:
                    elementwise_sum_node = node_tuple[1]
                    elementwise_input_nodes = graph.get_op_input_nodes(elementwise_sum_node)
                    constant_bias_node = [node for node in elementwise_input_nodes if node.op.TRANSLATION_KEY == 'constant'][0]

                matmul_input_buffer = graph.get_input_buffers(matmul_node)[0]
                if matmul_input_buffer.rank() > 2:
                    pre_reshape_name = matmul_node.op.name + "_pre_reshape"
                    pre_reshape_shape = [np.prod(matmul_input_buffer.shape[:-1]), matmul_input_buffer.shape[-1]]
                    pre_reshape_op = op_adapter.ReshapeOp(pre_reshape_name, shape=pre_reshape_shape)
                    graph.inject(pre_reshape_op, input_name=matmul_node.input_names[0], output_name=pre_reshape_name,
                                consumer_names=[matmul_node.op.name], axis_format=AxisTracker.AxisFormat.NF)
                matmul_op = matmul_node.op
                matmul_weight_buffer = graph.get_input_buffers(matmul_node)[1]

                # Currently, FC weight shape is IO in spatial-last axis order, OI in other axis orders
                # TODO: Remove axis order dependent code of FC weight after ONNX GEMM cleanup
                if isinstance(graph.src_axis_order, SpatialLastAxisOrder):
                    if matmul_op.transpose_in1 == True:
                        transpose_weight(matmul_weight_buffer)
                else:
                    if matmul_op.transpose_in1 == False:
                        transpose_weight(matmul_weight_buffer)
                fc_weights = constant_weights_node.op.tensor
                matmul_op_name = matmul_op.name
                fc_op_name = matmul_op_name

                if idx == 0:
                    bias = constant_bias_node.op.tensor.copy()
                    constant_bias_node.op.tensor = np.atleast_1d(np.squeeze(bias))
                    bias_name = constant_bias_node.op.name
                    # bias must be 1D tensor so directly update the axis format to ANY
                    # after modify the tensor, to avoid layout mismatch
                    graph.get_buffer(constant_bias_node.output_names[0]).axis_format \
                        = AxisTracker.AxisFormat.ANY
                    fc_op = op_adapter.FullyConnectedOp(name=fc_op_name, bias_op_name=bias_name)
                    graph.replace(matmul_op, fc_op)
                    fc_node = graph.get_node_by_name(fc_op.name)
                    bias_buffer = graph.get_buffer(bias_name)
                    bias_buffer.consumers.add(fc_node)
                    fc_node.input_names.append(bias_name)
                    # update data_axis_formats of the fc_node
                    fc_node.op.data_axis_formats.append(bias_buffer.axis_format)
                else:
                    fc_op = op_adapter.FullyConnectedOp(name=fc_op_name)
                    graph.replace(matmul_op, fc_op)
                    fc_node = graph.get_node_by_name(fc_op.name)


                if idx == 0:
                    bias_buffer.consumers.remove(elementwise_sum_node)
                    elementwise_sum_node.input_names.remove(bias_name)

                    # update order from [input, weight, fully_connected, bias] to [input, weight, bias, fully_connected]
                    idx_bias = graph.nodes_in_order.index(constant_bias_node)
                    idx_fc = graph.nodes_in_order.index(fc_node)
                    if idx_bias > idx_fc:
                        graph.nodes_in_order[idx_fc] = constant_bias_node
                        graph.nodes_in_order[idx_bias] = fc_node

                    graph.squash(elementwise_sum_node, graph.get_output_buffers(fc_node)[0].name)

                    elementwise_sum_output_buffer = graph.get_output_buffers(elementwise_sum_node)[0]
                    # Refetch fc_node after squash
                    fc_node = graph.get_node_by_name(fc_op.name)
                    # Transferring activation encoding of elementwise sum to fullyconnected.
                    q = graph.user_quantization_overrides
                    if q and 'activation_encodings' in q and elementwise_sum_output_buffer.name in q['activation_encodings']:
                        activations = q['activation_encodings']
                        version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                        act_encs = [IROptimizations.extract_encoding_dict(graph.get_output_buffers(fc_node)[0].name,
                                                                          activations[elementwise_sum_output_buffer.name], version)]
                        graph.add_quantization_params(fc_node.op.name, output_encodings=act_encs)

                    # adds the quantization params to the graph if present
                    q = graph.user_quantization_overrides
                    if q and 'param_encodings' in q and bias_name in q['param_encodings']:
                        params = q['param_encodings']
                        version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                        param_encs = [IROptimizations.extract_encoding_dict('bias', params[bias_name], version)]
                        graph.add_quantization_params(fc_op_name, param_encodings=param_encs)

                    if matmul_input_buffer.rank() > 2:
                        fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                        old_buffer_name = fc_node_output_buffer.name
                        new_buffer_name = fc_node_output_buffer.name + '_fc'
                        graph.change_buffer_name(old_buffer_name, new_buffer_name)
                        # Update fc name in quant params
                        q = graph.quantization_params
                        if fc_op.name in q and len(q[fc_op.name]['output_encodings']) > 0:
                            q[fc_op.name]['output_encodings'][0]['name'] = new_buffer_name

                        post_reshape_name = fc_node.op.name + '_post_reshape'
                        post_reshape_op = op_adapter.ReshapeOp(post_reshape_name,
                                                            shape=elementwise_sum_output_buffer.shape[:])
                        graph.inject(post_reshape_op, new_buffer_name, old_buffer_name,
                                    axis_format=matmul_input_buffer.axis_format)
                else:
                    if matmul_input_buffer.rank() > 2:
                        fc_node_output_buffer = graph.get_output_buffers(fc_node)[0]
                        old_buffer_name = fc_node_output_buffer.name
                        new_buffer_name = fc_node_output_buffer.name + '_fc'
                        graph.change_buffer_name(old_buffer_name, new_buffer_name)
                        # Update fc name in quant params
                        q = graph.quantization_params
                        if fc_op.name in q and len(q[fc_op.name]['output_encodings']) > 0:
                            graph.quantization_params[fc_op.name]['output_encodings'][0]['name'] = new_buffer_name

                        post_reshape_name = fc_node.op.name + '_post_reshape'
                        if isinstance(graph.src_axis_order, SpatialLastAxisOrder):
                            post_reshape_shape = [*(matmul_input_buffer.shape[:-1]), fc_weights.shape[-1]]
                        else:
                            post_reshape_shape = [*(matmul_input_buffer.shape[:-1]), fc_weights.shape[-2]]
                        post_reshape_op = op_adapter.ReshapeOp(post_reshape_name,
                                                            shape=post_reshape_shape)
                        graph.inject(post_reshape_op, new_buffer_name, old_buffer_name,
                                    axis_format=matmul_input_buffer.axis_format)

                # Refetch the fc_node to update the output buffer shape to 2D
                fc_node = graph.get_node_by_name(fc_op.name)
                fc_out_buf = graph.get_output_buffers(fc_node)[0]
                # np.prod assigns default platform integer int64
                # which is not consistent to the default int. So the dtypes we get are int64 and int
                # Cast to int() makes it consistent (int, int)
                if isinstance(graph.src_axis_order, SpatialLastAxisOrder):
                    fc_out_buf.shape = [int(np.prod(matmul_input_buffer.shape[:-1])), fc_weights.shape[-1]]
                else:
                    fc_out_buf.shape = [int(np.prod(matmul_input_buffer.shape[:-1])), fc_weights.shape[-2]]

    @staticmethod
    def squash_batchnorm(graph):
        def validate_squash(nodes_tuple):
            matmul_node = nodes_tuple[0]
            # if FC has output_encodings squashing is disabled for better alignment
            # with the expected/simulated accuracy
            # Note: This is not necessarily better accuracy
            if graph.has_quantization_param(matmul_node.op.name) and \
                graph.quantization_params[matmul_node.op.name]["output_encodings"]:
                return False

            if len(nodes_tuple) == 1:
                bn_node = next(iter(graph.get_output_buffers(matmul_node)[0].consumers))
            if len(nodes_tuple) == 2:
                reshape_node = nodes_tuple[1]
                bn_node = next(iter(graph.get_output_buffers(reshape_node)[0].consumers))

            # check if the shapes of weights and biases of FC, BN are broadcastable
            bn_node_weights =  graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
            bn_node_bias =  graph.get_buffer(bn_node.input_names[2]).producer.op.tensor
            weights = graph.get_buffer(matmul_node.input_names[1]).producer.op.tensor
            if(len(matmul_node.input_names)>=3):
                bias = graph.get_buffer(matmul_node.input_names[2]).producer.op.tensor
            broadcasted_tensor = np.zeros(len(bn_node_weights), dtype=np.float32)

            if matmul_node.op.transpose_in1:
                weight_tensor = np.transpose(weights, (1, 0)).copy()
            else:
                weight_tensor = weights.copy()
            if len(matmul_node.input_names) >= 3 and \
                translation_utils.broadcastable(weight_tensor.shape, broadcasted_tensor.shape) and \
                translation_utils.broadcastable(bias.shape, bn_node_bias.shape):
                return True
            elif len(matmul_node.input_names) == 2 and \
                translation_utils.broadcastable(weight_tensor.shape, broadcasted_tensor.shape):
                return True
            else:
                return False

        sequence1 = [
            ("MatMul",
                ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
                ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
             )
        ]
        sequence2 = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1)]),
             ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
             )
        ]

        sequence3 = [
            ("MatMul",
             ("MATCH_BUFS_AT_INDEX", [("constant", 1), ("constant", 2)]),
               ("MATCH_NUM_BUFS", [("Reshape", "ANY")])
             ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("MatMul", "ANY")]),
               ("MATCH_NUM_BUFS", [("Batchnorm", "ALL")])
            )
        ]

        sequences = [sequence1, sequence2, sequence3]
        for idx, sequence in enumerate(sequences):

            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_squash)

            for node_tuple in matched_node_list:
                # sanity check
                log_assert(len(node_tuple) == len(sequence),
                        "ERROR: Pattern matching for squash batchnorm returned extra nodes. Got {} nodes, Expected {}.",
                        len(node_tuple), len(sequence))

                matmul_node = node_tuple[0]
                if(idx != 2):
                    bn_node = next(iter(graph.get_output_buffers(matmul_node)[0].consumers))
                else:
                    reshape_node = node_tuple[1]
                    bn_node = next(iter(graph.get_output_buffers(reshape_node)[0].consumers))
                matmul_node_output_buffer = graph.get_output_buffers(matmul_node)[0]
                bn_node_weights =  graph.get_buffer(bn_node.input_names[1]).producer.op.tensor
                bn_node_bias =  graph.get_buffer(bn_node.input_names[2]).producer.op.tensor
                bn_input_buffer = graph.get_input_buffers(bn_node)[0]
                bn_output_buffer = graph.get_output_buffers(bn_node)[0]
                manage_shared_static_input(graph, matmul_node, 1)
                weights = graph.get_buffer(matmul_node.input_names[1]).producer.op.tensor
                manage_shared_static_input(graph, matmul_node, 2)
                if(len(matmul_node.input_names) >= 3):
                    bias = graph.get_buffer(matmul_node.input_names[2]).producer.op.tensor
                broadcasted_tensor = np.zeros(len(bn_node_weights), dtype=np.float32)
                if matmul_node.op.transpose_in1:
                    weight_tensor = np.transpose(weights, (1, 0)).copy()
                else:
                    weight_tensor = weights.copy()
                broadcasted_tensor = broadcasted_tensor + weight_tensor
                broadcasted_tensor = broadcasted_tensor * bn_node_weights
                if matmul_node.op.transpose_in1:
                    broadcasted_transpose = np.transpose(broadcasted_tensor, (1, 0)).copy()
                else:
                    broadcasted_transpose = broadcasted_tensor.copy()

                graph.get_buffer(matmul_node.input_names[1]).producer.op.tensor = broadcasted_transpose
                if(len(matmul_node.input_names) >= 3):
                    graph.get_buffer(matmul_node.input_names[2]).producer.op.tensor = bias * bn_node_weights + bn_node_bias
                graph.squash(bn_node, input_name=bn_input_buffer.name)
                log_debug2(code_to_message.get_debugging_message("DEBUG_BATCHNORM_SQUASH")(bn_node.op.name,
                                                                                        matmul_node.op.type,
                                                                                        matmul_node.op.name))
                # Transferring activation encoding of BN to fullyconnected.
                q = graph.user_quantization_overrides
                if q and 'activation_encodings' in q and bn_output_buffer.name in q['activation_encodings']:
                    activations = q['activation_encodings']
                    version = "1.0.0" if 'version' in q and q['version'] == "1.0.0" else "0.0.6"
                    act_encs = [IROptimizations.extract_encoding_dict(matmul_node_output_buffer.name, activations[bn_output_buffer.name], version)]
                    graph.add_quantization_params(matmul_node.op.name, output_encodings=act_encs)


@register_layer_optimization
class OptimizeMergedWeightsGruTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MergedWeightsGruOp.TRANSLATION_KEY
        self.register_method(UNROLL_GRU_TIME_STEPS, self.unroll_gru_time_steps)
        self.register_method(MULTI_TIME_STEPS_GRU, self.multi_time_steps_gru)

    def axes_to_spatial_first_order(self, node, graph):
        # GRU input axis format must be NTF
        input_name = node.input_names[0]
        input_bufs = graph.get_input_buffers(node)
        output_bufs = graph.get_output_buffers(node)

        # Input Data Buffer can be in NTF/TNF format.
        in_buf = input_bufs[0]
        # If MergedWeightsGru Op's input (X_t with TNF axis format) is also the model's input then the
        # input would have been converterd to NTF format after call to axes_spatial_first_order
        # in the OptimizeInputTranslation. Then time_major param should be False.
        if in_buf.axis_format == AxisTracker.AxisFormat.NTF:
            node.op.time_major = False
        elif in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            graph.inject_implicit_permute(in_buf.name, AxisTracker.AxisFormat.NTF,
                                          AxisTracker.AxisFormat.TNF_TO_NTF)
            # Make sure that the time major param is False if the buffer is in NTF format.
            node.op.time_major = False

        # Check that h input buffer is NONTRIVIAL
        for data_axis_format, in_buf in zip(node.op.data_axis_formats[1:], input_bufs[1:]):
            # We would like to revert the axis format of h input buffer to NONTRIVIAL if it isn't.
            if in_buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL:
                if in_buf.axis_format != data_axis_format and \
                        in_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
                    # Transpose the axis format to source's one
                    graph.inject_implicit_permute(
                        in_buf.name,
                        AxisTracker.AxisFormat.NONTRIVIAL,
                        spatial_first_format_to_channel_first_permute_order[in_buf.axis_format],
                        [node.op.name]
                    )
                in_buf = graph.get_buffer(list(in_buf.consumers)[0].output_names[0])
            log_assert(in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL,
                       "GRU h input buffer {} needs to have format NONTRIVIAL, got {}",
                       in_buf,
                       in_buf.axis_format)

        # Set up MergedWeightsGru outputs' axis formats
        # First output: NTF/TNF
        # Other outputs: NONTRIVIAL
        time_major_param = node.op.time_major
        for i, output_buf in enumerate(output_bufs):
            if i == 0:
                if time_major_param:
                    output_buf.axis_format = AxisTracker.AxisFormat.TNF
                    log_assert(input_bufs[0].axis_format == AxisTracker.AxisFormat.TNF,
                               "MergedWeightsGru X_t input buffer {} needs to have format TNF, got {}", input_bufs[0],
                               input_bufs[0].axis_format)
                else:
                    output_buf.axis_format = AxisTracker.AxisFormat.NTF
                    output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

    def align_to_source_output_names(self, graph, current_output_names, source_output_names, align_only_in_src_output=True):
        # If the current_output names are in graph.output_names, then we need to align
        if align_only_in_src_output:
            new_current, new_src = [], []
            for idx, name in enumerate(source_output_names):
                if name in graph.output_names:
                    new_current.append(current_output_names[idx])
                    new_src.append(name)
            current_output_names, source_output_names = new_current, new_src

        # Replace current name with source name for alignment
        for current_name, source_name in zip(current_output_names, source_output_names):
            # override encoding info update
            pre_node = graph.get_producer_node(current_name)
            if graph.has_quantization_param(pre_node.op.name):
                src_encodings = graph.quantization_params[pre_node.op.name]['output_encodings']
                for i in range(len(src_encodings)):
                    if (src_encodings[i]["name"] == current_name):
                        src_encodings[i]["name"] = source_name
                        break

            buf = graph.get_buffer(current_name)
            if source_name in graph.buffers:
                raise ValueError("Buffer {} already exists in graph, duplicate buffer name when replacing buffer {} with it".format(
                        source_name, current_name))

            # Update consumers input name
            for consumer in list(buf.consumers):
                # The consumer may have the same buffer as input twice
                consumer.input_names = [source_name if name == current_name else name for name in consumer.input_names]

            # Update producer output name
            producer_node = graph.get_producer_node(current_name)
            idx = producer_node.output_names.index(current_name)
            producer_node.output_names[idx] = source_name

            # Update buffer in graph
            buf.name = source_name
            graph.buffers[source_name] = graph.buffers.pop(current_name)

    def get_dims(self, buffer_shape, time_major_param=False):
        if time_major_param:
            return [buffer_shape[1], buffer_shape[0], buffer_shape[2]]
        else:
            return buffer_shape

    def unroll_gru_time_steps(self, graph):
        sequence = [
            (op_adapter.MergedWeightsGruOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            merged_weights_gru_node = nodes_tuple[0]
            merged_weights_gru_node_name = merged_weights_gru_node.op.name
            time_major_param = merged_weights_gru_node.op.time_major

            if len(merged_weights_gru_node.input_names) > gru_props.IR_RESET_IDX:
                # Current gru comes from stateful gru which is not allowed to do unroll function
                # Just ignore the 'reset' input
                log_warning("Gru Op {} with {} inputs is not allowed to call 'unroll_gru_time_steps'. "
                            "Just ignore the 'reset' input".format(merged_weights_gru_node_name,
                                                          len(merged_weights_gru_node.input_names)))

            DATA_IDX, HIDDEN_OUT_IDX = 0, 1
            shared_inputs_list = self.preprocess_merged_weights_gru_node(graph, merged_weights_gru_node)
            merged_weights_gru_node_input_name = merged_weights_gru_node.input_names[DATA_IDX]
            merged_weights_gru_node_output_name = merged_weights_gru_node.output_names[DATA_IDX]
            merged_weights_gru_node_idx = graph.nodes_in_order.index(merged_weights_gru_node)

            log_debug("Unrolling MergedWeightsGru node {}".format(merged_weights_gru_node_name))

            # Extract and validate inputs, outputs, and sizes
            number_of_outputs = len(merged_weights_gru_node.output_names)
            all_output_buffer = graph.get_buffer(merged_weights_gru_node_output_name)
            input_buffer_shape = self.get_dims(graph.get_buffer(merged_weights_gru_node_input_name).shape, time_major_param)
            batch_size, seq_length, input_size = input_buffer_shape[:]

            if number_of_outputs == 1:
                # Add dummy buffers for missing outputs
                output_size = graph.get_buffer(merged_weights_gru_node_output_name).shape[-1]
                num_units = merged_weights_gru_node.op.hidden_size
                hidden_output_dummy_name = merged_weights_gru_node_name + "_hidden_output_dummy"
                graph.add_output_buffer(merged_weights_gru_node, hidden_output_dummy_name,
                                        [batch_size, output_size], AxisTracker.AxisFormat.NONTRIVIAL)

            hidden_output_buffer = graph.get_buffer(merged_weights_gru_node.output_names[HIDDEN_OUT_IDX])

            time_step_axis = 0 if time_major_param else 1
            input_x_split_name_list = []
            if seq_length == 1:
                input_x_split_name_list.append(merged_weights_gru_node.input_names[DATA_IDX])
            else:
                for i in range(seq_length):
                    input_x_i_name = merged_weights_gru_node_name + "_" + merged_weights_gru_node_input_name + str(i)
                    input_x_split_name_list.append(input_x_i_name)
                    input_x_split_name = merged_weights_gru_node_name + "_" + merged_weights_gru_node_input_name + "_split"
                    input_x_split_op = op_adapter.SplitOp(name=input_x_split_name, axis=time_step_axis)

                # Split input to T inputs
                input_x_split_op_node = graph.add(input_x_split_op, input_names=[merged_weights_gru_node.input_names[0]],
                          output_names=input_x_split_name_list, idx=graph.nodes_in_order.index(merged_weights_gru_node))
                # Update trace info for new created node
                graph.update_trace_info(input_x_split_op_node, [merged_weights_gru_node])
                # set override encodings for split op
                pre_node = graph.get_producer_node(merged_weights_gru_node.input_names[DATA_IDX])
                for output_name in input_x_split_name_list:
                    graph.copy_quantization_param(pre_node.op.name,
                                                  input_x_split_op_node.op.name,
                                                  merged_weights_gru_node.input_names[DATA_IDX],
                                                  output_name)

                if merged_weights_gru_node.op.direction == ir_graph.QNN_OP_GRU_DIRECTION_REVERSE:
                    input_x_split_name_list.reverse()

            output_y_concat_name_list = []
            output_h_name_list = []
            for i in range(seq_length):
                output_y_i_name = merged_weights_gru_node_output_name + str(i)
                output_y_concat_name_list.append(output_y_i_name)
                output_h_i_name = merged_weights_gru_node.output_names[HIDDEN_OUT_IDX] + str(i)
                output_h_name_list.append(output_h_i_name)

            for i in range(seq_length):
                if i == 0:
                    _h_0_input_name = merged_weights_gru_node.input_names[HIDDEN_OUT_IDX]
                else:
                    _h_0_input_name = output_h_name_list[i-1]

                gru_time_step_i_op_name = merged_weights_gru_node_name + '_' + str(i)
                gru_time_step_i_op = op_adapter.GruOp(name=gru_time_step_i_op_name,
                                                      activation=merged_weights_gru_node.op.activation,
                                                      gate_activation=merged_weights_gru_node.op.gate_activation,
                                                      rec_gate_activation=merged_weights_gru_node.op.rec_gate_activation,
                                                      h_0_input_name=_h_0_input_name,
                                                      direction=merged_weights_gru_node.op.direction,
                                                      hidden_size=merged_weights_gru_node.op.hidden_size,
                                                      linear_before_reset=merged_weights_gru_node.op.linear_before_reset,
                                                      time_major=merged_weights_gru_node.op.time_major)

                gru_op_input_name_list = [input_x_split_name_list[i], _h_0_input_name]
                gru_op_input_name_list = gru_op_input_name_list[:1] + shared_inputs_list + gru_op_input_name_list[1:]
                gru_op_output_name_list = [output_y_concat_name_list[i], output_h_name_list[i]]
                gru_time_step_i_op_node = graph.add(gru_time_step_i_op, input_names=gru_op_input_name_list, output_names=gru_op_output_name_list, idx=graph.nodes_in_order.index(merged_weights_gru_node))
                # Update trace info for new created node
                graph.update_trace_info(gru_time_step_i_op_node, [merged_weights_gru_node])
                # Set output override encodings for single layer gru op
                graph.copy_quantization_param(merged_weights_gru_node.op.name,
                                              gru_time_step_i_op.name,
                                              merged_weights_gru_node_output_name,
                                              output_y_concat_name_list[i])
                if i == seq_length-1:
                    # Set hidden out override encodings for last layer gru op
                    graph.copy_quantization_param(merged_weights_gru_node.op.name,
                                                  gru_time_step_i_op.name,
                                                  merged_weights_gru_node.output_names[HIDDEN_OUT_IDX],
                                                  output_h_name_list[i])

                if merged_weights_gru_node.op.time_major:
                    graph.get_buffer(gru_op_input_name_list[0]).axis_format = AxisTracker.AxisFormat.TNF
                else:
                    graph.get_buffer(gru_op_input_name_list[0]).axis_format = AxisTracker.AxisFormat.NTF

            output_y_concat_name = merged_weights_gru_node_output_name + "_concat" if seq_length > 1 else \
                output_y_concat_name_list[0]

            if seq_length > 1:
                output_y_concat_op_name = merged_weights_gru_node_name + "_" + merged_weights_gru_node_output_name + "_concat"
                output_y_concat_op = op_adapter.ConcatOp(name=output_y_concat_op_name, axis=time_step_axis)

                # Concat output from T outputs
                output_y_concat_op_node = graph.add(output_y_concat_op, input_names=output_y_concat_name_list, output_names=output_y_concat_name, idx=graph.nodes_in_order.index(merged_weights_gru_node))
                # Update trace info for new created node
                graph.update_trace_info(output_y_concat_op_node, [merged_weights_gru_node])
                # set override encodings for concat op
                graph.copy_quantization_param(merged_weights_gru_node.op.name,
                                            output_y_concat_op_node.op.name,
                                            merged_weights_gru_node_output_name,
                                            output_y_concat_name)

            # Adjust merged_weights_gru output consumers
            # Replace all the consumers' input names coming from merged_weights_gru node
            # with the new corresponding output buffer names
            # Delete the consumers for merged_weights_gru node's output buffers
            for consumer in list(all_output_buffer.consumers).copy():
                output_y_concat_buffer = graph.get_buffer(output_y_concat_name)
                output_y_concat_buffer.consumers.add(consumer)
                h_all_idx = consumer.input_names.index(merged_weights_gru_node.output_names[DATA_IDX])
                consumer.input_names[h_all_idx] = output_y_concat_name
                all_output_buffer.consumers.remove(consumer)

            for consumer in list(hidden_output_buffer.consumers).copy():
                output_h_buffer = graph.get_buffer(output_h_name_list[seq_length-1])
                output_h_buffer.consumers.add(consumer)
                h_idx = consumer.input_names.index(merged_weights_gru_node.output_names[HIDDEN_OUT_IDX])
                consumer.input_names[h_idx] = output_h_name_list[seq_length-1]
                hidden_output_buffer.consumers.remove(consumer)

            # prune original merged_weights_gru_node
            source_output_names = merged_weights_gru_node.output_names
            graph.prune(merged_weights_gru_node, force_remove=True)

            # At this point, current output names are not aligned to source output names,
            # we need to restore the source output names from Gru node.
            current_output_names = [output_y_concat_name, output_h_name_list[seq_length-1]]
            self.align_to_source_output_names(graph, current_output_names, source_output_names)

    def multi_time_steps_gru(self, graph):
        DATA_IDX, HIDDEN_IN_IDX = 0, 1
        HIDDEN_ALL_OUT_IDX, HIDDEN_OUT_IDX = 0, 1
        sequence = [
            (op_adapter.MergedWeightsGruOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            merged_weights_gru_node = nodes_tuple[0]
            merged_weights_gru_node_name = merged_weights_gru_node.op.name
            log_debug("Converting MergedWeightsGru node {} to Multi-time step Gru node".format(merged_weights_gru_node_name))

            # Before preprocessing merged weights, get the reset input if it
            # exists. TODO: Move this to preprocess_merged_weights_gru_node
            # once reset input has been tested on unrolled GRU on HTP
            reset_input_name = ''
            if len(merged_weights_gru_node.input_names) > gru_props.IR_RESET_IDX:
                reset_input_name = merged_weights_gru_node.input_names[gru_props.IR_RESET_IDX]

            shared_inputs_name_list = self.preprocess_merged_weights_gru_node(graph, merged_weights_gru_node)
            gru_multi_time_step_op_name = merged_weights_gru_node_name + '_multi_seq'
            output_name_list = []
            for name in merged_weights_gru_node.output_names:
                output_name_list.append(name + "_multi_seq")

            gru_multi_time_step_op = op_adapter.GruOp(name=gru_multi_time_step_op_name,
                                                      activation=merged_weights_gru_node.op.activation,
                                                      gate_activation=merged_weights_gru_node.op.gate_activation,
                                                      rec_gate_activation=merged_weights_gru_node.op.rec_gate_activation,
                                                      h_0_input_name=merged_weights_gru_node.op.h_0_input_name,
                                                      direction=merged_weights_gru_node.op.direction,
                                                      hidden_size=merged_weights_gru_node.op.hidden_size,
                                                      linear_before_reset=merged_weights_gru_node.op.linear_before_reset,
                                                      time_major=merged_weights_gru_node.op.time_major)

            gru_all_inputs_name_list = [merged_weights_gru_node.input_names[DATA_IDX], merged_weights_gru_node.input_names[HIDDEN_IN_IDX]]
            gru_all_inputs_name_list = gru_all_inputs_name_list[:1] + shared_inputs_name_list + gru_all_inputs_name_list[1:]
            if reset_input_name != '':
                gru_all_inputs_name_list.append(reset_input_name)

            gru_multi_time_step_op_node = graph.add(gru_multi_time_step_op,
                      input_names=gru_all_inputs_name_list,
                      output_names=output_name_list,
                      idx=graph.nodes_in_order.index(merged_weights_gru_node))
            # Add trace info for new created node
            graph.update_trace_info(gru_multi_time_step_op_node, [merged_weights_gru_node])
            # add override encodings to multi_time_step gru op
            for idx, name in enumerate(merged_weights_gru_node.output_names):
                graph.copy_quantization_param(merged_weights_gru_node.op.name, gru_multi_time_step_op.name, name, output_name_list[idx])

            # Adjust merged_weights_gru output consumers
            # Replace all the consumers' input names coming from merged_weights_gru node
            # with the new corresponding output buffer names
            # Delete the consumers for merged_weights_gru node's output buffers
            all_output_buffer = graph.get_buffer(merged_weights_gru_node.output_names[HIDDEN_ALL_OUT_IDX])
            hidden_output_buffer = graph.get_buffer(merged_weights_gru_node.output_names[HIDDEN_OUT_IDX])
            for consumer in list(all_output_buffer.consumers).copy():
                output_y_all_hidden_buffer = graph.get_buffer(output_name_list[0])
                output_y_all_hidden_buffer.consumers.add(consumer)
                h_all_idx = consumer.input_names.index(merged_weights_gru_node.output_names[DATA_IDX])
                consumer.input_names[h_all_idx] = output_name_list[0]
                all_output_buffer.consumers.remove(consumer)

            for consumer in list(hidden_output_buffer.consumers).copy():
                output_h_buffer = graph.get_buffer(output_name_list[1])
                output_h_buffer.consumers.add(consumer)
                h_idx = consumer.input_names.index(merged_weights_gru_node.output_names[HIDDEN_OUT_IDX])
                consumer.input_names[h_idx] = output_name_list[1]
                hidden_output_buffer.consumers.remove(consumer)

            # input buf axis format gets modified during graph construction by MergedWeightsGruOp populate_axis_format()
            # Make sure that the first input of the MergedWeightsGru Op has correct axis format.
            if merged_weights_gru_node.op.time_major:
                graph.get_buffer(gru_all_inputs_name_list[0]).axis_format = AxisTracker.AxisFormat.TNF
            else:
                graph.get_buffer(gru_all_inputs_name_list[0]).axis_format = AxisTracker.AxisFormat.NTF
            # Prune original MergedWeightsGru Op
            source_output_names = merged_weights_gru_node.output_names
            graph.prune(merged_weights_gru_node, force_remove=True)

            # At this point, current output names are not aligned to source output names,
            # we need to restore the source output names from Gru node.
            current_output_names = output_name_list
            self.align_to_source_output_names(graph, current_output_names, source_output_names)

    def preprocess_merged_weights_gru_node(self, graph, merged_weights_gru_node):
        # Index for MergedWeightsGruOp inputs
        INPUT_WEIGHTS_IDX, REC_WEIGHTS_IDX, GATE_BIASES_IDX = 2, 3, 4

        def split_gru_tensor_per_gate(input_name, split_axis=0):
            producer_node = graph.get_producer_node(input_name)
            if producer_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                param_tensor = producer_node.op.tensor
                # Split weights so that they can be indexed by gate
                split_sections = int(param_tensor.shape[split_axis] / merged_weights_gru_node.op.hidden_size)
                param_split_tensor = np.split(param_tensor, indices_or_sections=split_sections, axis=split_axis)
                # Two different MergedWeightsGruOps may share the same weights or biases, so we need to extract the
                # weights by input name before we prune the const node from graph
                param_buf_consumers = graph.get_buffer(input_name).consumers
                param_buf_consumers.remove(merged_weights_gru_node)
                if not param_buf_consumers:
                    # Prune the unsplit weights node from the graph
                    graph.prune(producer_node, force_remove=True)
                return param_split_tensor
            else:
                raise ValueError("GruOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                        producer_node.op.name))

        def add_split_tensor_to_graph(tensor_name, tensor, desired_shape=None):
            merged_weights_gru_node_idx = graph.nodes_in_order.index(merged_weights_gru_node)
            # Share the tensor if they are already added in the graph
            if not graph.has_buffer(tensor_name):
                tensor = np.resize(tensor, desired_shape) if desired_shape else tensor
                const_op = op_adapter.ConstantOp(name=tensor_name, tensor=tensor)
                const_op_node = graph.add(const_op, input_names=[], output_names=[tensor_name], idx=merged_weights_gru_node_idx)
                # Add trace info for new created node
                graph.update_trace_info(const_op_node, [merged_weights_gru_node])
            elif graph.get_producer_op(tensor_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("GruOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                        graph.get_producer_op(tensor_name).name))

        # Must add all inputs derived from splitting the tensor per gate as ConstantOp to the graph.
        # Weights may already be 2D, but it is cleaner to resize anyway rather than check shape for each input.
        # The weights and biases are shared across unrolled gru nodes.
        def prepare_gru_all_inputs():
            input_size = graph.get_buffer(merged_weights_gru_node.input_names[0]).shape[-1]
            num_units = merged_weights_gru_node.op.hidden_size

            # Input weights are expected in [3*hidden_size, input_size]
            src_input_weights_name = merged_weights_gru_node.input_names[INPUT_WEIGHTS_IDX]
            input_split_weights = split_gru_tensor_per_gate(src_input_weights_name)

            input_w_to_update_gate_name = src_input_weights_name + '_input_w_to_update_gate'
            add_split_tensor_to_graph(input_w_to_update_gate_name, input_split_weights[0], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_update_gate_name, src_input_weights_name, input_w_to_update_gate_name)

            input_w_to_reset_gate_name = src_input_weights_name + '_input_w_to_reset_gate'
            add_split_tensor_to_graph(input_w_to_reset_gate_name, input_split_weights[1], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_reset_gate_name, src_input_weights_name, input_w_to_reset_gate_name)

            input_w_to_new_gate_name = src_input_weights_name + '_input_w_to_new_gate'
            add_split_tensor_to_graph(input_w_to_new_gate_name, input_split_weights[2], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_new_gate_name, src_input_weights_name, input_w_to_new_gate_name)

            # Recurrence weights are expected in [3*hidden_size, hidden_size]
            src_rec_weights_name = merged_weights_gru_node.input_names[REC_WEIGHTS_IDX]
            rec_split_weights = split_gru_tensor_per_gate(src_rec_weights_name)

            recurrent_w_to_update_gate_name = src_rec_weights_name + '_recurrent_w_to_update_gate'
            add_split_tensor_to_graph(recurrent_w_to_update_gate_name, rec_split_weights[0], desired_shape=(num_units, num_units))
            graph.copy_quantization_param(src_rec_weights_name, recurrent_w_to_update_gate_name, src_rec_weights_name, recurrent_w_to_update_gate_name)

            recurrent_w_to_reset_gate_name = src_rec_weights_name + '_recurrent_w_to_reset_gate'
            add_split_tensor_to_graph(recurrent_w_to_reset_gate_name, rec_split_weights[1], desired_shape=(num_units, num_units))
            graph.copy_quantization_param(src_rec_weights_name, recurrent_w_to_reset_gate_name, src_rec_weights_name, recurrent_w_to_reset_gate_name)

            recurrent_w_to_new_gate_name = src_rec_weights_name + '_recurrent_w_to_new_gate'
            add_split_tensor_to_graph(recurrent_w_to_new_gate_name, rec_split_weights[2], desired_shape=(num_units, num_units))
            graph.copy_quantization_param(src_rec_weights_name, recurrent_w_to_new_gate_name, src_rec_weights_name, recurrent_w_to_new_gate_name)

            # Gate biases are expected in [6*hidden_size]
            # Input Gate biases are expected in [3*hidden_size]
            src_gate_biases_name = merged_weights_gru_node.input_names[GATE_BIASES_IDX]
            gate_split_biases = split_gru_tensor_per_gate(src_gate_biases_name)

            input_b_to_update_gate_name = src_gate_biases_name + '_input_b_to_update_gate'
            add_split_tensor_to_graph(input_b_to_update_gate_name, gate_split_biases[0], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, input_b_to_update_gate_name, src_gate_biases_name, input_b_to_update_gate_name)

            input_b_to_reset_gate_name = src_gate_biases_name + '_input_b_to_reset_gate'
            add_split_tensor_to_graph(input_b_to_reset_gate_name, gate_split_biases[1], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, input_b_to_reset_gate_name, src_gate_biases_name, input_b_to_reset_gate_name)

            input_b_to_new_gate_name = src_gate_biases_name + '_input_b_to_new_gate'
            add_split_tensor_to_graph(input_b_to_new_gate_name, gate_split_biases[2], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, input_b_to_new_gate_name, src_gate_biases_name, input_b_to_new_gate_name)

            # Recurrence Gate biases are expected in [3*hidden_size]
            recurrent_b_to_update_gate_name = src_gate_biases_name + '_recurrent_b_to_update_gate'
            add_split_tensor_to_graph(recurrent_b_to_update_gate_name, gate_split_biases[3], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, recurrent_b_to_update_gate_name, src_gate_biases_name, recurrent_b_to_update_gate_name)

            recurrent_b_to_reset_gate_name = src_gate_biases_name + '_recurrent_b_to_reset_gate'
            add_split_tensor_to_graph(recurrent_b_to_reset_gate_name, gate_split_biases[4], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, recurrent_b_to_reset_gate_name, src_gate_biases_name, recurrent_b_to_reset_gate_name)

            recurrent_b_to_new_gate_name = src_gate_biases_name + '_recurrent_b_to_new_gate'
            add_split_tensor_to_graph(recurrent_b_to_new_gate_name, gate_split_biases[5], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, recurrent_b_to_new_gate_name, src_gate_biases_name, recurrent_b_to_new_gate_name)

            # Prepare the GruOp input names - inputs not captured by any FE are passed the empty string
            gru_all_inputs_name_list = [
                input_w_to_update_gate_name,
                input_w_to_reset_gate_name,
                input_w_to_new_gate_name,
                recurrent_w_to_update_gate_name,
                recurrent_w_to_reset_gate_name,
                recurrent_w_to_new_gate_name,
                input_b_to_update_gate_name,
                input_b_to_reset_gate_name,
                input_b_to_new_gate_name,
                recurrent_b_to_update_gate_name,
                recurrent_b_to_reset_gate_name,
                recurrent_b_to_new_gate_name
            ]

            # Update the MergedWeightsGruOp input names
            merged_weights_gru_node.input_names = merged_weights_gru_node.input_names[:INPUT_WEIGHTS_IDX]
            return gru_all_inputs_name_list

        def ensure_h_c_inputs_present():
            merged_weights_gru_node_name = merged_weights_gru_node.op.name
            merged_weights_gru_node_idx = graph.nodes_in_order.index(merged_weights_gru_node)
            time_major_param = merged_weights_gru_node.op.time_major
            input_buffer_shape = self.get_dims(graph.get_buffer(merged_weights_gru_node.input_names[0]).shape, time_major_param)
            batch_size = input_buffer_shape[0]
            num_units = merged_weights_gru_node.op.hidden_size

            # Requires initial_h input to be present
            # The following code adds zero valued tensor provided the conditions below are satisfied
            if not merged_weights_gru_node.input_names[1]:
                if merged_weights_gru_node.op.h_0_input_name:
                    raise ValueError('MergedWeightsGru node {} op attribute h_0_input_name {} mismatch with merged_weights_gru_node.input_names[1] {}.'.format(
                            merged_weights_gru_node_name, merged_weights_gru_node.op.h_0_input_name, merged_weights_gru_node.input_names[1]))

                # add zeros for initial h which is needed for QNN
                initial_hidden_state_name = merged_weights_gru_node_name + '_initial_hidden_state'
                initial_hidden_state_tensor = np.zeros((1, batch_size, num_units), dtype=np.float32)
                initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor)
                initial_hidden_state_op_node = graph.add(initial_hidden_state_op, input_names=[], output_names=[initial_hidden_state_name], idx=merged_weights_gru_node_idx)
                # Add trace info for new created node
                graph.update_trace_info(initial_hidden_state_op_node, [merged_weights_gru_node])

                merged_weights_gru_node.input_names[1] = initial_hidden_state_name
                merged_weights_gru_node.op.h_0_input_name = initial_hidden_state_name
                graph.get_buffer(initial_hidden_state_name).consumers.add(merged_weights_gru_node)

        log_debug("Preprocessing MergedWeightsGru node {} for QNN lowering.".format(merged_weights_gru_node.op.name))

        # Prepare QNN Gru all inputs and return the input name list
        gru_all_inputs_name_list = prepare_gru_all_inputs()
        ensure_h_c_inputs_present()

        return gru_all_inputs_name_list


@register_layer_optimization
class OptimizeElementwiseNeuronTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ElementwiseNeuronOp.TRANSLATION_KEY
        self.register_method(MATCH_GELU, self.match_gelu)
        self.register_method(MATCH_GELU_APPROX, self.match_gelu)
        self.register_method(MATCH_HARDSWISH, self.match_hardswish)
        self.register_method(EXPAND_SPARSE_OP_STRUCTURE, self.expand_sparse_op_structure)
        self.register_method(SQUASH_CONSTANT_INPUT, self.squash_constant_input)

    @staticmethod
    def squash_constant_input(node, graph):
        """
        Apply constant-folding on ElementWiseNeuron Op
        """
        input_buff = graph.get_buffer(node.input_names[0])
        if not input_buff.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            # cannot perform constant-folding if input is not constant node
            return

        def gelu_inference(tensor: np.ndarray) -> np.ndarray:
            return 0.5 * tensor * (1 + np.vectorize(math.erf)(tensor/np.sqrt(2)))

        # TODO: inferecne function for different ElementWiseNeuron Ops
        inference_map = {
            ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU: gelu_inference,
        }
        if node.op.operation in inference_map :
            # compute Neuron output from constant input
            folded_tensor = inference_map[node.op.operation](input_buff.producer.op.tensor)

            if len(input_buff.consumers) == 1:
                # Squash current op to input op.
                # Setting option `is_data_movement_node=False` trigger copying the quantization encodings of the squashed Neuron Op into the ConstantOp.
                input_buff.producer.op.tensor = folded_tensor
                graph.squash(node, input_name=input_buff.name, is_data_movement_node=False)
            else:
                # Since constant input node is used elsewhere
                # , we create a new constant node to replace Neuron Op.
                new_const = op_adapter.ConstantOp(name=node.op.name, tensor=folded_tensor)
                input_buff.consumers.remove(node)
                node.input_names.clear()
                graph.replace(node.op, new_const)

    @staticmethod
    def match_gelu(graph, is_approx=False):
        def replace_gelu(graph, matched_node_list):
            for node_tuple in matched_node_list:
                last_node = node_tuple[-1]
                # Squash all nodes except the last node in forward order and the last op will be replaced
                for node in node_tuple[:-1]:
                    input_names = node.input_names[:]
                    # pick squashable input based on whether it produced by constantOp
                    input_name = [name for name in input_names if (
                                not isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))][0]
                    input_names.remove(input_name)
                    for input_name_ in input_names:
                        # disconnect rest of inputs from node
                        # skip if current input_name_ equal to the input_name to squash
                        if input_name_ == input_name:
                            continue
                        input_buf_ = graph.get_buffer(input_name_)
                        input_buf_.consumers.remove(node)
                        node.input_names.remove(input_name_)
                    graph.squash(node, input_name=input_name, squash_into_next=True)

                # For the last_node, four different sequences correspond to two different processes:
                # Sequence1:
                # the inputs of last elementwise_product op will be [original input, constant(0.5)].
                # so we need to disconnect the constant input of the last op
                # Sequence2, sequence3, sequence6:
                # the last elementwise_product op will receive two duplicated input buffer
                # after squashing previous nodes, so we need pop one of them.
                const_input_bufs = [graph.get_buffer(name) for name in last_node.input_names if
                                graph.get_producer_op(name).type == op_adapter.ConstantOp.TRANSLATION_KEY]
                if len(const_input_bufs):
                    const_input_bufs[0].consumers.remove(last_node)
                    last_node.input_names.remove(const_input_bufs[0].name)
                else:
                    last_node.input_names.pop()

                # replace the first op with gelu
                last_node_op = last_node.op
                gelu_op_name =  graph.naming_policy.get_op_name_by_type(op_adapter.ElementwiseNeuronOp.TRANSLATION_KEY,
                                                                        op_adapter.ElementwiseNeuronOp.LEGACY_TRANSLATION_KEY,
                                                                        folded_op=True)
                gelu_op = op_adapter.ElementwiseNeuronOp(gelu_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU)
                graph.replace(last_node_op, gelu_op)

        def check_const_input_value(node, value):
            par_nodes = graph.get_parent_nodes(node)
            const_nodes = [node for node in par_nodes if
                           node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY]
            if len(const_nodes) != 1:
                return False
            const_input_value = const_nodes[0].op.tensor
            if len(const_input_value.flatten()) == 1:
                const_input_value = const_input_value.flatten()
            if not isinstance(value, np.ndarray):
                value = np.array([value])
            if not translation_utils.compare_values(const_input_value, value, rtol=1.e-3, atol=1.e-5):
                return False
            else:
                return True

        def is_valid_approx_gelu(node_tuple):
            pow_idx = 0
            tanh_node_idx = -4
            mul_or_div_node_idx = -2 # Mul with constant value "0.5"
            add_idx = -3  # Add with constant value "1"
            mul_1_idx = -5 # Mul with constant value "(2/pi)^0.5"
            mul_2_idx = 1 # Mul with constant value "0.044715"

            # For sequence7 mul comes first topologically
            if (node_tuple[0].op.type == "elementwise_product"):
                pow_idx = 1
                tanh_node_idx = -3
                mul_or_div_node_idx = 0 # Mul with constant value "0.5"
                add_idx = -2  # Add with constant value "1"
                mul_1_idx = -4 # Mul with constant value "(2/pi)^0.5"
                mul_2_idx = 2 # Mul with constant value "0.044715"



            tanh_node = node_tuple[tanh_node_idx]
            if tanh_node.op.operation != ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH:
                return False

            mul_or_div_node = node_tuple[mul_or_div_node_idx]
            div_status = check_const_input_value(mul_or_div_node, 2)
            mul_status = check_const_input_value(mul_or_div_node, 0.5)
            if (not mul_status) and (not div_status):
                return False

            add_node = node_tuple[add_idx]
            add_status = check_const_input_value(add_node, 1)
            if not add_status:
                return False

            mul_node = node_tuple[mul_1_idx]
            mul_status = check_const_input_value(mul_node, np.sqrt(2/np.pi))
            if not mul_status:
                return False

            mul_node = node_tuple[mul_2_idx]
            mul_status = check_const_input_value(mul_node, 0.044715)
            if not mul_status:
                return False

            pow_node = node_tuple[pow_idx]
            pow_status = check_const_input_value(pow_node, 3)
            if not pow_status:
                return False

            return True

        sequence1 = [
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Erf", "ALL")])
             ),
            ("Erf",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]
        sequence2 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Erf", "ALL")])
             ),
            ("Erf",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ()
             )
        ]
        sequence3 = [
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("Erf", "ALL")])
             ),
            ("Erf",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum", "ANY")]),
             ()
             )
        ]
        sequence4 = [
            ("elementwise_power",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_power", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("ElementWiseNeuron", "ALL")])
            ),
            ("ElementWiseNeuron",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("MATCH_BUFS_AT_INDEX", [("ElementWiseNeuron", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
             ()
            )
        ]
        sequence5 = [
            ("elementwise_power",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_power", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("ElementWiseNeuron", "ALL")])
            ),
            ("ElementWiseNeuron",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("MATCH_BUFS_AT_INDEX", [("ElementWiseNeuron", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
            ),
            ("elementwise_div",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_div", "ANY")]),
             ()
            )
        ]
        sequence6 = [
            ("elementwise_div",
                ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                ("MATCH_NUM_BUFS", [("Erf", "ALL")])
                ),
            ("Erf",
                ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
                ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
                ),
            ("elementwise_product",
                ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
                ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
                ),
            ("elementwise_sum",
                ("FLEXIBLE_NUM_BUFS", [("Erf", "ANY")]),
                ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
                ),
            ("elementwise_product",
                ("FLEXIBLE_NUM_BUFS", [("elementwise_sum", "ANY"), ("elementwise_product", "ANY")]),
                (),
                )
        ]

        sequence7 = [
            # This is the mul before the last mul in the sequence
            # Topologically it comes before the pow node, hence
            # added it to the top
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_power",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_power", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_sum", "ANY")]),
             ("MATCH_NUM_BUFS", [("ElementWiseNeuron", "ALL")])
            ),
            ("ElementWiseNeuron",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
            ),
            ("elementwise_sum",
             ("MATCH_BUFS_AT_INDEX", [("ElementWiseNeuron", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
            ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("elementwise_product", "ANY"), ("elementwise_sum", "ANY")]),
             ()
            )
        ]

        if is_approx:
            sequences_approx_gelu = [sequence4, sequence5, sequence7]
            for sequence in sequences_approx_gelu:
                matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_approx_gelu, ignore_constants=True)
                replace_gelu(graph, matched_node_list)
        else:
            sequences = [sequence1, sequence2, sequence3, sequence6]
            for sequence in sequences:
                matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True, use_dfs=True)
                replace_gelu(graph, matched_node_list)

    @staticmethod
    def match_hardswish(graph):
        def is_valid_hardswish(node_tuple):
            def check_for_valid_add_node(input_const_name):
                const_input_node = graph.get_producer_node(input_const_name)
                const_input_value = const_input_node.op.tensor
                const_input_length = reduce(lambda x,y:x * y, const_input_value.shape)
                temp = set(const_input_value.reshape(const_input_length))
                if len(temp) != 1 or int(temp.pop()) != 3:
                    return False
                return True

            def check_for_valid_neuron_node(node):
                if node.op.operation != ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX \
                        or int(node.op.min_value) != 0 \
                        or int(node.op.max_value) != 6:
                    return False
                return True

            def check_for_valid_div_node(node):
                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if np.array_equal(np.unique(const_input_value), [6]):
                  return True
                return False

            def check_for_valid_mul_node_with_const_input(node):
                def is_close_to_one_sixth(num):
                    return translation_utils.compare_values(float(num[0]), 1/6, rtol=1.e-3, atol=1.e-5)

                input_names = node.input_names
                const_input_nodes = get_input_const_nodes(input_names)
                const_input_value = const_input_nodes[0].op.tensor
                if const_input_value.shape != (1,) or not is_close_to_one_sixth(const_input_value):
                    return False
                return True

            add_node, neuron_node = node_tuple[0], node_tuple[1]
            add_non_const_input_name, add_const_input_name, mul_node, mul_node_const_input, div_node = [None] * 5
            for input_name in add_node.input_names:
                if graph.get_producer_op(input_name).type == op_adapter.ConstantOp.TRANSLATION_KEY:
                    add_const_input_name = input_name
                else:
                    add_non_const_input_name = input_name

            for node in node_tuple[2:]:
                if node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE]:
                    div_node = node
                else:
                    mul_input_names = node.input_names
                    if len(mul_input_names) != 2:
                        return False
                    if any(op_adapter.ConstantOp.TRANSLATION_KEY == graph.get_producer_op(input_name).type
                           for input_name in mul_input_names):
                        mul_node_const_input = node
                    else:
                        mul_node = node

            if not add_const_input_name or not mul_node or (not div_node and not mul_node_const_input):
                return False

            if add_non_const_input_name not in mul_node.input_names:
                # the add and mul must share same input_name to be matched as hswish
                return False

            return (check_for_valid_add_node(add_const_input_name) and
                    check_for_valid_neuron_node(neuron_node) and
                    (check_for_valid_div_node(div_node) if div_node else
                     check_for_valid_mul_node_with_const_input(mul_node_const_input)))

        def get_input_const_nodes(input_names):
            input_nodes = [graph.buffers[name].producer for name in input_names]
            const_nodes = [node for node in input_nodes if
                           node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY]
            return const_nodes

        def remove_const_nodes(node_tuple, matched_sequence_flag):
            if matched_sequence_flag[-1] in ['1', '3']:
                nodes_with_const_input = [node_tuple[0], node_tuple[3]]
            else:
                nodes_with_const_input = [node_tuple[0], node_tuple[2]]

            for node in nodes_with_const_input:
                const_node = get_input_const_nodes(node.input_names)[0]
                const_node_output_buf = graph.get_buffer(const_node.output_names[0])
                if len(const_node_output_buf.consumers) == 1:
                    # Only prune const_node if node is its only consumer
                    graph.prune(const_node, force_remove=True)
                else:
                    # Else, disconnect from node and leave const_node alone
                    const_node_output_buf.consumers.remove(node)
                    node.input_names.remove(const_node_output_buf.name)

        # Y = X*RELU6(X+3)*(1/6) or X*CLIP(X+3)*(1/6)
        sequence1 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")])
             ),
            (ir_graph.QNN_OP_ELEMENT_WISE_NEURON,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)*(1/6)) or X*(CLIP(X+3)*(1/6))
        sequence2 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")])
             ),
            (ir_graph.QNN_OP_ELEMENT_WISE_NEURON,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ()
             )
        ]

        # Y = X*RELU6(X+3)/6 or X*CLIP(X+3)/6
        sequence3 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")])
             ),
            (ir_graph.QNN_OP_ELEMENT_WISE_NEURON,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"),
                                 ("constant", "ANY")]),
             ()
             )
        ]

        # Y = X*(RELU6(X+3)/6) or X*(CLIP(X+3)/6)
        sequence4 = [
            ("elementwise_sum",
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ALL")])
             ),
            (ir_graph.QNN_OP_ELEMENT_WISE_NEURON,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_ELEMENT_WISE_NEURON, "ANY"),
                                 ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             (),
             ()
             )
        ]

        sequences = [sequence1, sequence2, sequence3, sequence4]

        for index, sequence in enumerate(sequences):
            matched_sequence_flag = 'matched_sequence' + str(index + 1)
            matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_hardswish, ignore_constants=True)

            for node_tuple in matched_node_list:
                # Record trace info before prune nodes
                node_tuple_trace_info = graph.get_trace_info_sub_graph(node_tuple,
                                                                       graph.get_output_buffers(node_tuple[-1]))
                remove_const_nodes(node_tuple, matched_sequence_flag)
                # Get hardswish_node's input_names and output_names
                input_buffers = graph.get_input_buffers(node_tuple[0])
                output_buffers = graph.get_output_buffers(node_tuple[-1])
                input_names = [buf.name for buf in input_buffers]
                output_names = [buf.name for buf in output_buffers]

                # Get previous input_name idx for output_buffers[0]
                output_consumers = output_buffers[0].consumers
                output_consumers_dict = {}
                for consumer in output_consumers:
                    buf_idx = consumer.input_names.index(output_names[0])
                    output_consumers_dict[consumer] = buf_idx

                # the output should be only one tensor
                output_trace_info = graph.get_trace_info(output_buffers[0])
                # Remove all the nodes in node_tuple
                for node in reversed(node_tuple):
                    graph.prune(node, force_remove=True)

                # Add new node
                add_op = node_tuple[0].op
                add_op_name = add_op.name
                hardswish_op_name = add_op_name + '_Hswish'
                hardswish_op = op_adapter.ElementwiseNeuronOp(hardswish_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH)
                idx_to_insert = 0
                for input_name in input_names:
                    buf = graph.get_buffer(input_name)
                    cur_idx = graph.nodes_in_order.index(buf.producer)
                    if idx_to_insert <= cur_idx:
                        idx_to_insert = cur_idx + 1
                hardswish_node = graph.add(hardswish_op, input_names=input_names, output_names=output_names,idx=idx_to_insert)
                # Set trace info for new created node
                graph.set_trace_info(hardswish_node, node_tuple_trace_info)
                graph.set_trace_info(graph.get_output_buffers(hardswish_node)[0], output_trace_info)

                # Restore the input_name for following nodes
                for consumer in output_consumers_dict.keys():
                    buf_idx = output_consumers_dict[consumer]
                    consumer.input_names.insert(buf_idx, output_names[0])

                # Add consumer for output_buffers[0]
                graph.get_output_buffers(hardswish_node)[0].consumers = output_consumers

                # Update hardswish output encodings
                last_op = node_tuple[-1].op
                if graph.has_quantization_param(last_op.name):
                    output_encodings = graph.get_layer_quantization_param(last_op.name)[op_graph.QuantParams.OUTPUT_ENCODINGS]
                    if len(output_encodings) > 0:
                        output_encoding = output_encodings[0].copy()
                        output_encoding['name'] = hardswish_node.output_names[0]
                        graph.add_quantization_params(hardswish_node.op.name, output_encodings=output_encoding)

    def check_static_equal_tensor_vals_input(self, node, graph):
        perform_optimization = True
        if graph.get_producer_op(node.input_names[0]).type == op_adapter.ConstantOp.TRANSLATION_KEY:
            tensor = graph.get_producer_op(node.input_names[0]).tensor
            constant_node = graph.get_node_by_name(node.input_names[0])
        elif graph.get_producer_op(node.input_names[1]).type == op_adapter.ConstantOp.TRANSLATION_KEY:
            tensor = graph.get_producer_op(node.input_names[1]).tensor
            constant_node = graph.get_node_by_name(node.input_names[1])
        else:
            perform_optimization = False

        if perform_optimization:
            # Perform some more verification on min and max clips by flattening and checking if all vals are the same
            # Get flattened 1D array for multidimensional tensor array
            flatten_arr = np.ravel(tensor)
            # Check if all value in min_val array are equal
            perform_optimization = np.all(tensor == flatten_arr[0])
            tensor_scalar_value = None
            if perform_optimization:
                tensor_scalar_value = flatten_arr[0]

        return perform_optimization, constant_node, tensor_scalar_value

    def relu_min_max_sequence_optimization(self, node_replace, node_delete, relu_min_max_op, graph):
        # Replace elementwise min or elementwise max op depending on sequence found with reluminmax op
        graph.replace(node_replace.op, relu_min_max_op)

        # Get output buffer of last op in sequence
        node_delete_output_buff_name = node_delete.output_names
        node_delete_buff = graph.get_buffer(node_delete_output_buff_name[0])

        # Update input_names of consumers of elementwise min or elementwise max output buffer depending on sequence
        node_delete_buff_consumers = node_delete_buff.consumers
        for consumer in node_delete_buff_consumers:
            # consumer.input_names = graph.nodes_by_name[relu_min_max_op.name].output_names
            for idx in range(len(consumer.input_names)):
                if consumer.input_names[idx] == node_delete_output_buff_name[0]:
                    consumer.input_names[idx] = graph.nodes_by_name[relu_min_max_op.name].output_names[0]
        node_delete_buff.consumers = set()
        graph.prune(node_delete, force_remove=True)
        graph.get_buffer(node_replace.output_names[0]).consumers = node_delete_buff_consumers

    def merge_low_level_ops_to_layers(self, graph):

        def validate_node(node_tuple):
            node_out_buff = graph.get_buffer(node_tuple[0].output_names[0])
            if node_tuple[1] not in node_out_buff.consumers or len(node_out_buff.consumers) > 1:
                return False

            if node_tuple[0].op.operation == ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM:
                min_node, max_node = node_tuple
            else:
                max_node, min_node = node_tuple

            # Check to see if elementwise min and max have one constant input
            # Verify if tensor values in min and max clip arrays are all same for optimization to work properly
            perform_optimization, check_tensor_vals_equal, _ = self.check_static_equal_tensor_vals_input(min_node, graph)
            perform_optimization_2, check_tensor_vals_equal_2, _ = self.check_static_equal_tensor_vals_input(max_node, graph)

            if not perform_optimization or not perform_optimization_2:
                return False
            return True

        # Elementwisemin -> Elementwisemax = Reluminmax
        sequence_1 = [
            ("elementwise_min",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_max", "ALL")])
             ),
            ("elementwise_max",
             ("MATCH_NUM_BUFS", [("elementwise_min", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence_2 = [
            ("elementwise_max",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_min", "ALL")])
             ),
            ("elementwise_min",
             ("MATCH_NUM_BUFS", [("elementwise_max", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequences = [sequence_1, sequence_2]
        for idx, sequence in enumerate(sequences):
            matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True, validator=validate_node)
            for node_tuple in matched_node_list:
                is_min_max_sequence = False
                if node_tuple[0].op.operation == ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM:
                    is_min_max_sequence = True

                if is_min_max_sequence:
                    min_node, max_node = node_tuple
                else:
                    max_node, min_node = node_tuple

                # Retrieve tensor and node information required for ReluMinMax optimization
                _, constant_node_min, min_op_scalar_value = self.check_static_equal_tensor_vals_input(min_node, graph)
                _, constant_node_max, max_op_scalar_value = self.check_static_equal_tensor_vals_input(max_node, graph)

                if max_op_scalar_value <= min_op_scalar_value:
                    # if min is greater than or equal to max, then assign min value(eltwise min) as max for ReluMinMax and
                    # max value(eltwise max)  as min for ReluMinMax
                    relu_max_value, relu_min_value = min_op_scalar_value, max_op_scalar_value
                else:
                    if is_min_max_sequence:
                        # When elementwise min is followed by elementwise max and min value is less than max value,
                        # assign both min and max for ReluMinMax to max value(elementwise max)
                        relu_min_value = max_op_scalar_value
                        relu_max_value = max_op_scalar_value
                    else:
                        # When elementwise max is followed by elementwise min and min value is less than max value,
                        # assign both min and max for ReluMinMax to min value(elementwise min)
                        relu_min_value = min_op_scalar_value
                        relu_max_value = min_op_scalar_value

                # Assign values of old sequence ops to new op
                relu_min_max_op = op_adapter.ElementwiseNeuronOp("",
                                                                 operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                                                                 min_value=relu_min_value,
                                                                 max_value=relu_max_value)
                relu_min_max_op.name = graph.naming_policy.get_op_name(relu_min_max_op)

                # Check to remove constant node from graph if consumer of it is only first node in sequence matched.
                # Second node's constant input will be automatically removed by remove_disconnected_nodes optimization
                # since second node in sequence is pruned
                if is_min_max_sequence:
                    # removing node as consumer should take care of removing the min node as a consumer of the constant
                    # regardless of however many consumers the constant node has besides the min node
                    graph.remove_node_as_consumer(min_node, graph.get_buffer(constant_node_min.output_names[0]).name)
                    self.relu_min_max_sequence_optimization(min_node, max_node, relu_min_max_op, graph)
                else:
                    # removing node as consumer should take care of removing the max node as a consumer of the constant
                    # regardless of however many consumers the constant node has besides the max node
                    graph.remove_node_as_consumer(max_node, graph.get_buffer(constant_node_max.output_names[0]).name)
                    self.relu_min_max_sequence_optimization(max_node, min_node, relu_min_max_op, graph)

    @staticmethod
    def expand_sparse_op_structure(node, graph):
        output_buf = graph.get_output_buffers(node)[0]
        consumer_nodes = output_buf.consumers
        node_input = node.input_names[0]
        list_ids = []

        for consumer in consumer_nodes:
            list_ids.append(consumer.input_names.index(str(output_buf)))

        sparse_params = graph.get_buffer(node_input).get_sparse_params()

        if sparse_params.layout != ir_graph.QNN_SPARSE_LAYOUT_UNDEFINED:
            post_expansion_idx = graph.nodes_in_order.index(node)
            # Record trace info before prune nodes
            origin_trace_info = graph.get_trace_info_sub_graph([node])
            # first prune and add back to adjust its input
            graph.prune(node, force_remove=True)
            sparse_indices_op_name = node.op.name + '_sparseIndices'
            sparse_values_op_name = node.op.name + '_sparseValues'
            sparse_indices_op_output_name = sparse_indices_op_name + '_out'
            sparse_values_op_output_name = sparse_values_op_name + '_out'
            post_expansion_node_output_name = node.op.name + '_out'
            create_sparse_op_name = node.op.name + '_createSparse'
            create_sparse_output_name = node.op.name + '_createSparseOut'
            create_sparse_output_shape = output_buf.get_buf_dims()
            sparse_indices_op = op_adapter.GetSparseIndicesOp(sparse_indices_op_name,
                                                              num_specified_elements = sparse_params.cooInfo.numSpecifiedElements)
            sparse_values_op = op_adapter.GetSparseValuesOp(sparse_values_op_name,
                                                            num_specified_elements = sparse_params.cooInfo.numSpecifiedElements)
            sparse_values_op_node = graph.add(sparse_values_op, [node_input], [sparse_values_op_output_name], idx=post_expansion_idx)
            sparse_indices_op_node = graph.add(sparse_indices_op, [node_input], [sparse_indices_op_output_name], idx=post_expansion_idx+1)
            post_expansion_op = op_adapter.ElementwiseNeuronOp(name=node.op.name, operation=node.op.operation)
            post_expansion_op_node = graph.add(post_expansion_op, [sparse_values_op_output_name], [post_expansion_node_output_name], idx=post_expansion_idx+2)
            create_sparse_op = op_adapter.CreateSparseOp(create_sparse_op_name, create_sparse_output_shape)
            create_sparse_op_node = graph.add(create_sparse_op, [sparse_indices_op_output_name, post_expansion_node_output_name],
                      [create_sparse_output_name], idx=post_expansion_idx+3, sparse_params=sparse_params)
            # Set trace info for new created nodes/tensors
            expanded_nodes = [sparse_values_op_node, sparse_indices_op_node, post_expansion_op_node, create_sparse_op_node]
            expanded_nodes_tensors = []
            for node in expanded_nodes:
                expanded_nodes_tensors.append(node)
                expanded_nodes_tensors.extend(graph.get_output_buffers(node))
            graph.set_trace_info(expanded_nodes_tensors, origin_trace_info)

            for i, consumer in enumerate(consumer_nodes):
                graph.get_buffer(create_sparse_output_name).consumers.add(consumer)
                consumer.input_names.insert(list_ids[i], create_sparse_output_name)


@register_layer_optimization
class OptimizeMeanTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MeanOp.TRANSLATION_KEY
        self.register_method(expand_elementwise_mean, self.expand_elementwise_mean)

    def expand_elementwise_mean(self, node, graph):
        sequence = [(op_adapter.MeanOp.TRANSLATION_KEY, (), ())]
        matched_node_list = graph.get_matched_nodes(sequence)

        for matched_node in matched_node_list:
            input_names = matched_node[0].input_names
            output_names = matched_node[0].output_names
            elementwise_mean_node = matched_node[0]
            op_name = elementwise_mean_node.op.name
            self.idx = graph.list_nodes().index(elementwise_mean_node)
            consumers = graph.get_buffer(output_names[0]).consumers.copy()
            consumers_dict = {}
            for c in consumers:
                indices = [i for i in range(len(c.input_names)) if c.input_names[i] == output_names[0]]
                consumers_dict[c.op.name] = indices
            graph.prune(elementwise_mean_node, force_remove=True)

            if len(input_names) == 1:
                op = op_adapter.IdentityOp(name=op_name)
                node = graph.inject(op, input_names[0], output_names[0])
                if graph.enable_trace:
                    graph.set_trace_info(node, [(op_name, op_graph.TraceType.OP)])
                continue

            op_input_names = [input_names[0], input_names[1]]
            name = op_name + '_add_' + str(1)
            op = op_adapter.ElementwiseBinaryOp(name=name,
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            op_output_names = [name]
            node = graph.add(op, op_input_names, op_output_names, idx = self.idx)
            if graph.enable_trace:
                graph.set_trace_info(node, [(op_name, op_graph.TraceType.OP)])
            self.idx += 1
            for i in range(2, len(input_names)):
                del op_input_names[0]
                op_input_names[0] =  op_output_names[0]
                op_input_names.append(input_names[i])
                name = op_name + '_add_' + str(i)
                op = op_adapter.ElementwiseBinaryOp(name=name,
                                                    operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
                op_output_names = [name]
                node = graph.add(op, op_input_names, op_output_names, idx = self.idx)
                if graph.enable_trace:
                    graph.set_trace_info(node, [(op_name, op_graph.TraceType.OP)])
                self.idx += 1

            N = len(input_names)
            input_len_op = op_adapter.ConstantOp(name = op_name+'_div_constant', tensor=np.array([N], dtype=np.float32))
            node = graph.add(input_len_op, [], op_name+'_div_constant', idx = self.idx)
            if graph.enable_trace:
                graph.set_trace_info(node, [(op_name, op_graph.TraceType.OP)])
            self.idx += 1
            div_op = op_adapter.ElementwiseBinaryOp(name=op_name+'_div', operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE)
            node = graph.add(div_op, [op.name, op_name+'_div_constant'], output_names, idx= self.idx)
            if graph.enable_trace:
                graph.set_trace_info(node, [(op_name, op_graph.TraceType.OP)])
            consumers = list(consumers)
            if len(consumers) > 0:
                for i in range(len(consumers)):
                    indices = consumers_dict[consumers[i].op.name]
                    for idx in indices:
                        consumers[i].input_names.insert(idx, elementwise_mean_node.output_names[0])
                graph.get_buffer(elementwise_mean_node.output_names[0]).consumers = consumers


@register_layer_optimization
class OptimizeNonZeroTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NonZeroOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        # input buffer should be in source framework order.
        input_buf = graph.get_input_buffers(node)[0]
        data_axis_format = node.op.data_axis_formats[0]
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                data_axis_format == AxisTracker.AxisFormat.NCDHW:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NCDHW,
                                            AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                data_axis_format == AxisTracker.AxisFormat.NCS:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NCS,
                                            AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                data_axis_format == AxisTracker.AxisFormat.NCF:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NCF,
                                            AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                data_axis_format == AxisTracker.AxisFormat.TNF:
            graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.TNF,
                                          AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        output_buf = graph.get_output_buffers(node)[0]
        output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        return True


@register_layer_optimization
class OptimizeOneHotTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.OneHotOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == node.op.data_axis_formats[0] and \
                input_buf.axis_format in AxisTracker.AxisFormat.get_valid_formats():
            return False

        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                input_buf.axis_format != node.op.data_axis_formats[0]:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                          AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                input_buf.axis_format != node.op.data_axis_formats[0]:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                          AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                input_buf.axis_format != node.op.data_axis_formats[0]:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                          AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                input_buf.axis_format != node.op.data_axis_formats[0]:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                          AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
        elif input_buf.axis_format in AxisTracker.AxisFormat.get_valid_formats():
            pass
        else:
            raise ValueError("OneHot Node {} got unexpected input_axis_formats {}".format(node, input_buf.axis_format))

        return True


@register_layer_optimization
class OptimizePadTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PadOp.TRANSLATION_KEY
        self.register_method(SQUASH_PAD, self.squash_pad)

    @staticmethod
    def squash_pad(graph):
        def validate_node(nodes_tuple):
            pad_node_ = nodes_tuple[0]
            pads = pad_node_.op.pad_amount
            # squash if all values are 0s
            if all(not (pad_0 or pad_1) for pad_0, pad_1 in pads) and \
                    len(graph.get_buffer(pad_node_.input_names[0]).consumers) == 1:
                return True
            return False

        sequence = [
            ("Pad", (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            pad_node = node_tuple[0]
            if not graph.is_output_node(pad_node):
                graph.squash_identity(pad_node)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.NCDHW_TO_NDHWC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.NCS_TO_NSC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.NCF_TO_NFC)
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            node.op.pad_amount = AxisTracker.permute_shape(node.op.pad_amount, AxisTracker.AxisFormat.TNF_TO_NTF)
        node.op.pad_amount = np.asarray(node.op.pad_amount, dtype=np.dtype('uint32'))
        return True


@register_layer_optimization
class OptimizePoolTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.Pool2dOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_buffers = graph.get_input_buffers(node)
        input_axis_formats = [buf.axis_format for buf in input_buffers]

        if any(axis_format in input_axis_formats for axis_format in [AxisTracker.AxisFormat.NDHWC,
                                                                     AxisTracker.AxisFormat.NCDHW,
                                                                     AxisTracker.AxisFormat.NSC,
                                                                     AxisTracker.AxisFormat.NCS,
                                                                     AxisTracker.AxisFormat.ANY,
                                                                     AxisTracker.AxisFormat.NONTRIVIAL]):
            AxisTracker.image_to_channel_last_order(node, graph)
            output_buffer = graph.get_output_buffers(node)[0]
            # image_to_channel_last_order function may set the output as NONTRIVIAL, when input is NONTRIVIAL
            # Enforce the output format here according to output buffer's rank
            output_buffer.axis_format = AxisOrder().get_axis_format(output_buffer.rank())
        else:
            raise ValueError("Pool Node {} got unexpected input_axis_formats {}".format(node, input_axis_formats))
        return True


@register_layer_optimization
class OptimizePool1DTranslation(Optimize1DNNTranslation):
    def __init__(self):
        Optimize1DNNTranslation.__init__(self)
        self.op_type = op_adapter.Pool1dOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def expand_1d_spatial_nn_nodes(self, node, graph):
        pool_op_name = node.op.name + "_2d"
        if node.op.pool_type == ir_graph.QNN_OP_POOL_MAX_2D:
            self.nn_2d_op = op_adapter.Pool2dOp(pool_op_name,
                                                pool_type=node.op.pool_type,
                                                size_x=node.op.filter_size,
                                                size_y=1,
                                                stride_x=node.op.stride,
                                                stride_y=1,
                                                padx_before=node.op.pad_amount[0],
                                                padx_after=node.op.pad_amount[1],
                                                pady_before=0,
                                                pady_after=0,
                                                padding_size_strategy=node.op.padding_size_strategy)
        elif node.op.pool_type == ir_graph.QNN_OP_POOL_AVG_2D:
            self.nn_2d_op = op_adapter.Pool2dOp(pool_op_name,
                                                pool_type=node.op.pool_type,
                                                size_x=node.op.filter_size,
                                                size_y=1,
                                                stride_x=node.op.stride,
                                                stride_y=1,
                                                padx_before=node.op.pad_amount[0],
                                                padx_after=node.op.pad_amount[1],
                                                pady_before=0,
                                                pady_after=0,
                                                padding_size_strategy=node.op.padding_size_strategy,
                                                count_pad_for_edges=node.op.count_pad_for_edges)
        elif node.op.pool_type == ir_graph.QNN_OP_L2_POOL_2D:
            self.nn_2d_op = op_adapter.Pool2dOp(pool_op_name,
                                                pool_type=node.op.pool_type,
                                                size_x=node.op.filter_size,
                                                size_y=1,
                                                stride_x=node.op.stride,
                                                stride_y=1,
                                                padx_before=node.op.pad_amount[0],
                                                padx_after=node.op.pad_amount[1],
                                                pady_before=0,
                                                pady_after=0,
                                                padding_size_strategy=node.op.padding_size_strategy,
                                                p=node.op.p)

        super().expand_1d_spatial_nn_nodes(node, graph)


@register_layer_optimization
class OptimizePool3dTranslation(OptimizePoolTranslation):
    def __init__(self):
        OptimizePoolTranslation.__init__(self)
        self.op_type = op_adapter.Pool3dOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeTransposeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TransposeOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)
        self.register_method(SQUASH_CONSTANT_INPUT, self.squash_constant_input)
        self.register_method(FOLD_MULTIPLE_TRANSPOSE, self.fold_multiple_transpose)

    def replace_6d_operation(self, node, graph):
        """Replace 6D Transpose by inserting Reshape around.

        In order to getting rid of >6D shapes, this optimization tries to reduce dimensions by
        merging those axes which are still consecutive after Transpose. Taking a Transpose with
        perm [0,1,3,4,5,2] for example, axes [0,1] and [3,4,5] are consucutive even after Transpose,
        and therefore they can be respectively merged into single dimension beforehand and recovered
        afterwards. Note that Transpose perm must be updated accordingly.
        """
        input_buf = graph.get_buffer(node.input_names[0])
        output_buf = graph.get_buffer(node.output_names[0])
        if input_buf.rank() < 6:
            return

        check_if_6d_supportable(node, graph)

        # Calculate target shape by merging consecutive axes.
        target_shape, remaining_axes = [input_buf.shape[0]], [0]
        for idx in range(1, input_buf.rank()):
            if node.op.perm.index(idx - 1) + 1 == node.op.perm.index(idx):
                target_shape[-1] *= input_buf.shape[idx]
            else:
                target_shape.append(input_buf.shape[idx])
                remaining_axes.append(idx)

        # Current solution only supports shapes that could be merged into < 6D ones. For those
        # non-mergable cases should be handled by splitting and concating which is much more
        # complicated and therefore is left as future work.
        converter_utils.log_assert(
            len(target_shape) < 6,
            f'Failed to resolve 6D tensor by merging consecutive axes for Transpose {node.op.name}'
            f'with input shape {input_buf.shape[0]} and permutation {node.op.perm}.'
        )

        # Calculate updated perm according to target shape and original perm.
        target_perm = list(range(len(target_shape)))
        target_perm.sort(key=lambda axis: node.op.perm.index(remaining_axes[axis]))
        node.op.perm = target_perm

        # Insert pre-Reshape.
        pre_reshape = op_adapter.ReshapeOp(f'{node.op.name}_6d_pre_reshape', shape=target_shape)
        graph.inject(pre_reshape, input_buf.name, pre_reshape.name, consumer_names=[node.op.name])
        # Insert post-Reshape.
        post_reshape_insertion(
            node, graph, [[target_shape[axis] for axis in target_perm]], [output_buf.shape]
        )

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        # check for trivial cases first, which will end up
        # in removal.
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NDHWC:
                # This permute changes from NDHWC to NCDHW which is opposite of desired, so skip this node
                if output_buf.axis_format == AxisTracker.AxisFormat.NCDHW and node.op.perm == [0, 4, 1, 2, 3]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                        node.op.perm == [0, 1, 2, 3, 4]:
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
            elif node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
                # This permute changes from NCDHW to NDHWC but input has already changed to NDHWC, so skip
                if output_buf.axis_format == AxisTracker.AxisFormat.NDHWC and node.op.perm == [0, 2, 3, 4, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NCDHW and \
                        node.op.perm == [0, 1, 2, 3, 4]:
                    output_buf.axis_format = AxisTracker.AxisFormat.NDHWC
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                                  AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                                  consumers=[node.op.name])
            else:
                # going to nontrivial, hoping for the best.
                log_warning("Op {} with Permute order {} has Unsupported input data format {}".format(node,
                                                                                               node.op.perm,
                                                                                               node.op.data_axis_formats[0]))
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCDHW and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NONTRIVIAL]:
            if output_buf.axis_format == AxisTracker.AxisFormat.NDHWC and node.op.perm == [0, 2, 3, 4, 1]:
                return
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NSC:
                # This permute changes from NSC to NCS which is opposite of desired, so skip this node
                if output_buf.axis_format == AxisTracker.AxisFormat.NCS and node.op.perm == [0, 3, 1, 2]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                        node.op.perm == [0, 1, 2, 3]:
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
            elif node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
                # This permute changes from NCS to NSC but input has already changed to NSC, so skip
                if output_buf.axis_format == AxisTracker.AxisFormat.NSC and node.op.perm == [0, 2, 3, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NCS and \
                        node.op.perm == [0, 1, 2, 3]:
                    output_buf.axis_format = AxisTracker.AxisFormat.NSC
                    output_shape = AxisTracker.permute_shape(output_buf.get_buf_dims(), AxisTracker.AxisFormat.NCS_TO_NSC)
                    output_buf.set_buf_dims(output_shape)
                    return
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                                  AxisTracker.AxisFormat.NSC_TO_NCS,
                                                  consumers=[node.op.name])
            else:
                # going to nontrivial, hoping for the best.
                log_warning("Op {} with Permute order {} has Unsupported input data format {}".format(node,
                                                                                                      node.op.perm,
                                                                                                      node.op.data_axis_formats[0]))
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCS and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NONTRIVIAL]:
            if output_buf.axis_format == AxisTracker.AxisFormat.NSC and node.op.perm == [0, 2, 3, 1]:
                return
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NFC:
                # This permute changes from NFC to NCF which is opposite of desired, so skip this node
                if output_buf.axis_format == AxisTracker.AxisFormat.NCF and node.op.perm == [0, 2, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    output_buf.axis_format = AxisTracker.AxisFormat.NFC
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                        node.op.perm == [0, 1, 2]:
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
            elif node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
                # This permute changes from NCF to NFC but input has already changed to NFC, so skip
                if output_buf.axis_format == AxisTracker.AxisFormat.NFC and node.op.perm == [0, 2, 1]:
                    graph.replace(node.op,
                                  op_adapter.IdentityOp(node.op.name))
                    return
                elif output_buf.axis_format == AxisTracker.AxisFormat.NCF and \
                        node.op.perm == [0, 1, 2]:
                    output_buf.axis_format = AxisTracker.AxisFormat.NFC
                    # Nothing to be done, Remove_Identity will handle the squashing of this node
                    return
                else:
                    graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                                  AxisTracker.AxisFormat.NFC_TO_NCF,
                                                  consumers=[node.op.name])
            else:
                # going to nontrivial, hoping for the best.
                log_warning("Op {} with Permute order {}: Unknown input data format {}".format(node,
                                                                                               node.op.perm,
                                                                                               node.op.data_axis_formats[0]))
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF,
                                              consumers=[node.op.name])
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NCF and \
                node.op.data_axis_formats[0] in [AxisTracker.AxisFormat.NCF, AxisTracker.AxisFormat.NONTRIVIAL]:
            if output_buf.axis_format == AxisTracker.AxisFormat.NFC and node.op.perm == [0, 2, 1]:
                return
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF:
            if node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF and node.op.perm == [1, 0, 2]:
                node.op.perm = [0, 1, 2]
                output_buf.axis_format = AxisTracker.AxisFormat.NTF
            else:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.TNF_TO_NTF,
                                              consumers=[node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            if output_buf.axis_format == AxisTracker.AxisFormat.OIHW:
                output_buf.axis_format = AxisTracker.AxisFormat.OIHW
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                input_buf.axis_format == AxisTracker.AxisFormat.NF:
            output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
        else:
            raise ValueError("Permute Op {} got unexpected params: input format {} saved data format {}".format(
                node, input_buf.axis_format, node.op.data_axis_formats[0]))

        return True

    @staticmethod
    def squash_constant_input(node, graph):
        """
        if the constant input buff only has one consumer, merge the input constant op and this transpose op.
        """
        input_buff = graph.get_buffer(node.input_names[0])
        if input_buff.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY and len(input_buff.consumers) == 1:
            # Transpose input tensor
            tensor = input_buff.producer.op.tensor
            tensor = np.ascontiguousarray(np.transpose(tensor, node.op.perm))
            input_buff.producer.op.tensor = tensor
            input_buff.shape = list(tensor.shape)
            # Squash current op to input op.
            graph.squash(node, input_name=input_buff.name)
            q = graph.quantization_params
            if (input_buff.name in q and (node.op.name not in q or
                                          (node.op.name in q and not q[node.op.name]['output_encodings']))):
                quant_param = graph.get_layer_quantization_param(input_buff.name)
                graph.add_quantization_params(node.op.name,
                                              bn_params=quant_param['bn_params'],
                                              output_encodings=quant_param['output_encodings'],
                                              param_encodings=quant_param['param_encodings'])
                graph.quantization_params[node.op.name]['output_encodings'][0]['name'] = node.op.name

    @staticmethod
    def remove_identity(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        if input_buffer.axis_format == output_buffer.axis_format and node.op.perm == list(range(len(node.op.perm))):
            # this permute is trivial, remove it
            try:
                graph.squash(node, input_name=input_buffer.name, is_data_movement_node=True)
            except RuntimeError:
                converter_utils.log_debug(f'Unable to squash identity Transpose {node.op.name}.')
        return True

    @staticmethod
    def fold_multiple_transpose(node, graph):
        input_buf = graph.get_input_buffers(node)[0]
        input_buf_producer = input_buf.producer
        node_output = node.output_names[0]
        node_consumers = list(graph.get_buffer(node_output).consumers)
        nodes_to_remove = []

        for consumer in node_consumers:
            if (consumer.op.type == op_adapter.TransposeOp.TRANSLATION_KEY
                    and graph.get_buffer(consumer.output_names[0]).shape == input_buf.shape):
                nodes_to_remove.append(consumer)

        # Remove transpose nodes when input shape is equal to output shape after two transposes in a row
        for transpose_node in nodes_to_remove:
            transpose_node_output = transpose_node.output_names[0]
            transpose_node_consumers = list(graph.get_buffer(transpose_node_output).consumers)
            for consumer in transpose_node_consumers.copy():
                consumer_in_idx = consumer.input_names.index(transpose_node_output)
                consumer.input_names[consumer_in_idx] = input_buf.name
                input_buf.consumers.add(consumer)
                graph.get_buffer(transpose_node_output).consumers.remove(consumer)
                graph.update_trace_info(consumer, [input_buf_producer])
                transpose_node_consumers.remove(consumer)

            if len(transpose_node_consumers) == 0:
                graph.prune(transpose_node, force_remove=True)
                node_consumers.remove(transpose_node)


        # Squash transpose op into one when input shape is not equal to output shape after two transposes in a row
        first_permute_order = node.op.perm
        for consumer in node_consumers.copy():
            if consumer.op.type == op_adapter.TransposeOp.TRANSLATION_KEY:
                second_permute_order = consumer.op.perm
                new_order = first_permute_order[:]
                for i, val in enumerate(second_permute_order):
                    new_order[i] = first_permute_order[val]

                consumer.op.perm = new_order
                consumer_in_idx = consumer.input_names.index(node_output)
                consumer.input_names[consumer_in_idx] = input_buf.name
                input_buf.consumers.add(consumer)
                graph.get_buffer(node_output).consumers.remove(consumer)
                graph.update_trace_info(consumer, [input_buf_producer])
                node_consumers.remove(consumer)

                if len(node_consumers) == 0:
                    graph.prune(node)


@register_layer_optimization
class OptimizePreluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PreluOp.TRANSLATION_KEY
        self.register_method(PREPARE_INPUTS_AS_PARAMS, self.prepare_inputs_as_params)

    @classmethod
    def _permute_coeff(cls, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        coeff_buf = graph.get_buffer(node.input_names[1])
        coeff_shape = coeff_buf.shape

        # Storing the coeff_buf axis format
        current_axis_format = coeff_buf.axis_format
        # determine the permute order(if any) after spatial first transformation
        # Note: only NDHWC, NSC, NFC, and NTF formats imply permute was done.
        input_permute_order = None
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            input_permute_order = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            current_axis_format = AxisTracker.AxisFormat.NDHWC
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            input_permute_order = AxisTracker.AxisFormat.NCS_TO_NSC
            current_axis_format = AxisTracker.AxisFormat.NSC
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            input_permute_order = AxisTracker.AxisFormat.NCF_TO_NFC
            current_axis_format = AxisTracker.AxisFormat.NFC
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            input_permute_order = AxisTracker.AxisFormat.TNF_TO_NTF
            current_axis_format = AxisTracker.AxisFormat.NTF

        if len(coeff_buf.shape) != 1 and len(coeff_buf.shape) != len(input_buf.shape):
            raise ValueError("Prelu coefficient rank must equal either 1 or input rank {} for node {}. Got {} instead."
                             .format(len(input_buf.shape), node.op.name, len(coeff_buf.shape)))

        # If coeff_buf is a shared buffer, then it will be read once again for another PRelu op.
        # In such a case it will already be in one of NHWC, NFC, NDHWC or NTF.
        if input_permute_order is not None and len(coeff_shape) > 1 and coeff_buf.axis_format != current_axis_format:
            # The input has been permuted hence we also need to permute coeff so that broadcasting persists.
            # Here, the coeff_buf is not a shared buffer or it is being read for the first time.
            coeff_buf.producer.op.tensor = np.ascontiguousarray(np.transpose(coeff_buf.producer.op.tensor, input_permute_order))
            coeff_shape = coeff_buf.producer.op.shape
            coeff_buf.shape = coeff_shape # Update the buffer shape
            coeff_buf.axis_format = current_axis_format # Update the axis format, so that it is not permuted again.
        if not translation_utils.broadcastable(input_buf.shape, coeff_shape):
            raise ValueError(code_to_message.get_error_message("ERROR_OPERATION_INPUTS_NOT_BROADCASTABLE")
                             (node.op.name, input_buf.name, "coeff", input_buf.shape, coeff_shape))

    def axes_to_spatial_first_order(self, node, graph):
        ret = super(OptimizePreluTranslation, self).axes_to_spatial_first_order(node, graph)
        if ret:
            # Input buffer axis might have been transformed, coeff need to be transformed as well
            OptimizePreluTranslation._permute_coeff(node, graph)
        return ret

    def merge_low_level_ops_to_layers(self, graph):
        def validate(node_tuple):
            first_node = node_tuple[0]
            if first_node.op.type == 'elementwise_min':
                min_node = first_node
                mul_node = node_tuple[1]
                max_node = node_tuple[2]
            else:
                max_node = first_node
                min_node = node_tuple[1]
                mul_node = node_tuple[2]

            min_input_buffer = graph.get_input_buffers(min_node)
            max_input_buffer = graph.get_input_buffers(max_node)
            mul_output_buffer = graph.get_output_buffers(mul_node)
            max_output_buffer = graph.get_output_buffers(max_node)

            if min_input_buffer[1] == max_input_buffer[1] and mul_output_buffer[0].consumers == max_output_buffer[0].consumers:
                return True
            return False

        sequence1 = [
            ("elementwise_min",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_max",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ()
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_max", "ANY"), ("elementwise_product", "ANY")]),
             ()
             ),
        ]
        sequence2 = [
            ("elementwise_max",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_min",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY")])
             ),
            ("elementwise_product",
             ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ANY")])
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_max", "ANY"), ("elementwise_product", "ANY")]),
             ()
             ),
        ]
        sequences = [sequence1, sequence2]
        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate, ignore_constants=True)
            for node_tuple in matched_node_list:
                first_node = node_tuple[0]
                if first_node.op.type == 'elementwise_min':
                    mul_node = node_tuple[1]
                else:
                    mul_node = node_tuple[2]
                const_mul_node = graph.get_node_by_name(mul_node.input_names[0])
                add_node = node_tuple[3]

                # get the prelu coeff from the constant tensor and create a prelu op
                # Change the coeff to a constant node
                prelu_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.PreluOp.TRANSLATION_KEY,
                                                                        op_adapter.PreluOp.LEGACY_TRANSLATION_KEY,
                                                                        folded_op=True)
                prelu_op = op_adapter.PreluOp(prelu_op_name)

                # replace the last op in seq with prelu_op
                graph.replace(add_node.op, prelu_op)
                # min_input_names = min_node.input_names

                # get the buffer of first node in the sequence
                first_node_buf = graph.get_buffer(first_node.input_names[1])

                # prune all the nodes in the sequence except the last one
                for node in node_tuple[:-1]:
                    graph.prune(node, force_remove=True)

                # update the input names of the prelu node
                prelu_node = graph.nodes_by_name[prelu_op.name]
                prelu_node.input_names = first_node.input_names[1:]
                prelu_node.input_names.append(const_mul_node.op.name)
                graph.get_buffer(const_mul_node.op.name).consumers.add(prelu_node)

                # update the consumers of the first node buffer
                first_node_buf.consumers.add(prelu_node)

    def prepare_inputs_as_params(self, node, graph):
        coeff_buffer = graph.get_buffer(node.input_names[1])
        coeff_node = coeff_buffer.producer
        if coeff_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
            node.op.coeff = coeff_node.op.tensor
            # Remove the coeff inputs from the IR graph
            graph.remove_node_as_consumer(node, coeff_buffer.name)
            node.input_names = [node.input_names[0]]

@register_layer_optimization
class OptimizeProposalTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ProposalOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        output_buffer = graph.get_output_buffers(node)[0]

        # change input dims to 4D as required by snpe. Handling this here since converter allows for
        # non-4D inputs. Note: only change dimensions if it is input and no other node is consuming it
        # TODO: how should this be really handled
        im_info_input_buf = graph.get_input_buffers(node)[-1]
        if im_info_input_buf.producer.op.type == op_adapter.InputOp.TRANSLATION_KEY \
                and len(im_info_input_buf.consumers) == 1 \
                and im_info_input_buf.rank() != 4:
            shape = translation_utils.expand_to_rank(im_info_input_buf.shape, 4)
            im_info_input_buf.shape = shape
            im_info_input_buf.producer.op.shape = shape
            im_info_input_buf.axis_format = AxisTracker.AxisFormat.NSC
            output_buffer.axis_format = AxisTracker.AxisFormat.NSC
            return True
        else:
            return super(OptimizeProposalTranslation, self).axes_to_spatial_first_order(node, graph)


@register_layer_optimization
class OptimizeQuantizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.QuantizeOp.TRANSLATION_KEY
        self.register_method(REMOVE_QUANT_NODES, self.remove_quant_nodes)
        self.register_method(SQUASH_QUANT_NODES, self.squash_quant_dequant)

    @staticmethod
    def squash_quant_dequant(graph):
        sequence = [
                    (op_adapter.QuantizeOp.TRANSLATION_KEY, (), ()),
                    (op_adapter.DequantizeOp.TRANSLATION_KEY, (), ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, ignore_constants=True)
        for node_tuple in matched_node_list:
            # We found a quant/dequant combo, extract the nodes.
            first, second = node_tuple

            diff_value = []
            if first.op.name in graph.quantization_params.keys() and second.op.name in graph.quantization_params.keys():
                quantize_encoding = graph.quantization_params[first.op.name]['output_encodings'][0]
                dequantize_encoding = graph.quantization_params[second.op.name]['output_encodings'][0]
                for key in quantize_encoding:
                    if key in dequantize_encoding and quantize_encoding[key] != dequantize_encoding[key] and key not in ["dtype", "name"] :
                        diff_value.append(key)
                if len(diff_value) > 0:
                    raise ValueError("The quantize and dequantize op have an different value in {}.".format(diff_value))
            else:
                raise ValueError("quantize:{} and dequantize:{} ops miss quantization params, please insert the info when translation."
                                 .format(first.op.name, second.op.name))

            previous_node = graph.get_buffer(first.input_names[0]).producer
            if previous_node.op.type != op_adapter.InputOp.TRANSLATION_KEY:
                first_input_buffer = graph.get_input_buffers(first)[0]
                producer = first_input_buffer.producer
                # Fold these nodes into a convert op. Quant params are folded as part of squashing
                convert_name = producer.output_names[0] + "_convert_quant_dequant"
                convert_op = op_adapter.ConvertOp(convert_name)
                graph.inject(convert_op, input_name=first_input_buffer.name, output_name=convert_name, consumer_names=[first.op.name])
                convert_input_buffer = graph.get_output_buffers(producer)[0]
                log_debug('Injecting convert op {} with input {} and output {}'.format(convert_name, convert_input_buffer.name, convert_name))
                log_debug('Found {} and {} nodes to squash into {} '.format(first.op.name,second.op.name,convert_op.name))
            else:
                # only squash not work for the case input->quantize->dequantize, so need the change here
                # and need set overrride is true here
                output_encodings = quantize_encoding.copy()
                output_encodings["name"] = first.input_names[0]
                output_encodings.update({"overridden":True})
                graph.add_quantization_params(first.input_names[0],  output_encodings=output_encodings)
                log_debug('Found the encoding info of {} and {} are overridden into {}'.format(first.op.name, second.op.name, first.input_names[0]))
            graph.squash(second, input_name=second.input_names[0])
            graph.squash(first, input_name=first.input_names[0])

    @staticmethod
    def remove_quant_nodes(node, graph):
        # Squash the quant node. The quant params are folded as part of squashing
        graph.squash(node, input_name=node.input_names[0])
        log_debug("Remove quantize op {}".format(node.op.name))


class OptimizeReduceTranslationBase(OptimizationTranslationBase):
    def __init__(self, op_type):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_type

    def replace_6d_operation(self, node, graph):
        """
        replace 6D ReduceOp by inserting reshapes around the op
        """
        # only single input
        input_shape = graph.get_input_shapes(node)[0]
        output_shapes = graph.get_output_shapes(node)
        in_rank = len(input_shape)
        if in_rank <= 5:
            return
        check_if_6d_supportable(node, graph)
        # Currently only support the case of ReduceOp with len(axes) == 1
        log_assert(
            len(node.op.axes) == 1,
            f"Currently we do not support 6d tensor on ReduceOp with len(axes) > 1"
        )
        axis = node.op.axes[0]
        # new_input_shape
        left_shape = [] if axis == 0 else [np.prod(input_shape[:axis])]
        right_shape = [] if axis == (in_rank-1) else [np.prod(input_shape[axis+1:])]
        new_input_shape = left_shape + [input_shape[axis]] + right_shape

        # insert pre-reshape before ReduceOp
        pre_reshape_op_name = node.op.name + '_6d_pre_reshape'
        pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_op_name, shape=new_input_shape)
        graph.inject(
            pre_reshape_op, input_name=node.input_names[0],
            output_name=pre_reshape_op_name, consumer_names=[node.op.name]
        )

        # replace ReduceOp
        new_reduce_op = op_adapter.ReduceOp(
            name=node.op.name,
            axes=[0] if axis == 0 else [1],
            keep_dims=node.op.keep_dims,
            reduce_type=node.op.reduce_type
        )
        new_reduce_op_output_shape = left_shape + [1] + right_shape if node.op.keep_dims else left_shape + right_shape
        graph.replace(node.op, new_reduce_op)

        # insert post-reshape after ReduceOp
        post_reshape_insertion(
            node, graph,
            new_out_shapes=[new_reduce_op_output_shape],
            orig_out_shapes=output_shapes
        )

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            target_format = spatial_first_format_to_channel_first_format[input_buf.axis_format]
            permute_order = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            # If keep dims = 0 we must permute as it will remove dimensions
            if not node.op.keep_dims:
                if output_buf.axis_format != AxisTracker.AxisFormat.NC:
                    graph.inject_implicit_permute(input_name, target_format,
                                                  permute_order, [node.op.name])
                    output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL
                else:
                    axis_map = permute_order
                    node.op.axes = [axis_map[axis] for axis in node.op.axes]
            else:
                input_axis_formats_before = graph.get_input_axis_formats(node)
                AxisTracker.alter_axis_format_to_ir_order(node, graph)
                input_axis_formats_after = graph.get_input_axis_formats(node)
                input_buffers = graph.get_input_buffers(node)
                for i, buf in enumerate(input_buffers):
                    if input_axis_formats_before[i] != input_axis_formats_after[i]:
                        transpose_node = buf.producer
                        graph.update_trace_info(transpose_node, [node])
                        graph.update_trace_info(buf, [node])
                axis_map = permute_order
                node.op.axes = [axis_map[axis] for axis in node.op.axes]

        return True


@register_layer_optimization
class OptimizeReduceMaxTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MAX])


@register_layer_optimization
class OptimizeReduceMeanTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MEAN])


@register_layer_optimization
class OptimizeReduceMinTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_MIN])


@register_layer_optimization
class OptimizeReduceProdTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_PROD])


@register_layer_optimization
class OptimizeReduceSumTranslation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.QNN_OP_REDUCE_SUM])


@register_layer_optimization
class OptimizeReduceL2Translation(OptimizeReduceTranslationBase):
    def __init__(self):
        OptimizeReduceTranslationBase.__init__(self, op_adapter.ReduceOp.ir_to_legacy_type[ir_graph.IR_OP_REDUCE_L2])
        self.register_method(EXPAND_REDUCE_L2_OP_STRUCTURE, self.expand_reduce_l2_op_structure)

    def expand_reduce_l2_op_structure(self, node, graph):
        reduce_l2_op_name = node.op.name
        input_names = node.input_names
        output_names = node.output_names
        axes = getattr(node.op, ir_graph.IR_OP_REDUCE_L2_PARAM_AXES)
        keep_dims = getattr(node.op, ir_graph.IR_OP_REDUCE_L2_PARAM_KEEP_DIMS)

        # Get the index of the reduce_l2 node
        idx = graph.list_nodes().index(node)

        # Storing the consumers of the reduce_l2 node before pruning it
        consumers = graph.get_buffer(output_names[0]).consumers.copy()
        # Get previous input_name idx for output_buffers[0]
        output_consumers_dict = {}
        for consumer in consumers:
            buf_idx = consumer.input_names.index(output_names[0])
            output_consumers_dict[consumer] = buf_idx

        # Pruning the node
        graph.prune(node, force_remove=True)

        # Generate ReduceSumSquare Op
        mul_op = op_adapter.ElementwiseBinaryOp(reduce_l2_op_name + "_mul",
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
        reduce_sum_op = op_adapter.ReduceOp(reduce_l2_op_name + "_reduced_sum",
                                            reduce_type=ir_graph.QNN_OP_REDUCE_SUM,
                                            axes=axes,
                                            keep_dims=keep_dims)

        # Generate ElementWiseSquareRoot Op
        sqrt_op_name = reduce_l2_op_name + '_sqrt'
        sqrt_op = op_adapter.ElementwiseUnaryOp(sqrt_op_name,
                                                operation=ir_graph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT)

        # Add multiply, reduce_sum, sqrt op to the graph and update trace info for new created node
        mul_output_name = mul_op.name
        mul_input_names = [input_names[0], input_names[0]]
        mul_node = graph.add(mul_op, mul_input_names, [mul_output_name], idx=idx)
        graph.update_trace_info(mul_node, [node])

        idx += 1
        reduce_sum_output_name = reduce_sum_op.name
        reduce_sum_node = graph.add(reduce_sum_op, [mul_output_name], [reduce_sum_output_name], idx=idx)
        graph.update_trace_info(reduce_sum_node, [node])

        idx += 1
        sqrt_node = graph.add(sqrt_op, [reduce_sum_output_name], output_names, idx=idx)
        graph.update_trace_info(sqrt_node, [node])

        if len(consumers) > 0:
            # Restore the input_name for following nodes
            for consumer in output_consumers_dict.keys():
                buf_idx = output_consumers_dict[consumer]
                consumer.input_names.insert(buf_idx, output_names[0])
            graph.get_buffer(output_names[0]).consumers = consumers

        # Transfer output encodings of ReduceL2 node to the last node.
        if graph.has_quantization_param(node.op.name):
            output_encodings = graph.get_layer_quantization_param(node.op.name)[op_graph.QuantParams.OUTPUT_ENCODINGS]
            if len(output_encodings) > 0:
                output_encoding = output_encodings[0].copy()
                output_encoding['name'] = output_names[0]
                graph.add_quantization_params(sqrt_op_name, output_encodings=output_encoding)


@register_layer_optimization
class OptimizeReshapeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ReshapeOp.TRANSLATION_KEY
        self.register_method(MATCH_CHANNELSHUFFLE, self.match_channelshuffle)
        self.register_method(REMOVE_IDENTITY, self.remove_identity)
        self.register_method(SQUASH_RESHAPE, self.squash_reshape)
        self.register_method(FOLD_RESHAPES, self.fold_reshapes)
        self.register_method(ADD_TRANSPOSE_AFTER_OUTPUT_RESHAPE, self.add_transpose_after_output_reshape)
        self.register_method(SQUASH_TRANSPOSE_RESHAPE, self.squash_transpose_reshape)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if AxisTracker.input_axis_formats_intact(graph, node) and \
                input_buf.axis_format in AxisTracker.AxisFormat.get_valid_formats():
            return False

        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.TransposeOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NC or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.TNF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

            return True

    @staticmethod
    def add_transpose_after_output_reshape(node, graph):
        """
        when output of graph is reshape, we need to add a transpoe after reshape so graph output axis format is consistent with graph input axis format
        e.g.,
                    op                                                         op (Channel_last)
                    | (Channel_first)                                          | (Channel_last)
                 reshape (Channel_first)      axes_to_spatial_first_orde   transpose (Channel_last)
                    | (Channel_first) (output)    ---------------->            | (Channel_first)
                                                                            reshape (Channel_first)
                                                                               | (Channel_first)
                                                                           transpose (Channel_first) <- should add this node so it has consistent axis format
                                                                               | (Channel_last) (output)
        """
        output_buf = graph.get_output_buffers(node)[0]
        input_node = graph.get_input_buffers(node)[0].producer
        axis_formats = [output_buf.axis_format]
        if len(output_buf.consumers) == 0:
            perm = None
            if output_buf.axis_format == AxisTracker.AxisFormat.NCF:
                perm = AxisTracker.AxisFormat.NCF_TO_NFC
            elif output_buf.axis_format == AxisTracker.AxisFormat.NCS:
                perm = AxisTracker.AxisFormat.NCS_TO_NSC
            elif output_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                perm = AxisTracker.AxisFormat.NCDHW_TO_NDHWC
            elif output_buf.axis_format == AxisTracker.AxisFormat.TNF:
                perm = AxisTracker.AxisFormat.TNF_TO_NTF
            if perm is not None:
                transpose_op_name = node.op.name + '_transpose'
                transpose_op = op_adapter.TransposeOp(transpose_op_name, perm=perm)
                post_reshape_idx = graph.nodes_in_order.index(node)
                # first prune and add back to adjust its output name
                node_trace_info = graph.get_trace_info(node)
                graph.prune(node)
                post_reshape_output_name = node.output_names[0]
                new_post_reshape_output_name = input_node.output_names[0] + "." + output_buf.axis_format.lower()
                post_reshape_op = node.op
                node = graph.add(post_reshape_op, input_node.output_names, [new_post_reshape_output_name], idx=post_reshape_idx, axis_formats=axis_formats)
                graph.set_trace_info(node, node_trace_info)
                graph.set_trace_info(graph.get_buffer(new_post_reshape_output_name), node_trace_info)
                transpose_node = graph.add(transpose_op, node.output_names, [post_reshape_output_name], idx=post_reshape_idx+1)
                graph.set_trace_info(transpose_node, node_trace_info)
                graph.set_trace_info(graph.get_buffer(post_reshape_output_name), node_trace_info)

    @staticmethod
    def match_channelshuffle(graph):
        def is_valid_channelshuffle(nodes_tuple):
            # *********************** For NCHW layout ***********************
            # Reshape_1:
                # input shape [N, C, H, W]
                # Output shape should be [N, G, C/G, H, W]
            # Transpose:
                # perm = [0, 2, 1, 3, 4]
            # Reshape_2:
                # input shape [N, C/G, G, H, W]
                # Output shape should be [N, C, H, W]
            #  *********************** For NHWC layout ***********************
            # Reshape_1:
                # input shape [N, H, W, C]
                # Output shape should be [N, H, W, G, C/G]
            # Transpose:
                # perm = [0, 1, 2, 4, 3]
            # Reshape_2:
                # input shape [N, H, W, C/G, G]
                # Output shape should be [N, H, W, C]

            def check_for_nchw_reshape_1(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_1_input_shape = input_buffer.shape
                reshape_1_output_shape = output_buffer.shape

                return (len(reshape_1_input_shape) == 4 and len(reshape_1_output_shape) == 5 and
                        reshape_1_input_shape[0] == reshape_1_output_shape[0] and
                        reshape_1_input_shape[2] == reshape_1_output_shape[3] and
                        reshape_1_input_shape[3] == reshape_1_output_shape[4])

            def check_for_nhwc_reshape_1(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_1_input_shape = input_buffer.shape
                reshape_1_output_shape = output_buffer.shape

                return (len(reshape_1_input_shape) == 4 and len(reshape_1_output_shape) == 5 and
                        reshape_1_input_shape[0] == reshape_1_output_shape[0] and
                        reshape_1_input_shape[1] == reshape_1_output_shape[1] and
                        reshape_1_input_shape[2] == reshape_1_output_shape[2])

            def check_for_nchw_permute(node):
                return node.op.type == op_adapter.TransposeOp.TRANSLATION_KEY and node.op.perm == [0, 2, 1, 3, 4]

            def check_for_nhwc_permute(node):
                return node.op.type == op_adapter.TransposeOp.TRANSLATION_KEY and node.op.perm == [0, 1, 2, 4, 3]

            def check_for_nchw_reshape_2(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_2_input_shape = input_buffer.shape
                reshape_2_output_shape = output_buffer.shape

                return (len(reshape_2_input_shape) == 5 and len(reshape_2_output_shape) == 4 and
                            reshape_2_input_shape[0] == reshape_2_output_shape[0] and
                            reshape_2_input_shape[3] == reshape_2_output_shape[2] and
                            reshape_2_input_shape[4] == reshape_2_output_shape[3])

            def check_for_nhwc_reshape_2(node):
                input_buffer = graph.get_input_buffers(node)[0]
                output_buffer = graph.get_output_buffers(node)[0]
                reshape_2_input_shape = input_buffer.shape
                reshape_2_output_shape = output_buffer.shape

                return (len(reshape_2_input_shape) == 5 and len(reshape_2_output_shape) == 4 and
                            reshape_2_input_shape[0] == reshape_2_output_shape[0] and
                            reshape_2_input_shape[1] == reshape_2_output_shape[1] and
                            reshape_2_input_shape[2] == reshape_2_output_shape[2])

            first_, second_, third_ = nodes_tuple
            input_shape_ = graph.get_input_buffers(first_)[0].shape
            output_shape_ = graph.get_output_buffers(third_)[0].shape

            return ((output_shape_ == input_shape_)
                    and
                    ((check_for_nchw_reshape_1(first_) and
                    check_for_nchw_permute(second_) and
                    check_for_nchw_reshape_2(third_))
                    or
                    (check_for_nhwc_reshape_1(first_) and
                    check_for_nhwc_permute(second_) and
                    check_for_nhwc_reshape_2(third_))))

        sequence = [
                    ("Reshape",
                        (),
                        ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
                     ),
                    ("Transpose",
                        (),
                        ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
                     ),
                    ("Reshape",
                        (),
                        ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_channelshuffle, ignore_constants=True)

        for node_tuple in matched_node_list:
            #  ChannelShuffle Op found, Squash Permute and 2nd Reshape Op and Replace 1st ReshapeOp with ShuffleOp
            first, second, third = node_tuple
            output_shape = graph.get_output_shapes(first)[0]

            # The shape may be N[GC']HW or NHW[GC']
            groups = output_shape[1] if second.op.perm == [0, 2, 1, 3, 4] else output_shape[3]
            axis = 1 if second.op.perm == [0, 2, 1, 3, 4] else 3

            third_input_buffer = graph.get_input_buffers(third)[0]
            graph.squash(third, input_name=third_input_buffer.name)

            second_input_buffer = graph.get_input_buffers(second)[0]
            graph.squash(second, input_name=second_input_buffer.name)

            shuffle_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.ChannelShuffleOp.TRANSLATION_KEY,
                                                                      op_adapter.ChannelShuffleOp.LEGACY_TRANSLATION_KEY,
                                                                      folded_op=True)

            shuffle_op = op_adapter.ChannelShuffleOp(shuffle_op_name, num_groups=groups, axis=axis)

            graph.replace(first.op, shuffle_op)
            shuffle_op.data_axis_formats = [AxisTracker.AxisFormat.NCS] if second.op.perm == [0, 2, 1, 3, 4] else \
                [AxisTracker.AxisFormat.NSC]

            log_debug2(code_to_message.get_debugging_message("DEBUG_CHANNEL_SHUFFLE_REPLACE")(first.op.name,
                                                                                              second.op.name,
                                                                                              third.op.name,
                                                                                              shuffle_op.name))

    @staticmethod
    def remove_identity(node, graph):
        input_buffer = graph.get_input_buffers(node)[0]
        output_buffer = graph.get_output_buffers(node)[0]
        consumers = list(graph.get_buffer(node.output_names[0]).consumers)
        ret = False
        # Remove reshape if same shape as input as this reshape has no effect, remove it
        if input_buffer.shape == output_buffer.shape and len(input_buffer.consumers) == 1:
            try:
                status = graph.squash_identity(node, is_data_movement_node=True)
                if status:
                    log_debug("Squash Reshape op {} due to identity. "
                              "Input shape {}, shape after {}".format(node.op.name,
                                                                      input_buffer.shape,
                                                                      output_buffer.shape))
            except RuntimeError as e:
                log_debug("Squash Reshape op {} due to identity not possible ".format(node.op.name))
        # Remove reshape  if the batch dimension is maintained through the reshape when consumer of reshape is
        # fc layer
        elif len(consumers) == 1 and isinstance(consumers[0].op, op_adapter.FullyConnectedOp) and \
                 input_buffer.shape[0] == output_buffer.shape[0]:
            try:
                status = graph.squash(node, input_name=input_buffer.name, squash_into_next=True, is_data_movement_node=True)
                if status:
                    log_debug("Squash Reshape op {} due to identity. "
                              "Input shape {}, shape after {}".format(node.op.name,
                                                                      input_buffer.shape,
                                                                      output_buffer.shape))
            except RuntimeError as e:
                log_debug("Squash Reshape op {} due to identity not possible ".format(node.op.name))

    @staticmethod
    def squash_reshape(graph):
        def validate_node(nodes_tuple):
            input_buffer = graph.get_buffer(nodes_tuple[0].input_names[0])
            return len(input_buffer.consumers) == 1

        def squash_reshape_into_constant(graph, node):
            constant_buffer = graph.get_buffer(node.input_names[0])

            const_tensor = constant_buffer.producer.op.tensor
            const_tensor_shape = graph.get_output_shapes(node)[0]
            const_tensor = np.reshape(const_tensor, const_tensor_shape)

            constant_buffer.producer.op.tensor = const_tensor
            constant_buffer.shape = const_tensor_shape

            log_debug("Squashed {} node {} into constant node {}"
                       .format(node.op.type, node.op.name, constant_buffer.name))
            graph.squash(node, input_name=constant_buffer.name)

        sequence = [
            ("Reshape",
                        ("MATCH_BUFS_AT_INDEX", [("constant", 0)]),
                        ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            squash_reshape_into_constant(graph, node_tuple[0])

    @staticmethod
    def fold_reshapes(graph):
        def validate_node(nodes_tuple):
            input_buffer = graph.get_buffer(nodes_tuple[0].input_names[0])
            #if input bufferis one of the graph outputs then we skip folding
            return len(input_buffer.consumers) == 1 and input_buffer.name not in graph.output_names

        sequence = [
                    ("Reshape",
                     ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                     ()
                     )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            reshape_node = node_tuple[0]
            reshape_node_input_buf = graph.get_input_buffers(reshape_node)[0]
            reshape_node_input_names = reshape_node.input_names

            if reshape_node_input_buf.producer.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY:
                prev_reshape_node = reshape_node_input_buf.producer
                prev_reshape_node_input_names = prev_reshape_node.input_names

                # update shape attribute
                prev_reshape_node.op.shape = ir_graph.IrStaticTensor(
                    ir_graph.IR_OP_RESHAPE_PARAM_SHAPE,
                    [len(reshape_node.op.shape)],
                    reshape_node.op.shape,
                    ir_graph.QNN_DATATYPE_INT_32
                )
                # squash next reshape node into previous
                status = graph.squash(reshape_node, reshape_node_input_names[0], squash_into_next=False, is_data_movement_node=True)
                if status:
                    log_debug2("Folded Reshape:{} into Reshape:{}".format(reshape_node.op.name, prev_reshape_node.op.name))

    @staticmethod
    def squash_transpose_reshape(graph):
        """ When there are Transpose and Reshape with data on single dimension and with same shape around sequence,
            they are redundant and can be removed. Any other constant nodes in between them can be ignored.
            This function handles both Transpose-Reshape and Reshape-Transpose Op sequences."""
        def validate(nodes_tuple):
            def data_on_single_dimension(shape):
                return len([s for s in shape if s != 1]) == 1
            input_buffer = graph.get_buffer(nodes_tuple[0].input_names[0])
            output_buffer = graph.get_buffer(nodes_tuple[1].output_names[0])
            return (input_buffer.shape == output_buffer.shape and
                    data_on_single_dimension(input_buffer.shape) and
                    data_on_single_dimension(output_buffer.shape))

        sequence = [[
                     ("Transpose",
                        (),
                         ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
                      ),
                     ("Reshape",
                         ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
                         ()
                      )],
                     [("Reshape",
                         (),
                         ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
                      ),
                     ("Transpose",
                         ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
                         ()
                      )
                    ]]

        for seq in sequence:
            matched_node_list = graph.get_matched_nodes(seq, validator=validate, ignore_constants=True)
            for node_tuple in matched_node_list:
                first_node, second_node = node_tuple
                graph.squash(second_node, first_node.output_names[0])
                graph.squash(first_node, first_node.input_names[0])

    def replace_6d_operation(self, node, graph):
        """ Replace 6D Reshape by cloning it for each consumer
            so that fold_reshapes can then take effect to remove 6d tensors."""
        in_shapes = graph.get_input_shapes(node)
        out_shapes = graph.get_output_shapes(node)
        # Reshape has only one input and one output
        if len(in_shapes[0]) <= 5 and len(out_shapes[0]) <= 5:
            return
        check_if_6d_supportable(node, graph)
        # TODO: Handle 6d inputs for Reshape
        if len(out_shapes[0]) >= 6:
            # Handle 6d outputs for Reshape: clone the ReshapeOp for each consumer
            in_buf = graph.get_input_buffers(node)[0]
            out_buf = graph.get_output_buffers(node)[0]
            consumers = out_buf.consumers.copy()
            orig_axis_format = out_buf.axis_format
            # Step 1: squash the original ReshapeOp
            graph.squash(
                node, input_name=node.input_names[0], squash_into_next=True, is_data_movement_node=True
            )
            # Step 2: inject the cloned ReshapeOp before each consumer
            for idx, consumer in enumerate(consumers):
                clone_reshape_op_name = node.op.name + f'_6d_clone{idx}'
                clone_reshape_op = op_adapter.ReshapeOp(
                    name=clone_reshape_op_name, shape=node.op.shape
                )
                graph.inject(
                    clone_reshape_op, input_name=in_buf.name,
                    output_name=clone_reshape_op_name, consumer_names=[consumer.op.name],
                    axis_format=orig_axis_format
                )

@register_layer_optimization
class OptimizeResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ResizeOp.TRANSLATION_KEY
        self.register_method(expand_1d_spatial_nn_nodes, self.expand_1d_spatial_nn_nodes)

    def axes_to_spatial_first_order(self, node, graph):
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        return True

    def expand_1d_spatial_nn_nodes(self, node, graph):
        input_shape = graph.get_input_buffers(node)[0].shape
        if len(input_shape) == 3:
            optimize_1dnn_translation = Optimize1DNNTranslation()
            optimize_1dnn_translation.nn_2d_op = op_adapter.ResizeOp(node.op.name + "_2d",
                                                                     exclude_outside=node.op.exclude_outside,
                                                                     transformation_mode=node.op.transformation_mode,
                                                                     interpolation_mode=node.op.interpolation_mode,
                                                                     nearest_mode=node.op.nearest_mode,
                                                                     cubic_coeff=node.op.cubic_coeff,
                                                                     scale_depth=node.op.scale_depth,
                                                                     scale_height=node.op.scale_height,
                                                                     scale_width=node.op.scale_width)

            optimize_1dnn_translation.expand_1d_spatial_nn_nodes(node, graph)

    def merge_low_level_ops_to_layers(self, graph):
        # Match following pattern to resize (upsampling) op:
        #     input:    3-5d    [1, 256, 13, 13]        (in this case 4d)
        #     reshape:    nd    [1, 256, 13, 1, 13, 1]  (larger than input rank, in this case 6d)
        #     expand:     nd    [1, 256, 13, 2, 13, 2]  (larger than input rank, in this case 6d)
        #     reshape:  3-5d    [1, 256, 26, 26]        (in this case 4d)
        # In the section below, rename 3-5d to supported_rank

        def validate(nodes_tuple_):
            def check_shape_at_n_c(shape_1, shape_2, index_n_c):
                for idx in index_n_c:
                    if shape_1[idx] != shape_2[idx]:
                        return False
                return True

            def check_reshape_between_supported_rank_and_nd(shape_supported_rank, shape_nd, index_n_c):
                if not check_shape_at_n_c(shape_supported_rank, shape_nd, index_n_c):
                    return False
                if np.prod(shape_supported_rank) != np.prod(shape_nd):
                    return False
                return True

            reshape_to_nd_node = nodes_tuple_[0]
            reshape_to_supported_rank_node = nodes_tuple_[2]

            reshape_nd_input_shape = graph.get_input_shapes(reshape_to_nd_node)[0]
            reshape_nd_output_shape = graph.get_output_shapes(reshape_to_nd_node)[0]
            reshape_supported_rank_input_shape = graph.get_input_shapes(reshape_to_supported_rank_node)[0]
            reshape_supported_rank_output_shape = graph.get_output_shapes(reshape_to_supported_rank_node)[0]

            index_n_c = [0, 1] if isinstance(graph.src_axis_order, SpatialLastAxisOrder) else [0, -1]

            # The length of input and output shape should be equal and 3-5d.
            # The final output shape should not be equal to the input shape,
            # which means it should not be an identity op.
            if len(reshape_nd_input_shape) not in [3, 4, 5] or len(reshape_supported_rank_output_shape) not in [3, 4, 5] \
                    or len(reshape_nd_input_shape) != len(reshape_supported_rank_output_shape) \
                    or not check_shape_at_n_c(reshape_nd_input_shape, reshape_supported_rank_output_shape, index_n_c) \
                    or reshape_nd_input_shape == reshape_supported_rank_output_shape:
                return False

            # Check the expansion from supported_rank to nd (larger than input rank) and
            # Check reshape from nd to supported_rank
            if not check_reshape_between_supported_rank_and_nd(reshape_nd_input_shape, reshape_nd_output_shape, index_n_c) \
                    or not check_shape_at_n_c(reshape_nd_output_shape, reshape_supported_rank_input_shape, index_n_c) \
                    or not check_reshape_between_supported_rank_and_nd(reshape_supported_rank_output_shape,
                                                                       reshape_supported_rank_input_shape, index_n_c):
                return False

            return True

        sequence = [
                    (ir_graph.QNN_OP_RESHAPE,
                        (),
                        (("MATCH_NUM_BUFS", [(ir_graph.IR_OP_EXPAND, "ALL")]))
                    ),
                    (ir_graph.IR_OP_EXPAND,
                        ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
                        ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")])
                    ),
                    (ir_graph.QNN_OP_RESHAPE,
                        ("MATCH_NUM_BUFS", [(ir_graph.IR_OP_EXPAND, "ALL")]),
                        ()
                    )
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            reshape_to_nd_node = nodes_tuple[0]
            expand_node = nodes_tuple[1]
            reshape_to_supported_rank_node = nodes_tuple[2]

            reshape_nd_input_shape = graph.get_input_shapes(reshape_to_nd_node)[0]
            reshape_supported_rank_output_shape = graph.get_output_shapes(reshape_to_supported_rank_node)[0]

            # Squash the reshape_supported_rank node
            reshape_supported_rank_input_buffer = graph.get_input_buffers(reshape_to_supported_rank_node)[0]
            graph.squash(reshape_to_supported_rank_node, input_name=reshape_supported_rank_input_buffer.name)

            # Squash the expand node
            expand_input_buffer = graph.get_input_buffers(expand_node)[0]
            graph.squash(expand_node, input_name=expand_input_buffer.name)

            # Replace the reshapeND OpNode to a Resize OpNode
            # - calculate the scales of resize first
            if isinstance(graph.src_axis_order, SpatialLastAxisOrder):
                # NCDHW
                index_w, index_h, index_d = -1, -2, -3
            else:
                # NDHWC
                index_w, index_h, index_d = -2, -3, -4
            scale_w = int(reshape_supported_rank_output_shape[index_w]) / int(reshape_nd_input_shape[index_w])
            scale_h = int(reshape_supported_rank_output_shape[index_h]) / int(reshape_nd_input_shape[index_h]) \
                        if len(reshape_nd_input_shape) in [4, 5] else None
            scale_d = int(reshape_supported_rank_output_shape[index_d]) / int(reshape_nd_input_shape[index_d]) \
                        if len(reshape_nd_input_shape) in [5] else None
            # Since expand op is used to broadcast the input tensor,
            # the replaced operator should have upsampling behavior rather than downsampling behavior.
            if (scale_w and scale_w < 1) or (scale_h and scale_h < 1) or (scale_d and scale_d < 1):
                raise ValueError("scale_w, scale_h and scale_d should be greater than or equal to 1 to have "
                                 "upsampling behavior, but got scale_w: {}, scale_h: {} and scale_d: {}"
                                 .format(scale_w, scale_h, scale_d))

            resize_op = op_adapter.ResizeOp(expand_node.op.name + '_resize',
                                            # should always be nearest in this case
                                            interpolation_mode=ir_graph.QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST,
                                            scale_depth=scale_d,
                                            scale_height=scale_h,
                                            scale_width=scale_w)
            graph.replace(reshape_to_nd_node.op, resize_op)


@register_layer_optimization
class OptimizeRMSNormTranslation(OptimizeNormTranslationBase):
    def __init__(self):
        OptimizeNormTranslationBase.__init__(self)
        self.op_type = op_adapter.RMSNormOp.TRANSLATION_KEY
        self.register_method(MATCH_RMSNORM, self.match_rms_norm)

    def is_already_squeezed(self, input_name, graph, node_tuple):
        buffer = graph.get_buffer(input_name)
        main_input_buffer = graph.get_input_buffers(node_tuple[1])[0]
        input_rank = main_input_buffer.rank()
        return buffer.rank() < input_rank

    def validate_epsilon(self, node_tuple, graph):
        for node in node_tuple:
            if node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
                # Check this is the add node providing epsilon and not the one providing beta
                if isinstance(graph.get_producer_node(node.input_names[0]).op, op_adapter.ReduceOp):
                    constant_node = self.get_constant_input_node(node, graph)
                    if constant_node.op.tensor.size > 1:
                        # If tensor has more than one elements
                        tensor = constant_node.op.tensor.flatten()
                        # Check that all the elements are same
                        if not np.all(tensor == tensor[0]):
                            return False
        return True


    def match_rms_norm(self, graph):
        # --------------------- Sequences to be matched -----------------------
        sequence1 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_sqrt", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence2 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_sqrt", "ANY")]),
             ()
             )
        ]

        sequence3 = [
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_product", "ANY")]),
             ()
             )
        ]

        sequence4 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_sqrt", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence5 = [
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("elementwise_product", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence6 = [
            ("Transpose",
             (),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")]),
             ),
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
             ),
            ("Transpose",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_sqrt", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("Transpose", "ANY"), ("Transpose", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ()
             )
        ]

        sequence7 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum","ANY")]),
            ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
            ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
            ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"), ("constant", "ANY")]),
             ()
             )
       ]

        sequence8 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum","ANY")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY"), ("cast", "ANY")]),
             ("MATCH_NUM_BUFS", [("cast", "ALL")]),
             ),
            ("cast",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("cast", "ANY"), ("constant", "ANY")]),
             ()
             )
        ]

        sequence9 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [("elementwise_sum","ANY")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_unary_sqrt", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ),
            ("elementwise_product",
             ("MATCH_NUM_BUFS", [("elementwise_div", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ANY"), ("constant", "ANY")]),
             (),
             )
        ]

        sequence10 = [
            ("elementwise_product",
             (),
             ("MATCH_NUM_BUFS", [("reduce_mean", "ALL")])
             ),
            ("reduce_mean",
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ),
            ("elementwise_sum",
             ("MATCH_NUM_BUFS", [("reduce_mean", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ALL")])
             ),
            ("elementwise_unary_sqrt",
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [("elementwise_div", "ALL")])
             ),
            ("elementwise_div",
             ("MATCH_NUM_BUFS", [("elementwise_unary_sqrt", "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")])
             ),
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("elementwise_div", "ANY")]),
             ()
             )
        ]

        def get_epsilon(node_tuple):
            for node in node_tuple:
                if node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
                    constant_node = self.get_constant_input_node(node, graph)
                    if constant_node.op.tensor.size == 1:
                        epsilon = constant_node.op.tensor[0]
                        return epsilon
                    elif constant_node.op.tensor.size > 1:
                        # If tensor has more than one elements
                        epsilon = constant_node.op.tensor.flatten()[0]
                        return epsilon
            return op_adapter.RMSNormOp.EPSILON

        def make_rms_norm_op(node_tuple):
            rms_norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.RMSNormOp.type,
                                                                       op_adapter.RMSNormOp.LEGACY_TRANSLATION_KEY)
            axes = self.get_axes(node_tuple)
            epsilon = get_epsilon(node_tuple)
            rms_norm_op = op_adapter.RMSNormOp(name=rms_norm_op_name,
                                               axes=axes,
                                               epsilon=epsilon)
            return rms_norm_op

        def get_rms_norm_input_names(node_tuple):
            first_node = node_tuple[0]
            mul_node = node_tuple[-2]
            sum_node = node_tuple[-1]

            main_input_name = first_node.input_names[0]
            beta_input_name = self.get_affine_scalar_input(sum_node, node_tuple, graph)
            gamma_input_name = self.get_affine_scalar_input(mul_node, node_tuple, graph)

            rms_norm_op_name = graph.naming_policy.get_op_name_by_type(op_adapter.RMSNormOp.type,
                                                                       op_adapter.RMSNormOp.LEGACY_TRANSLATION_KEY)

            input_buf = graph.get_input_buffers(node_tuple[0])[0]
            axes = self.get_axes(node_tuple)
            input_shapes_axes = [input_buf.shape.dims[i] for i in axes]

            beta_input_name = self.broadcast_axes(beta_input_name, 0, input_shapes_axes, graph, input_buf,
                                                  rms_norm_op_name, node_tuple)
            gamma_input_name = self.broadcast_axes(gamma_input_name, 1, input_shapes_axes, graph, input_buf,
                                                   rms_norm_op_name, node_tuple)

            rms_norm_input_names = [main_input_name, gamma_input_name, beta_input_name]
            return rms_norm_input_names

        def match_base_rms_norm(node_tuple, origin_node_tuple=None):
            last_node = node_tuple[-1]
            last_node_consumers = set(graph.get_op_output_nodes(last_node))

            rms_norm_op = make_rms_norm_op(node_tuple)
            rms_norm_input_names = get_rms_norm_input_names(node_tuple)
            rms_norm_output_names = last_node.output_names
            # Record trace info before prune nodes
            if not origin_node_tuple:
                origin_node_tuple = node_tuple
            node_tuple_trace_info = graph.get_trace_info_sub_graph(origin_node_tuple,
                                                                   graph.get_output_buffers(last_node))
            output_trace_info = graph.get_trace_info(graph.get_buffer(rms_norm_output_names[0]))
            self.prune_nodes(node_tuple, graph)
            insertion_index = self.get_norm_op_insertion_index(rms_norm_input_names, graph)

            rms_norm_node = graph.add(op=rms_norm_op,
                                      input_names=rms_norm_input_names,
                                      output_names=rms_norm_output_names,
                                      idx=insertion_index)
            graph.set_trace_info(rms_norm_node, node_tuple_trace_info)
            graph.set_trace_info(graph.get_buffer(rms_norm_output_names[0]),output_trace_info)

            self.update_norm_op_output_buffer(rms_norm_node, last_node_consumers, graph)
            graph.replace_quantization_param(last_node.op.name, rms_norm_node.op.name)
            self.update_consumer_data_axis_formats(graph, rms_norm_node)

            return rms_norm_node

        def match_no_beta(node_tuple, beta_tensor=None, origin_node_tuple=None):
            mul_node = node_tuple[-1]
            gamma_buffer_name = self.get_affine_scalar_input(mul_node, node_tuple, graph)
            gamma_buffer = graph.get_buffer(gamma_buffer_name)
            gamma_node = gamma_buffer.producer

            bias_name = mul_node.op.name + "_bias"
            tensor_dimensions = self.get_weight_bias_dimensions(node_tuple, graph)
            if beta_tensor is not None:
                bias_tensor = beta_tensor
            else:
                bias_tensor = np.zeros(tensor_dimensions)
            bias_op = op_adapter.ConstantOp(bias_name, tensor=bias_tensor)
            bias_index = graph.list_nodes().index(gamma_node) + 1
            bias_node = graph.add(bias_op,
                                  [],
                                  [bias_name],
                                  axis_formats=[AxisTracker.AxisFormat.ANY],
                                  idx=bias_index)
            # update trace info of new added bias op and output
            if not origin_node_tuple:
                origin_node_tuple = node_tuple
            graph.set_trace_info([bias_node, graph.get_buffer(bias_name)],
                                 graph.get_trace_info_sub_graph(origin_node_tuple))

            add_name = mul_node.op.name + "_add"
            add_op = op_adapter.ElementwiseBinaryOp(name=add_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
            add_input_names = bias_node.output_names
            add_output_names = mul_node.output_names
            # Copy the quant params from the mul node to the dummy add node
            if graph.has_quantization_param(mul_node.op.name):
                quant_param = graph.get_layer_quantization_param(mul_node.op.name)
                graph.add_quantization_params(add_op.name,
                                              output_encodings=quant_param['output_encodings'])
            dummy_add_node = op_graph.OpNode(add_op, add_input_names, add_output_names)

            new_tuple = node_tuple + (dummy_add_node,)
            return match_base_rms_norm(new_tuple, origin_node_tuple)

        def match_no_affine_transformation(node_tuple, gamma_tensor=None, beta_tensor=None):
            last_node = node_tuple[-1]

            weight_name = last_node.op.name + "_weight"
            tensor_dimensions = self.get_weight_bias_dimensions(node_tuple, graph)
            if gamma_tensor is not None:
                weight_tensor = gamma_tensor
            else:
                weight_tensor = np.ones(tensor_dimensions)
            weight_op = op_adapter.ConstantOp(weight_name, tensor=weight_tensor)
            weight_index = graph.list_nodes().index(last_node) + 1
            weight_node = graph.add(weight_op,
                                    [],
                                    [weight_name],
                                    axis_formats=[AxisTracker.AxisFormat.ANY],
                                    idx=weight_index)
            # update trace info of new added weight op and output
            graph.set_trace_info([weight_node, graph.get_buffer(weight_name)],
                                 graph.get_trace_info_sub_graph(node_tuple))

            mul_output_name = last_node.op.name + "_mul"
            mul_op = op_adapter.ElementwiseBinaryOp(name=mul_output_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
            mul_input_names = weight_node.output_names
            mul_output_names = last_node.output_names
            dummy_mul_node = op_graph.OpNode(mul_op, mul_input_names, mul_output_names)

            new_tuple = node_tuple + (dummy_mul_node,)
            return match_no_beta(new_tuple, beta_tensor, node_tuple)

        def match_rms_norm_with_transpose(node_tuple):
            def get_axes_after_permute(transpose_node, original_axes):
                perm = transpose_node.op.perm
                return [perm.index(original_axes)]

            # check if both transpose nodes at input of div have broadcastable shape
            transpose_1_node = node_tuple[0]
            transpose_2_node = node_tuple[-3]
            reduce_node = node_tuple[2]
            transpose_1_node_output_buffer = graph.get_output_buffers(transpose_1_node)[0]
            transpose_2_node_output_buffer = graph.get_output_buffers(transpose_2_node)[0]
            if not translation_utils.broadcastable(transpose_1_node_output_buffer.shape,
                                                   transpose_2_node_output_buffer.shape):
                return

            # Move Transpose at input of Div node to input of RMSNorm node
            square_node = node_tuple[1]
            graph.get_buffer(square_node.input_names[0]).consumers.remove(square_node)
            square_node.input_names[0] = transpose_1_node.output_names[0]
            square_node.input_names[1] = transpose_1_node.output_names[0]
            graph.get_buffer(square_node.input_names[0]).consumers.add(square_node)
            reduce_node.op.axes = get_axes_after_permute(transpose_1_node, reduce_node.op.axes)

            # remove transpose at input[1] of Div node
            graph.prune(transpose_2_node, force_remove=True)

            # remove both transposes from node_tuple;
            node_tuple = list(node_tuple)
            origin_node_tuple = copy.deepcopy(node_tuple)
            del node_tuple[-3], node_tuple[0]

            #  Now node_tuple contains only RMSNorm nodes
            return match_no_beta(tuple(node_tuple), origin_node_tuple=origin_node_tuple)

        def match_rms_with_gamma_before_div(node_tuple):
            gamma_node = node_tuple[0]
            gamma_tensor = None
            beta_tensor = None

            #Check if a Mul node having the gamma parameter already exist
            if (gamma_node.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY]
                    and node_tuple[1].op.type == gamma_node.op.type):
                gamma_input_buffer = graph.get_buffer(gamma_node.input_names[1])
                gamma_producer_op = gamma_input_buffer.producer.op
                gamma_tensor = gamma_producer_op.tensor

            #Check if a Add node having the beta parameter already exist
            if node_tuple[-1].op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
                beta_node = node_tuple[-1]
                beta_input_buffer = graph.get_buffer(beta_node.input_names[1])
                beta_producer_op = beta_input_buffer.producer.op
                beta_tensor = beta_producer_op.tensor

            return match_no_affine_transformation(node_tuple, gamma_tensor=gamma_tensor, beta_tensor=beta_tensor)

        matching_strategy_map = {
            match_base_rms_norm: [sequence9],
            match_rms_norm_with_transpose: [sequence6],
            match_no_beta: [sequence1, sequence7, sequence8],
            match_rms_with_gamma_before_div: [sequence3, sequence4, sequence5],
            match_no_affine_transformation: [sequence2, sequence10],
        }

        for matching_function, sequences in matching_strategy_map.items():
            sequences_in_descending_length = sorted(sequences,
                                                    key=len,
                                                    reverse=True)
            for sequence in sequences_in_descending_length:
                matched_node_list = graph.get_matched_nodes(sequence,
                                                            validator=self.validate_reduce_mean_ops,
                                                            ignore_constants=True,
                                                            use_dfs=True)

                for node_tuple in matched_node_list:
                   if self.validate_epsilon(node_tuple, graph):
                       matching_function(node_tuple)


@register_layer_optimization
class OptimizeRoiAlignTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiAlignOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == node.op.data_axis_formats[0] and \
                input_buf.axis_format != AxisTracker.AxisFormat.NCS:
            # No change
            return False

        AxisTracker.enforce_input_axis_format(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                              AxisTracker.AxisFormat.NCS_TO_NSC, valid_input_axis_formats=[AxisTracker.AxisFormat.NCS],
                                              consumers=[node.op.name])
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC
        return True


@register_layer_optimization
class OptimizeRoiPoolingTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RoiPoolingOp.TRANSLATION_KEY
        self.register_method("PREPROCESS_ROI_POOL_INPUTS", self.preprocess_roi_pool_inputs)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if input_buf.axis_format == node.op.data_axis_formats[0]:
            # No change
            return False

        AxisTracker.enforce_input_axis_format(graph, node.input_names[0], AxisTracker.AxisFormat.NSC,
                                              AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf = graph.get_output_buffers(node)[0]
        node.op.output_shape = output_buf.shape = AxisTracker.permute_shape(output_buf.shape,
                                                                            AxisTracker.AxisFormat.NCS_TO_NSC)
        output_buf.axis_format = AxisTracker.AxisFormat.NSC
        return True

    @staticmethod
    def preprocess_roi_pool_inputs(graph):
        def validate_node(nodes_tuple):
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])
            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY or len(roi_node.input_names) == 3:
                return True
            return False

        sequence = [(op_adapter.RoiPoolingOp.TRANSLATION_KEY, (), ())]

        matched_nodes_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for nodes_tuple in matched_nodes_list:
            roi_node = nodes_tuple[0]
            roi_buf = graph.get_buffer(roi_node.input_names[1])

            # Batch indices are embedded in the ROI input for some frameworks
            # as (batch_index, x1, y1, x2, y2....). In this case the ROI must be static
            # so that the batch index input can be extracted
            if roi_buf.producer.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                if roi_buf.shape[-1] == 5:
                    # QNN needs roi values to be separated from batch index
                    roi_values = roi_buf.producer.op.tensor
                    roi_values_no_batch = roi_values[:, 1:]

                    # Update ROI values in constant op to new values
                    roi_buf.producer.op.tensor = roi_values_no_batch

                    # Set batch indices to first sub-tensor of ROI values
                    batch_indices_name = roi_buf.name + "_batch_indices"
                    batch_indices = np.asarray(roi_values[:, 0], np.int32)

                    # Add a new constant op to capture batch indices

                    # constant op needs to be added before roi node
                    roi_idx = graph.nodes_in_order.index(roi_node)
                    const_node = graph.add(op_adapter.ConstantOp(batch_indices_name, batch_indices, quantizable=False), [],
                                           [batch_indices_name], idx=roi_idx)
                    graph.update_trace_info(const_node, roi_buf)
                    graph.update_trace_info(graph.get_buffer(const_node), roi_buf)

                    # add input name to roi node
                    roi_node.input_names.append(batch_indices_name)

                else:
                    raise ValueError("Expected 5 dimensions for static ROI buffer: {}, instead got {}"
                                     .format(roi_buf.name, roi_buf.shape[-1]))
            elif len(roi_node.input_names) != 3:
                raise AttributeError("Missing batch indices input. "
                                     "Expected 3 inputs for ROI operation instead got: {}"
                                     .format(len(roi_node.input_names)))


@register_layer_optimization
class OptimizeRolledLstmTranslation(OptimizationTranslationBase):
    # Index for QNN RolledLstmOp inputs
    DATA_IDX = 0
    LSTM_HIDDEN_IN_IDX = op_adapter.LstmOp.HIDDEN_IN_IDX
    LSTM_CELL_IN_IDX = op_adapter.LstmOp.CELL_IN_IDX
    # Output tensors at index HIDDEN_OUT_IDX and HIDDEN_ALL_OUT_IDX are effectively the same
    # for single timestep LstmOp in QNN.
    HIDDEN_ALL_OUT_IDX, CELL_OUT_IDX, HIDDEN_OUT_IDX = 0, 1, 2

    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.RolledLstmOp.TRANSLATION_KEY
        self.register_method(EXPAND_LSTM_OP_STRUCTURE, self.expand_lstm_op_structure)
        self.register_method(UNROLL_LSTM_TIME_STEPS, self.unroll_lstm_time_steps)
        self.register_method(MULTI_TIME_STEPS_LSTM, self.multi_time_steps_lstm)

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        output_bufs = graph.get_output_buffers(node)

        # Input Data Buffer can be in NTF/TNF format.
        in_buf = input_bufs[0]
        # If RolledLstm Op's input (X_t with TNF axis format) is also the model's input then the
        # input would have been converterd to NTF format after call to axes_spatial_first_order
        # in the OptimizeInputTranslation. Then time_major param should be False.
        if in_buf.axis_format == AxisTracker.AxisFormat.NTF:
            node.op.time_major = False
        elif in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
            graph.inject_implicit_permute(in_buf.name, AxisTracker.AxisFormat.NTF,
                                          AxisTracker.AxisFormat.TNF_TO_NTF)
            # Make sure that the time major param is False if the buffer is in NTF format.
            node.op.time_major = False

        # Check that h/c input buffers are NONTRIVIAL
        for data_axis_format, in_buf in zip(node.op.data_axis_formats[1:3], input_bufs[1:3]):
            if in_buf.type != op_graph.BufferType.NULL:
                # We would like to revert the axis format of h/c input buffers to NONTRIVIAL if it isn't.
                if in_buf.axis_format != AxisTracker.AxisFormat.NONTRIVIAL:
                    if in_buf.axis_format != data_axis_format and \
                            in_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
                        # Transpose the axis format to source's one
                        graph.inject_implicit_permute(
                            in_buf.name,
                            AxisTracker.AxisFormat.NONTRIVIAL,
                            spatial_first_format_to_channel_first_permute_order[in_buf.axis_format],
                            [node.op.name]
                        )
                    in_buf = graph.get_buffer(list(in_buf.consumers)[0].output_names[0])
                log_assert(in_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL,
                           "LSTM h/c input buffer {} needs to have format NONTRIVIAL, got {}",
                           in_buf,
                           in_buf.axis_format)

        # Set up LSTM outputs' axis formats
        # First output: NTF/TNF
        # Other outputs: NONTRIVIAL
        time_major_param = node.op.time_major
        for i, output_buf in enumerate(output_bufs):
            if i == 0:
                if time_major_param:
                    output_buf.axis_format = AxisTracker.AxisFormat.TNF
                    log_assert(input_bufs[0].axis_format == AxisTracker.AxisFormat.TNF,
                               "RolledLstm Op X_t input buffer {} needs to have format TNF, got {}", input_bufs[0],
                               input_bufs[0].axis_format)
                else:
                    output_buf.axis_format = AxisTracker.AxisFormat.NTF
                    output_buf.shape = AxisTracker.permute_shape(output_buf.shape, AxisTracker.AxisFormat.TNF_TO_NTF)
            else:
                output_buf.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        return True

    def get_dims(self, buffer_shape, time_major_param=False):
        # In case the previous node is a reshape node,
        # The reshape node will be squashed and the current input buffer is a 2D tensor.
        if len(buffer_shape) == 2:
            return [buffer_shape[0], 1, buffer_shape[1]]
        if time_major_param:
            return [buffer_shape[1], buffer_shape[0], buffer_shape[2]]
        else:
            return buffer_shape

    def align_to_source_output_names(self, graph, current_output_names, source_output_names, align_only_in_src_output=True):
        # If the current_output names are in graph.output_names, then we need to align
        if align_only_in_src_output:
            new_current, new_src = [], []
            for idx, name in enumerate(source_output_names):
                if name in graph.output_names:
                    new_current.append(current_output_names[idx])
                    new_src.append(name)
            current_output_names, source_output_names = new_current, new_src

        # Replace current name with source name for alignment
        for current_name, source_name in zip(current_output_names, source_output_names):
            # override encoding info update
            pre_node = graph.get_producer_node(current_name)
            if graph.has_quantization_param(pre_node.op.name):
                src_encodings = graph.quantization_params[pre_node.op.name]['output_encodings']
                for i in range(len(src_encodings)):
                    if (src_encodings[i]["name"] == current_name):
                        src_encodings[i]["name"] = source_name
                        break

            buf = graph.get_buffer(current_name)
            if source_name in graph.buffers:
                raise ValueError("Buffer {} already exists in graph, duplicate buffer name when replacing buffer {} with it".format(
                        source_name, current_name))

            # Update consumers input name
            for consumer in list(buf.consumers):
                # The consumer may have the same buffer as input twice
                consumer.input_names = [source_name if name == current_name else name for name in consumer.input_names]

            # Update producer output name
            producer_node = graph.get_producer_node(current_name)
            idx = producer_node.output_names.index(current_name)
            producer_node.output_names[idx] = source_name

            # Update buffer in graph
            buf.name = source_name
            graph.buffers[source_name] = graph.buffers.pop(current_name)

    def unroll_lstm_time_steps(self, graph):

        def split_input(rolled_lstm_node, time_step_axis=1):
            rolled_lstm_node_name = rolled_lstm_node.op.name
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[self.DATA_IDX]).shape
            seq_length = input_shape[1] if time_step_axis else input_shape[0]
            input_x_split_name_list = []

            for i in range(seq_length):
                input_x_i_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[self.DATA_IDX] + str(i)
                input_x_split_name_list.append(input_x_i_name)
            input_x_split_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[self.DATA_IDX] + "_split"
            # If split_index is not specified, we split equally between the number of outputs
            input_x_split_op = op_adapter.SplitOp(name=input_x_split_name, axis=time_step_axis)

            # The split T inputs have same rank as original input, so reshape is needed to squeeze the timestep dimension
            split_node = graph.add(input_x_split_op, input_names=[rolled_lstm_node.input_names[self.DATA_IDX]],
                                   output_names=input_x_split_name_list, idx=rolled_lstm_node_idx)
            # set override encodings for split op
            pre_node = graph.get_producer_node(rolled_lstm_node.input_names[self.DATA_IDX])
            graph.update_trace_info(split_node, rolled_lstm_node)
            for split_output_name in input_x_split_name_list:
                graph.update_trace_info(graph.get_buffer(split_output_name), rolled_lstm_node)
                graph.copy_quantization_param(pre_node.op.name, split_node.op.name, rolled_lstm_node.input_names[self.DATA_IDX], split_output_name)

            return input_x_split_name_list

        def reshape_input(rolled_lstm_node, input_x_name_list):
            time_major_param = rolled_lstm_node.op.time_major
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[self.DATA_IDX]).shape
            input_buffer_shape = self.get_dims(input_shape, time_major_param)
            batch_size, _, input_size = input_buffer_shape[:]

            for i, input_x_name in enumerate(input_x_name_list):
                input_x_reshape_name = input_x_name + "_reshape"
                # Bidirectional lstm share the same input X, so add a check here
                if not graph.has_buffer(input_x_reshape_name):
                    input_x_reshape_output_shape = [batch_size, input_size]
                    input_x_reshape_op = op_adapter.ReshapeOp(name=input_x_reshape_name,
                                                              shape=input_x_reshape_output_shape)
                    reshape_node = graph.add(input_x_reshape_op, input_names=[input_x_name],
                                             output_names=[input_x_reshape_name], idx=rolled_lstm_node_idx)

                    graph.update_trace_info(reshape_node, rolled_lstm_node)
                    graph.update_trace_info(graph.get_buffer(input_x_reshape_name), rolled_lstm_node)
                    # Update the RolledLstm index for adding a reshape to the graph
                    rolled_lstm_node_idx += 1

                input_x_name_list[i] = input_x_reshape_name

        def prepare_lstm_output_name_list(rolled_lstm_node):
            time_major_param = rolled_lstm_node.op.time_major
            output_y_name_list = []
            output_y_reshape_name_list = []
            output_h_name_list = []
            output_c_name_list = []
            input_buffer_shape = self.get_dims(graph.get_buffer(rolled_lstm_node.input_names[self.DATA_IDX]).shape, time_major_param)
            seq_length = input_buffer_shape[1]
            for i in range(seq_length):
                output_y_i_name = rolled_lstm_node.output_names[self.HIDDEN_ALL_OUT_IDX] + str(i)
                output_y_name_list.append(output_y_i_name)
                output_y_reshape_i_name = output_y_i_name + "_reshape"
                output_y_reshape_name_list.append(output_y_reshape_i_name)
                output_h_i_name = rolled_lstm_node.output_names[self.HIDDEN_OUT_IDX] + str(i)
                output_h_name_list.append(output_h_i_name)
                output_c_i_name = rolled_lstm_node.output_names[self.CELL_OUT_IDX] + str(i)
                output_c_name_list.append(output_c_i_name)

            return output_y_name_list, output_y_reshape_name_list, output_h_name_list, output_c_name_list

        def add_single_timestep_lstm_op(rolled_lstm_node, reset_state_at_time_step_0, h_0_input_name, c_0_input_name,
                                        lstm_time_step_i_op_name, lstm_i_node_input_name_list, lstm_i_node_output_name_list):
            lstm_time_step_i_op = op_adapter.LstmOp(name=lstm_time_step_i_op_name,
                                                    hidden_size=rolled_lstm_node.op.hidden_size,
                                                    direction=ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD,
                                                    reset_state_at_time_step_0=reset_state_at_time_step_0,
                                                    h_0_input_name=h_0_input_name,
                                                    c_0_input_name=c_0_input_name,
                                                    sequence_continuation_name=rolled_lstm_node.op.sequence_continuation_name,
                                                    x_static_name=rolled_lstm_node.op.x_static_name,
                                                    cell_clip_threshold=rolled_lstm_node.op.cell_clip_threshold,
                                                    output_clip_threshold=rolled_lstm_node.op.output_clip_threshold,
                                                    time_major=False)
            lstm_time_step_i_node = graph.add(lstm_time_step_i_op,
                                              input_names=lstm_i_node_input_name_list,
                                              output_names=lstm_i_node_output_name_list,
                                              idx=graph.nodes_in_order.index(rolled_lstm_node))
            graph.update_trace_info(lstm_time_step_i_node, rolled_lstm_node)
            # Set output override encodings for single layer lstm op
            graph.copy_quantization_param(rolled_lstm_node.op.name, lstm_time_step_i_op.name, rolled_lstm_node.output_names[0], lstm_i_node_output_name_list[0])
            return lstm_time_step_i_node


        def add_lstm_output_reshape(rolled_lstm_node, output_name, output_reshape_name):
            time_major_param = rolled_lstm_node.op.time_major
            input_buffer_shape = self.get_dims(graph.get_buffer(rolled_lstm_node.input_names[self.DATA_IDX]).shape, time_major_param)
            batch_size = input_buffer_shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[self.HIDDEN_ALL_OUT_IDX]).shape[-1]
            seq_length = input_buffer_shape[1]

            # Setting up reshape output shape based on time_major_param
            output_all_h_reshape_output_shape = [1, batch_size, output_size] if time_major_param else [batch_size, 1, output_size]
            output_all_h_reshape_op = op_adapter.ReshapeOp(name=output_reshape_name,
                                                           shape=output_all_h_reshape_output_shape)
            graph.inject(output_all_h_reshape_op,
                         input_name=output_name,
                         output_name=output_reshape_name,
                         consumer_names=[consumer.op.name for consumer in list(graph.get_buffer(output_name).consumers)])

            # Setting up reshape output buffer axis format based on time_major_param
            graph.get_buffer(output_reshape_name).axis_format = AxisTracker.AxisFormat.TNF if time_major_param else AxisTracker.AxisFormat.NTF
            # Change output buffer shape to 2D
            graph.get_buffer(output_name).shape = [batch_size, output_size]
            graph.get_buffer(output_name).axis_format = AxisTracker.AxisFormat.NONTRIVIAL

        def concat_multi_timestep_outputs(rolled_lstm_node, concat_input_name_list, concat_output_name, time_step_axis=1):
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            output_y_concat_op = op_adapter.ConcatOp(name=concat_output_name, axis=time_step_axis)
            if rolled_lstm_node.op.direction == ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE:
                concat_input_name_list.reverse()
            output_y_concat_node = graph.add(output_y_concat_op, input_names=concat_input_name_list, output_names=[concat_output_name], idx=rolled_lstm_node_idx)
            # set override encodings for concat op
            graph.copy_quantization_param(rolled_lstm_node.op.name, output_y_concat_node.op.name, rolled_lstm_node.output_names[0], concat_output_name)

            graph.update_trace_info(output_y_concat_node, rolled_lstm_node)
            graph.update_trace_info(graph.get_buffer(concat_output_name), rolled_lstm_node)

        sequence = [
            (op_adapter.RolledLstmOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            rolled_lstm_node = nodes_tuple[0]
            rolled_lstm_node_name = rolled_lstm_node.op.name
            log_debug("Unrolling RolledLstm node {}".format(rolled_lstm_node_name))

            # Extract and validate sizes
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[self.DATA_IDX]).shape
            time_major_param = rolled_lstm_node.op.time_major
            input_buffer_shape = self.get_dims(input_shape, time_major_param)
            seq_length = input_buffer_shape[1]
            time_step_axis = 0 if time_major_param else 1
            if len(input_shape) != 3:
                raise ValueError('Unsupported input rank for RolledLstm node {}, expected 3, got {}.'.format(
                     rolled_lstm_node_name, len(input_shape)))

            if len(rolled_lstm_node.input_names) > lstm_props.IR_RESET_IDX:
                # Current lstm comes from stateful lstm which is not allowed to do unroll function
                # Just ignore the 'reset' input
                log_warning("LSTM Op {} with {} inputs is not allowed to call 'unroll_lstm_time_steps'. "
                            "Just ignore the 'reset' input".format(rolled_lstm_node_name,
                                                                   len(rolled_lstm_node.input_names)))

            # The name list of 2D input X_i for lstm_i node(s) at timestep i
            input_x_name_list = []

            if seq_length == 1:
                # Add the name to the input name list
                input_x_name_list.append(rolled_lstm_node.input_names[self.DATA_IDX])
            else:
                input_x_split_name_list = split_input(rolled_lstm_node, time_step_axis=time_step_axis)
                # Add the input x split names to input name list
                input_x_name_list.extend(input_x_split_name_list)

            # Adding reshape nodes to squeeze sequence length dimensions from input if input is 3D
            reshape_input(rolled_lstm_node, input_x_name_list)

            # Pre-process RolledLstm node and return the input name list for LstmOp
            lstm_all_inputs_name_list = self.preprocess_rolled_lstm_node(graph, rolled_lstm_node)
            output_y_name_list, output_y_reshape_name_list, output_h_name_list, output_c_name_list = prepare_lstm_output_name_list(rolled_lstm_node)

            # Add LstmOp to the graph per timestep
            for i in range(seq_length):
                # Prepare name of LstmOp at timestep i
                lstm_time_step_i_op_name = rolled_lstm_node_name + '_step_' + str(i)
                # Prepare necessary attributes for lstm_i
                reset_state_at_time_step_0 = rolled_lstm_node.op.reset_state_at_time_step_0 if i == 0 else False
                h_0_input_name = rolled_lstm_node.op.h_0_input_name if i == 0 else output_h_name_list[i-1]
                c_0_input_name = rolled_lstm_node.op.c_0_input_name if i == 0 else output_c_name_list[i-1]

                # Share weights and biases across Lstm nodes by using the same input name list
                lstm_i_node_input_name_list = lstm_all_inputs_name_list[:]
                # Update the specific inputs for lstm_i
                curr_idx = i if rolled_lstm_node.op.direction == ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD else seq_length-1-i
                lstm_i_node_input_name_list[self.DATA_IDX] = input_x_name_list[curr_idx]
                lstm_i_node_input_name_list[self.LSTM_HIDDEN_IN_IDX] = h_0_input_name
                lstm_i_node_input_name_list[self.LSTM_CELL_IN_IDX] = c_0_input_name
                # Prepare output name list for lstm_i
                lstm_i_node_output_name_list = [output_y_name_list[i], output_c_name_list[i], output_h_name_list[i]]

                lstm_time_step_i_node = add_single_timestep_lstm_op(rolled_lstm_node, reset_state_at_time_step_0, h_0_input_name, c_0_input_name,
                                                                    lstm_time_step_i_op_name, lstm_i_node_input_name_list, lstm_i_node_output_name_list)
                # For the last single layer lstm, we need to override output f_c and f_h
                if i == seq_length-1:
                    graph.copy_quantization_param(rolled_lstm_node.op.name,
                                                  lstm_time_step_i_node.op.name,
                                                  rolled_lstm_node.output_names[self.HIDDEN_OUT_IDX],
                                                  lstm_i_node_output_name_list[self.HIDDEN_OUT_IDX])
                    graph.copy_quantization_param(rolled_lstm_node.op.name,
                                                  lstm_time_step_i_node.op.name,
                                                  rolled_lstm_node.output_names[self.CELL_OUT_IDX],
                                                  lstm_i_node_output_name_list[self.CELL_OUT_IDX])
                # Reshape is added to unsqueeze the timestep dimension if output buffer is not 2D and
                # it is necessary to restore the NTF axis format regarding the output shape of 2D from QNN LstmOp
                add_lstm_output_reshape(rolled_lstm_node, lstm_i_node_output_name_list[self.HIDDEN_ALL_OUT_IDX], output_y_reshape_name_list[i])

            output_y_concat_name = rolled_lstm_node.output_names[self.HIDDEN_ALL_OUT_IDX] + "_reshape_concat" if seq_length > 1 else \
                                   output_y_reshape_name_list[0]
            # Concat output from T outputs
            if seq_length > 1:
                concat_multi_timestep_outputs(rolled_lstm_node, output_y_reshape_name_list, output_y_concat_name, time_step_axis=time_step_axis)

            self.adjust_lstm_output_consumers(graph, rolled_lstm_node, output_y_concat_name, output_h_name_list[seq_length-1], output_c_name_list[seq_length-1])
            source_output_names = rolled_lstm_node.output_names

            # Prune original RolledLstm node
            graph.prune(rolled_lstm_node, force_remove=True)

            current_output_names = [output_y_concat_name, output_c_name_list[seq_length-1], output_h_name_list[seq_length-1]]
            # At this point, current output names are not aligned to source output names, we need to
            # restore the source output names from RolledLstm node
            self.align_to_source_output_names(graph, current_output_names, source_output_names)

    def multi_time_steps_lstm(self, graph):

        sequence = [
            (op_adapter.RolledLstmOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            rolled_lstm_node = nodes_tuple[0]
            rolled_lstm_node_name = rolled_lstm_node.op.name
            log_debug("Converting RolledLstm node {} to Multi-time step Lstm node".format(rolled_lstm_node_name))

            if len(rolled_lstm_node.input_names) > lstm_props.IR_RESET_IDX:
                reset_input = rolled_lstm_node.input_names[lstm_props.IR_RESET_IDX]
            else:
                reset_input = ''
            lstm_all_inputs_name_list = self.preprocess_rolled_lstm_node(graph, rolled_lstm_node)
            lstm_all_inputs_name_list.append(reset_input)
            lstm_multi_time_step_op_name = rolled_lstm_node_name + '_multi_time_step'
            output_name_list = []
            for idx, name in enumerate(rolled_lstm_node.output_names):
                output_name_list.append(name + "_multi_time_step")

            lstm_multi_time_step_op = op_adapter.LstmOp(name=lstm_multi_time_step_op_name,
                                                    hidden_size=rolled_lstm_node.op.hidden_size,
                                                    direction=rolled_lstm_node.op.direction,
                                                    reset_state_at_time_step_0=rolled_lstm_node.op.reset_state_at_time_step_0,
                                                    h_0_input_name=rolled_lstm_node.op.h_0_input_name,
                                                    c_0_input_name=rolled_lstm_node.op.c_0_input_name,
                                                    sequence_continuation_name=rolled_lstm_node.op.sequence_continuation_name,
                                                    x_static_name=rolled_lstm_node.op.x_static_name,
                                                    cell_clip_threshold=rolled_lstm_node.op.cell_clip_threshold,
                                                    output_clip_threshold=rolled_lstm_node.op.output_clip_threshold,
                                                    time_major=rolled_lstm_node.op.time_major)
            lstm_all_inputs_name_list[self.LSTM_HIDDEN_IN_IDX] = rolled_lstm_node.op.h_0_input_name
            lstm_all_inputs_name_list[self.LSTM_CELL_IN_IDX] = rolled_lstm_node.op.c_0_input_name

            lstm_multi_time_step_op_node = graph.add(lstm_multi_time_step_op,
                      input_names=lstm_all_inputs_name_list,
                      output_names=output_name_list,
                      idx=graph.nodes_in_order.index(rolled_lstm_node))

            # add override encodings to multi_time_step lstm op
            for idx, name in enumerate(rolled_lstm_node.output_names):
                graph.copy_quantization_param(rolled_lstm_node.op.name, lstm_multi_time_step_op.name, name, output_name_list[idx])
            # Update trace info for new created node
            graph.update_trace_info(lstm_multi_time_step_op_node, [rolled_lstm_node])

            self.adjust_lstm_output_consumers(graph, rolled_lstm_node, output_name_list[0], output_name_list[2], output_name_list[1])
            # input buf axis format gets modified during graph construction by LstmOp populate_axis_format()
            # Make sure that the first input of the Lstm Op has correct axis format.
            if rolled_lstm_node.op.time_major:
                graph.get_buffer(lstm_all_inputs_name_list[0]).axis_format = AxisTracker.AxisFormat.TNF
            else:
                graph.get_buffer(lstm_all_inputs_name_list[0]).axis_format = AxisTracker.AxisFormat.NTF

            source_output_names = rolled_lstm_node.output_names
            # Prune original RolledLstm node
            graph.prune(rolled_lstm_node, force_remove=True)

            current_output_names = output_name_list
            # At this point, current output names are not aligned to source output names, we need to
            # restore the source output names from RolledLstm node
            self.align_to_source_output_names(graph, current_output_names, source_output_names)

    def adjust_lstm_output_consumers(self, graph, rolled_lstm_node, output_all_hidden_concat_name='', output_hidden_name='', output_cell_name=''):
        """
        Add original output buffers consumers to the last lstm_i node
        """
        if output_all_hidden_concat_name:
            all_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[self.HIDDEN_ALL_OUT_IDX])
            for consumer in list(all_output_buffer.consumers).copy():
                output_all_h_concat_buffer = graph.get_buffer(output_all_hidden_concat_name)
                output_all_h_concat_buffer.consumers.add(consumer)
                h_all_idx = consumer.input_names.index(rolled_lstm_node.output_names[self.HIDDEN_ALL_OUT_IDX])
                consumer.input_names[h_all_idx] = output_all_hidden_concat_name
                all_output_buffer.consumers.remove(consumer)

        if output_hidden_name:
            hidden_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[self.HIDDEN_OUT_IDX])
            for consumer in list(hidden_output_buffer.consumers).copy():
                output_h_buffer = graph.get_buffer(output_hidden_name)
                output_h_buffer.consumers.add(consumer)
                h_idx = consumer.input_names.index(rolled_lstm_node.output_names[self.HIDDEN_OUT_IDX])
                consumer.input_names[h_idx] = output_hidden_name
                hidden_output_buffer.consumers.remove(consumer)

        if output_cell_name:
            cell_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[self.CELL_OUT_IDX])
            for consumer in list(cell_output_buffer.consumers).copy():
                output_c_buffer = graph.get_buffer(output_cell_name)
                output_c_buffer.consumers.add(consumer)
                c_idx = consumer.input_names.index(rolled_lstm_node.output_names[self.CELL_OUT_IDX])
                consumer.input_names[c_idx] = output_cell_name
                cell_output_buffer.consumers.remove(consumer)

    # TODO Move to QNN-specific graph transformations once work on GraphTransformer is complete
    # Preprocesses LstmOp inputs, outputs, and attributes for QNN consumption
    def preprocess_rolled_lstm_node(self, graph, rolled_lstm_node):
        def split_lstm_tensor_per_gate(input_name, split_axis=0):
            producer_node = graph.get_producer_node(input_name)
            if producer_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                param_tensor = producer_node.op.tensor
                # Split weights so that they can be indexed by gate
                split_sections = int(param_tensor.shape[split_axis] / rolled_lstm_node.op.hidden_size)
                param_split_tensor = np.split(param_tensor, indices_or_sections=split_sections, axis=split_axis)
                # Two different RolledLstmOps may share the same weights or biases, so we need to extract the
                # weights by input name before we prune the const node from graph
                param_buf_consumers = graph.get_buffer(input_name).consumers
                param_buf_consumers.remove(rolled_lstm_node)
                if not param_buf_consumers:
                    # Prune the unsplit weights node from the graph
                    graph.prune(producer_node, force_remove=True)
                return param_split_tensor
            else:
                raise ValueError("LstmOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                        producer_node.op.name))

        def add_split_tensor_to_graph(tensor_name, tensor, desired_shape=None):
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            # Share the tensor if they are already added in the graph
            if not graph.has_buffer(tensor_name):
                tensor = np.resize(tensor, desired_shape) if desired_shape else tensor
                const_op = op_adapter.ConstantOp(name=tensor_name, tensor=tensor)
                const_node = graph.add(const_op, input_names=[], output_names=[tensor_name], idx=rolled_lstm_node_idx)
                graph.update_trace_info(const_node, rolled_lstm_node)
                graph.update_trace_info(graph.get_buffer(tensor_name), rolled_lstm_node)
            elif graph.get_producer_op(tensor_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("LstmOp requires weights and biases to be constant, got dynamic tensor from {}".format(
                        graph.get_producer_op(tensor_name).name))

        # Must add all inputs derived from splitting the tensor per gate as ConstantOp to the graph.
        # Weights may already be 2D (from TF as an example), but it is cleaner to resize anyway
        # rather than check shape for each input.
        # The weights and biases are shared across unrolled lstm nodes
        def prepare_lstm_all_inputs():
            input_size = graph.get_buffer(rolled_lstm_node.input_names[0]).shape[-1]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            # Input weights are expected in [4*hidden_size, input_size] in IFOC format
            src_input_weights_name = rolled_lstm_node.input_names[lstm_props.IR_INPUT_WEIGHTS_IDX]
            input_split_weights = split_lstm_tensor_per_gate(src_input_weights_name)

            input_w_to_forget_gate_name = src_input_weights_name + '_input_w_to_forget_gate'
            add_split_tensor_to_graph(input_w_to_forget_gate_name, input_split_weights[1], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_forget_gate_name, src_input_weights_name, input_w_to_forget_gate_name)

            input_w_to_cell_gate_name = src_input_weights_name + '_input_w_to_cell_gate'
            add_split_tensor_to_graph(input_w_to_cell_gate_name, input_split_weights[3], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_cell_gate_name, src_input_weights_name, input_w_to_cell_gate_name)

            input_w_to_output_gate_name = src_input_weights_name + '_input_w_to_output_gate'
            add_split_tensor_to_graph(input_w_to_output_gate_name, input_split_weights[2], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_output_gate_name, src_input_weights_name, input_w_to_output_gate_name)

            # Hidden state weights are expected in [4*hidden_size, hidden_size] in IFOC format
            src_hidden_state_weights_name = rolled_lstm_node.input_names[lstm_props.IR_HIDDEN_STATE_WEIGHTS_IDX]
            hidden_state_split_weights = split_lstm_tensor_per_gate(src_hidden_state_weights_name)

            recurrent_w_to_forget_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_forget_gate'
            add_split_tensor_to_graph(recurrent_w_to_forget_gate_name, hidden_state_split_weights[1], desired_shape=(num_units, output_size))
            graph.copy_quantization_param(src_hidden_state_weights_name, recurrent_w_to_forget_gate_name, src_hidden_state_weights_name, recurrent_w_to_forget_gate_name)

            recurrent_w_to_cell_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_cell_gate'
            add_split_tensor_to_graph(recurrent_w_to_cell_gate_name, hidden_state_split_weights[3], desired_shape=(num_units, output_size))
            graph.copy_quantization_param(src_hidden_state_weights_name, recurrent_w_to_cell_gate_name, src_hidden_state_weights_name, recurrent_w_to_cell_gate_name)

            recurrent_w_to_output_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_output_gate'
            add_split_tensor_to_graph(recurrent_w_to_output_gate_name, hidden_state_split_weights[2], desired_shape=(num_units, output_size))
            graph.copy_quantization_param(src_hidden_state_weights_name, recurrent_w_to_output_gate_name, src_hidden_state_weights_name, recurrent_w_to_output_gate_name)

            # Gate biases are expected in [4*hidden_size] in IFOC format
            src_gate_biases_name = rolled_lstm_node.input_names[lstm_props.IR_GATE_BIASES_IDX]
            gate_split_biases = split_lstm_tensor_per_gate(src_gate_biases_name)

            b_to_forget_gate_name = src_gate_biases_name + '_b_to_forget_gate'
            add_split_tensor_to_graph(b_to_forget_gate_name, gate_split_biases[1], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, b_to_forget_gate_name, src_gate_biases_name, b_to_forget_gate_name)

            b_to_cell_gate_name = src_gate_biases_name + '_b_to_cell_gate'
            add_split_tensor_to_graph(b_to_cell_gate_name, gate_split_biases[3], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, b_to_cell_gate_name, src_gate_biases_name, b_to_cell_gate_name)

            b_to_output_gate_name = src_gate_biases_name + '_b_to_output_gate'
            add_split_tensor_to_graph(b_to_output_gate_name, gate_split_biases[2], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, b_to_output_gate_name, src_gate_biases_name, b_to_output_gate_name)

            # Normalization weights are expected in [4*hidden_size] in IFOC format
            src_norm_weights_name = rolled_lstm_node.input_names[lstm_props.IR_NORM_WEIGHTS_IDX]
            norm_split_weights = split_lstm_tensor_per_gate(src_norm_weights_name) if src_norm_weights_name else None

            norm_w_to_input_gate_name = src_norm_weights_name + '_norm_w_to_input_gate' if norm_split_weights else ''
            if norm_w_to_input_gate_name:
                add_split_tensor_to_graph(norm_w_to_input_gate_name, norm_split_weights[0], desired_shape=(num_units,))
                graph.copy_quantization_param(src_norm_weights_name, norm_w_to_input_gate_name, src_norm_weights_name, norm_w_to_input_gate_name)

            norm_w_to_forget_gate_name = src_norm_weights_name + '_norm_w_to_forget_gate' if norm_split_weights else ''
            if norm_w_to_forget_gate_name:
                add_split_tensor_to_graph(norm_w_to_forget_gate_name, norm_split_weights[1], desired_shape=(num_units,))
                graph.copy_quantization_param(src_norm_weights_name, norm_w_to_forget_gate_name, src_norm_weights_name, norm_w_to_forget_gate_name)

            norm_w_to_cell_gate_name = src_norm_weights_name + '_norm_w_to_cell_gate' if norm_split_weights else ''
            if norm_w_to_cell_gate_name:
                add_split_tensor_to_graph(norm_w_to_cell_gate_name, norm_split_weights[3], desired_shape=(num_units,))
                graph.copy_quantization_param(src_norm_weights_name, norm_w_to_cell_gate_name, src_norm_weights_name, norm_w_to_cell_gate_name)

            norm_w_to_output_gate_name = src_norm_weights_name + '_norm_w_to_output_gate' if norm_split_weights else ''
            if norm_w_to_output_gate_name:
                add_split_tensor_to_graph(norm_w_to_output_gate_name, norm_split_weights[2], desired_shape=(num_units,))
                graph.copy_quantization_param(src_norm_weights_name, norm_w_to_output_gate_name, src_norm_weights_name, norm_w_to_output_gate_name)

            input_w_to_input_gate_name = src_input_weights_name + '_input_w_to_input_gate'
            add_split_tensor_to_graph(input_w_to_input_gate_name, input_split_weights[0], desired_shape=(num_units, input_size))
            graph.copy_quantization_param(src_input_weights_name, input_w_to_input_gate_name, src_input_weights_name, input_w_to_input_gate_name)

            recurrent_w_to_input_gate_name = src_hidden_state_weights_name + '_recurrent_w_to_input_gate'
            add_split_tensor_to_graph(recurrent_w_to_input_gate_name, hidden_state_split_weights[0], desired_shape=(num_units, output_size))
            graph.copy_quantization_param(src_hidden_state_weights_name, recurrent_w_to_input_gate_name, src_hidden_state_weights_name, recurrent_w_to_input_gate_name)

            # Cell state weights are expected in [3*hidden_size] in IFO format
            src_cell_state_weights_name = rolled_lstm_node.input_names[lstm_props.IR_CELL_STATE_WEIGHTS_IDX]
            cell_state_split_weights = split_lstm_tensor_per_gate(src_cell_state_weights_name) if src_cell_state_weights_name else None

            cell_w_to_input_gate_name = src_cell_state_weights_name + '_cell_w_to_input_gate' if cell_state_split_weights else ''
            if cell_w_to_input_gate_name:
                add_split_tensor_to_graph(cell_w_to_input_gate_name, cell_state_split_weights[0], desired_shape=(num_units,))
                graph.copy_quantization_param(src_cell_state_weights_name, cell_w_to_input_gate_name, src_cell_state_weights_name, cell_w_to_input_gate_name)

            cell_w_to_forget_gate_name = src_cell_state_weights_name + '_cell_w_to_forget_gate' if cell_state_split_weights else ''
            if cell_w_to_forget_gate_name:
                add_split_tensor_to_graph(cell_w_to_forget_gate_name, cell_state_split_weights[1], desired_shape=(num_units,))
                graph.copy_quantization_param(src_cell_state_weights_name, cell_w_to_forget_gate_name, src_cell_state_weights_name, cell_w_to_forget_gate_name)

            cell_w_to_output_gate_name = src_cell_state_weights_name + '_cell_w_to_output_gate' if cell_state_split_weights else ''
            if cell_w_to_output_gate_name:
                add_split_tensor_to_graph(cell_w_to_output_gate_name, cell_state_split_weights[2], desired_shape=(num_units,))
                graph.copy_quantization_param(src_cell_state_weights_name, cell_w_to_output_gate_name, src_cell_state_weights_name, cell_w_to_output_gate_name)

            b_to_input_gate_name = src_gate_biases_name + '_b_to_input_gate'
            add_split_tensor_to_graph(b_to_input_gate_name, gate_split_biases[0], desired_shape=(num_units,))
            graph.copy_quantization_param(src_gate_biases_name, b_to_input_gate_name, src_gate_biases_name, b_to_input_gate_name)

            # The projection weights and bias do not need to be split, and they are added to the graph in frontend if provided
            proj_w_name = rolled_lstm_node.input_names[lstm_props.IR_PROJ_WEIGHTS_IDX]
            proj_b_name = rolled_lstm_node.input_names[lstm_props.IR_PROJ_BIAS_IDX]

            # TODO: Add reset input handling here and remove it from multi-time-step LSTM. See
            # AISW-106286 for info on why this was removed. See AISW-106948 for bug raised on HTP
            # backend to fix this. Once that bug is resolved, todo can be resolved.
            # Prepare the LstmOp input names - inputs not captured by any FE are passed the empty string
            lstm_all_inputs_name_list = [
                rolled_lstm_node.input_names[0],
                input_w_to_forget_gate_name,
                input_w_to_cell_gate_name,
                input_w_to_output_gate_name,
                recurrent_w_to_forget_gate_name,
                recurrent_w_to_cell_gate_name,
                recurrent_w_to_output_gate_name,
                b_to_forget_gate_name,
                b_to_cell_gate_name,
                b_to_output_gate_name,
                rolled_lstm_node.input_names[1],
                rolled_lstm_node.input_names[2],
                norm_w_to_input_gate_name,
                norm_w_to_forget_gate_name,
                norm_w_to_cell_gate_name,
                norm_w_to_output_gate_name,
                input_w_to_input_gate_name,
                recurrent_w_to_input_gate_name,
                cell_w_to_input_gate_name,
                cell_w_to_forget_gate_name,
                cell_w_to_output_gate_name,
                b_to_input_gate_name,
                proj_w_name,
                proj_b_name,
            ]

            # Update the RolledLstmOp input names
            new_input_names = rolled_lstm_node.input_names[:lstm_props.IR_INPUT_WEIGHTS_IDX]
            rolled_lstm_node.input_names = new_input_names
            return lstm_all_inputs_name_list

        def ensure_h_c_inputs_present():
            rolled_lstm_node_name = rolled_lstm_node.op.name
            rolled_lstm_node_idx = graph.nodes_in_order.index(rolled_lstm_node)
            time_major_param = rolled_lstm_node.op.time_major
            input_buffer_shape = self.get_dims(graph.get_buffer(rolled_lstm_node.input_names[0]).shape, time_major_param)
            batch_size = input_buffer_shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            # Requires initial_h and initial_c inputs to be present
            # The following code adds zero valued tensors provided the conditions below are satisfied
            if not rolled_lstm_node.input_names[1] and not rolled_lstm_node.input_names[2]:
                if rolled_lstm_node.op.h_0_input_name:
                    raise ValueError('RolledLstm node {} op attribute h_0_input_name {} mismatch with rolled_lstm_node.input_names[1] {}.'.format(
                            rolled_lstm_node_name, rolled_lstm_node.op.h_0_input_name, rolled_lstm_node.input_names[1]))
                if rolled_lstm_node.op.c_0_input_name:
                    raise ValueError('RolledLstm node {} op attribute c_0_input_name {} mismatch with rolled_lstm_node.input_names[2] {}.'.format(
                            rolled_lstm_node_name, rolled_lstm_node.op.c_0_input_name, rolled_lstm_node.input_names[2]))

                # add zeros for initial h and c inputs since there are needed for QNN
                initial_hidden_state_name = rolled_lstm_node_name + '_initial_hidden_state'
                initial_hidden_state_tensor = np.zeros((batch_size, output_size), dtype=np.float32)
                initial_hidden_state_op = op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor)
                initial_hidden_state_node = graph.add(initial_hidden_state_op, input_names=[], output_names=[initial_hidden_state_name], idx=rolled_lstm_node_idx)
                graph.update_trace_info(initial_hidden_state_node, rolled_lstm_node)
                graph.update_trace_info(graph.get_buffer(initial_hidden_state_name), rolled_lstm_node)
                rolled_lstm_node.input_names[1] = initial_hidden_state_name
                rolled_lstm_node.op.h_0_input_name = initial_hidden_state_name
                graph.get_buffer(initial_hidden_state_name).consumers.add(rolled_lstm_node)

                initial_cell_state_name = rolled_lstm_node_name + '_initial_cell_state'
                initial_cell_state_tensor = np.zeros((batch_size, num_units), dtype=np.float32)
                initial_cell_state_op = op_adapter.ConstantOp(name=initial_cell_state_name, tensor=initial_cell_state_tensor)
                initial_cell_state_node = graph.add(initial_cell_state_op, input_names=[], output_names=[initial_cell_state_name], idx=rolled_lstm_node_idx+1)
                graph.update_trace_info(initial_cell_state_node, rolled_lstm_node)
                graph.update_trace_info(graph.get_buffer(initial_cell_state_name), rolled_lstm_node)
                rolled_lstm_node.input_names[2] = initial_cell_state_name
                rolled_lstm_node.op.c_0_input_name = initial_cell_state_name
                graph.get_buffer(initial_cell_state_name).consumers.add(rolled_lstm_node)

        def add_h_c_inputs_reshape_if_needed():
            rolled_lstm_node_name = rolled_lstm_node.op.name
            time_major_param = rolled_lstm_node.op.time_major
            input_buffer_shape = self.get_dims(graph.get_buffer(rolled_lstm_node.input_names[0]).shape, time_major_param)
            batch_size = input_buffer_shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            # If the initial hidden state shape (and implicitly initial cell state shape)
            # is not 2D then it should be reshaped
            initial_h_shape = graph.get_buffer(rolled_lstm_node.input_names[1]).shape
            initial_state_reshape_needed = len(initial_h_shape) != 2 or initial_h_shape != [batch_size, output_size]
            if initial_state_reshape_needed:
                input_h_reshape_node_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[1] + "_reshape"
                input_h_reshape_output_shape = [batch_size, output_size]
                input_h_reshape_op = op_adapter.ReshapeOp(name=input_h_reshape_node_name,
                                                          shape=input_h_reshape_output_shape)
                graph.inject(input_h_reshape_op, input_name=rolled_lstm_node.input_names[1],
                             output_name=input_h_reshape_node_name, consumer_names=[rolled_lstm_node_name])
                rolled_lstm_node.op.h_0_input_name = input_h_reshape_node_name

                input_c_reshape_node_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[2] + "_reshape"
                input_c_reshape_output_shape = [batch_size, num_units]
                input_c_reshape_op = op_adapter.ReshapeOp(name=input_c_reshape_node_name,
                                                          shape=input_c_reshape_output_shape)
                graph.inject(input_c_reshape_op, input_name=rolled_lstm_node.input_names[2],
                             output_name=input_c_reshape_node_name, consumer_names=[rolled_lstm_node_name])
                rolled_lstm_node.op.c_0_input_name = input_c_reshape_node_name

        def handle_missing_outputs():
            rolled_lstm_node_name = rolled_lstm_node.op.name
            time_major_param = rolled_lstm_node.op.time_major
            input_buffer_shape = self.get_dims(graph.get_buffer(rolled_lstm_node.input_names[0]).shape, time_major_param)
            batch_size = input_buffer_shape[0]
            output_size = graph.get_buffer(rolled_lstm_node.output_names[0]).shape[-1]
            num_units = rolled_lstm_node.op.hidden_size

            number_of_outputs = len(rolled_lstm_node.output_names)
            all_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[0])

            if number_of_outputs == 3:
                # Modify existing output buffers for QNN specification
                hidden_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[2])
                hidden_output_buffer.shape = [batch_size, output_size]
                hidden_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

                cell_output_buffer = graph.get_buffer(rolled_lstm_node.output_names[1])
                cell_output_buffer.shape = [batch_size, num_units]
                cell_output_buffer.axis_format = AxisTracker.AxisFormat.NONTRIVIAL

                # Prepare output names and keep the first output as all_hidden
                # Output tensor at index HIDDEN_OUT_IDX and HIDDEN_ALL_OUT_IDX are effectively the same in QNN LstmOp
                rolled_lstm_node.output_names = [all_output_buffer.name, cell_output_buffer.name, hidden_output_buffer.name]
            elif number_of_outputs == 1:
                # Add dummy buffers for missing outputs - QNN requires 3
                hidden_output_dummy_name = rolled_lstm_node_name + "_hidden_output_dummy"
                graph.add_output_buffer(rolled_lstm_node, hidden_output_dummy_name,
                                        [batch_size, output_size], AxisTracker.AxisFormat.NONTRIVIAL)

                cell_output_dummy_name = rolled_lstm_node_name + "_cell_output_dummy"
                graph.add_output_buffer(rolled_lstm_node, cell_output_dummy_name,
                                        [batch_size, num_units], AxisTracker.AxisFormat.NONTRIVIAL)

                rolled_lstm_node.output_names = [all_output_buffer.name, cell_output_dummy_name, hidden_output_dummy_name]
            else:
                # Only 1 or 3 outputs are supported for this optimization
                raise ValueError("Unsupported number of outputs for RolledLstm node {}, expected 1 or 3, got {}.".format(
                    rolled_lstm_node_name, number_of_outputs))

        log_debug("Preprocessing RolledLstm node {} for QNN lowering.".format(rolled_lstm_node.op.name))

        # Prepare QNN Lstm all inputs and return the input name list
        lstm_all_inputs_name_list = prepare_lstm_all_inputs()
        ensure_h_c_inputs_present()
        add_h_c_inputs_reshape_if_needed()
        handle_missing_outputs()

        return lstm_all_inputs_name_list

    def expand_lstm_op_structure(self, graph):
        sequence = [
            (op_adapter.RolledLstmOp.TRANSLATION_KEY, (), ())
        ]

        matched_node_list = graph.get_matched_nodes(sequence)

        for nodes_tuple in matched_node_list:
            rolled_lstm_node = nodes_tuple[0]
            rolled_lstm_node_name = rolled_lstm_node.op.name
            log_debug('Expand RolledLstm node {}'.format(rolled_lstm_node_name))

            # Extract and validate sizes
            input_shape = graph.get_buffer(rolled_lstm_node.input_names[self.DATA_IDX]).shape
            time_major_param = rolled_lstm_node.op.time_major
            hidden_size = rolled_lstm_node.op.hidden_size
            output_size = hidden_size
            input_buffer_shape = self.get_dims(input_shape, time_major_param)
            batch_size, seq_length, input_size = input_buffer_shape
            time_step_axis = 0 if time_major_param else 1
            if len(input_shape) != 3:
                raise ValueError('Unsupported input rank for RolledLstm node {}, expected 3, got {}.'.format(
                    rolled_lstm_node_name, len(input_shape)))

            # The input Xi name list for lstm_i node at time step i
            input_x_name_list = [rolled_lstm_node.input_names[self.DATA_IDX]]
            if seq_length != 1:
                input_x_split_op_name = rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[self.DATA_IDX] + "_split"
                input_x_split_output_name_list = [rolled_lstm_node_name + "_" + rolled_lstm_node.input_names[self.DATA_IDX] + str(i)
                                                  for i in range(seq_length)]
                # The split input Xi has same rank as original input, so reshape is needed to squeeze the time step dimension
                graph.add(op_adapter.SplitOp(name=input_x_split_op_name, axis=time_step_axis),
                          input_names=[rolled_lstm_node.input_names[self.DATA_IDX]],
                          output_names=input_x_split_output_name_list,
                          idx=graph.nodes_in_order.index(rolled_lstm_node))
                # Update input X name list
                input_x_name_list = input_x_split_output_name_list

            input_x_reshape_name_list = []
            for input_x_name in input_x_name_list:
                input_x_reshape_name = input_x_name + "_reshape"
                input_x_reshape_name_list.append(input_x_reshape_name)
                # Bidirectional lstm share the same input X, so check if the ReshapeOp is already added to the graph
                if not graph.has_buffer(input_x_reshape_name):
                    graph.add(op_adapter.ReshapeOp(name=input_x_reshape_name,
                                                   shape=[batch_size, input_size]),
                              input_names=[input_x_name],
                              output_names=[input_x_reshape_name],
                              idx=graph.nodes_in_order.index(rolled_lstm_node))

            if rolled_lstm_node.op.h_0_input_name != rolled_lstm_node.input_names[lstm_props.IR_INITIAL_H_IDX]:
                raise ValueError('RolledLstm node {} op attribute h_0_input_name {} mismatch with rolled_lstm_node initial_h input name {}.'.format(
                    rolled_lstm_node_name, rolled_lstm_node.op.h_0_input_name, rolled_lstm_node.input_names[lstm_props.IR_INITIAL_H_IDX]))
            if rolled_lstm_node.op.c_0_input_name != rolled_lstm_node.input_names[lstm_props.IR_INITIAL_C_IDX]:
                raise ValueError('RolledLstm node {} op attribute c_0_input_name {} mismatch with rolled_lstm_node initial_c input name {}.'.format(
                    rolled_lstm_node_name, rolled_lstm_node.op.c_0_input_name, rolled_lstm_node.input_names[lstm_props.IR_INITIAL_C_IDX]))

            # Require initial_h and initial_c inputs to be present.
            # Add zero-value tensor for initial_h and initial_c regarding the condition below is satisfied
            if not rolled_lstm_node.op.h_0_input_name:
                initial_hidden_state_name = rolled_lstm_node_name + '_initial_hidden_state'
                initial_hidden_state_tensor = np.zeros((batch_size, hidden_size), dtype=np.float32)
                graph.add(op_adapter.ConstantOp(name=initial_hidden_state_name, tensor=initial_hidden_state_tensor),
                          input_names=[],
                          output_names=[initial_hidden_state_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))
                rolled_lstm_node.op.h_0_input_name = initial_hidden_state_name
                rolled_lstm_node.input_names[lstm_props.IR_INITIAL_H_IDX] = initial_hidden_state_name
                graph.get_buffer(initial_hidden_state_name).consumers.add(rolled_lstm_node)

            if not rolled_lstm_node.op.c_0_input_name:
                initial_cell_state_name = rolled_lstm_node_name + '_initial_cell_state'
                initial_cell_state_tensor = np.zeros((batch_size, hidden_size), dtype=np.float32)
                graph.add(op_adapter.ConstantOp(name=initial_cell_state_name, tensor=initial_cell_state_tensor),
                          input_names=[],
                          output_names=[initial_cell_state_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))
                rolled_lstm_node.op.c_0_input_name = initial_cell_state_name
                rolled_lstm_node.input_names[lstm_props.IR_INITIAL_C_IDX] = initial_cell_state_name
                graph.get_buffer(initial_cell_state_name).consumers.add(rolled_lstm_node)

            # Add input h/c reshape op if needed
            if rolled_lstm_node.op.h_0_input_name and graph.get_buffer(rolled_lstm_node.op.h_0_input_name).shape != [batch_size, hidden_size]:
                input_h_reshape_node_name = rolled_lstm_node.op.h_0_input_name + '_reshape'
                if not graph.has_buffer(input_h_reshape_node_name):
                    input_h_reshape_op = op_adapter.ReshapeOp(name=input_h_reshape_node_name,
                                                              shape=[batch_size, hidden_size])
                    graph.inject(input_h_reshape_op,
                                 input_name=rolled_lstm_node.op.h_0_input_name,
                                 output_name=input_h_reshape_node_name,
                                 consumer_names=[rolled_lstm_node_name])
                rolled_lstm_node.op.h_0_input_name = input_h_reshape_node_name

            if rolled_lstm_node.op.c_0_input_name and graph.get_buffer(rolled_lstm_node.op.c_0_input_name).shape != [batch_size, hidden_size]:
                input_c_reshape_node_name = rolled_lstm_node.op.c_0_input_name + '_reshape'
                if not graph.has_buffer(input_c_reshape_node_name):
                    input_c_reshape_op = op_adapter.ReshapeOp(name=input_c_reshape_node_name,
                                                              shape=[batch_size, hidden_size])
                    graph.inject(input_c_reshape_op,
                                 input_name=rolled_lstm_node.op.c_0_input_name,
                                 output_name=input_c_reshape_node_name,
                                 consumer_names=[rolled_lstm_node_name])
                rolled_lstm_node.op.c_0_input_name = input_c_reshape_node_name

            # Initial h/c input names in each single time step lstm
            # h/c input names would be updated during the iteration for each time step
            h_input_name = rolled_lstm_node.op.h_0_input_name
            c_input_name = rolled_lstm_node.op.c_0_input_name

            # Input Xt list in-place reversal for backward direction
            if rolled_lstm_node.op.direction == ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE:
                input_x_reshape_name_list.reverse()

            ht_output_reshape_name_list = list()
            # Add decomposed ops to the graph from rolled lstm per timestep iteration
            for time_step, input_x_reshape_name in enumerate(input_x_reshape_name_list):
                # Prepare name of LstmOp at time step t
                lstm_time_step_t_op_name = rolled_lstm_node_name + '_step_' + str(time_step)

                src_input_weights_name = rolled_lstm_node.input_names[lstm_props.IR_INPUT_WEIGHTS_IDX]
                # Input weights are expected in [4*hidden_size, input_size] in IFOC format
                if not isinstance(graph.get_producer_node(src_input_weights_name).op, op_adapter.ConstantOp):
                    raise ValueError('RolledLstm node {} requires input weights to be constant, got dynamic tensor from {}'.format(
                        rolled_lstm_node.op.name, src_input_weights_name))

                src_hidden_state_weights_name = rolled_lstm_node.input_names[lstm_props.IR_HIDDEN_STATE_WEIGHTS_IDX]
                # Hidden state weights are expected in [4*hidden_size, hidden_size] in IFOC format
                if not isinstance(graph.get_producer_node(src_hidden_state_weights_name).op, op_adapter.ConstantOp):
                    raise ValueError('RolledLstm node {} requires hidden state weights to be constant, got dynamic tensor from {}'.format(
                        rolled_lstm_node.op.name, src_hidden_state_weights_name))

                src_gate_biases_name = rolled_lstm_node.input_names[lstm_props.IR_GATE_BIASES_IDX]
                # Gate biases are expected in [4*hidden_size] in IFOC format
                if not isinstance(graph.get_producer_node(src_gate_biases_name).op, op_adapter.ConstantOp):
                    raise ValueError('RolledLstm node {} requires gate biases to be constant, got dynamic tensor from {}'.format(
                        rolled_lstm_node.op.name, src_gate_biases_name))

                # Add matmul for [Xt]*[Wi]
                xt_wi_matmul_op_name = lstm_time_step_t_op_name + '_xt_wi_matmul'
                xt_wi_matmul_op = op_adapter.MatMulOp(name=xt_wi_matmul_op_name,
                                                      transpose_in0=False,
                                                      transpose_in1=True)
                graph.add(xt_wi_matmul_op,
                          input_names=[input_x_reshape_name, src_input_weights_name],
                          output_names=[xt_wi_matmul_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Add matmul for [Ht-1]*[Wh]
                ht_1_wh_matmul_op_name = lstm_time_step_t_op_name + '_ht_1_wh_matmul'
                ht_1_wh_matmul_op = op_adapter.MatMulOp(name=ht_1_wh_matmul_op_name,
                                                        transpose_in0=False,
                                                        transpose_in1=True)
                graph.add(ht_1_wh_matmul_op,
                          input_names=[h_input_name, src_hidden_state_weights_name],
                          output_names=[ht_1_wh_matmul_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                ifoc_gate_matmul_add_op_name = lstm_time_step_t_op_name + '_ifoc_gate_matmul_add'
                ifoc_gate_matmul_add_op = op_adapter.ElementwiseBinaryOp(name=ifoc_gate_matmul_add_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
                graph.add(ifoc_gate_matmul_add_op,
                          input_names=[xt_wi_matmul_op_name, ht_1_wh_matmul_op_name],
                          output_names=[ifoc_gate_matmul_add_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Add gate biases
                ifoc_gate_matmul_bias_add_op_name = ifoc_gate_matmul_add_op_name + '_bias_add'
                ifoc_gate_matmul_bias_add_op = op_adapter.ElementwiseBinaryOp(name=ifoc_gate_matmul_bias_add_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
                graph.add(ifoc_gate_matmul_bias_add_op,
                          input_names=[ifoc_gate_matmul_add_op_name, src_gate_biases_name],
                          output_names=[ifoc_gate_matmul_bias_add_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Split concated IFOC-gate output to (IF, O, C)
                if_o_c_split_op_name = lstm_time_step_t_op_name + '_if_o_c_gate_split'
                if_o_c_split_op_output_names = [lstm_time_step_t_op_name + '_if_gate_output',
                                                lstm_time_step_t_op_name + '_o_gate_output',
                                                lstm_time_step_t_op_name + '_c_gate_output']
                if_o_c_split_op = op_adapter.SplitOp(name=if_o_c_split_op_name,
                                                     split_index=[hidden_size*2, hidden_size*3], axis=1)
                graph.add(if_o_c_split_op,
                          input_names=[ifoc_gate_matmul_bias_add_op_name],
                          output_names=if_o_c_split_op_output_names,
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Apply sigmoid to IF-gate output
                if_gate_sigmoid_op_name = if_o_c_split_op_output_names[0] + '_sigmoid'
                if_gate_sigmoid_op = op_adapter.ElementwiseNeuronOp(name=if_gate_sigmoid_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID)
                graph.add(if_gate_sigmoid_op,
                          input_names=[if_o_c_split_op_output_names[0]],
                          output_names=[if_gate_sigmoid_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Apply tanh to C-gate output
                c_gate_tanh_op_name = if_o_c_split_op_output_names[2] + '_tanh'
                c_gate_tanh_op = op_adapter.ElementwiseNeuronOp(name=c_gate_tanh_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH)
                graph.add(c_gate_tanh_op,
                          input_names=[if_o_c_split_op_output_names[2]],
                          output_names=[c_gate_tanh_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Concat C-gate tanh output with Ct-1
                c_gate_tanh_ct_1_concat_op_name = lstm_time_step_t_op_name + '_c_gate_tanh_ct_1_concat'
                c_gate_tanh_ct_1_concat_op = op_adapter.ConcatOp(name=c_gate_tanh_ct_1_concat_op_name, axis=1)
                graph.add(c_gate_tanh_ct_1_concat_op,
                          input_names=[c_gate_tanh_op_name, c_input_name],
                          output_names=[c_gate_tanh_ct_1_concat_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Perform elementwise dot activation:
                # (1) (i-gate sigmoid output) * (c-gate tanh output)
                # (2) (f-gate sigmoid output) * (Ct-1)
                # and then add above outputs together
                ifc_gate_ct_1_dot_activation_op_name = lstm_time_step_t_op_name + '_ifc_gate_ct_1_dot_activation'
                ifc_gate_ct_1_dot_activation_op = op_adapter.ElementwiseBinaryOp(name=ifc_gate_ct_1_dot_activation_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                graph.add(ifc_gate_ct_1_dot_activation_op,
                          input_names=[if_gate_sigmoid_op_name, c_gate_tanh_ct_1_concat_op_name],
                          output_names=[ifc_gate_ct_1_dot_activation_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                ifc_gate_ct_1_dot_activation_split_op_name = ifc_gate_ct_1_dot_activation_op_name + '_split'
                ifc_gate_ct_1_dot_activation_split_output_names = [ifc_gate_ct_1_dot_activation_split_op_name + '_output_1',
                                                                   ifc_gate_ct_1_dot_activation_split_op_name + '_output_2']
                ifc_gate_ct_1_dot_activation_split_op = op_adapter.SplitOp(name=ifc_gate_ct_1_dot_activation_split_op_name,
                                                                           split_index=[hidden_size], axis=1)
                graph.add(ifc_gate_ct_1_dot_activation_split_op,
                          input_names=[ifc_gate_ct_1_dot_activation_op_name],
                          output_names=ifc_gate_ct_1_dot_activation_split_output_names,
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                ct_output_op_name = ifc_gate_ct_1_dot_activation_op_name + '_split_output_elementwise_add'
                ct_output_op = op_adapter.ElementwiseBinaryOp(name=ct_output_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD)
                graph.add(ct_output_op,
                          input_names=ifc_gate_ct_1_dot_activation_split_output_names,
                          output_names=[ct_output_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Add clip function if provided
                ct_output_clip_op_name = ct_output_op_name
                tcell = np.float32(rolled_lstm_node.op.cell_clip_threshold)
                if tcell:
                    ct_output_clip_op_name = ct_output_op_name + '_clip'
                    ct_output_clip_op = op_adapter.ElementwiseNeuronOp(name=ct_output_clip_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX,
                                                                       min_value=-tcell, max_value=tcell)
                    graph.add(ct_output_clip_op,
                              input_names=[ct_output_op_name],
                              output_names=[ct_output_clip_op_name],
                              idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Apply sigmoid to O-gate output
                # TODO: Add support for peephole optimization
                o_gate_sigmoid_op_name = if_o_c_split_op_output_names[1] + '_sigmoid'
                o_gate_sigmoid_op = op_adapter.ElementwiseNeuronOp(name=o_gate_sigmoid_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID)
                graph.add(o_gate_sigmoid_op,
                          input_names=[if_o_c_split_op_output_names[1]],
                          output_names=[o_gate_sigmoid_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Apply tanh to Ct
                ct_output_tanh_op_name = ct_output_clip_op_name + '_tanh'
                ct_output_tanh_op = op_adapter.ElementwiseNeuronOp(name=ct_output_tanh_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH)
                graph.add(ct_output_tanh_op,
                          input_names=[ct_output_clip_op_name],
                          output_names=[ct_output_tanh_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Perform elementwise dot activation:
                # (o-gate sigmoid output) * (Ct tanh output)
                ht_output_op_name = lstm_time_step_t_op_name + '_o_gate_output_ct_tanh_dot_activation'
                ht_output_op = op_adapter.ElementwiseBinaryOp(name=ht_output_op_name, operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY)
                graph.add(ht_output_op,
                          input_names=[ct_output_tanh_op_name, o_gate_sigmoid_op_name],
                          output_names=[ht_output_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                # Update h/c input name for next time step
                h_input_name = ht_output_op_name
                c_input_name = ct_output_clip_op_name

                # Add reshape to unsqueeze the time step dimension
                # TODO: Replace hidden_size with output_size if proj weights is provided
                ht_output_reshape_op_name = ht_output_op_name + '_reshape'
                ht_output_reshape_output_shape = [1, batch_size, hidden_size] if time_major_param else [batch_size, 1, hidden_size]
                graph.add(op_adapter.ReshapeOp(name=ht_output_reshape_op_name,
                                               shape=ht_output_reshape_output_shape),
                          input_names=[ht_output_op_name],
                          output_names=[ht_output_reshape_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

                ht_output_reshape_name_list.append(ht_output_reshape_op_name)

            final_ht_name = h_input_name
            final_ct_name = c_input_name
            ht_output_reshape_concat_op_name = rolled_lstm_node.op.name + '_ht_output_reshape_concat' if seq_length > 1 else \
                                               ht_output_reshape_name_list[0]
            # Concat Ht output reshape in each time step
            if seq_length > 1:
                if rolled_lstm_node.op.direction == ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE:
                    ht_output_reshape_name_list.reverse()
                graph.add(op_adapter.ConcatOp(name=ht_output_reshape_concat_op_name, axis=time_step_axis),
                          input_names=ht_output_reshape_name_list,
                          output_names=[ht_output_reshape_concat_op_name],
                          idx=graph.nodes_in_order.index(rolled_lstm_node))

            if len(rolled_lstm_node.output_names) == 1:
                self.adjust_lstm_output_consumers(graph, rolled_lstm_node, ht_output_reshape_concat_op_name)
            else:
                self.adjust_lstm_output_consumers(graph, rolled_lstm_node, ht_output_reshape_concat_op_name, final_ht_name, final_ct_name)

            # Prune RolledLstm node from the graph
            graph.prune(rolled_lstm_node, force_remove=True)

            # Two different RolledLstm nodes could share the same weights or biases.
            # Need to check whether the weights have consumer before pruned from the graph
            input_weights_buf_consumers = graph.get_buffer(rolled_lstm_node.input_names[lstm_props.IR_INPUT_WEIGHTS_IDX]).consumers
            if not input_weights_buf_consumers:
                graph.prune(graph.get_producer_node(rolled_lstm_node.input_names[lstm_props.IR_INPUT_WEIGHTS_IDX]), force_remove=True)

            hidden_weights_buf_consumers = graph.get_buffer(rolled_lstm_node.input_names[lstm_props.IR_HIDDEN_STATE_WEIGHTS_IDX]).consumers
            if not hidden_weights_buf_consumers:
                graph.prune(graph.get_producer_node(rolled_lstm_node.input_names[lstm_props.IR_HIDDEN_STATE_WEIGHTS_IDX]), force_remove=True)

            current_output_names = [ht_output_reshape_concat_op_name] if len(rolled_lstm_node.output_names) == 1 else \
                [ht_output_reshape_concat_op_name, final_ct_name, final_ht_name]
            source_output_names = rolled_lstm_node.output_names
            # At this point, current output names are not aligned to source output names,
            # we need to restore the source output names from graph.
            self.align_to_source_output_names(graph, current_output_names, source_output_names)


@register_layer_optimization
class OptimizeLstmTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.LstmOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeScatterDenseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SparseToDenseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeScatterElementsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScatterElementsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name, updates_name = node.input_names
        input_buf,  indices_buf,  updates_buf = graph.get_input_buffers(node)

        output_buf = graph.get_output_buffers(node)[0]

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                return
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        # Check if any of the buffers has been changed into NDHWC, NSC, NFC, NTF order and revert if so
        # All inputs need to be in source framework order
        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])
        set_input_axis_format(updates_name, updates_buf.axis_format, node.op.data_axis_formats[2])

        input_buf = graph.get_input_buffers(node)[0]
        # set output buf axis format to input[0] axis format since data buffer's one is unchanged with ScatterElements
        output_buf.axis_format = input_buf.axis_format

        return True


@register_layer_optimization
class OptimizeScatterNDTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ScatterNDOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name, indices_name, updates_name = node.input_names
        input_buf,  indices_buf,  updates_buf = graph.get_input_buffers(node)

        output_buf = graph.get_output_buffers(node)[0]

        def set_input_axis_format(buf_name, buf_axis_format, data_axis_format):
            if buf_axis_format == data_axis_format:
                return
            elif buf_axis_format == AxisTracker.AxisFormat.NDHWC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NSC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NFC and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif buf_axis_format == AxisTracker.AxisFormat.NTF and \
                    buf_axis_format != data_axis_format:
                graph.inject_implicit_permute(buf_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False

        # Check if any of the buffers have been changed into NDHWC, NSC, NFC, NTF order and revert if so
        # All inputs need to be in source framework order
        set_input_axis_format(input_name, input_buf.axis_format, node.op.data_axis_formats[0])
        set_input_axis_format(indices_name, indices_buf.axis_format, node.op.data_axis_formats[1])
        set_input_axis_format(updates_name, updates_buf.axis_format, node.op.data_axis_formats[2])

        input_buf = graph.get_input_buffers(node)[0]
        # set output buf axis format to input[0] axis format since data format is unchanged with ScatterND
        output_buf.axis_format = input_buf.axis_format

        return True


    def replace_6d_operation(self, node, graph):
        # ScatterND has three inputs: data, indice, update.
        # Rank of data and indice should be less than 6 and rank of update should be less than 7.
        data_buf, indice_buf, val_buf = graph.get_input_buffers(node)
        log_assert(
            data_buf.rank() <= 5, f"data of {node.op.name} is not supported in rank >= 6"
        )
        log_assert(
            val_buf.rank() <= 5, f"update value of {node.op.name} is not supported in rank >= 6"
        )
        log_assert(
            indice_buf.rank() <= 6, f"indices of {node.op.name} is not supported in rank >= 7"
        )


@register_layer_optimization
class OptimizeSplitTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SplitOp.TRANSLATION_KEY
        self.register_method(REMOVE_IDENTITY, self.remove_identity)

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)

        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            axis_map = spatial_first_format_to_channel_first_permute_order[input_buf.axis_format]
            node.op.axis = axis_map[node.op.axis]
        input_axis_formats_before = graph.get_input_axis_formats(node)
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_axis_formats_after = graph.get_input_axis_formats(node)
        input_buffers = graph.get_input_buffers(node)
        for i, buf in enumerate(input_buffers):
            if input_axis_formats_before[i] != input_axis_formats_after[i]:
                transpose_node = buf.producer
                graph.update_trace_info(transpose_node, [node])
                graph.update_trace_info(buf, [node])
        return True

    @staticmethod
    def remove_identity(node, graph):
        if not len(node.op.split_index):
            graph.squash(node, input_name=node.input_names[0])


@register_layer_optimization
class OptimizeSoftmaxTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SoftmaxOp.TRANSLATION_KEY
        self.register_method(FOLD_SOFTMAX, self.fold_softmax)
        self.register_method(REPLACE_6D_OPERATION, self.replace_6d_operation)

    def replace_6d_operation(self, node, graph):
        in_shapes = graph.get_input_shapes(node)
        out_shapes = graph.get_output_shapes(node)
        rank = len(in_shapes[0])
        if rank <= 5:
            return
        check_if_6d_supportable(node, graph)
        # insert pre-reshape before SoftmaxOp
        # SoftmaxOp has an attribute 'axis'
        # Say, input_dims = [a, b, c, d, e, f] and axis = 2,
        # insert ReshapeOp to reshape to [a*b, c, d*e*f]
        # Do softmax with this 3D input tensor
        # Insert post-reshape op after softmax
        # Reshape back to original dims [a, b, c, d, e, f]
        def reduce_softmax_input_shape(input_shape, axis):
            left_shape = [] if node.op.axis == 0 else [np.prod(input_shape[:node.op.axis])]
            right_shape = [] if node.op.axis == (rank-1) else [np.prod(input_shape[node.op.axis+1:])]
            new_in_shape = left_shape + [input_shape[node.op.axis]] + right_shape

            new_axis = 0 if node.op.axis == 0 else 1
            return new_in_shape, new_axis
        new_in_shape, new_axis = reduce_softmax_input_shape(list(in_shapes[0]), axis=node.op.axis)
        pre_reshape_op_name = node.op.name + '_6d_pre_reshape'
        pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_op_name, shape=new_in_shape)
        graph.inject(pre_reshape_op, input_name=node.input_names[0],
                     output_name=pre_reshape_op_name, consumer_names=[node.op.name])
        new_out_shape = new_in_shape
        node.op.axis = new_axis
        # insert post-reshape
        post_reshape_insertion(node, graph, new_out_shapes=[new_out_shape], orig_out_shapes=out_shapes)

    def axes_to_spatial_first_order(self, node, graph):
        def update_output_info(node, graph):
            # Softmax doesn't change the axis format and shape of input_buf
            # Just use input_buf's axis_format and shape for output_buf
            new_input_buf = graph.get_buffer(node.input_names[0])
            output_buf = graph.get_buffer(node.output_names[0])
            output_buf.axis_format = new_input_buf.axis_format
            output_buf.shape = new_input_buf.shape

        channel_first_to_spatial_first_permute_order = {'NCDHW': AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                                        'NCHW': AxisTracker.AxisFormat.NSC_TO_NCS,
                                                        'NCF': AxisTracker.AxisFormat.NFC_TO_NCF,
                                                        'TNF': AxisTracker.AxisFormat.NTF_TO_TNF,
                                                        'NDHWC': AxisTracker.AxisFormat.NCDHW_TO_NDHWC,
                                                        'NHWC': AxisTracker.AxisFormat.NCS_TO_NSC,
                                                        'NFC': AxisTracker.AxisFormat.NCF_TO_NFC,
                                                        'NTF': AxisTracker.AxisFormat.TNF_TO_NTF,
                                                        'NF': [1, 0]}

        spatial_first_to_channel_first_permute_order = {'NDHWC': AxisTracker.AxisFormat.NDHWC_TO_NCDHW,
                                                        'NHWC': AxisTracker.AxisFormat.NSC_TO_NCS,
                                                        'NFC': AxisTracker.AxisFormat.NFC_TO_NCF,
                                                        'NTF': AxisTracker.AxisFormat.NTF_TO_TNF,
                                                        'NCDHW': AxisTracker.AxisFormat.NCDHW_TO_NDHWC,
                                                        'NCHW': AxisTracker.AxisFormat.NCS_TO_NSC,
                                                        'NCF': AxisTracker.AxisFormat.NCF_TO_NFC,
                                                        'TNF': AxisTracker.AxisFormat.TNF_TO_NTF,
                                                        'NF': [1, 0]}

        has_change = False
        input_buf = graph.get_buffer(node.input_names[0])

        # If input_buf.axis has changed, update axis according to input_buf.axis
        if not AxisTracker.input_axis_formats_intact(graph, node):
            if input_buf.axis_format in spatial_first_to_channel_first_permute_order:
                axis_map = spatial_first_to_channel_first_permute_order[input_buf.axis_format]
                log_debug('Mapping axis from {} to {}: '.format(node.op.axis, axis_map[node.op.axis]))
                node.op.axis = axis_map[node.op.axis]
                update_output_info(node, graph)
                has_change = True
            # Added this check for any 4D input for frcnn_vgg_compressed model
            # where it expects a permute after reshape
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL:
                update_output_info(node, graph)
                has_change = True

        # Make the axis be -1
        rank = input_buf.rank()
        # Check if current axis is expected
        if (node.op.axis == rank - 1):
            return True if has_change else False
        else:
            # Need to add transpose to make axis=rank-1
            if input_buf.axis_format in channel_first_to_spatial_first_permute_order:
                axis_map = channel_first_to_spatial_first_permute_order[input_buf.axis_format]
                if axis_map[node.op.axis] == rank - 1:
                    # Add transpose to change axis to rank-1 for normal axis_format
                    if input_buf.axis_format in [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF]:
                        AxisTracker.enforce_channel_last_input(input_buf, node, graph)
                    elif input_buf.axis_format in [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC]:
                        AxisTracker.enforce_spatial_last_input(input_buf, node, graph)
                    elif input_buf.axis_format in [AxisTracker.AxisFormat.NFC]:
                        AxisTracker.enforce_feature_last_input(input_buf, node, graph)
                    elif input_buf.axis_format in [AxisTracker.AxisFormat.NF]:
                        graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL, [1, 0], consumers=[node.op.name])
                    else:
                        target_format = AxisTracker.AxisFormat.TNF if input_buf.axis_format == AxisTracker.AxisFormat.NTF \
                                        else AxisTracker.AxisFormat.NTF
                        permute_order = AxisTracker.AxisFormat.NTF_TO_TNF if input_buf.axis_format == AxisTracker.AxisFormat.NTF \
                                        else AxisTracker.AxisFormat.TNF_TO_NTF
                        graph.inject_implicit_permute(input_buf.name, target_format, permute_order, consumers=[node.op.name])
                    node.op.axis = rank - 1
                    update_output_info(node, graph)
            # Add transpose to change axis to rank-1 for special axis_format
            if node.op.axis != rank - 1:
                permute_order = [idx for idx in range(rank)]
                permute_order[rank-1], permute_order[node.op.axis] = node.op.axis, rank - 1
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL, permute_order, consumers=[node.op.name])
                node.op.axis = rank - 1
                update_output_info(node, graph)
                # Restore the axis format after Softmax node in case the NONTRIVIAL would be considered as NCHW
                softmax_output_buf = graph.get_buffer(node.output_names[0])
                consumer_names = [consumer.op.name for consumer in list(softmax_output_buf.consumers)]
                graph.inject_implicit_permute(softmax_output_buf.name, input_buf.axis_format, permute_order, consumers=consumer_names)

        return True


    @staticmethod
    def fold_softmax(graph):
        def validate_multiplier(node_tuple):
            def is_scalar(constant_node):
                tensor = constant_node.op.tensor
                return len(tensor.shape) == tensor.size == 1

            def is_positive(constant_node):
                constant_value = constant_node.op.tensor[0]
                return constant_value > 0

            mul_node = node_tuple[0]
            mul_input_nodes = graph.get_op_input_nodes(mul_node)
            constant_index = 0 if isinstance(mul_input_nodes[0].op, op_adapter.ConstantOp) else 1
            constant_node = mul_input_nodes[constant_index]

            return is_scalar(constant_node) and \
                   is_positive(constant_node) and \
                   not graph.has_quantization_param(mul_node)

        sequence = [
            ("elementwise_product",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SOFTMAX, "ALL")])
            ),
            (ir_graph.QNN_OP_SOFTMAX,
             ("MATCH_NUM_BUFS", [("elementwise_product", "ALL")]),
             ()
            )
        ]

        matched_node_list = graph.get_matched_nodes(sequence, validate_multiplier)
        for node_tuple in matched_node_list:
            softmax_node = node_tuple[1]
            mul_node = node_tuple[0]

            mul_input_nodes = graph.get_op_input_nodes(mul_node)
            constant_index = 0 if isinstance(mul_input_nodes[0].op, op_adapter.ConstantOp) else 1
            main_index = 0 if constant_index == 1 else 1

            constant_node = mul_input_nodes[constant_index]
            constant_value = constant_node.op.tensor[0]
            constant_buffer_name = constant_node.output_names[0]
            main_buffer_name = mul_node.input_names[main_index]

            graph.remove_node_as_consumer(mul_node, constant_buffer_name)
            graph.squash(mul_node, main_buffer_name, squash_into_next=True)
            current_beta = softmax_node.op.__getattr__(ir_graph.QNN_OP_SOFTMAX_PARAM_BETA)
            new_beta = current_beta * constant_value
            softmax_node.op.__setattr__(ir_graph.QNN_OP_SOFTMAX_PARAM_BETA, new_beta)


@register_layer_optimization
class OptimizeSqueezeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SqueezeOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if AxisTracker.input_axis_formats_intact(graph, node) and \
                input_buf.axis_format in AxisTracker.AxisFormat.get_valid_formats():
            return False

        # force convergence if necessary
        # use the 'backwards' permute orders because they are self-inverses.
        # Check if input is a permute, if so this means the source framework deliberately added the permute
        # and we do not want to inject another one.
        if input_buf.producer.op.type != op_adapter.TransposeOp.TRANSLATION_KEY:
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[0]:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NC or \
                    input_buf.axis_format == AxisTracker.AxisFormat.ANY or \
                    input_buf.axis_format == AxisTracker.AxisFormat.TNF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCS or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCF or \
                    input_buf.axis_format == AxisTracker.AxisFormat.NCDHW:
                pass
            else:
                raise ValueError(code_to_message.get_error_message("ERROR_RESHAPE_UNEXPECTED_INPUT_ORDER")
                                 (input_buf.axis_format))

            return True


@register_layer_optimization
class OptimizeUdlTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UdlOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_names = node.input_names
        for input_name in input_names:
            input_buf = graph.get_buffer(input_name)
            current_input_order = input_buf.get_axis_annotations()
            expected_input_order = []
            for dims in node.op.expected_input_axis_orders:
                if len(dims) == input_buf.rank():
                    expected_input_order = dims
            target_input_type = AxisTracker.get_axis_format_from_annotation(expected_input_order)
            permute_order = AxisTracker.compute_permute_order(current_input_order, expected_input_order)
            if len(permute_order) and permute_order != list(range(len(permute_order))):
                graph.inject_implicit_permute(input_name, target_input_type,
                                              permute_order, [node.op.name])

            target_output_order = []
            output_buffers = graph.get_output_buffers(node)
            for output_buf in output_buffers:
                for dims in node.op.expected_output_axis_orders:
                    if len(dims) == output_buf.rank():
                        target_output_order = dims
                output_buf.axis_format = AxisTracker.get_axis_format_from_annotation(target_output_order)
        return True


@register_layer_optimization
class OptimizeCropAndResizeTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CropAndResizeOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExtractGlimpseTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExtractGlimpseOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeExtractPatchesTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ExtractPatchesOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeImageProjectiveTransformTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ImageProjectiveTransformOp.TRANSLATION_KEY


@register_layer_optimization
class OptimizeMomentTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MomentOp.TRANSLATION_KEY

@register_layer_optimization
class OptimizeCombinedNmsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.CombinedNmsOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        for i, input_buf in enumerate(input_bufs):
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        input_bufs = graph.get_input_buffers(node)
        boxes_batch, num_boxes = input_bufs[0].shape[:2]

        for i, input_buf in enumerate(input_bufs):
            # the first input is boxes, the other inputs need to be 3D but NONTRIVIAL
            if (i == 0):
                continue
            # handle case where buf in NON TRIVIAL but not in expected order
            if input_buf.rank() == 3 and input_buf.shape[2] == num_boxes:
                graph.inject_implicit_permute(node.input_names[i], AxisTracker.AxisFormat.NONTRIVIAL,
                                              [0, 2, 1], [node.op.name])
                input_buf = graph.get_input_buffers(node)[i]
            # verify each input meets spec [batch, num_boxes] spec
            log_assert(input_buf.shape[:2] == [boxes_batch, num_boxes],
                       "Unable to get proper axis order for {} to expected prefix [batch, num_boxes]. Cannot match "
                       "input shapes [{}] with boxes input shapes [{}] for nms node {}."
                       .format(input_buf.name, input_buf.shape, input_bufs[0].shape, node.op.name))

        return True


@register_layer_optimization
class OptimizeMultiClassNmsTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.MultiClassNmsOp.TRANSLATION_KEY
        self.register_method(ADJUST_NMS_FEATURE_DIMS, self.adjust_nms_feature_dimensions)

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        for i, input_buf in enumerate(input_bufs):
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        input_bufs = graph.get_input_buffers(node)
        boxes_batch, num_boxes = input_bufs[0].shape[:2]

        for i, input_buf in enumerate(input_bufs):
            # the first input is boxes, the other inputs need to be 3D but NONTRIVIAL
            if (i == 0):
                continue
            # handle case where buf in NON TRIVIAL but not in expected order
            if input_buf.rank() == 3 and input_buf.shape[2] == num_boxes:
                graph.inject_implicit_permute(node.input_names[i], AxisTracker.AxisFormat.NONTRIVIAL,
                                              [0, 2, 1], [node.op.name])
                input_buf = graph.get_input_buffers(node)[i]
            # verify each input meets spec [batch, num_boxes] spec
            log_assert(input_buf.shape[:2] == [boxes_batch, num_boxes],
                       "Unable to get proper axis order for {} to expected prefix [batch, num_boxes]. Cannot match "
                       "input shapes [{}] with boxes input shapes [{}] for nms node {}."
                       .format(input_buf.name, input_buf.shape, input_bufs[0].shape, node.op.name))

        return True

    @staticmethod
    def adjust_nms_feature_dimensions(graph):
        """
        By default nms requires 2 inputs for boxes and score whose input and output shape is handled in
        TF translation. With the extra input_features they do not typically come with batch dimensions, so handle
        here by verifying required second dimension equality with num_boxes
        TODO: remove once backend consolidate input/output shapes of features to MultiClassNms. This should be
        handled during TF translation similar to the boxes and scores input.
        """

        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            # adjustment of features only needed if features are given as inputs
            if len(nms_node_.input_names) > 2 and len(nms_node_.output_names) > 4 and \
                    "scale_y" not in nms_node_.op.attrs:
                return True
            return False

        sequence = [
            (ir_graph.QNN_OP_MULTI_CLASS_NMS,
             (),
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)
        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_input_names = nms_node.input_names
            nms_output_names = nms_node.output_names
            num_boxes = graph.get_buffer(nms_node.input_names[0]).shape[1]
            for i in range(2, len(nms_node.input_names)):
                input_feature_buf = graph.get_buffer(nms_input_names[i])
                input_feature_shape = input_feature_buf.shape
                if len(input_feature_shape) == 1 or input_feature_shape[1] != num_boxes:
                    input_feature_node = graph.get_producer_node(nms_input_names[i])
                    # add reshape node to add batch dimension to the input features
                    expected_input_feature_shape = [1, *input_feature_shape]
                    # verify this is will result in expected input
                    log_assert(expected_input_feature_shape[1] == num_boxes,
                               "Unable to adjust input feature to match expected num_boxes on second dimension. "
                               "Got: {}, Expected num_boxes {}".format(expected_input_feature_shape, num_boxes))

                    if input_feature_node.op.type == op_adapter.ReshapeOp.TRANSLATION_KEY and \
                            graph.get_buffer(input_feature_node.input_names[0]).shape == expected_input_feature_shape:
                        # there was a squeeze done to remove batch dim, remove it and adjust to expected
                        # input feature instead.
                        graph.squash(input_feature_node, input_name=input_feature_node.input_names[0])
                        graph.get_buffer(input_feature_node.output_names[0]).set_buf_dims(expected_input_feature_shape)
                    else:
                        # add the reshape to add batch dim
                        input_feature_reshape_node_name = nms_input_names[i] + "_reshape_batch_add"
                        input_feature_reshape_op = op_adapter.ReshapeOp(name=input_feature_reshape_node_name,
                                                                        shape=expected_input_feature_shape)
                        graph.inject(input_feature_reshape_op, input_name=nms_input_names[i],
                                     output_name=input_feature_reshape_node_name,
                                     consumer_names=[nms_node.op.name])

                    # since we are reshaping input, output from nms will need to be adjusted as intermediate and
                    # will require a post reshape to remove batch dimension added.
                    output_name_idx = i + 2  # accounting for class and num_det output
                    output_feature_name = nms_output_names[output_name_idx]
                    output_feature_buf = graph.get_buffer(output_feature_name)
                    # replace the nms output as intermediate and the post reshaped output as the src fw output_feature
                    graph.delete_buffer(output_feature_name)
                    output_feature_reshape_op = op_adapter.ReshapeOp(name=output_feature_name,
                                                                     shape=output_feature_buf.shape)
                    # adjust to expected buffer shape for nms feature output(i.e with batch dim) and rename buffer as
                    # intermediate
                    output_feature_buf.set_buf_dims([1, *output_feature_buf.shape])
                    intermediate_output_name = output_feature_name + "_intermediate"
                    output_feature_buf.name = intermediate_output_name
                    graph.add_buffer(output_feature_buf)
                    nms_output_names[output_name_idx] = intermediate_output_name
                    graph.inject(output_feature_reshape_op, input_name=intermediate_output_name,
                                 output_name=output_feature_name)

                    # Addition of a const tensor to features should not be quantized
                    # TODO: add conditional that it should be set non quantizable based on tensortype and
                    #       quantization info of input tensor when irgraph supports these info
                    output_feature_reshape_buf = graph.get_buffer(output_feature_name)
                    for consumer in output_feature_reshape_buf.consumers:
                        if isinstance(consumer.op, op_adapter.ElementwiseBinaryOp):
                            for input_name in consumer.input_names:
                                eltwise_input_node = graph.get_producer_node(input_name)
                                if eltwise_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                                    eltwise_input_node.op.quantizable = False


@register_layer_optimization
class OptimizeNonMaxSuppressionTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.NonMaxSuppressionOp.TRANSLATION_KEY
        self.register_method(MERGE_LOW_LEVEL_OPS_TO_LAYERS, self.merge_low_level_ops_to_layers)

    def axes_to_spatial_first_order(self, node, graph):
        input_bufs = graph.get_input_buffers(node)
        for i, input_buf in enumerate(input_bufs):
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    input_buf.axis_format != node.op.data_axis_formats[i]:
                graph.inject_implicit_permute(input_buf.name, AxisTracker.AxisFormat.NONTRIVIAL,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])

        return True

    def merge_low_level_ops_to_layers(self, graph):
        validate_node = None

        sequence1 = [
            (ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            (ir_graph.QNN_OP_GATHER,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_NON_MAX_SUPPRESSION, "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_STRIDED_SLICE, "ALL")])
            ),
            (ir_graph.QNN_OP_STRIDED_SLICE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            )
        ]
        sequence2 = [
            (ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
            (ir_graph.QNN_OP_GATHER,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_NON_MAX_SUPPRESSION, "ANY"), ("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_RESHAPE, "ALL")]),
            ),
            (ir_graph.QNN_OP_RESHAPE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            ),
        ]
        sequence3 = [
            (ir_graph.QNN_OP_NON_MAX_SUPPRESSION,
             (),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_STRIDED_SLICE, "ALL")])
            ),
            (ir_graph.QNN_OP_STRIDED_SLICE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_NON_MAX_SUPPRESSION, "ALL")]),
             ("FLEXIBLE_NUM_BUFS", [(ir_graph.QNN_OP_GATHER, "ALL")])
            )
        ]

        sequences = [sequence1, sequence2, sequence3]

        for sequence in sequences:
            matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node, ignore_constants=True)
            for node_tuple in matched_node_list:
                onnx_nms_node = node_tuple[0]
                if len(onnx_nms_node.output_names) != 1:
                    continue
                onnx_nms_op = onnx_nms_node.op
                nms_output_names = ['{}_boxes'.format(onnx_nms_op.name),
                                    '{}_scores'.format(onnx_nms_op.name),
                                    '{}_classes'.format(onnx_nms_op.name),
                                    '{}_num_detections'.format(onnx_nms_op.name)]

                nms_max_total_detections = onnx_nms_node.op.max_boxes_selected
                nms_iou_threshold = onnx_nms_node.op.iou_threshold
                nms_score_threshold = onnx_nms_node.op.score_threshold

                # Replace to MultiClassNmsOp
                nms_op_name = onnx_nms_op.name + '_gather'
                nms_op = op_adapter.MultiClassNmsOp(nms_op_name,
                                                    max_total_detections=nms_max_total_detections,
                                                    iou_threshold=nms_iou_threshold,
                                                    score_threshold=nms_score_threshold
                                                    )
                nms_input_names = onnx_nms_node.input_names.copy()
                last_node = node_tuple[-1]
                last_output_buf = graph.get_output_buffers(last_node)[0]

                pruned_nodes = []
                box_n_class_succors = []
                box_n_class_succors_input = []
                feature_consumer_succors = []
                for consumer in last_output_buf.consumers:
                    if consumer.op.type == ir_graph.QNN_OP_GATHER:
                        consumer_input_names = consumer.input_names
                        gather_data_inputs = [input_name for input_name in consumer_input_names if input_name != last_output_buf.name]
                        # boxes and classes nodes have been done in nms_output_names
                        # therefore no need to create an extra output from gather op
                        if gather_data_inputs[0] in nms_input_names[:2]:
                            box_n_class_succors_input.append(nms_output_names[nms_input_names.index(gather_data_inputs[0])])
                            box_n_class_succors.append(graph.get_output_buffers(consumer)[0].consumers)
                        # feature parts, which need to be added as extra outputs
                        # connected the graph by nms output[4:]
                        else:
                            nms_input_names.extend(gather_data_inputs)
                            nms_output_names.extend(consumer.output_names)
                            # gather has only one output buffer
                            feature_consumer_succors.append(graph.get_output_buffers(consumer)[0].consumers)
                        pruned_nodes.append(consumer)

                # Record trace info before prune nodes
                orig_trace_info = graph.get_trace_info_sub_graph(node_tuple)

                for node in pruned_nodes:
                    graph.prune(node, force_remove=True)

                # Prune the nodes after extract required information
                for node_in_tuple in reversed(node_tuple):
                    graph.prune(node_in_tuple, force_remove=True)
                idx_to_insert = 0
                for input_name in nms_input_names:
                    buf = graph.get_buffer(input_name)
                    cur_idx = graph.nodes_in_order.index(buf.producer)
                    if idx_to_insert <= cur_idx:
                        idx_to_insert = cur_idx + 1
                nms_node = graph.add(nms_op, input_names=nms_input_names, output_names=nms_output_names,idx=idx_to_insert)
                # Set trace info for new created node
                nms_node_outputs = [nms_node]
                nms_node_outputs.extend(graph.get_output_buffers(nms_node))
                graph.set_trace_info(nms_node_outputs, orig_trace_info)

                # re-connected the nodes after gather
                # box, scores part
                for idx, succs in enumerate(box_n_class_succors):
                    for succ_node in succs:
                        succ_node.input_names.append(box_n_class_succors_input[idx])
                        nms_output_buf = graph.get_buffer(nms_output_names[idx])
                        nms_output_buf.consumers.add(succ_node)
                # feature part
                for idx, succs in enumerate(feature_consumer_succors):
                    succ_input_name = nms_output_names[4+idx]
                    for succ_node in succs:
                        succ_node.input_names.append(succ_input_name)
                        nms_output_buf = graph.get_buffer(nms_output_names[4+idx])
                        nms_output_buf.consumers.add(succ_node)


@register_layer_optimization
class OptimizePackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.PackOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        for idx, input_name in enumerate(node.input_names):
            input_buf = graph.get_buffer(input_name)
            # Pack needs to happen in src format, so in case the current input format and the data_axis_format
            # are different, inject permute to change it back to src format
            if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.NCDHW:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                              AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.NCS:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                              AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.NCF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                              AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
            elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                    node.op.data_axis_formats[idx] == AxisTracker.AxisFormat.TNF:
                graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                              AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
            else:
                log_debug2("No axes change for Op {}".format(node.op.name))


@register_layer_optimization
class OptimizeBufferTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BufferOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        # Mutate the output dimensions and update the concat/buffer dim,
        # if the input dimensions are transposed
        input_name = node.input_names[0]
        input_buf = graph.get_input_buffers(node)[0]
        output_buf = graph.get_output_buffers(node)[0]

        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            axis_map = AxisTracker.AxisFormat.NDHWC_TO_NCDHW
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            node.op.buffer_dim = axis_map[node.op.buffer_dim]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            axis_map = AxisTracker.AxisFormat.NSC_TO_NCS
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            node.op.buffer_dim = axis_map[node.op.buffer_dim]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            axis_map = AxisTracker.AxisFormat.NFC_TO_NCF
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            node.op.buffer_dim = axis_map[node.op.buffer_dim]
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            axis_map = AxisTracker.AxisFormat.NTF_TO_TNF
            AxisTracker.alter_axis_format_to_ir_order(node, graph)
            node.op.buffer_dim = axis_map[node.op.buffer_dim]
        # Don't change the output domensions and axis for NONTRIVIAL layout
        elif input_buf.axis_format == AxisTracker.AxisFormat.NONTRIVIAL and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NONTRIVIAL:
            pass
        # Don't change anything if the layouts match
        elif input_buf.axis_format == node.op.data_axis_formats[0]:
            pass
        else:
            raise ValueError("Error while handling axis layouts in Buffer op named {}."
                " Only handling NDHWC, NCS, NFC, NTF and NONTRIVIAL axis formats now."
                .format(node.op.name))

        return True

@register_layer_optimization
class OptimizeDepthToSpaceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.DepthToSpaceOp.TRANSLATION_KEY
        self.register_method(MATCH_DEPTHTOSPACE, self.match_depthtospace)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.rank() != 4:
            raise ValueError("Backend only support DepthToSpace with rank 4, but got an input rank with {}."
                             .format(input_buf.rank()))
        # To ensure the depthToSpace's input and output format as NSC
        AxisTracker.image_to_channel_last_order(node, graph)
        return True

    @staticmethod
    def match_depthtospace(graph):
        # To validate the getting node tuple match the optimiaztion solution.
        def validate_node_tuple(node_tuple):
            for d2s_node in node_tuple[3:5]:
                if d2s_node.op.mode != ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR:
                    return False
            conv_op = node_tuple[0]
            conv_out_channel_idx = AxisTracker.get_axis_annotation_from_format(
                graph.get_buffer(conv_op.input_names[1]).axis_format
                ).index(AxisTracker.AxisAnnotations.OUTPUT_CHANNELS)
            split_node = node_tuple[2]
            split_axis = split_node.op.__getattr__(ir_graph.QNN_OP_SPLIT_PARAM_AXIS)
            concat_node = node_tuple[6]
            concat_axis = concat_node.op.__getattr__(ir_graph.QNN_OP_CONCAT_PARAM_AXIS)
            if split_axis != concat_axis or split_axis != conv_out_channel_idx:
                return False
            return True

        # rearrange the conv data to reorder the channel axis from CRD to DCR
        def rearrange_conv(weight, bias, conv_out_channel_idx, block_size):
            block_height = block_size[0]
            block_width = block_size[1]
            new_weight = weight.copy()
            new_bias = bias.copy()
            conv_out_channel_dim = weight.shape[conv_out_channel_idx]
            depth = int(conv_out_channel_dim / (block_height * block_width))
            idxes = np.zeros((block_height, block_width, depth), dtype=np.dtype("int32"))
            idx = 0
            # Reorder the channel axis from CRD to DCR
            for k in range(depth):
                for i in range(block_height):
                    for j in range(block_width):
                        idxes[i,j,k] = idx
                        idx = idx + 1
            idx_list = idxes.flatten()
            for i in range(conv_out_channel_dim):
                if weight.ndim == 4:
                    if conv_out_channel_idx == 1:
                        new_weight[:,i,:,:] = weight[:,idx_list[i],:,:]
                    if conv_out_channel_idx == 3:
                        new_weight[:,:,:,i] = weight[:,:,:,idx_list[i]]
                elif weight.ndim == 5:
                    if conv_out_channel_idx == 1:
                        new_weight[:,i,:,:,:] = weight[:,idx_list[i],:,:,:]
                    if conv_out_channel_idx == 4:
                        new_weight[:,:,:,:,i] = weight[:,:,:,:,idx_list[i]]
                new_bias[i] = bias[idx_list[i]]
            return new_weight, new_bias

        sequence = [
            (ir_graph.QNN_OP_CONV_2D,
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")])
             ),
            ("elementwise_sum",
             ("FLEXIBLE_NUM_BUFS", [("constant", "ANY")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")])
             ),
            (ir_graph.QNN_OP_SPLIT,
             ("MATCH_NUM_BUFS", [("elementwise_sum", "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY")])
             ),
            (ir_graph.QNN_OP_DEPTH_TO_SPACE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_CONCAT, "ALL")])
             ),
            (ir_graph.QNN_OP_DEPTH_TO_SPACE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_CONCAT, "ALL")])
             ),
            (ir_graph.QNN_OP_DEPTH_TO_SPACE,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_SPLIT, "ALL")]),
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_CONCAT, "ALL")])
             ),
            (ir_graph.QNN_OP_CONCAT,
             ("MATCH_NUM_BUFS", [(ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY"), (ir_graph.QNN_OP_DEPTH_TO_SPACE, "ANY")]),
             ()
             ),
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validate_node_tuple, ignore_constants=True)

        for node_tuple in matched_node_list:
            conv_op = node_tuple[0]
            conv_out_channel_idx = AxisTracker.get_axis_annotation_from_format(
                graph.get_buffer(conv_op.input_names[1]).axis_format
                ).index(AxisTracker.AxisAnnotations.OUTPUT_CHANNELS)
            add_op = node_tuple[1]
            split_node = node_tuple[2]
            block_size = node_tuple[3].op.block_size
            weight_producer = graph.get_producer_op(conv_op.input_names[1])
            bias_producer = graph.get_producer_op(conv_op.input_names[2])
            if bias_producer.tensor.max() == bias_producer.tensor.min() == 0:
                bias_producer = graph.get_producer_op(add_op.input_names[1])
            new_weights, new_bias = rearrange_conv(weight_producer.tensor, bias_producer.tensor, conv_out_channel_idx, block_size)
            weight_producer.tensor = new_weights
            bias_producer.tensor = new_bias

            # merge the three DCR D2S to one DCR D2s
            for node in node_tuple[:2:-1]:
                input_names = node.input_names[:]
                # pick squashable input based on whether current node is only consumer and input is not network input
                input_name = [name for name in input_names if (len(graph.get_buffer(name).consumers) == 1 and
                              not isinstance(graph.get_producer_op(name), op_adapter.InputOp))][0]
                input_names.remove(input_name)
                for input_name_ in input_names:
                    # disconnect rest of inputs from node
                    input_buf_ = graph.get_buffer(input_name_)
                    input_buf_.consumers.remove(node)
                    node.input_names.remove(input_name_)
                graph.squash(node, input_name=input_name)

            redundant_outputs = [name for name in split_node.output_names if len(graph.get_buffer(name).consumers) == 0]
            for name in redundant_outputs:
                graph.delete_buffer(name)
                split_node.output_names.remove(name)

            # replace the split + 3 DCR D2S op with 1 DCR D2S
            split_op = split_node.op
            split_op_name = graph.naming_policy.get_op_name(split_op)
            DCR_D2S_op_name = split_op_name + '_DCR_DepthToSpace'
            DCR_D2S_op = op_adapter.DepthToSpaceOp(DCR_D2S_op_name, block_size=block_size, mode=ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR)
            graph.replace(split_op, DCR_D2S_op)

    def merge_low_level_ops_to_layers(self, graph):
        # DCR mode: elements along the depth dimension are rearranged in the order of depth, column, and then row.
        #     input: [n, c, h, w]
        #     reshape: [n, blk_h, blk_w, c/(blk_h*blk_w), h, w]
        #     transpose: [n, c/(blk_h*blk_w), h, blk_h, w, blk_w] with [0, 3, 4, 1, 5, 2]
        #     reshape: [n, c/(blk_h*blk_w), h*blk_h, w*blk_w]
        #
        # CRD mode: elements along the depth dimension are rearranged in the order of column, row, and then depth.
        #     input: [n, c, h, w]
        #     reshape: [n, c/(blk_h*blk_w), blk_h, blk_w, h, w]
        #     transpose: [n, c/(blk_h*blk_w), h, blk_h, w, blk_w] with [0, 1, 4, 2, 5, 3]
        #     reshape: [n, c/(blk_h*blk_w), h*blk_h, w*blk_w]

        def validate(nodes_tuple_):
            reshape_to_6D_node = nodes_tuple_[0]
            permute_node = nodes_tuple_[1]
            reshape_to_4D_node = nodes_tuple_[2]

            reshape6d_input_shape = graph.get_input_shapes(reshape_to_6D_node)[0]
            reshape6d_output_shape = graph.get_output_shapes(reshape_to_6D_node)[0]
            reshape4d_input_shape = graph.get_input_shapes(reshape_to_4D_node)[0]
            reshape4d_output_shape = graph.get_output_shapes(reshape_to_4D_node)[0]

            # Check the output shape should be 4
            if len(reshape4d_output_shape) != 4:
                return False

            if len(reshape6d_output_shape) == 5:
                # Check the Channel dimension is split into blocks
                if np.prod(reshape6d_output_shape[1:3]) != reshape6d_input_shape[1]:
                    return False
                # Check the permute order for CRD or DCR mode
                if permute_node.op.perm not in [[0,1,3,4,2]]:
                    return False
                # Check that the block_size was reshaped into H and W
                if reshape4d_output_shape[2] != np.prod(reshape4d_input_shape[2:3]) or \
                        reshape4d_output_shape[3] != np.prod(reshape4d_input_shape[3:]):
                    return False
                return True

            if len(reshape6d_output_shape) == 6:
                # Check the Channel dimension is split into blocks
                if np.prod(reshape6d_output_shape[1:4]) != reshape6d_input_shape[1]:
                    return False
                # Check the permute order for CRD or DCR mode
                if permute_node.op.perm not in [[0,1,4,2,5,3], [0,3,4,1,5,2]]:
                    return False
                # Check that the block_size was reshaped into H and W
                if reshape4d_output_shape[2] != np.prod(reshape4d_input_shape[2:4]) or \
                        reshape4d_output_shape[3] != np.prod(reshape4d_input_shape[4:]):
                    return False
                return True

            return False

        sequence = [
                    (ir_graph.QNN_OP_RESHAPE, (), ()),
                    (ir_graph.QNN_OP_TRANSPOSE,
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_RESHAPE, 0)]),
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_RESHAPE, 0)]),),
                    (ir_graph.QNN_OP_RESHAPE,
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_TRANSPOSE, 0)]),
                        ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            reshape_to_6D_node = nodes_tuple[0]
            permute_node = nodes_tuple[1]
            reshape_to_4D_node = nodes_tuple[2]

            reshape4d_input_shape = graph.get_input_shapes(reshape_to_4D_node)[0]
            upscale_factor_height = 0
            upscale_factor_width = 0

            if len(reshape4d_input_shape) == 5:
                upscale_factor_height = 1
                upscale_factor_width = reshape4d_input_shape[4]
            elif len(reshape4d_input_shape) == 6:
                upscale_factor_height = reshape4d_input_shape[3]
                upscale_factor_width = reshape4d_input_shape[5]
            else:
                continue

            if permute_node.op.perm == [0, 1, 4, 2, 5, 3]:
                d2s_mode = ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD
            elif permute_node.op.perm == [0, 1, 3, 4, 2]:
                d2s_mode = ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_CRD
            else:
                d2s_mode = ir_graph.QNN_OP_DEPTH_TO_SPACE_MODE_DCR

            # Squashes the reshape_4d node
            reshape4d_input_buffer = graph.get_input_buffers(reshape_to_4D_node)[0]
            graph.squash(reshape_to_4D_node, input_name=reshape4d_input_buffer.name)

            # Squashes the permute node
            permute_input_buffer = graph.get_input_buffers(permute_node)[0]
            graph.squash(permute_node, input_name=permute_input_buffer.name)

            # Replace the reshape6D OpNode to a DepthToSpace OpNode
            d2s_op = op_adapter.DepthToSpaceOp(name=permute_node.op.name,
                                               block_size=[upscale_factor_height, upscale_factor_width],
                                               mode=d2s_mode)
            graph.replace(reshape_to_6D_node.op, d2s_op)

            # Update consumers' data_axis_formats
            d2s_node = graph.nodes_by_name[d2s_op.name]
            output_buffer = graph.get_output_buffers(d2s_node)[0]
            for consumer in output_buffer.consumers:
                consumer.op.populate_data_axis_formats(graph, graph.get_input_buffers(consumer))


@register_layer_optimization
class OptimizeStridedSliceTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.StridedSliceOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if not super(OptimizeStridedSliceTranslation, self).axes_to_spatial_first_order(node, graph):
            # No change in input formats, and none of the input formats are NonTrivial
            return False

        input_buf = graph.get_buffer(node.input_names[0])

        spatial_last_axis_formats = [AxisTracker.AxisFormat.NCDHW, AxisTracker.AxisFormat.NCS, AxisTracker.AxisFormat.NCF, AxisTracker.AxisFormat.TNF]
        spatial_first_axis_formats = [AxisTracker.AxisFormat.NDHWC, AxisTracker.AxisFormat.NSC, AxisTracker.AxisFormat.NFC, AxisTracker.AxisFormat.NTF]

        # if data_axis_formats is spatial-last order and input_buf.axis_format is spatial-first, transform the attributes
        if (node.op.data_axis_formats[0], input_buf.axis_format) in list(zip(spatial_last_axis_formats, spatial_first_axis_formats)) or \
            (node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NONTRIVIAL and input_buf.axis_format in spatial_first_axis_formats):

            begins, ends, strides = list(map(list, zip(*node.op.ranges.tolist())))

            # tranform begins/ends/strides from spatial-last format to spatial-first format
            begins = SpatialLastAxisOrder().permute_shape_to_ir(begins, input_buf.axis_format)
            ends = SpatialLastAxisOrder().permute_shape_to_ir(ends, input_buf.axis_format)
            strides = SpatialLastAxisOrder().permute_shape_to_ir(strides, input_buf.axis_format)

            ranges_data = np.array(list(map(list, zip(begins, ends, strides))), dtype=np.int32)
            ranges = ir_graph.IrStaticTensor(ir_graph.QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                             list(ranges_data.shape),
                                             ranges_data,
                                             ir_graph.QNN_DATATYPE_INT_32)
            node.op.ranges = ranges

        return True


@register_layer_optimization
class OptimizeSpaceToDepthTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.SpaceToDepthOp.TRANSLATION_KEY
        self.register_method(MATCH_SPACETODEPTH, self.match_spacetodepth)

    def axes_to_spatial_first_order(self, node, graph):
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.rank() != 4:
            raise ValueError("Backend only support SpaceToDepth with rank 4, but got an input rank with {}."
                             .format(input_buf.rank()))
        # To ensure the SpaceToDepthOp's input and output format as NSC
        AxisTracker.image_to_channel_last_order(node, graph)
        return True

    @staticmethod
    def match_spacetodepth(graph):
        # check shapes in the reshape layers
        # [n, c, h, w] -> [n, c * blk**2, h/blk, w/blk]
        def is_valid_spacetodepth(node_tuple):
            input_buf = graph.get_input_buffers(node_tuple[0])[0]
            input_shape = input_buf.shape
            first_reshape_output_shape = graph.get_output_shapes(node_tuple[0])[0]
            if len(input_shape) == 4 and len(first_reshape_output_shape) == 6:
                blocksize = first_reshape_output_shape[3]
                sequence_output_shape = graph.get_output_shapes(node_tuple[-1])[0]

                batch, height, width, channel = graph.src_axis_order.extract_2d_spatial_dims(input_shape)
                expected_shape = graph.src_axis_order.format_2d_spatial_output_shape(batch_size=batch,
                                                                                     channel=channel * (blocksize**2),
                                                                                     height=height//blocksize,
                                                                                     width=width//blocksize)
                return sequence_output_shape == expected_shape
            else:
                return False

        # reshape:   [n, c, h/blk1, blk1, w/blk2, blk2], blk1 == blk2, number is for transpose order.
        # transpose: [n, c, h/blk1, w/blk2, blk1, blk2]
        # reshape:   [n, c, h/blk * w/blk, blk ** 2]
        # transpose: [n, c, blk ** 2, h/blk * w/blk]
        # reshape:   [n, c, blk ** 2, h/blk, w/blk]
        # transpose: [n, blk ** 2, c, h/blk, w/blk]
        # reshape:   [n, c*(blk**2), h/blk, w/blk]
        sequence = [
            ("Reshape",
             (),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
            ),
            ("Transpose",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")])
            ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
            ),
            ("Transpose",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
            ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")])
            ),
            ("Transpose",
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
             ("MATCH_NUM_BUFS", [("Reshape", "ALL")]),
            ),
            ("Reshape",
             ("MATCH_NUM_BUFS", [("Transpose", "ALL")]),
             ()
            )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=is_valid_spacetodepth, ignore_constants=True)
        for node_tuple in matched_node_list:
            downscale_factor = graph.get_output_shapes(node_tuple[0])[0][3]
            block_size = [downscale_factor] * 2
            reshape_node = node_tuple[0]
            # Squash all nodes except the first reshape in reverse order
            # the first reshape op will be replaced
            for node in node_tuple[:0:-1]:
                for input_name in node.input_names:
                    graph.squash(node, input_name=input_name)
            reshape_op = reshape_node.op
            reshape_op_name = reshape_op.name
            spacetodepth_op_name = reshape_op_name + '_space_to_depth'
            spacetodepth_op = op_adapter.SpaceToDepthOp(spacetodepth_op_name, block_size=block_size)
            graph.replace(reshape_op, spacetodepth_op)

    def merge_low_level_ops_to_layers(self, graph):
        # CRD mode: elements along the depth dimension are rearranged in the order of column, row, and then depth.
        #     input:     [n, c, h, w]
        #     reshape:   [n, c, h/blk1, blk1, w/blk2, blk2]
        #     transpose: [n, c, blk1, blk2, h/blk1, w/blk2] with [0, 1, 3, 5, 2, 4]
        #     reshape:   [n, c*(blk1*blk2), h/blk, w/blk]

        # DCR mode: elements along the depth dimension are rearranged in the order of depth, column, and then row.
        #     input:     [n, c, h, w]
        #     reshape:   [n, c, h/blk1, blk1, w/blk2, blk2]
        #     transpose: [n, blk1, blk2, c, h/blk1, w/blk2] with [0, 3, 5, 1, 2, 4]
        #     reshape:   [n, (blk1*blk2)*c, h/blk, w/blk]

        def validate(nodes_tuple_):
            reshape_to_6D_node = nodes_tuple_[0]
            permute_node = nodes_tuple_[1]
            reshape_to_4D_node = nodes_tuple_[2]

            reshape6d_input_shape = graph.get_input_shapes(reshape_to_6D_node)[0]
            reshape6d_output_shape = graph.get_output_shapes(reshape_to_6D_node)[0]
            reshape4d_input_shape = graph.get_input_shapes(reshape_to_4D_node)[0]
            reshape4d_output_shape = graph.get_output_shapes(reshape_to_4D_node)[0]

            # Check the output shape should be 4
            if len(reshape4d_output_shape) != 4:
                return False

            if len(reshape6d_input_shape) == 4 and len(reshape6d_output_shape) == 5:
                # Check H and W is split into blocks
                if reshape6d_input_shape[2] != np.prod(reshape6d_output_shape[2:3]) and \
                    reshape6d_input_shape[3] != np.prod(reshape6d_output_shape[3:]) :
                    return False
                # Check the permute order for CRD mode (split W into blocks)
                # TODO: Need to add check for all 5D cases
                if permute_node.op.perm not in [[0,1,4,2,3]]:
                    return False
                # Check that the block_size is reshaped into Channel
                if reshape4d_output_shape[1] != np.prod(reshape4d_input_shape[1:3]):
                    return False
                return True

            if len(reshape6d_input_shape) == 4 and len(reshape6d_output_shape) == 6:
                # Check H and W is split into blocks
                if reshape6d_input_shape[2] != np.prod(reshape6d_output_shape[2:4]) and \
                    reshape6d_input_shape[3] != np.prod(reshape6d_output_shape[4:6]) :
                    return False
                # Check the permute order for CRD or DCR mode
                if permute_node.op.perm not in [[0,1,3,5,2,4], [0,3,5,1,2,4]]:
                    return False
                # Check that the block_size is reshaped into Channel
                if reshape4d_output_shape[1] != np.prod(reshape4d_input_shape[1:4]):
                    return False
                return True

            return False

        sequence = [
                    (ir_graph.QNN_OP_RESHAPE, (), ()),
                    (ir_graph.QNN_OP_TRANSPOSE,
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_RESHAPE, 0)]),
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_RESHAPE, 0)]),),
                    (ir_graph.QNN_OP_RESHAPE,
                        ("MATCH_BUFS_AT_INDEX", [(ir_graph.QNN_OP_TRANSPOSE, 0)]),
                        ())
                   ]

        matched_node_list = graph.get_matched_nodes(sequence, validator=validate)
        for nodes_tuple in matched_node_list:
            reshape_to_6D_node = nodes_tuple[0]
            permute_node = nodes_tuple[1]
            reshape_to_4D_node = nodes_tuple[2]

            reshape6d_output_shape = graph.get_output_shapes(reshape_to_6D_node)[0]
            downscale_factor_height = 0
            downscale_factor_width = 0

            if len(reshape6d_output_shape) == 5:
                # TODO: Once the checks for all 5D cases are added, need to assign the
                # downscale factor based on the axis which is splited into blocks
                downscale_factor_height = 1
                downscale_factor_width = reshape6d_output_shape[4]
            elif len(reshape6d_output_shape) == 6:
                downscale_factor_height = reshape6d_output_shape[3]
                downscale_factor_width = reshape6d_output_shape[5]
            else:
                continue

            if permute_node.op.perm == [0, 1, 3, 5, 2, 4]:
                s2d_mode = ir_graph.QNN_OP_SPACE_TO_DEPTH_MODE_CRD
            elif permute_node.op.perm == [0, 1, 4, 2, 3]:
                s2d_mode = ir_graph.QNN_OP_SPACE_TO_DEPTH_MODE_CRD
            else:
                s2d_mode = ir_graph.QNN_OP_SPACE_TO_DEPTH_MODE_DCR

            # Squashes the reshape_4d node
            reshape4d_input_buffer = graph.get_input_buffers(reshape_to_4D_node)[0]
            graph.squash(reshape_to_4D_node, input_name=reshape4d_input_buffer.name)

            # Squashes the permute node
            permute_input_buffer = graph.get_input_buffers(permute_node)[0]
            graph.squash(permute_node, input_name=permute_input_buffer.name)

            # Replace the reshape6D OpNode to a SpaceToDepth OpNode
            s2d_op = op_adapter.SpaceToDepthOp(name=permute_node.op.name,
                                               block_size=[downscale_factor_height, downscale_factor_width],
                                               mode=s2d_mode)
            graph.replace(reshape_to_6D_node.op, s2d_op)


@register_layer_optimization
class OptimizeSsdTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.BoxDecoderOp.TRANSLATION_KEY
        self.register_method(SQUASH_BOX_DECODER, self.squash_box_decoder)

    @staticmethod
    def squash_box_decoder(graph):
        def validate_node(nodes_tuple):
            nms_node_ = nodes_tuple[0]
            nms_input_names_ = nms_node_.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_input_names_[0]).type:
                # remove optional reshape input to check if previous is box decoder(ssd) below
                reshape_node_ = graph.get_producer_node(nms_node_.input_names[0])
                nms_input_names_ = [nms_input_names_[1], *reshape_node_.input_names]

            if any(op_adapter.BoxDecoderOp.TRANSLATION_KEY == graph.get_producer_node(name_).op.TRANSLATION_KEY
                   for name_ in nms_input_names_):
                return True

            return False

        sequence = [
            (ir_graph.QNN_OP_MULTI_CLASS_NMS,
             (),
             ()
             )
        ]
        matched_node_list = graph.get_matched_nodes(sequence, validator=validate_node)

        for node_tuple in matched_node_list:
            nms_node = node_tuple[0]
            nms_op = nms_node.op
            # update the boxes input of nms to be box decoder's inputs along with box decoder's op attributes.
            #  [boxes]_______[anchor or priorboxes]
            #            |
            #       [box_decoder(ssd_op)]   <- remove
            #                  |
            #        remove->([Reshape] (optional))_______[scores]
            #                                         |
            #                                 [non_max_suppression]   <- replace by [detection_output]
            #                                         |
            #                                   [detection_output]
            # Updated input for nms will be: [scores, boxes, anchor(priorboxes)]

            nms_boxes_input_name, nms_scores_input_name = nms_node.input_names
            if op_adapter.ReshapeOp.TRANSLATION_KEY == graph.get_producer_op(nms_boxes_input_name).type:
                # update inputs for nms and subsequently the boxes_node
                reshape_node = graph.get_producer_node(nms_boxes_input_name)
                reshape_buf = graph.get_buffer(nms_boxes_input_name)
                nms_boxes_input_name = reshape_node.input_names[0]

                # update consumer relation with reshape buf and prune if applicable
                reshape_buf.consumers.remove(nms_node)
                graph.update_trace_info(nms_node, [reshape_buf, reshape_node])
                if len(reshape_buf.consumers) == 0:
                    graph.prune(reshape_node)

            # fold box_decoder(ssd) node
            box_decoder_node = graph.get_producer_node(nms_boxes_input_name)
            box_decoder_buf = graph.get_buffer(nms_boxes_input_name)
            # Copy over input_names and all op attrs to nms op
            nms_node.input_names = [nms_scores_input_name, *box_decoder_node.input_names]

            # update consumer relation with nms node, box_decoder node and input to box_decoder and
            # prune if applicable
            for name in box_decoder_node.input_names:
                buf = graph.get_buffer(name)
                buf.consumers.add(nms_node)
            if nms_node in box_decoder_buf.consumers:
                box_decoder_buf.consumers.remove(nms_node)
            graph.update_trace_info(nms_node, [box_decoder_buf, box_decoder_node])
            if len(box_decoder_buf.consumers) == 0:
                graph.prune(box_decoder_node)

            # replace nms_node(non_maximum_suppress)
            attrs = dict()
            attrs['delta_scaling_factors'] = [box_decoder_node.op.scale_y, box_decoder_node.op.scale_x, box_decoder_node.op.scale_h, box_decoder_node.op.scale_w]
            attrs['confidence_threshold'] = nms_op.score_threshold
            attrs['iou_threshold'] = nms_op.iou_threshold
            attrs['detection_limit'] = nms_op.max_total_detections
            attrs['use_bg_in_nms'] = 1
            output_dims = []
            for output_name in nms_node.output_names:
                output_dims.append(graph.get_buffer(output_name).shape)
            # nms outputs are [box, score, pred_cls, num_detection]
            # detection outputs [score, box, pred_cls, num_detection]
            output_dims = [
                output_dims[1],
                output_dims[0],
                output_dims[2],
                output_dims[3]
            ]
            attrs['output_dims'] = output_dims
            detection_op = op_adapter.DetectionOutputOp(name=nms_op.name, **attrs)
            graph.replace(nms_op, detection_op)
            detection_node = graph.get_node_by_name(detection_op.name)

            # nms outputs are [box, score, pred_cls, num_detection]
            # detection outputs [score, box, pred_cls, num_detection]
            detection_node.output_names = [
                detection_node.output_names[1],
                detection_node.output_names[0],
                detection_node.output_names[2],
                detection_node.output_names[3]
            ]

            # Update Anchors inputs to fit DetectionOut spec
            anchor_buf = graph.get_buffer(nms_node.input_names[-1])
            anchor_data = anchor_buf.producer.op.tensor

            # TF style (decodeBox+nms) comes as CORNER_SIZE spec requires CENTER_SIZE
            for batch in range(0, anchor_buf.shape[0]):
                for i in range(0, anchor_buf.shape[1]):
                    y_min, x_min, y_max, x_max = anchor_data[batch][i]
                    height = (y_max - y_min)
                    width = (x_max - x_min)
                    anchor_data[batch][i][0] = y_min + height / 2.  # center_y
                    anchor_data[batch][i][1] = x_min + width / 2.  # center_x
                    anchor_data[batch][i][2] = height  # height
                    anchor_data[batch][i][3] = width

            # Addition of a const tensor to class labels should not be quantized
            classes_buf = graph.get_buffer(nms_node.output_names[2])
            for consumer in classes_buf.consumers:
                if consumer.op.type == op_adapter.ElementwiseBinaryOp.operation_to_legacy[ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD]:
                    for input_name in consumer.input_names:
                        add_input_node = graph.get_producer_node(input_name)
                        if add_input_node.op.type == op_adapter.ConstantOp.TRANSLATION_KEY:
                            add_input_node.op.quantizable = False

            # change shape for anchor input from [batch, num_anchors, 4] to [batch * num_anchors, 4] per spec
            anchor_buf.shape = [anchor_buf.shape[0] * anchor_buf.shape[1], anchor_buf.shape[2]]
            anchor_buf.producer.op.tensor = anchor_data.reshape(anchor_buf.shape)

            log_debug2(code_to_message.get_debugging_message("DEBUG_BOXDECODER_SQUASH")(box_decoder_node.op.name,
                                                                                        nms_node.op.name))


@register_layer_optimization
class OptimizeTileTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TileOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if AxisTracker.input_axis_formats_intact(graph, node):
            # No change
            return False
        input_axis_formats_before = graph.get_input_axis_formats(node)
        AxisTracker.alter_axis_format_to_ir_order(node, graph)
        input_axis_formats_after = graph.get_input_axis_formats(node)
        input_buffers = graph.get_input_buffers(node)
        for i, buf in enumerate(input_buffers):
            if input_axis_formats_before[i] != input_axis_formats_after[i]:
                transpose_node = buf.producer
                graph.update_trace_info(transpose_node, [node])
                graph.update_trace_info(buf, [node])
        input_buf = graph.get_buffer(node.input_names[0])
        if input_buf.axis_format in spatial_first_format_to_channel_first_permute_order:
            node.op.multiples = graph.src_axis_order.permute_shape_to_ir(node.op.multiples,
                                                                         input_buf.axis_format)

        return True

    def replace_6d_operation(self, node, graph):
        """
        replace 6D Tile by inserting reshapes around the op
        """
        input_shapes = graph.get_input_shapes(node)
        output_shapes = graph.get_output_shapes(node)
        if len(input_shapes[0]) <= 5:
            return
        check_if_6d_supportable(node, graph)
        # Given input A and TileOp's multiples,
        # We can flatten (A[i], A[i+1], ..., A[j])
        # if multiples[i+1] == multiples[i+2] == ... == multiples[j] == 1
        # Note that multiples[i] is not required to be 1.
        #
        # Case 1:
        #   A with shape [1,128] multiples [2,1]
        #   We can flatten (A[0], A[1]) since multiples[1] == 1
        #   => new_input_shape [128] new_multiples [2]
        # Case 2:
        #   A with shape [1,128] multiples [1,2]
        #   We can not flatten this
        #   => new_input_shape [1,128] new_multiples [1,2]

        src_multiples = node.op.multiples.tolist()
        src_shape = input_shapes[0].dims
        new_multiples = []
        new_input_shape, new_output_shape = [], []
        while len(src_multiples):
            if len(src_multiples) >= 2 and src_multiples[-1] == 1:
                src_multiples = src_multiples[:-1]
                src_shape = src_shape[:-2] + [src_shape[-2]* src_shape[-1]]
            else:
                new_multiples.insert(0, src_multiples.pop())
                new_input_shape.insert(0, src_shape.pop())
                new_output_shape.insert(0, new_input_shape[0]* new_multiples[0])
        log_assert(
            len(new_input_shape) <= 5,
            f"inputs of {node.op.name} are not supported 6D inputs and cannot use reshapes to support the op"
        )
        # insert pre-reshape to reduce the rank of output
        pre_reshape_op_name = node.op.name + '_6d_pre_reshape'
        pre_reshape_op = op_adapter.ReshapeOp(name=pre_reshape_op_name, shape=new_input_shape)
        graph.inject(
            pre_reshape_op, input_name=node.input_names[0],
            output_name=pre_reshape_op_name, consumer_names=[node.op.name]
        )

        # replace TileOp
        new_tile_op = op_adapter.TileOp(
            name=node.op.name, multiples=np.asarray(new_multiples)
        )
        graph.replace(node.op, new_tile_op)

        # insert post-reshape to recover the rank (shape) of output
        post_reshape_insertion(
            node, graph,
            new_out_shapes=[new_output_shape], orig_out_shapes=output_shapes
        )

@register_layer_optimization
class OptimizeTopKTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.TopKOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        if AxisTracker.input_axis_formats_intact(graph, node, input_nontrivial_as_changed=True):
            # No change in input formats, and none of the input formats are NonTrivial
            # Nothing to do in this case
            return False

        #Because QNN TopK only support do it in the last dimension
        #Transpose back to src format if the input buffer was changed
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        if node.op.data_axis_formats[0] != input_buf.axis_format and \
                input_buf.axis_format in AxisOrder().axis_formats:
            # Transpose to maintain src format
            graph.inject_implicit_permute(
                input_buf.name,
                spatial_first_format_to_channel_first_format[input_buf.axis_format],
                spatial_first_format_to_channel_first_permute_order[input_buf.axis_format],
                [node.op.name]
            )

        return True


@register_layer_optimization
class OptimizeThesholdedReluTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.ThresholdedReluOp.TRANSLATION_KEY
        self.register_method(expand_thresholded_relu, self.expand_thresholded_relu)

    def expand_thresholded_relu(self, node,graph):
        sequence = [(op_adapter.ThresholdedReluOp.TRANSLATION_KEY, (), ())]
        matched_node_list = graph.get_matched_nodes(sequence)

        for matched_node in matched_node_list:
            thresholded_relu_node = matched_node[0]

            #getting input buffer names
            inputs = thresholded_relu_node.input_names

            #getting output buffer names
            outputs= thresholded_relu_node.output_names

            #getting index of the inputs to the thresholded_relu node.
            self.idx = graph.list_nodes().index(thresholded_relu_node) - 1

            # Generating alpha tensor
            alpha_name = thresholded_relu_node.op.name + '_alpha'
            alpha = thresholded_relu_node.op.attrs.get("alpha")
            alpha_tensor = np.ones(1) * alpha
            alpha_tensor = op_adapter.ConstantOp(name=alpha_name, tensor=alpha_tensor)

            # Storing the consumers of the ThresholdedRelu node before pruning it
            consumers = graph.get_buffer(thresholded_relu_node.output_names[0]).consumers.copy()

            # Record trace info before prune nodes
            orig_trace_info = graph.get_trace_info(thresholded_relu_node)
            orig_output_trace_info = graph.get_trace_info(graph.get_output_buffers(thresholded_relu_node))

            # Pruning the node
            graph.prune(thresholded_relu_node, force_remove=True)

            inputs.append(alpha_tensor.name)
            elementwise_greater_op_name = inputs[0] + '_gt_' + inputs[1]

            # Generate ElementwiseGreater Op
            eltw_binary_op = op_adapter.ElementwiseBinaryOp(elementwise_greater_op_name,
                                                  operation=ir_graph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER)

            # Add ConstantOp alpha to graph
            alpha_tensor_node = graph.add(alpha_tensor, [], [alpha_tensor.name], axis_formats=[op_graph.AxisTracker.AxisFormat.ANY], idx=self.idx)
            # Set trace info for new created node
            graph.set_trace_info([alpha_tensor_node, graph.get_buffer(alpha_tensor.name)], orig_trace_info)

            # Update index
            self.idx += 1

            # Add ElementwiseGreater Op to graph
            eltw_binary_op_node = graph.add(eltw_binary_op, inputs, [elementwise_greater_op_name], idx=self.idx+1)
            # Set trace info for new created node
            graph.set_trace_info([eltw_binary_op_node, graph.get_buffer(elementwise_greater_op_name)], orig_trace_info)

            # Update index
            self.idx += 1

            # Zero tensor
            zero_tensor_name = thresholded_relu_node.op.name + "_zeros"
            zero_tensor = np.zeros(1)
            zero_tensor = op_adapter.ConstantOp(name=zero_tensor_name, tensor=zero_tensor)
            inputs.append(zero_tensor.name)

            op_name = "ternary_" + str(self.idx)
            eltw_ternary_op = op_adapter.ElementwiseTernaryOp(op_name, eltwise_type=ir_graph.QNN_OP_ELEMENT_WISE_SELECT)

            # Add ConstantOp zeros to graph
            zero_tensor_node = graph.add(zero_tensor, [], [zero_tensor.name], axis_formats=[op_graph.AxisTracker.AxisFormat.ANY], idx=self.idx)
            # Set trace info for new created node
            graph.set_trace_info([zero_tensor_node, graph.get_buffer(zero_tensor.name)], orig_trace_info)

            # Update index
            self.idx += 1

            # Add ElementwiseTernary Op to Graph
            eltw_ternary_op_node = graph.add(eltw_ternary_op, [elementwise_greater_op_name, inputs[0], zero_tensor.name], outputs, idx=self.idx+1)
            # Set trace info for new created node
            graph.set_trace_info(eltw_ternary_op_node, orig_trace_info)
            graph.set_trace_info(graph.get_buffer(outputs[0]), orig_output_trace_info)

            # ThresholdRelu op is replaced. Now set the consumers of the ternary op to those of threshold op
            if len(consumers) > 0:
                for c in consumers:
                    c.input_names.insert(0, thresholded_relu_node.output_names[0])
                graph.get_buffer(thresholded_relu_node.output_names[0]).consumers = consumers


@register_layer_optimization
class OptimizeUnpackTranslation(OptimizationTranslationBase):
    def __init__(self):
        OptimizationTranslationBase.__init__(self)
        self.op_type = op_adapter.UnpackOp.TRANSLATION_KEY

    def axes_to_spatial_first_order(self, node, graph):
        input_name = node.input_names[0]
        input_buf = graph.get_buffer(input_name)
        # Pack needs to happen in src format, so in case the current input format and the data_axis_format
        # are different, inject permute to change it back to src format
        if input_buf.axis_format == AxisTracker.AxisFormat.NDHWC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCDHW:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCDHW,
                                          AxisTracker.AxisFormat.NDHWC_TO_NCDHW, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NSC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCS:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCS,
                                          AxisTracker.AxisFormat.NSC_TO_NCS, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NFC and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.NCF:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.NCF,
                                          AxisTracker.AxisFormat.NFC_TO_NCF, [node.op.name])
        elif input_buf.axis_format == AxisTracker.AxisFormat.NTF and \
                node.op.data_axis_formats[0] == AxisTracker.AxisFormat.TNF:
            graph.inject_implicit_permute(input_name, AxisTracker.AxisFormat.TNF,
                                          AxisTracker.AxisFormat.NTF_TO_TNF, [node.op.name])
        else:
            log_debug2("No axes change for Unpack Op named {}".format(node.op.name))
