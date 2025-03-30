# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import faulthandler
import sys
import traceback
import argparse
import numpy as np
from argparse import Namespace
from typing import Union
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.utils.converter_utils import log_debug,log_debug1,log_warning,log_error,log_info
from qti.aisw.converters.common.converter_ir.op_graph import TraceType

try:
    import onnx
except ImportError as e:
    raise Exception(code_to_message.get_error_message("ERROR_ONNX_NOT_FOUND")(str(e), str(sys.path)))

from qti.aisw.converters.common.converter_ir import op_policies
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.converter_ir.op_graph import InputLayout
from qti.aisw.converters.common.converter_base import ConverterFrontend
from qti.aisw.converters.common.input_shape import InputShapeArgParser, InputShapeInfo
import qti.aisw.converters.onnx.composable_custom_op_utils as ComposableCustomOp
from .composable_custom_op_utils import ComposableCustomOpCollection
from .util import *
from .os_compat_util import *
from qti.aisw.converters.onnx import onnx_translations
from qti.aisw.converters.onnx.onnx_loader import ONNXLoader
from qti.aisw.converters.common.utils.converter_utils import converter_type, convert_args_dict_to_namespace
from qti.aisw.converters.onnx.onnx_model_runtime import ONNXModelRuntime

class OnnxConverterContext(object):
    def __init__(self, graph, backend_info_obj=None):
        """
        This class contains information regarding the weights obtained from WeightProvider.
        Any Other information that needs to be propagated can be added here without changing
        the graph class.
        :type graph: IROpGraph
        :type weights: WeightProvider
        """

        self.ir_graph = graph
        self.weights = []
        self.tensor_to_np_dtype = {}
        # TODO: deprecate it after 0d tensor is fully supported
        self.scalar_tensor = set()
        self.opset_version = 0
        self.composable_custom_op_collection = None
        self.custom_op_factory = None
        # when converter context update weights but not create constant op
        # it is needed to store src information to tracing these ops
        self.weights_trace_info = dict()
        # to store thoes op trace_info before add_op and this op will include
        # the trace_info from weights_trace_info
        self.weights_trace_info_for_op = dict()
        # Backend awareness
        self.backend_info_obj = backend_info_obj

    def update_weights_trace_info(self, src_op_names, src_tensor_names, weight_name):
        """
        Function to update the weights_trace_info, which is to store the framework tracing
        information of translation_context.weights since it may be changed before add_op
        :param src_op_names: list, the source op names
        :param src_tensor_names: list, the source tensor names, usually the inputs/outputs of source op
        :param weight_name: the new weight name, usually the output of source op
        """
        if self.ir_graph.enable_trace:
            weight_trace = []
            for name in src_op_names:
                weight_trace.append((name, TraceType.OP))
                if name in self.weights_trace_info:
                    weight_trace.extend(self.weights_trace_info[name])
            for name in src_tensor_names:
                weight_trace.append((name, TraceType.TENSOR))
                if name in self.weights_trace_info:
                    weight_trace.extend(self.weights_trace_info[name])
            weight_trace.append((weight_name, TraceType.TENSOR))
            if weight_name in self.weights_trace_info:
                weight_trace.extend(self.weights_trace_info[weight_name])
            weight_trace = list(set(weight_trace))
            self.weights_trace_info[weight_name] = weight_trace

    def update_weights_trace_info_for_op(self, op_name, weight_name):
        """
        Function to update the weights_trace_info for an op. Sometimes in op there is parameter
        from weights but no constant op will be created, it should be recorded
        :param op_name: string, the op name
        :param weight_name: string, the weight name
        """
        if self.ir_graph.enable_trace and weight_name in self.weights_trace_info:
            trace_info = self.weights_trace_info[weight_name]
            if op_name in self.weights_trace_info_for_op:
                trace_info.extend(self.weights_trace_info_for_op[op_name])
            self.weights_trace_info_for_op[op_name] = list(set(trace_info))

    def update_trace_info_from_context_weights(self):
        """
        Function to update the framework tracing from weights_trace_info, it should be
        called once after framework translation done.
        """
        if self.ir_graph.enable_trace:
            for name in self.weights_trace_info:
                for op_name in self.ir_graph.trace_dict[TraceType.OP]:
                    update_trace = []
                    if (name, TraceType.TENSOR) in self.ir_graph.trace_dict[TraceType.OP][op_name]:
                        update_trace.extend(self.weights_trace_info[name])
                    self.ir_graph.trace_dict[TraceType.OP][op_name].extend(list(set(update_trace)))
                    self.ir_graph.trace_dict[TraceType.OP][op_name] = list(set(self.ir_graph.trace_dict[TraceType.OP][op_name]))
                for tensor_name in self.ir_graph.trace_dict[TraceType.TENSOR]:
                    update_trace = []
                    if (name, TraceType.TENSOR) in self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name]:
                        update_trace.extend(self.weights_trace_info[name])
                    self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name].extend(list(set(update_trace)))
                    self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name] = list(set(self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name]))
            for name in self.weights_trace_info_for_op:
                for op_name in self.ir_graph.trace_dict[TraceType.OP]:
                    update_trace = []
                    if (name, TraceType.OP) in self.ir_graph.trace_dict[TraceType.OP][op_name]:
                        update_trace.extend(self.weights_trace_info_for_op[name])
                    self.ir_graph.trace_dict[TraceType.OP][op_name].extend(list(set(update_trace)))
                    self.ir_graph.trace_dict[TraceType.OP][op_name] = list(set(self.ir_graph.trace_dict[TraceType.OP][op_name]))
                for tensor_name in self.ir_graph.trace_dict[TraceType.TENSOR]:
                    update_trace = []
                    if (name, TraceType.OP) in self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name]:
                        update_trace.extend(self.weights_trace_info_for_op[name])
                    self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name].extend(list(set(update_trace)))
                    self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name] = list(set(self.ir_graph.trace_dict[TraceType.TENSOR][tensor_name]))

    def insert_weights(self, key, weights, was_scalar=False, src_op_names=[], src_tensor_names=[]):
        """
        This function is to insert new weights and record the weights source infomation
        :param key: insert weight's name
        :param weights: insert weight's data
        :param was_scalar: insert weight's is scalar or not
        :src_op_names: the source op names of insert weight
        :src_tensor_names: the source tensor names of insert weight
        """
        self.weights.insert(key, weights, was_scalar)
        if self.ir_graph.enable_trace:
            self.update_weights_trace_info(
                src_op_names, src_tensor_names, key)

# ------------------------------------------------------------------------------
#   The Converter Class
# ------------------------------------------------------------------------------
class OnnxConverterFrontend(ConverterFrontend):
    class ArgParser(ConverterFrontend.ArgParser):
        def __init__(self, **kwargs):
            super(OnnxConverterFrontend.ArgParser, self).__init__(**kwargs)
            # add command-line options custom to onnx converter
            self.add_optional_argument("--dry_run", type=str, nargs='?', const='info', default=None,
                                       help='Evaluates the model without actually converting any ops, and '
                                            'returns unsupported ops/attributes as well as unused inputs and/or '
                                            'outputs if any. Leave empty or specify "info" to see dry run as a '
                                            'table, or specify "debug" to show more detailed messages only"')
            self.add_optional_argument('-d', '--input_dim', nargs=2, action='append',
                                       metavar=('INPUT_NAME', 'INPUT_DIM'),
                                       help="The name and dimension of all the input buffers to the network specified in\n"
                                            "the format [input_name comma-separated-dimensions],\n"
                                            "for example: 'data' 1,224,224,3. \n"
                                            "Note that the quotes should always be included in order to handle special\n"
                                            "characters, spaces, etc.\n"
                                            "NOTE: This feature works only with Onnx 1.6.0 and above")
            self.add_optional_argument('-n', '--no_simplification', action='store_true', default=False,
                                       help="Do not attempt to simplify the model automatically. This may prevent some models from properly converting \n"
                                            "when sequences of unsupported static operations are present.")
            self.add_optional_argument('-b', '--batch', type=int, default=None,
                                       help="The batch dimension override. This will take the first dimension of all "
                                            "inputs and treat it as a batch dim, overriding it with the value provided "
                                            "here. For example:\n"
                                            "--batch 6\n"
                                            "will result in a shape change from [1,3,224,224] to [6,3,224,224].\n"
                                            "If there are inputs without batch dim this should not be used and each input "
                                            "should be overridden independently using -d option for input dimension overrides.")
            self.add_optional_argument('-s', '--define_symbol', nargs=2, action='append',
                                       metavar=('SYMBOL_NAME', 'VALUE'),
                                       help="This option allows overriding specific input dimension symbols. For instance you "
                                            "might see input shapes specified with variables such as :\n"
                                            "data: [1,3,height,width]\n"
                                            "To override these simply pass the option as:\n"
                                            "--define_symbol height 224 --define_symbol width 448\n"
                                            "which results in dimensions that look like:\n"
                                            "data: [1,3,224,448]")
            self.add_optional_argument('--dump_inferred_model', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--dump_value_info', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--defer_loading', action='store_true', default=False,
                                       help=argparse.SUPPRESS)
            self.add_optional_argument('--dump_custom_io_config_template', type=str, default="",
                                 help='Dumps the yaml template for Custom I/O configuration. This file can'
                                 'be edited as per the custom requirements and passed using the option --custom_io'
                                 'Use this option to specify a yaml file to which the custom IO config template is dumped.')



    def __init__(self, args: Union[Namespace, dict], *, custom_op_factory=None, validator=None, backend_info_obj=None):
        """
        Initialize a new instance of OnnxConverterFrontend class.
        :param args: Namespace or dict of converter arguments
        :param custom_op_factory: CustomOpFactory instance, defaults to None.
        :param validator: Validator instance, defaults to None.
        """
        # Convert args dict to OnnxConverterContext Namespace instance
        if isinstance(args, dict):
            args = convert_args_dict_to_namespace(args, self.ArgParser())
        super(OnnxConverterFrontend, self).__init__(args,
                                                    naming_policy=OnnxNamePolicy(),
                                                    shape_inference_policy=OnnxShapeInferencePolicy(),
                                                    axis_order=AxisOrders.ONNX,
                                                    )
        self.loader = ONNXLoader(args, custom_op_factory=custom_op_factory)
        self.input_network = args.input_network
        self.translations = onnx_translations.OnnxTranslations
        self.dry_run = args.dry_run
        self.no_simplification = args.no_simplification
        self.dump_inferred_model = args.dump_inferred_model
        self.dump_value_info = args.dump_value_info
        self.defer_loading = args.defer_loading
        self.op_info = onnx_translations.OpVersionInfo()
        self.converter_op_package_lib = self.loader.converter_op_package_lib

        self.dump_custom_io_config_template = ''
        if hasattr(args, 'dump_custom_io_config_template'):
            self.dump_custom_io_config_template = args.dump_custom_io_config_template

        self.validator = validator

        # parse input shape provided by the user
        self.input_shape_info = InputShapeArgParser(args.input_dim).input_shape_info
        log_info("Input shape info ", self.input_shape_info)
        self.graph.input_dynamic_axes = self.input_shape_info.input_dynamic_axes

        if self.input_shape_info.has_dynamic_shapes and (args.define_symbol or args.batch or args.custom_op_config_paths):
            raise RuntimeError("Graph conversion with dynamic shapes is not supported along with define_symbol, batch or custom-ops option.")

        self.define_symbols = None
        if args.define_symbol is not None:
            self.define_symbols = {item[0]: item[1] for item in args.define_symbol}

        self.batch = None
        if args.batch is not None:
            self.batch = args.batch

        if self.validator:
            input_dims_dict = None
            if args.input_dim is not None:
                input_dims_dict = dict(zip(self.input_shape_info.input_names, self.input_shape_info.input_dims))
            org_rt_session = ONNXModelRuntime(args.input_network, input_dims_dict, self.define_symbols, self.batch)
            self.validator.add_runtime_sessions("Original Onnx Model", org_rt_session)

        self.converter_context = OnnxConverterContext(self.graph, backend_info_obj=backend_info_obj)

        # We need this if custom ops are present in any of the subgraphs of the If Op
        self.converter_context.custom_op_factory = self.loader.custom_op_factory

        # We can't run simplification and quantization overrides/custom ops as the simplification process
        # could possibly squash layers preventing the custom ops or quantization overrides from being used
        # simplification runs on static shape information, so simplification is disabled when graph is
        # converted with dynamic shapes
        if not self.no_simplification and (args.quantization_overrides or args.custom_op_config_paths or self.input_shape_info.has_dynamic_shapes):
            self.no_simplification = True
            log_warning("Can't simplify the model when custom ops, quantization overrides or dynamic input shapes are specified, converting without simplification.")


    def populate_composable_custom_op_collection(self, model):
        """
        Create a collection of all the ONNX functions present in the given model
        :param model: a ONNX ModelProto
        :return: a ComposableCustomOpCollection object
        """
        self.composable_custom_op_collection = None
        if hasattr(model, "functions") and len(model.functions) > 0:
            self.composable_custom_op_collection = ComposableCustomOpCollection()
            self.composable_custom_op_collection.parse_functions_from_model(model)

        # We need this if custom ops are present in any of the subgraphs of the If Op
        self.converter_context.composable_custom_op_collection = self.composable_custom_op_collection

    def add_composable_op(self, src_op, model):
        """
        Expand the composable custom op node and add all the elementary nodes in the IR graph
        :param src_op: a Composable Custom op node
        :param model: a ONNX Model Proto
        :return:
        """
        expanded_nodes = ComposableCustomOp.expand(src_op, self.composable_custom_op_collection)

        # sub model is only required for Custom op. Sub Model will be created only if there is a programmable custom op
        # in the expansion
        sub_model = None
        custom_op_factory = self.loader.custom_op_factory
        for elem_op in expanded_nodes:
            # collect framework op/tensor names for decompsed op
            self.graph.fw_op_names.append(elem_op.name)
            if elem_op.input:
                self.graph.fw_tensor_names.extend([i for i in elem_op.input])
            if elem_op.output:
                self.graph.fw_tensor_names.extend([o for o in elem_op.output])
            # Check whether the op is a Block Op. It is a Block Op if and only
            # if domain is qti_aisw.
            if elem_op.domain == "qti_aisw":
                src_type = converter_type(f'block_op_{elem_op.op_type}', "onnx")
                self.translations.apply_method_to_op(src_type,
                                                     onnx_translations.OnnxTranslationBase.ADD_OP,
                                                     elem_op,
                                                     self.converter_context)
            # check whether the op is a custom op or not
            elif custom_op_factory and elem_op.op_type in [operator.type_name for operator in custom_op_factory.custom_opdefs]:

                # create a ModelProto from the sub graph of the composable custom op
                if sub_model is None:
                    sub_model = ComposableCustomOp.create_model_from_function(src_op, expanded_nodes, self.composable_custom_op_collection, model)

                if sub_model is None:
                    log_warning("Shape inference library should be provided for the programmable custom operations "
                                "of op type {} using --converter_op_package_lib option".format(elem_op.op_type))

                elem_op_type = converter_type('custom', "onnx")
                # dynamic flag should be true in this case since Custom onnx op for this node will not be present
                # in the custom op collection. We need to create a new custom onnx op from operator and src op.
                node = self.translations.apply_method_to_op(elem_op_type,
                                                            onnx_translations.OnnxTranslationBase.ADD_OP,
                                                            elem_op,
                                                            self.converter_context,
                                                            dynamic=True,
                                                            model=sub_model)
                self.graph.add_src_op_info(node.op.name, [i for i in elem_op.input], [o for o in elem_op.output])
            elif elem_op.domain in ['org.pytorch._caffe2']:
                elem_op_type = converter_type(elem_op.op_type, "onnx_caffe2")
                self.translations.apply_method_to_op(elem_op_type,
                                                     onnx_translations.OnnxTranslationBase.ADD_OP,
                                                     elem_op,
                                                     self.converter_context)
            else:
                elem_op_type = converter_type(elem_op.op_type, "onnx")
                supported_version = self.translations.apply_method_to_op(elem_op_type,
                                                                         onnx_translations.OnnxTranslationBase.SUPPORTED_VERSION,
                                                                         elem_op.op_type)
                self.op_info.validate_op_ver(elem_op, supported_version)

                self.translations.apply_method_to_op(elem_op_type,
                                                     onnx_translations.OnnxTranslationBase.ADD_OP,
                                                     elem_op,
                                                     self.converter_context)


    def dump_io_config_yaml_template(self):
        yaml_data = []
        i_str = "# For complete graph or subgraph conversion\n"
        i_str += "Converted Graph:\n"
        i_str += "  - Input Tensors:\n"
        i_str += "  - Output Tensors:\n\n"

        supported_datatypes = [np.float32, np.float16, np.uint8, np.int8, np.int32, np.uint32, np.bool_, np.int64]

        i_str += "# Input tensors specified in this section should be subset of subgraph (if specified)\n"
        i_str += "Input Tensor Configuration:\n"
        input_num = 0
        for node in self.graph.get_input_nodes_to_graph():
            for buffer_name in node.output_names:
                if self.converter_context.tensor_to_np_dtype[buffer_name] not in supported_datatypes:
                    continue
                input_num += 1
                i_str += "  # Input " + str(input_num) + "\n"
                i_str += "  - Name: " + buffer_name + "\n"
                i_str += "    Src Model Parameters:\n"
                i_str += "        DataType:\n"
                i_str += "        Layout:\n"
                i_str += "    Desired Model Parameters:\n"
                i_str += "        DataType:\n"
                i_str += "        Layout:\n"
                i_str += "        Shape:\n"
                i_str += "        Color Conversion:\n"
                i_str += "        QuantParams:\n          Scale:\n          Offset:\n\n"
        yaml_data.append(i_str)

        output_num = 0
        o_str = "Output Tensor Configuration:\n"
        for node in self.graph.get_output_nodes_of_graph():
            for buffer_name in node.output_names:
                if self.converter_context.tensor_to_np_dtype[buffer_name] not in supported_datatypes:
                    continue
                output_num += 1
                o_str += "  # Output " + str(output_num) + "\n"
                o_str += "  - Name: " + buffer_name + "\n"
                o_str += "    Src Model Parameters:\n"
                o_str += "        DataType:\n"
                o_str += "        Layout:\n"
                o_str += "    Desired Model Parameters:\n"
                o_str += "        DataType:\n"
                o_str += "        Layout:\n"
                o_str += "        QuantParams:\n          Scale:\n          Offset:\n\n"
        yaml_data.append(o_str)

        f = open(self.dump_io_config_template, 'w')
        f.write('\n'.join(yaml_data))
        log_info("Dumped IO config template in file %s" % (self.dump_io_config_template))
        f.close()


    def convert(self):
        self.loader.utils.update_onnx_define_symbols(self.define_symbols, self.batch)

        if self.input_shape_info:
            self.loader.utils.update_input_node(self.input_shape_info.input_names, self.input_shape_info.input_dims)

        # create a collection of all the onnx functions present in the model. This step needs to be run before
        # the onnx simplifier.
        # skip shape inference if simplifier not required
        if self.no_simplification == False:
            self.loader.utils.native_shape_inference()

        self.populate_composable_custom_op_collection(self.loader.model)

        static_input_shapes = {}
        if self.input_shape_info and not self.input_shape_info.has_dynamic_shapes:
            for i in range(len(self.input_shape_info.input_names)):
                static_input_shapes[self.input_shape_info.input_names[i]] = [
                    int(k) for k in self.input_shape_info.input_dims[i].split(",")
                ]

        self.loader.utils.optimize(
            static_input_shapes=static_input_shapes,
            skip_optimization=self.no_simplification,
        )

        faulthandler.enable()

        self.op_info.set_global_op_ver(self.loader.model)
        self.converter_context.opset_version = self.op_info.model_opset_version

        self.loader.native_checker(self.dry_run)

        self.loader.model_wrapper.get_full_model(inplace=True)

        graphs = get_graphs(self.loader.model)
        for graph in graphs:
            dtype_map = self._track_tensor_type(graph)
            self.converter_context.tensor_to_np_dtype.update(dtype_map)

        call_update_output_names = True
        if self.loader.model.graph.output:
            user_provided_output = set(self.output_names)
            graph_output = set()
            for value_info in self.loader.model.graph.output:
                graph_output.add(str(value_info.name))
            if graph_output == user_provided_output:
                log_info("user_provided_output is same as graph_output..skipping update_output_names call")
                call_update_output_names = False

        if self.output_names and call_update_output_names == True:
            # Trims the existing graph to the output nodes specified
            log_debug("update_output_names using user provided names")
            self.loader.utils.update_output_names(self.output_names)

        elif self.loader.model.graph.output:
            # Add the Onnx model outputs to IR Graph
            log_debug("populate graph.output_names")
            for value_info in self.loader.model.graph.output:
                self.graph.output_names.append(str(value_info.name))

        if self.graph.preserve_io_datatype_passed:
            # --custom_io has higher precedence than --preserve_io. Skip the tensors for which dtype is
            # supplied using the --custom_io option.
            tensors_having_custom_dtypes = []
            if self.graph.user_custom_io:
                for entry in self.graph.user_custom_io:
                    if "Datatype" in entry:
                        tensors_having_custom_dtypes.append(str(entry['IOName']))

            for arg in self.graph.preserve_io:
                if self.graph.preserve_io_datatype_passed == 1 and arg[0] == 'datatype':
                    for buffer_name in arg[1:]:
                        if buffer_name not in tensors_having_custom_dtypes:
                            self.graph.preserve_datatype_tensors[buffer_name] = None

            # self.graph.preserve_io_datatype_passed = 1 indicates that user intends to preserve datatype only for the specified tensors
            # self.graph.preserve_io_datatype_passed = 2 indicates that user intends to preserve datatype for all the input and output tensors
            for value_info in self.loader.model.graph.input:
                if ((self.graph.preserve_io_datatype_passed == 1 and value_info.name in self.graph.preserve_datatype_tensors) or \
                    self.graph.preserve_io_datatype_passed == 2) and value_info.name not in tensors_having_custom_dtypes:
                    if value_info.type.tensor_type.elem_type == TensorProto.INT64:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(np.dtype('int64'))
                    else:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type))

            for value_info in self.loader.model.graph.output:
                if ((self.graph.preserve_io_datatype_passed == 1 and value_info.name in self.graph.preserve_datatype_tensors) or \
                    self.graph.preserve_io_datatype_passed == 2) and value_info.name not in tensors_having_custom_dtypes:
                    if value_info.type.tensor_type.elem_type == TensorProto.INT64:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(np.dtype('int64'))
                    else:
                        self.graph.preserve_datatype_tensors[value_info.name] = str(onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type))

            # Throw an error if there is a conflict between the dtype passed using the --input_dtype option and the original dtype
            for k in self.graph.input_dtypes_dict:
                if k in self.graph.preserve_datatype_tensors and self.graph.input_dtypes_dict[k] != self.graph.preserve_datatype_tensors[k]:
                    log_error("Datatype mismatch for tensor %s. %s datatype set with --input_dtype and %s datatype set with --preserve_io!" \
                            % (k, str(self.graph.input_dtypes_dict[k]), self.graph.preserve_datatype_tensors[k]))
                    sys.exit(-1)

            for k in self.graph.preserve_datatype_tensors:
                if self.graph.preserve_datatype_tensors[k] == None:
                    log_error("Graph does not have the tensor %s" % (k))
                    sys.exit(-1)

        # Dumps the trimmed and inferred model, if it was requested
        if self.dump_inferred_model:
            inferred_model_filename = self.input_model_path.split('.')[0] + "_inferred.onnx"
            self.loader.save_model(inferred_model_filename)

        # Dumps the value_info field of the ONNX graph after trimming, for debugging purposes
        if self.dump_value_info and self.loader.model.graph.value_info:
            original_stdout = sys.stdout
            with open(self.input_model_path.split('.')[0] + "_value_info.info", "w") as file:
                sys.stdout = file
                print(self.loader.model.graph.value_info)
                sys.stdout = original_stdout
        elif self.dump_value_info:
            log_warning("Unable to dump value info because field is not populated.")

        # extract inputs
        parameter_names = set()
        for tensor in self.loader.model.graph.initializer:
            parameter_names.add(str(tensor.name))

        for value_info in self.loader.model.graph.input:
            name = str(value_info.name)
            if name in parameter_names:
                # weights are usually listed as inputs too.
                continue
            self.translations.apply_method_to_op(converter_type("input", "onnx"),
                                                 onnx_translations.OnnxTranslationBase.ADD_INPUT_OP, value_info, self.graph)

        # Do not remove the weight in case of Custom ops or validation.
        remove_framework_model_weights = True
        if self.loader.has_custom_op or self.composable_custom_op_collection or self.validator:
            remove_framework_model_weights = False
        self.converter_context.weights = WeightProvider(self.loader.model,
                                                        not self.loader.model_wrapper.weights_loaded,
                                                        remove_framework_model_weights)

        # extract parameters, infer shapes, etc.
        for i, src_op in enumerate(self.loader.get_nodes(include_subgraphs=False)):
            log_debug(code_to_message.get_debugging_message("DEBUG_CONVERTING_NODE")(i, src_op.op_type))
            src_type = converter_type(src_op.op_type, "onnx")
            is_composable_op = False

            try:
                # Check whether the op is a Block Op. It is a Block Op if and
                # only if domain is 'qti_aisw'
                if src_op.domain == "qti_aisw":
                    src_type = converter_type(f'block_op_{src_op.op_type}', 'onnx')
                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)
                # check whether the op is a composable op or not
                # If so, then expand the composable operation and add individual nodes in the expansion.
                elif self.composable_custom_op_collection and self.composable_custom_op_collection.is_composable_op(src_op):
                    self.add_composable_op(src_op, self.loader.model)
                    is_composable_op = True

                # check if layer is a registered custom op in an op collection.
                # If so, the layer is added and the outer loop continues.
                elif self.loader.custom_op_factory and src_op.op_type in self.loader.custom_op_factory.op_collection:
                    src_type = converter_type('custom', "onnx")
                    node = self.translations.apply_method_to_op(src_type,
                                                                onnx_translations.OnnxTranslationBase.ADD_OP,
                                                                src_op,
                                                                self.converter_context)
                    self.graph.add_src_op_info(node.op.name, [i for i in src_op.input], [o for o in src_op.output])
                elif src_op.domain in ['org.pytorch._caffe2']:
                    src_type = converter_type(src_op.op_type, "onnx_caffe2")
                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)
                elif src_op.domain in ['spconv']:
                    src_type = converter_type(src_op.op_type, "spconv")
                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)
                elif src_op.domain in ['com.microsoft']:
                    src_type = converter_type(src_op.op_type, "onnx")
                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)
                else:
                    # If the op is not a custom operation, check the version and use the
                    # native converter translation
                    supported_version = self.translations.apply_method_to_op(src_type,
                                                                             onnx_translations.OnnxTranslationBase.SUPPORTED_VERSION,
                                                                             src_op.op_type)
                    self.op_info.validate_op_ver(src_op, supported_version)

                    self.translations.apply_method_to_op(src_type,
                                                         onnx_translations.OnnxTranslationBase.ADD_OP,
                                                         src_op,
                                                         self.converter_context)
                # collect framework op/tensor names
                if not is_composable_op:
                    self.graph.fw_op_names.append(src_op.name)
                    if src_op.input:
                        self.graph.fw_tensor_names.extend([i for i in src_op.input])
                    if src_op.output:
                        self.graph.fw_tensor_names.extend([o for o in src_op.output])
            except Exception as e:
                if self.debug:
                    traceback.print_exc()
                log_error("Node %s: %s" % (src_op.name, e))
                sys.exit(-1)

        # handle the framework tracing from graph.context_weight_src
        self.converter_context.update_trace_info_from_context_weights()
        # self.graph.insert_context_weights_trace_info()

        self.graph.validate_trace_info_completeness(self.graph.fw_op_names, self.graph.fw_tensor_names, "onnx_translation")

        self.graph.eval_macs_params()

        if self.dump_custom_io_config_template:
            axis_format_to_layout_dict = {AxisTracker.AxisFormat.NDHWC: InputLayout.NDHWC,
                                        AxisTracker.AxisFormat.NCDHW: InputLayout.NCDHW,
                                        AxisTracker.AxisFormat.NSC: InputLayout.NHWC,
                                        AxisTracker.AxisFormat.NCS: InputLayout.NCHW,
                                        AxisTracker.AxisFormat.NFC: InputLayout.NFC,
                                        AxisTracker.AxisFormat.NCF: InputLayout.NCF,
                                        AxisTracker.AxisFormat.NTF: InputLayout.NTF,
                                        AxisTracker.AxisFormat.TNF: InputLayout.TNF,
                                        AxisTracker.AxisFormat.NF: InputLayout.NF,
                                        AxisTracker.AxisFormat.NC: InputLayout.NC,
                                        AxisTracker.AxisFormat.NONTRIVIAL: InputLayout.NONTRIVIAL,
                                        AxisTracker.AxisFormat.ANY: InputLayout.FEATURE}

            yaml_data = []
            comments = "# Custom I/O configuration template for the provided model.\n\n" \
                "# Layout field (optional) has two sub fields : Model and Custom. \n" \
                "# Model: Specify the layout of the buffer in the original model. Default value is obatained from the model \n" \
                "#        This is equivalent to the --input_layout option and both cannot be used together. \n" \
                "# Custom: Specify the custom layout desired for the buffer. Needs to be filled by the user. \n" \
                "# Model and Custom fields support valid QNN Layout. Accepted values are:\n" \
                "# NCDHW, NDHWC, NCHW, NHWC, NFC, NCF, NTF, TNF, NF, NC, F, NONTRIVIAL\n" \
                "# where, N = Batch, C = Channels, D = Depth, H = Height, W = Width, F = Feature, T = Time\n\n" \
                "# Datatype field (optional) supports float32, float16 and uint8 datatypes. Default values for input buffer are obtained from the model \n" \
                "# This field is left empty for the output buffers. \n\n" \
                "# QuantParam field (optional) has three sub fields: Type, Scale and Offset \n" \
                "# Type: Set to QNN_DEFINITION_DEFINED (default) if the scale and offset are provided by the user else set to QNN_DEFINITION_UNDEFINED \n" \
                "# Scale and Offset fields are populated with dummy values as part of this template. Scale and Offset fields will be ignored for an I/O \n" \
                "# if the precision field corresponding to that I/O is not set to uint8 \n\n\n" \
                "# Model Inputs"

            yaml_data.append(comments)
            supported_datatypes = [np.float32, np.float16, np.uint8, np.int8, np.int32, np.uint32, np.bool_, np.int64]

            for node in self.graph.get_input_nodes_to_graph():
                for buffer_name in node.output_names:
                    if self.converter_context.tensor_to_np_dtype[buffer_name] not in supported_datatypes:
                        continue
                    io_str = " - IOName: " + buffer_name + "\n"
                    io_str += "   Layout:\n     Model: " + axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n     Custom: " +  axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n"
                    io_str += "   Datatype: " + str(self.converter_context.tensor_to_np_dtype[buffer_name]) + "\n"
                    io_str += "   QuantParam:\n     Type: QNN_DEFINITION_DEFINED\n     Scale: 1.0\n     Offset: 0\n"
                    yaml_data.append(io_str)

            yaml_data.append("\n# Model Outputs")

            for node in self.graph.get_output_nodes_of_graph():
                for buffer_name in node.output_names:
                    if self.converter_context.tensor_to_np_dtype[buffer_name] not in supported_datatypes:
                        continue
                    io_str = " - IOName: " + buffer_name + "\n"
                    io_str += "   Layout:\n     Model: " + axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n     Custom: " +  axis_format_to_layout_dict[self.graph.buffers[buffer_name].axis_format] + "\n"
                    io_str += "   Datatype: " + str(self.converter_context.tensor_to_np_dtype[buffer_name]) + "\n"
                    io_str += "   QuantParam:\n     Type: QNN_DEFINITION_DEFINED\n     Scale: 1.0\n     Offset: 0\n"
                    yaml_data.append(io_str)

            f = open(self.dump_custom_io_config_template, 'w')
            f.write('\n'.join(yaml_data))
            log_info("Dumped IO config template in file %s" % (self.dump_custom_io_config_template))
            f.close()
            sys.exit(0)

        if self.dump_io_config_template:
            self.dump_io_config_yaml_template()
            sys.exit(0)

        # remove weight map from the converter context as it's not required in the further steps
        del self.converter_context.weights.weight_map

        if self.validator:
            optimized_rt_session = ONNXModelRuntime(self.loader.model)
            self.validator.add_runtime_sessions("Optimized Onnx Model", optimized_rt_session)

        return self.graph

    def _track_tensor_type(self, graph):
        tensor_to_np_dtype = {}

        for value_info in graph.input:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.value_info:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        for value_info in graph.output:
            tensor_to_np_dtype[value_info.name] = onnx_to_np_dtype.get(value_info.type.tensor_type.elem_type)

        return tensor_to_np_dtype
# ------------------------------------------------------------------------------
#   Policies
# ------------------------------------------------------------------------------
class OnnxNamePolicy(op_policies.ConversionNamePolicy):
    def __init__(self):
        op_policies.ConversionNamePolicy.__init__(self)

    def get_op_name(self, op):
        count = self.type_count.get(op.type, 0)
        self.type_count[op.type] = count + 1
        if hasattr(op, 'LEGACY_TRANSLATION_KEY'):
            name_prefix_str = str(op.LEGACY_TRANSLATION_KEY)
        else:
            name_prefix_str = str(op.type)
        if op.name:
            return str(op.name)
        elif op.type == 'custom':
            return "%s_%s_%d" % (str(op.custom_type), name_prefix_str, count)
        else:
            return "%s_%d" % (name_prefix_str, count)

    def get_op_name_by_type(self, op_type, legacy_translation_key, custom_op_type="", **kwargs):
        count = self.type_count.get(op_type, 0)
        self.type_count[op_type] = count + 1
        if legacy_translation_key:
            name_prefix_str = str(legacy_translation_key)
        else:
            name_prefix_str = str(op_type)

        #if it is a folded op then we add _ at start to distinguish with other ops
        if 'folded_op' in kwargs:
            if kwargs['folded_op']:
                name_prefix_str = '_' + name_prefix_str

        if custom_op_type:
            return "%s_%s_%d" % (str(custom_op_type), name_prefix_str, count)
        else:
            return "%s_%d" % (name_prefix_str, count)


class OnnxShapeInferencePolicy(op_policies.ConversionShapeInferencePolicy):

    def infer_shape(self, op, input_shapes):
        return onnx_translations.OnnxTranslations.apply_method_to_op(op.type,
                                                                     onnx_translations.OnnxTranslationBase.INFER_SHAPE,
                                                                     op,
                                                                     input_shapes)
