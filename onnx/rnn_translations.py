# ==============================================================================
#
#  Copyright (c) 2018-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np

from qti.aisw.converters.common import ir_graph
from .onnx_translations import *
from qti.aisw.converters.common.converter_ir import op_adapter
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisOrders, AxisTracker
from qti.aisw.converters.common.utils import code_to_message
from qti.aisw.converters.common.converter_ir.op_graph import TraceType
import qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm as lstm_op_props
import qti.aisw.converters.common.converter_ir.op_properties.stateful_gru as gru_op_props

# ------------------------------------------------------------------------------
#  RNNTranslationBase
# ------------------------------------------------------------------------------
OPTIONAL_INPUTS = NamedDict(initial_h='', initial_c='', norm_weights='', cell_state_weights='', proj_weights='', proj_bias='')
RNN_INPUT_TYPES = ('_initial_h', '_initial_c')
QNN_LSTM_OUTPUT_TYPES = ('_all_hidden', '_final_cell', '_final_hidden')
QNN_GRU_OUTPUT_TYPES = ('_all_hidden', '_final_hidden')



class OnnxRnnTranslationsBase(OnnxTranslationBase):
    def __init__(self):
        OnnxTranslationBase.__init__(self)
        self.input_names = []
        self.weights = []
        self.rec_weights = []
        self.bias = None
        self.cell_state_weights = None
        self.params = NamedDict()
        self.no_of_gates = 1
        self.direction = ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD
        self.output_names = []
        self.num_directions = 1
        self.timesteps = []
        self.weights_src_info = []
        self.rec_weights_src_info = []
        self.bias_src_info = []
        self.src_input_dic = {}

    def extract_params_for_type(self, src_op, converter_context, rnn_type='Rnn'):
        graph = converter_context.ir_graph

        self.input_names = list(map(str, src_op.input))
        self.output_names = self.extract_output_names(src_op, converter_context)
        self.params.direction = str(self.params.direction).lower()
        self.direction = ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD if self.params.direction == 'forward' else ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE
        if rnn_type == 'Gru':
            self.direction = ir_graph.QNN_OP_GRU_DIRECTION_FORWARD if self.params.direction == 'forward' else ir_graph.QNN_OP_GRU_DIRECTION_REVERSE
        self.num_directions = 1

        # Ensure weights or rec_weights are not passed in dynamically
        weights_input_name = self.input_names[1]
        if not converter_context.weights.has(weights_input_name) or \
            (graph.has_buffer(weights_input_name) and not isinstance(graph.get_producer_op(weights_input_name), op_adapter.ConstantOp)):
                raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(weights_input_name,
                                                                                                  src_op.name))
        self.weights = converter_context.weights.fetch(weights_input_name)
        self.weights_src_info.append((self.input_names[1], TraceType.TENSOR))

        rec_weights_input_name = self.input_names[2]
        if not converter_context.weights.has(rec_weights_input_name) or \
            (graph.has_buffer(rec_weights_input_name) and not isinstance(graph.get_producer_op(rec_weights_input_name), op_adapter.ConstantOp)):
                raise ValueError("Unsupported dynamic weights for input {} of RNN node {}".format(rec_weights_input_name,
                                                                                                  src_op.name))
        self.rec_weights = converter_context.weights.fetch(rec_weights_input_name)
        self.rec_weights_src_info.append((self.input_names[2], TraceType.TENSOR))

        # ONNX may use empty string as a placeholder
        # So add an and-condition to further check it.
        if len(self.input_names) >= 4 and self.input_names[3]:
            bias_input_name = self.input_names[3]
            if not converter_context.weights.has(bias_input_name) or \
                (graph.has_buffer(bias_input_name) and not isinstance(graph.get_producer_op(bias_input_name), op_adapter.ConstantOp)):
                    raise ValueError("Unsupported dynamic bias for input {} of RNN node {}".format(bias_input_name,
                                                                                                   src_op.name))
            self.bias = converter_context.weights.fetch(bias_input_name)
            self.bias_src_info.append((self.input_names[3], TraceType.TENSOR))

        if len(self.input_names) >= 5 and self.input_names[4]:
            self.timesteps = int(converter_context.weights.fetch(self.input_names[4]))

        # Fetch cell_state_weights by source op input names if provided
        self.cell_state_weights = None

        # If it is bi-directional check that weights and rec_weights
        # have the right shape
        if self.params.direction == "bidirectional":
            log_assert(self.weights.shape[0] == 2 and self.rec_weights.shape[0] == 2,
                       "Node {}: Bidirectional input requires two sets of weights and recurrent "
                       "weights each. Got only {} set of weights",
                       src_op.name, self.weights.shape[0])

    def prepare_params_as_constants(self, graph, params):
        for param_name, tensor in params.items():
            if not graph.has_buffer(param_name):
                constant_node = graph.add(op_adapter.ConstantOp(name=param_name, tensor=tensor),
                                          input_names=[], output_names=[param_name],
                                          axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                graph.add_src_op_info(param_name, [], [param_name])
                src_info = self.src_input_dic[param_name] if param_name in self.src_input_dic.keys() else []
                if len(src_info) > 0:
                    graph.set_trace_info([constant_node, graph.get_output_buffers(constant_node)[0]], src_info)
            elif graph.get_producer_op(param_name).type != op_adapter.ConstantOp.TRANSLATION_KEY:
                raise ValueError("lstm requires weights and bias to be constant, got dynamic tensor from {}".format(
                    graph.get_producer_op(param_name).name))

    def add_constant_reset_node(self, converter_context, input_names, rnn_type='Rnn'):
        graph = converter_context.ir_graph
        if rnn_type == 'Gru':
            if len(input_names) == gru_op_props.IR_NUM_INPUTS and input_names[gru_op_props.IR_RESET_IDX]:
                if converter_context.weights.has(input_names[gru_op_props.IR_RESET_IDX]) and not graph.has_buffer(input_names[gru_op_props.IR_RESET_IDX]):
                    tensor = np.squeeze(converter_context.weights.fetch(input_names[gru_op_props.IR_RESET_IDX], prunable=False))
                    constant_node = graph.add(op_adapter.ConstantOp(input_names[gru_op_props.IR_RESET_IDX], tensor),
                                              [],
                                              input_names[gru_op_props.IR_RESET_IDX])
                    self.insert_constant_trace_info(input_names[gru_op_props.IR_RESET_IDX], constant_node, graph)
        elif rnn_type == 'LSTM':
            if len(input_names) == lstm_op_props.IR_NUM_INPUTS and input_names[lstm_op_props.IR_RESET_IDX]:
                if converter_context.weights.has(input_names[lstm_op_props.IR_RESET_IDX]) and \
                                    not graph.has_buffer(input_names[lstm_op_props.IR_RESET_IDX]):
                    # Convert reset constant tensor from 1D to 0D to align with backend validation requirements
                    tensor = np.squeeze(converter_context.weights.fetch(input_names[lstm_op_props.IR_RESET_IDX],
                                                                                 prunable=False))
                    constant_node = graph.add(op_adapter.ConstantOp(input_names[lstm_op_props.IR_RESET_IDX], tensor),
                                      [],
                                      input_names[lstm_op_props.IR_RESET_IDX])
                    self.insert_constant_trace_info(input_names[lstm_op_props.IR_RESET_IDX], constant_node, graph)

    def extract_output_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        onnx_output_names = [output for i, output in enumerate(src_op.output)]
        dummy_name_list = ["_all_output_dummy", "_hidden_output_dummy", "_cell_output_dummy"]

        # If the length of onnx_output_names less then 3
        # Firstly, add '' to the onnx_output_names to make it has 3 names
        # Then, use the dummy_name to fill with the empty names
        # Cause we can't make sure the first output name is alway not empty
        # we use the op.name to distinguish different empty name buffer in different op.
        # ---------------------------------------------------------------------------------------------------------------------------------
        # | src_op.outputs(src_op.name='Lstm') |    onnx_output_names - 1   |                   onnx_output_names - 2                     |
        # ---------------------------------------------------------------------------------------------------------------------------------
        # | ['Y'], ['Y', ''], ['Y', '', '']    |    ['Y', '', '']           | ['Y','Lstm_cell_output_dummy','Lstm_hidden_output_dummy']   |
        # | ['', 'Y_h'],['', 'Y_h', '']        |    ['', 'Y_h', '']         | ['Lstm_all_output_dummy', 'Lstm_cell_output_dummy', 'Y_h']  |
        # | ['', '', 'Y_c']                    |    ['', '', 'Y_c']         | ['Lstm_all_output_dummy', 'Y_c', 'Lstm_hidden_output_dummy']|
        # | ['Y', 'Y_h'], ['Y', 'Y_h', '']     |    ['Y', 'Y_h', '']        | ['Y','Lstm_cell_output_dummy','Y_h']                        |
        # | ['Y', '', 'Y_c']                   |    ['Y', '', 'Y_c']        | ['Y','Y_c','Lstm_hidden_output_dummy']                      |
        # | ['', 'Y_h', 'Y_c']                 |    ['', 'Y_h', 'Y_c']      | ['Lstm_all_output_dummy','Y_c','Y_h']                       |
        # | ['Y', 'Y_h', 'Y_c']                |    ['Y', 'Y_h', 'Y_c']     | ['Y','Y_c','Y_h']                                           |
        # ---------------------------------------------------------------------------------------------------------------------------------
        while len(onnx_output_names) < 3:
            onnx_output_names.append('')

        src_name = src_op.name if len(src_op.name) != 0 else \
                onnx_output_names[0] if onnx_output_names[0] else \
                    graph.naming_policy.get_op_name_by_type(op_adapter.LstmOp.TRANSLATION_KEY,
                                                            op_adapter.LstmOp.LEGACY_TRANSLATION_KEY)
        for idx, name in enumerate(onnx_output_names):
            if not name:
                onnx_output_names[idx] = src_name + dummy_name_list[idx]

        # ONNX LSTM output order: Y, Y_h, Y_c
        # SNPE LSTM output order: Y, Y_c, Y_h
        output_names = onnx_output_names
        output_names[1], output_names[2] = onnx_output_names[2], onnx_output_names[1]
        return output_names

    def extract_src_output_tuples(self, src_op):
        # Since extract_output_names may change the outputs names and their orders. Follow the changes
        # to construct a source output names for framework trace
        src_onnx_output_tuples = []
        for output in src_op.output:
            if output:
                src_onnx_output_tuples.append((output, TraceType.TENSOR))
            else:
                src_onnx_output_tuples.append((src_op.name, TraceType.OP))
        while len(src_onnx_output_tuples) < 3:
            src_onnx_output_tuples.append((src_op.name, TraceType.OP))

        src_onnx_output_tuples[1], src_onnx_output_tuples[2] = src_onnx_output_tuples[2], src_onnx_output_tuples[1]
        return src_onnx_output_tuples

    def create_rnn(self, src_op, converter_context, create_unidirectional_func, create_bidirectional_func, rnn_type='Rnn'):
        graph = converter_context.ir_graph
        QNN_RNN_OUTPUT_TYPES = QNN_LSTM_OUTPUT_TYPES
        if rnn_type == 'Gru':
            QNN_RNN_OUTPUT_TYPES = QNN_GRU_OUTPUT_TYPES

        if self.params.direction == "bidirectional":
            create_bidirectional_func(src_op, converter_context)
        else:
            # set up naming so that the buffers are all different and tagged correctly
            input_names = self.extract_input_names(src_op, converter_context)
            output_names = self.extract_output_names(src_op, converter_context)
            trace_src_onnx_output_tuples = self.extract_src_output_tuples(src_op)
            # QNN GruOp has two mandatory outputs. If the model has only one output then add
            # an additional output which signifies the final hidden output.
            if rnn_type == 'Gru' and input_names[1] and len(output_names) == 1:
                final_hidden_output_name = output_names[0] + '_unidirection'
                output_names.append(final_hidden_output_name)
            output_names_len = len(output_names)
            module_name = src_op.name if len(src_op.name) != 0 else \
                output_names[0] if output_names_len != 0 else \
                    src_op.op_type

            # set up rnn ops
            rnn_op = create_unidirectional_func(graph, module_name, input_names)

            input_x_name = input_names[0]
            if graph.has_buffer(input_x_name):
                input_x_buf = graph.get_buffer(input_x_name)
                input_x_buf.set_axis_format(AxisTracker.AxisFormat.TNF)
            else:
                raise KeyError("Graph has no buffer {} for {} Op input x", input_x_name, rnn_type)

            for idx, input_name in enumerate(input_names[1:3]):
                # GruOp only has initial_h.
                if rnn_type == 'Gru' and idx == 1:
                    break
                # Add constant op if one of the initial c/h inputs is Initializer and not added to graph
                if converter_context.weights.has(input_name) and not graph.has_buffer(input_name):
                    tensor = converter_context.weights.fetch(input_name, prunable=False)
                    constant_node = graph.add(op_adapter.ConstantOp(input_name, tensor),
                                              [],
                                              input_name,
                                              axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                    self.insert_constant_trace_info(input_name, constant_node, converter_context)
                elif graph.has_buffer(input_name):
                    # set axis format as NONTRIVIAL for dynamic c/h inputs
                    input_buf = graph.get_buffer(input_name)
                    input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

            # Add constant node for reset input if it is static
            self.add_constant_reset_node(converter_context, input_names, rnn_type)

            # set up reshape ops
            reshape_ops = []
            input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
            batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
            hidden_size = self.rec_weights.shape[-1]

            output_shapes = []
            for i in range(0, output_names_len):
                if i == 0:
                    output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
                else:
                    output_shapes.append([self.num_directions, batch_size, hidden_size])
            for i in range(0, output_names_len):
                # the outputs are reshaped to be ONNX format
                reshape_ops.append(op_adapter.ReshapeOp(output_names[i] + QNN_RNN_OUTPUT_TYPES[i] + '_reshape',
                                                        shape=output_shapes[i]))

            rnn_output_names = [str(name) + QNN_RNN_OUTPUT_TYPES[i] + "_rnn"
                                for i, name in enumerate(output_names)]

            reshape_output_names = [str(name) for name in output_names]

            # set override encodings for rnn op
            # ONNX LSTM output order: Y, Y_h, Y_c
            if graph.user_quantization_overrides:
                for id in range(0, len(src_op.output)):
                    encoding = graph.get_overridden_encoding(src_op.output[id], False)
                    if encoding is not None and rnn_type == 'Gru':
                        graph.set_overridden_encoding(rnn_output_names[id], encoding, False)
                    elif encoding is not None and id == 1 and len(rnn_output_names) > 2:
                        graph.set_overridden_encoding(rnn_output_names[2], encoding, False)
                    elif encoding is not None and id == 2:
                        graph.set_overridden_encoding(rnn_output_names[1], encoding, False)
                    elif encoding is not None:
                        graph.set_overridden_encoding(rnn_output_names[id], encoding, False)

            self.add_src_op_info(rnn_op.name, src_op, graph)
            rnn_node = graph.add(rnn_op, input_names, rnn_output_names)

            self.insert_trace_info(rnn_node, [(src_op.name, TraceType.OP)], converter_context)
            for i, rnn_output_name in enumerate(rnn_output_names):
                self.insert_trace_info(graph.get_buffer(rnn_output_name), [trace_src_onnx_output_tuples[i]], converter_context)

            for i in reversed(range(0, output_names_len)):
                # add reshape for outputs to make shape to be ONNX's format
                log_debug("Adding reshape op {} while creating unidirectional RNN unit".format(reshape_ops[i].name))
                reshape_node = graph.add(reshape_ops[i], [rnn_output_names[i]], [reshape_output_names[i]])
                graph.add_src_op_info(reshape_ops[i].name, [rnn_output_names[i]], [reshape_output_names[i]])
                self.insert_trace_info([reshape_node, graph.get_buffer(rnn_output_names[i]), graph.get_buffer(reshape_output_names[i])],
                                       [(src_op.name, TraceType.OP), trace_src_onnx_output_tuples[i]], converter_context)

            return rnn_node

    def create_bidirectional_module(self, src_op, converter_context, weights, rec_weights, bias, params,
                                    create_rnn_type):
        graph = converter_context.ir_graph
        # set up naming so that the buffers are all different and tagged correctly
        src_op_inputs = list(src_op.input)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        module_name = src_op.name if len(src_op.name) != 0 else output_names[0]
        initial_h_c_inputs_present = True if OPTIONAL_INPUTS.initial_h and OPTIONAL_INPUTS.initial_c else False

        output_names_len = len(output_names)
        if output_names_len > 3:
            raise ValueError("Unsupported number of outputs for source bidirectional LSTM op {}. Expected number of outputs between 0 and 3, got {}".
                             format(src_op.name, output_names_len))

        # bidirectional lstm shares the same source weights name of forward and backward, so we need to specify them in graph
        forward_op_input_names = input_names[:]
        backward_op_input_names = input_names[:]
        # Do not create separate tensors for reset input, it is shared between forward and backwards
        last_idx_to_split = min(len(input_names), lstm_op_props.NUM_INPUTS - 1)
        for i in range(3, last_idx_to_split):
            curr_input_name = input_names[i]
            if curr_input_name:
                forward_op_input_names[i] = curr_input_name + '_forward'
                backward_op_input_names[i] = curr_input_name + '_backward'
                # Set override param encodings for new name input_tensor
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(curr_input_name)
                    if encoding is not None:
                        graph.set_overridden_encoding(forward_op_input_names[i], encoding, False)
                        graph.set_overridden_encoding(backward_op_input_names[i], encoding, False)

        h_0_forward_split_name, h_0_backward_split_name = '', ''
        c_0_forward_split_name, c_0_backward_split_name = '', ''
        # set up split ops for initial_h and initial_c along `num_directions` dimension if present
        if initial_h_c_inputs_present:
            src_h_c_input_names = src_op_inputs[5:7]
            h_c_split_output_names = list()
            for i, name in enumerate(src_h_c_input_names):
                # add initial_h and initial_c to graph
                if converter_context.weights.has(name) and not graph.has_buffer(name):
                    tensor = converter_context.weights.fetch(name, prunable=False)
                    initial_h_c_node = graph.add(op_adapter.ConstantOp(name, tensor),
                                                [],
                                                name,
                                                axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                    self.insert_constant_trace_info(name, initial_h_c_node, converter_context)
                elif graph.has_buffer(name):
                    input_buf = graph.get_buffer(name)
                    input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

                split_op_name = f"{module_name}_{name}_{RNN_INPUT_TYPES[i]}_split"
                # If split_index is not given, split equally along the specified axis
                split_op = op_adapter.SplitOp(split_op_name, axis=0)
                split_output_names = [f"{split_op_name}_forward",
                                      f"{split_op_name}_backward"]
                h_c_split_output_names.append(split_output_names)
                log_debug("Adding split op {} while creating bidirectional RNN unit".format(split_op.name))
                split_node = graph.add(split_op, [name], split_output_names)
                graph.add_src_op_info(split_op.name, [name], split_output_names)
                split_targets = [split_node].extend(graph.get_output_buffers(split_node))
                self.insert_trace_info(split_targets, (name, TraceType.TENSOR), converter_context)

                # Set override param encodings for split op
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(name)
                    if encoding is not None:
                        for output_name in split_output_names:
                            graph.set_overridden_encoding(output_name, encoding, False)

            h_0_forward_split_name, h_0_backward_split_name = h_c_split_output_names[0]
            c_0_forward_split_name, c_0_backward_split_name = h_c_split_output_names[1]

        # set up forward op
        forward_op = create_rnn_type(graph,
                                     module_name + '_forward',
                                     forward_op_input_names,
                                     weights=weights[0, :, :],
                                     rec_weights=rec_weights[0, :, :],
                                     bias=bias[0, :] if bias is not None else bias,
                                     hidden_size=params.hidden_size,
                                     direction=ir_graph.QNN_OP_LSTM_DIRECTION_FORWARD,
                                     h_0_input_name=h_0_forward_split_name,
                                     c_0_input_name=c_0_forward_split_name)

        # set up backward op
        backward_op = create_rnn_type(graph,
                                      module_name + '_backward',
                                      backward_op_input_names,
                                      weights=weights[1, :, :],
                                      rec_weights=rec_weights[1, :, :],
                                      bias=bias[1, :] if bias is not None else bias,
                                      hidden_size=params.hidden_size,
                                      direction=ir_graph.QNN_OP_LSTM_DIRECTION_REVERSE,
                                      h_0_input_name=h_0_backward_split_name,
                                      c_0_input_name=c_0_backward_split_name)

        # set up reshape ops
        reshape_ops = []
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
        hidden_size = params.hidden_size
        output_shapes = []
        for i in range(0, output_names_len):
            # The shape is for unidirectional lstm, so we set num_directions to 1 here
            # Outputs from forward and backward lstm will then be concatenated together
            if i == 0:
                output_shapes.append([time_steps, 1, batch_size, hidden_size])
            else:
                output_shapes.append([1, batch_size, hidden_size])
        for i in range(0, output_names_len):
            # the outputs are reshaped to be ONNX format
            reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + QNN_LSTM_OUTPUT_TYPES[i] + '_reshape_forward',
                                                     shape=output_shapes[i]),
                                op_adapter.ReshapeOp(output_names[i] + QNN_LSTM_OUTPUT_TYPES[i] + '_reshape_backward',
                                                     shape=output_shapes[i])])
        # set up concat ops
        concat_ops = []
        for i in range(0, output_names_len):
            # The first output is used to concat all hidden output values at last axis
            axis = 1 if i == 0 else 0
            concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

        # Add constant node for reset input if it is static
        self.add_constant_reset_node(converter_context, input_names, 'LSTM')

        forward_output_names = [module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_forward" for i, name in
                                enumerate(output_names)]
        backward_output_names = [module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_backward" for i, name in
                                 enumerate(output_names)]
        reshape_output_names = [[module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_reshape_forward",
                                 module_name + str(name) + QNN_LSTM_OUTPUT_TYPES[i] + "_reshape_backward"]
                                for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]
        forward_input_names = [input_names[0], h_0_forward_split_name, c_0_forward_split_name] + forward_op_input_names[3:]
        backward_input_names = [input_names[0], h_0_backward_split_name, c_0_backward_split_name] + backward_op_input_names[3:]

        log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
        forward_node = graph.add(forward_op, forward_input_names, forward_output_names)
        graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)
        forward_targets = [forward_node].extend(graph.get_output_buffers(forward_node))
        self.insert_trace_info(forward_targets, [(src_op.name, TraceType.OP)], converter_context)

        log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
        backward_node = graph.add(backward_op, backward_input_names, backward_output_names)
        graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)
        backward_targets = [backward_node].extend(graph.get_output_buffers(backward_node))
        self.insert_trace_info(backward_targets, [(src_op.name, TraceType.OP)], converter_context)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for id in range(0, output_names_len):
            # add reshape for outputs to make shape to be ONNX's format
            log_debug("Adding reshape op {} and {} while creating bidirectional RNN unit".format(reshape_ops[id][0].name, reshape_ops[id][1].name))
            backward_reshape_node = graph.add(reshape_ops[id][0], [forward_output_names[id]], [reshape_output_names[id][0]])
            forward_reshape_node = graph.add(reshape_ops[id][1], [backward_output_names[id]], [reshape_output_names[id][1]])
            graph.add_src_op_info(reshape_ops[id][0].name, [forward_output_names[id]], [reshape_output_names[id][0]])
            graph.add_src_op_info(reshape_ops[id][1].name, [backward_output_names[id]], [reshape_output_names[id][1]])
            self.insert_trace_info([backward_reshape_node, graph.get_output_buffers(backward_reshape_node)[0]], [(src_op.name, TraceType.OP)], converter_context)
            self.insert_trace_info([forward_reshape_node, graph.get_output_buffers(forward_reshape_node)[0]], [(src_op.name, TraceType.OP)], converter_context)

            log_debug("Adding concat op {} while creating bidirectional RNN unit".format(concat_ops[id].name))
            concat_node = graph.add(concat_ops[id], [reshape_output_names[id][0], reshape_output_names[id][1]], [concat_output_names[id]])
            graph.add_src_op_info(concat_ops[id].name, [reshape_output_names[id][0], reshape_output_names[id][1]], [concat_output_names[id]])
            self.insert_trace_info([concat_node], [(src_op.name, TraceType.OP)], converter_context)
            self.insert_trace_info(graph.get_output_buffers(concat_node), [(concat_output_names[id], TraceType.TENSOR)], converter_context)

            # Set override output encodings for rnn op
            if id < len(src_op.output) and graph.user_quantization_overrides:
                encoding = graph.get_overridden_encoding(src_op.output[id], False)
                if encoding is not None and id == 1 and len(forward_output_names) > 2:
                    graph.set_overridden_encoding(forward_output_names[2], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[2], encoding, False)
                elif encoding is not None and id == 2:
                    graph.set_overridden_encoding(forward_output_names[1], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[1], encoding, False)
                elif encoding is not None:
                    graph.set_overridden_encoding(forward_output_names[id], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[id], encoding, False)


# ------------------------------------------------------------------------------
#  GRU
# ------------------------------------------------------------------------------

class OnnxGruTranslation(OnnxRnnTranslationsBase):
    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('GRU', [1, 7, 14], [['clip',
                                                               'activation_alpha',
                                                               'activation_beta',
                                                               'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid','Tanh'])
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 3
        self.linear_before_reset = 0
        self.NUM_INPUTS = gru_op_props.NUM_INPUTS - 1
        self.IR_NUM_INPUTS = gru_op_props.IR_NUM_INPUTS - 1
        self.NUM_REQUIRED_INPUTS = gru_op_props.NUM_REQUIRED_INPUTS

    def extract_parameters(self, src_op, converter_context):
        self.params = extract_attributes(src_op, schema=self.op_schema(op_type=src_op.op_type))
        self.extract_params_for_type(src_op, converter_context, rnn_type='Gru')

        if len(self.input_names) >= 6:
            # check if initial_h is included
            OPTIONAL_INPUTS.initial_h = self.input_names[5]

        self.linear_before_reset = self.params.linear_before_reset

    def extract_input_names(self, src_op, converter_context):
        input_weights_name = self.input_names[1]
        rec_weights_name = self.input_names[2]
        # Onnx may use empty string as a placeholder, so add an and-condition to further check it
        bias_name = self.input_names[3] if len(self.input_names) >= 4 and self.input_names[3] else 'gru_zero_bias'
        formatted_input_names = [self.input_names[0], OPTIONAL_INPUTS.initial_h, input_weights_name,
                                 rec_weights_name, bias_name]
        return formatted_input_names

    def extract_output_names(self, src_op, converter_context):
        graph = converter_context.ir_graph
        onnx_output_names = [output for i, output in enumerate(src_op.output)]
        dummy_name_list = ["_all_output_dummy", "_hidden_output_dummy"]

        # If the length of onnx_output_names less then 2
        # Firstly, add '' to the onnx_output_names to make it has 2 names
        # Then, use the dummy_name to fill with the empty names
        # Cause we can't make sure the first output name is alway not empty
        # we use the op.name to distinguish different empty name buffer in different op.
        # -------------------------------------------------------------------------------------------------------------
        # | src_op.outputs(src_op.name='Gru') |       onnx_output_names - 1       |      onnx_output_names - 2        |
        # -------------------------------------------------------------------------------------------------------------
        # |         ['Y'], ['Y', '']          |             ['Y', '']             |  ['Y', 'Gru_hidden_output_dummy'] |
        # |         ['', 'Y_h']               |             ['', 'Y_h']           |  ['Gru_all_output_dummy', 'Y_h']  |
        # |         ['Y', 'Y_h']              |             ['Y', 'Y_h']          |  ['Y', 'Y_h']                     |
        # -------------------------------------------------------------------------------------------------------------
        while len(onnx_output_names) < 2:
            onnx_output_names.append('')

        src_name = src_op.name if len(src_op.name) != 0 else \
                onnx_output_names[0] if onnx_output_names[0] else \
                    graph.naming_policy.get_op_name_by_type(op_adapter.GruOp.TRANSLATION_KEY,
                                                            op_adapter.GruOp.LEGACY_TRANSLATION_KEY)

        for idx, name in enumerate(onnx_output_names):
            if not name:
                onnx_output_names[idx] = src_name + dummy_name_list[idx]

        return onnx_output_names

    def extract_src_output_tuples(self, src_op):
        # Since extract_output_names may change the outputs names and their orders. Follow the changes
        # to construct a source output names for framework trace
        src_onnx_output_tuples = []
        for output in src_op.output:
            if output:
                src_onnx_output_tuples.append((output, TraceType.TENSOR))
            else:
                src_onnx_output_tuples.append((src_op.name, TraceType.OP))
        if len(src_onnx_output_tuples) < 2 :
            src_onnx_output_tuples.append((src_op.name, TraceType.OP))
        return src_onnx_output_tuples

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):
        if bias is None:
            # Bias tensor was not provided, so create an all zeroes bias
            bias = np.zeros((6 * hidden_size), dtype=np.float32)
        else:
            # Flatten incoming tensors for consistency
            if len(bias.shape) > 1:
                bias = np.reshape(bias, (-1,))
            # Due to the linear_before_reset parameter, we may need the
            # recurrent biases split out from the other biases. We need the bias
            # tensor to hold the concatenation of Wb and Rb.
            if bias.shape[0] != 6*hidden_size:
                raise ValueError(f'GRU node bias has incorrect shape. Expected ({6*hidden_size}, ), '
                                 f'got {bias.shape}')

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size) respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (self.no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (self.no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.extract_parameters(src_op, converter_context)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, converter_context, self.create_unidirectional_gru,
                               self.create_bidirectional_gru, rnn_type='Gru')

    def create_unidirectional_gru(self, graph, name, input_name_list, **kwargs):

        if kwargs:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(kwargs['weights'],
                                                                       kwargs['rec_weights'],
                                                                       kwargs['bias'],
                                                                       kwargs['hidden_size'])
            h_0_input_name = kwargs['h_0_input_name']
        else:
            [weights, rec_weights, bias] = self.convert_params_to_snpe(self.weights,
                                                                       self.rec_weights,
                                                                       self.bias,
                                                                       self.params.hidden_size)
            h_0_input_name = OPTIONAL_INPUTS.initial_h

        # gru specific organization into separate gates
        activations = self.params.activations

        # MergedWeightsGru specific organization into gate format
        gate_bias = bias.reshape(-1, )
        gate_rec_weights = rec_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
        gate_weights = weights.reshape(self.no_of_gates * self.params.hidden_size, -1)

        params_dict = {}
        # Extract the input names for weights and bias.
        if input_name_list is not None:
            input_weights_name, rec_weights_name, bias_name = input_name_list[2:5]
            params_dict[input_weights_name] = gate_weights
            params_dict[rec_weights_name] = gate_rec_weights
            params_dict[bias_name] = gate_bias
            self.prepare_params_as_constants(graph, params_dict)

        return op_adapter.MergedWeightsGruOp(name,
                                             activation=activations[0],
                                             gate_activation=activations[0],
                                             rec_gate_activation=activations[1],
                                             h_0_input_name=h_0_input_name,
                                             direction = kwargs['direction'] if kwargs else self.direction,
                                             hidden_size=self.params.hidden_size,
                                             linear_before_reset = self.params.linear_before_reset,
                                             time_major=True)

    def create_bidirectional_gru(self, src_op, converter_context):
        graph = converter_context.ir_graph

        src_op_inputs = list(src_op.input)
        input_names = self.extract_input_names(src_op, converter_context)
        output_names = self.extract_output_names(src_op, converter_context)
        # If the initial_h input is not present. QNN GruOp has two mandatory outputs.
        # If the model has only one output then add an additional output which signifies the final hidden output.
        if input_names[1] and len(output_names) == 1:
            final_hidden_output_name = output_names[0] + '_bidirection'
            output_names.append(final_hidden_output_name)
        module_name = src_op.name if len(src_op.name) != 0 else output_names[0]

        src_op_inputs_len = len(src_op_inputs)
        if src_op_inputs_len > self.NUM_INPUTS or src_op_inputs_len < self.NUM_REQUIRED_INPUTS:
            raise ValueError("Unsupported number of inputs on source op {}. Expected between {} and {}, got {}".
                             format(src_op.name, self.NUM_REQUIRED_INPUTS, self.NUM_INPUTS, src_op_inputs_len))

        input_names_len = len(input_names)
        if input_names_len > self.IR_NUM_INPUTS:
            raise ValueError("Unsupported number of inputs for bidirectional unit. Expected <= {}, got {}".
                             format(self.IR_NUM_INPUTS, input_names_len))

        output_names_len = len(output_names)
        if output_names_len > 2:
            raise ValueError("Unsupported number of outputs for bidirectional unit. Expected < 2, got {}".
                             format(output_names_len))

        forward_op_input_names = input_names[:]
        backward_op_input_names = input_names[:]

        # Rename the weights, rec_weights, bias for 2 different directions by adding '_forward' or '_backward'.
        # Later, each above tensor will splite into 2 tensors. Forward uses the 1st batch and the backward 2nd.
        # Do not create separate tensors for reset input, it is shared between forward and backward
        last_idx_to_split = min(input_names_len, gru_op_props.IR_NUM_INPUTS - 1)
        for i in range(2, last_idx_to_split):
            curr_input_name = input_names[i]
            if curr_input_name:
                forward_op_input_names[i] = curr_input_name + '_forward'
                backward_op_input_names[i] = curr_input_name + '_backward'
                # Set override param encodings for new name input_tensor
                if graph.user_quantization_overrides:
                    encoding = graph.get_overridden_encoding(curr_input_name)
                    if encoding is not None:
                        graph.set_overridden_encoding(forward_op_input_names[i], encoding, False)
                        graph.set_overridden_encoding(backward_op_input_names[i], encoding, False)

        # set up split ops
        split_ops = []
        split_ops.append(op_adapter.SplitOp(str(input_names[1]) + RNN_INPUT_TYPES[0] + "_split", axis=0))
        # set up forward op
        forward_op = self.create_unidirectional_gru(graph,
                                                    module_name + '_forward',
                                                    forward_op_input_names,
                                                    weights=self.weights[0, :, :],
                                                    rec_weights=self.rec_weights[0, :, :],
                                                    bias=self.bias[0, :] if self.bias is not None else self.bias,
                                                    hidden_size=self.params.hidden_size,
                                                    direction=ir_graph.QNN_OP_GRU_DIRECTION_FORWARD,
                                                    h_0_input_name=module_name + str(input_names[1]) + RNN_INPUT_TYPES[0] + "_forward_split" if
                                                    input_names_len > 1 else OPTIONAL_INPUTS.initial_h)

        # set up backward op
        backward_op = self.create_unidirectional_gru(graph,
                                                     module_name + '_backward',
                                                     backward_op_input_names,
                                                     weights=self.weights[1, :, :],
                                                     rec_weights=self.rec_weights[1, :, :],
                                                     bias=self.bias[1, :] if self.bias is not None else self.bias,
                                                     hidden_size=self.params.hidden_size,
                                                     direction=ir_graph.QNN_OP_GRU_DIRECTION_REVERSE,
                                                     h_0_input_name=module_name + str(input_names[1]) + RNN_INPUT_TYPES[0] + "_backward_split" if
                                                     input_names_len > 1 else OPTIONAL_INPUTS.initial_h)

        # set up reshape ops
        reshape_ops = []
        input_shapes = [graph.get_buffer(src_op.input[0]).shape[:]]
        batch_size, time_steps, input_size = AxisOrders.ONNX.extract_time_series_dims(input_shapes[0])
        hidden_size = self.params.hidden_size
        output_shapes = []
        for i in range(0, output_names_len):
            if i == 0:
                output_shapes.append([time_steps, self.num_directions, batch_size, hidden_size])
            else:
                output_shapes.append([self.num_directions, batch_size, hidden_size])
        for i in range(0, output_names_len):
            # the outputs are reshaped to be ONNX format
            reshape_ops.append([op_adapter.ReshapeOp(output_names[i] + QNN_GRU_OUTPUT_TYPES[i] + '_reshape_forward', shape=output_shapes[i]),
                                op_adapter.ReshapeOp(output_names[i] + QNN_GRU_OUTPUT_TYPES[i] + '_reshape_backward', shape=output_shapes[i])])
        # set up concat ops
        concat_ops = []
        for i in range(0, output_names_len):
            # The first output is used to concat all hidden output values at last axis
            axis = 1 if i == 0 else 0
            concat_ops.append(op_adapter.ConcatOp(output_names[i] + '_concat', axis=axis))

        for input_name in input_names[1:2]:
            # Add constant op if one of the initial h input is Initializer and not added to graph
            if converter_context.weights.has(input_name) and not graph.has_buffer(input_name):
                tensor = converter_context.weights.fetch(input_name, prunable=False)
                constant_node = graph.add(op_adapter.ConstantOp(input_name, tensor),
                                          [],
                                          input_name,
                                          axis_formats=[AxisTracker.AxisFormat.NONTRIVIAL])
                self.insert_constant_trace_info(input_name, constant_node, converter_context)
            elif graph.has_buffer(input_name):
                # set axis format as NONTRIVIAL for dynamic h input
                input_buf = graph.get_buffer(input_name)
                input_buf.set_axis_format(AxisTracker.AxisFormat.NONTRIVIAL)

        # Add constant node for reset input if it is static
        self.add_constant_reset_node(converter_context, input_names, 'Gru')

        split_output_names = [[str(input_names[1]) + RNN_INPUT_TYPES[0] + "_forward_split",
                               str(input_names[1]) + RNN_INPUT_TYPES[0] + "_backward_split"]]
        forward_output_names = [str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_forward" for i, name in
                                enumerate(output_names)]
        backward_output_names = [str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_backward" for i, name in
                                 enumerate(output_names)]
        reshape_output_names = [[str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_reshape_forward",
                                 str(name) + QNN_GRU_OUTPUT_TYPES[i] + "_reshape_backward"]
                                for i, name in enumerate(output_names)]
        concat_output_names = [str(name) for name in output_names]

        # add split ops to the graph, each split op should have two outputs, one going into each RNN
        # inputs[i] where i >= 1 must be split along num_directions dimension in ONNX
        split_src_op_inputs = src_op_inputs[5:]
        split_input_names = [input_names[1]]

        if len(split_src_op_inputs) < len(split_input_names):
            raise ValueError("Source op {} requires initial_h.".format(src_op.name))

        for i in range(0, len(split_input_names)):
            log_debug("Adding split op {} while creating bidirectional RNN unit".format(split_ops[i].name))
            split_node = graph.add(split_ops[i], [split_input_names[i]], split_output_names[i])
            graph.add_src_op_info(split_ops[i].name, split_src_op_inputs[i], split_output_names[i])

            self.insert_constant_trace_info(split_input_names[i], split_node, converter_context)

            # Set override param encodings for split op
            if graph.user_quantization_overrides:
                encoding = graph.get_overridden_encoding(split_input_names[i])
                if encoding is not None:
                    for output_name in split_output_names[i]:
                        graph.set_overridden_encoding(output_name, encoding, False)


        # Modify input names to be different according to split
        forward_input_names = [input_names[0]] + [split_output_names[i][0] for i in range(len(split_input_names))] + forward_op_input_names[2:]
        backward_input_names = [input_names[0]] + [split_output_names[i][1] for i in range(len(split_input_names))] + backward_op_input_names[2:]

        log_debug("Adding forward RNN op {} while creating bidirectional RNN unit".format(forward_op.name))
        forward_node = graph.add(forward_op, forward_input_names, forward_output_names)
        graph.add_src_op_info(forward_op.name, forward_input_names, forward_output_names)

        forward_targets = [forward_node]
        forward_targets.extend(graph.get_output_buffers(forward_node))
        self.insert_trace_info(forward_targets, [(src_op.name, TraceType.OP)], converter_context)

        log_debug("Adding backward RNN op {} while creating bidirectional RNN unit".format(backward_op.name))
        backward_node = graph.add(backward_op, backward_input_names, backward_output_names)
        graph.add_src_op_info(backward_op.name, backward_input_names, backward_output_names)

        backward_targets = [backward_node].extend(graph.get_output_buffers(backward_node))
        self.insert_trace_info(backward_targets, [(src_op.name, TraceType.OP)], converter_context)

        # add concat op to the graph, should end up as a child of both forward and backward ops.
        # we need more than one concat node, depending on the number of outputs.
        for id in range(0, output_names_len):
            # add reshape for outputs to make shape to be ONNX's format
            log_debug("Adding reshape op {} and {} while creating bidirectional RNN unit".format(reshape_ops[id][0].name, reshape_ops[id][1].name))
            forward_reshape_node = graph.add(reshape_ops[id][0], [forward_output_names[id]], [reshape_output_names[id][0]])
            backward_reshape_node = graph.add(reshape_ops[id][1], [backward_output_names[id]], [reshape_output_names[id][1]])
            graph.add_src_op_info(reshape_ops[id][0].name, [forward_output_names[id]], [reshape_output_names[id][0]])
            graph.add_src_op_info(reshape_ops[id][1].name, [backward_output_names[id]], [reshape_output_names[id][1]])
            self.insert_trace_info([forward_reshape_node, graph.get_output_buffers(forward_reshape_node)[0]], [(src_op.name, TraceType.OP)], converter_context)
            self.insert_trace_info([backward_reshape_node, graph.get_output_buffers(backward_reshape_node)[0]], [(src_op.name, TraceType.OP)], converter_context)

            log_debug("Adding concat op {} while creating bidirectional RNN unit".format(concat_ops[id].name))
            concat_node = graph.add(concat_ops[id], [reshape_output_names[id][0], reshape_output_names[id][1]], [concat_output_names[id]])
            graph.add_src_op_info(concat_ops[id].name, [reshape_output_names[id][0], reshape_output_names[id][1]], [concat_output_names[id]])
            self.insert_trace_info([concat_node], [(src_op.name, TraceType.OP)], converter_context)
            self.insert_trace_info(graph.get_output_buffers(concat_node), [(concat_output_names[id], TraceType.TENSOR)], converter_context)

            # Set override output encodings for gru ops
            if id < len(src_op.output) and graph.user_quantization_overrides:
                encoding = graph.get_overridden_encoding(src_op.output[id], False)
                if encoding is not None:
                    graph.set_overridden_encoding(forward_output_names[id], encoding, False)
                    graph.set_overridden_encoding(backward_output_names[id], encoding, False)


    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'linear_before_reset':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)
        elif attr_name == "activations":
            gru_supported_activations = ["Sigmoid", "Tanh"]
            if len(attr_value) == 4:
                # bidirectional case has double the number of activations
                gru_supported_activations = gru_supported_activations*2
            log_assert(attr_value == gru_supported_activations,
                       "Received activations list: {} differs from GRU supported activations list: {}",
                       attr_value, gru_supported_activations)



OnnxTranslations.register_translation(OnnxGruTranslation(),
                                      converter_type('GRU', 'onnx'),
                                      op_adapter.GruOp.TRANSLATION_KEY)


# ------------------------------------------------------------------------------
#  LSTM
# ------------------------------------------------------------------------------
class OnnxLSTMTranslation(OnnxRnnTranslationsBase):
    from qti.aisw.converters.common.converter_ir.op_properties.stateful_lstm\
        import IR_INPUT_IDX, IR_INPUT_WEIGHTS_IDX, IR_HIDDEN_STATE_WEIGHTS_IDX,\
        IR_GATE_BIASES_IDX, IR_NORM_WEIGHTS_IDX, IR_CELL_STATE_WEIGHTS_IDX,\
        IR_PROJ_WEIGHTS_IDX, IR_PROJ_BIAS_IDX, IR_RESET_IDX,\
        IR_INITIAL_H_IDX, IR_INITIAL_C_IDX, IR_NUM_INPUTS,\
        IR_TO_ONNX_INDICES

    def __init__(self):
        OnnxRnnTranslationsBase.__init__(self)
        schema_dict = self.register_op_schema('LSTM', [1, 7, 14], [['clip',
                                                                'activation_alpha',
                                                                'activation_beta',
                                                                'output_sequence']])
        schema_dict.replace_default_values(activations=['Sigmoid', 'Tanh', 'Tanh'])
        schema_dict.register_method(self.validate_attribute_values)
        self.no_of_gates = 4
        self.peephole_weights = []
        # Number of inputs for standard ONNX LSTM just excludes the reset input
        # TODO: Update when op properties are refactored into C++
        self.NUM_INPUTS = lstm_op_props.NUM_INPUTS - 1

    def extract_parameters(self, src_op, converter_context):
        schema = self.op_schema(op_type=src_op.op_type)
        self.params = extract_attributes(src_op, schema=schema, validate=True)

        if "layout" in self.params and self.params.layout == 1:
            raise ValueError("ERROR: Unsupported value - 1 for layout attribute for {} op".format(src_op.name))

        # set parameters
        self.extract_params_for_type(src_op, converter_context)

        # check if initial_h and initial_c are included
        # snpe requires that if they are included, then all
        # 3 outputs will be returned.
        ONNX_INITIAL_H_IDX = self.IR_TO_ONNX_INDICES[self.IR_INITIAL_H_IDX]
        ONNX_INITIAL_C_IDX = self.IR_TO_ONNX_INDICES[self.IR_INITIAL_C_IDX]
        if len(self.input_names) > ONNX_INITIAL_H_IDX:
            OPTIONAL_INPUTS.initial_h = self.input_names[ONNX_INITIAL_H_IDX]
        if len(self.input_names) > ONNX_INITIAL_C_IDX:
            OPTIONAL_INPUTS.initial_c = self.input_names[ONNX_INITIAL_C_IDX]

    def extract_input_names(self, src_op, converter_context):
        ONNX_INPUT_IDX = self.IR_TO_ONNX_INDICES[self.IR_INPUT_IDX]
        ONNX_INPUT_WEIGHTS_IDX = self.IR_TO_ONNX_INDICES[self.IR_INPUT_WEIGHTS_IDX]
        ONNX_HIDDEN_STATE_WEIGHTS_IDX = self.IR_TO_ONNX_INDICES[self.IR_HIDDEN_STATE_WEIGHTS_IDX]
        ONNX_GATE_BIASES_IDX = self.IR_TO_ONNX_INDICES[self.IR_GATE_BIASES_IDX]

        src_op_inputs_len = len(list(src_op.input))
        if src_op_inputs_len < lstm_op_props.NUM_REQUIRED_INPUTS or src_op_inputs_len > self.NUM_INPUTS:
            raise ValueError(f"Unsupported number of inputs for source LSTM op {src_op.name}. "
                             f"Expected number of inputs between {lstm_op_props.NUM_REQUIRED_INPUTS} "
                             f"and {self.NUM_INPUTS}, got {src_op_inputs_len}")

        input_weights_name = self.input_names[ONNX_INPUT_WEIGHTS_IDX]
        rec_weights_name = self.input_names[ONNX_HIDDEN_STATE_WEIGHTS_IDX]
        if len(self.input_names) > ONNX_GATE_BIASES_IDX:
            bias_name = self.input_names[ONNX_GATE_BIASES_IDX]
            self.bias_src_info = [(self.input_names[ONNX_GATE_BIASES_IDX], TraceType.TENSOR)]
        else:
            bias_name = ''
            self.bias_src_info = []
        # If no bias_name input provided, replace with dummy all-zero bias
        bias_name = bias_name if bias_name else 'lstm_dummy_bias_all_zeros'

        self.weights_src_info = [(self.input_names[ONNX_INPUT_WEIGHTS_IDX], TraceType.TENSOR)]
        self.rec_weights_src_info = [(self.input_names[ONNX_HIDDEN_STATE_WEIGHTS_IDX], TraceType.TENSOR)]

        # Exclude reset input from being set to a default value, as the base
        # ONNX LSTM does not have the reset input. If it's a Block Op, the reset
        # input is added by the OnnxBlockOpStatefulLstmTranslation
        formatted_input_names = [''] * (self.IR_NUM_INPUTS - 1)
        formatted_input_names[self.IR_INPUT_IDX] = self.input_names[ONNX_INPUT_IDX]
        formatted_input_names[self.IR_INITIAL_H_IDX] = OPTIONAL_INPUTS.initial_h
        formatted_input_names[self.IR_INITIAL_C_IDX] = OPTIONAL_INPUTS.initial_c
        formatted_input_names[self.IR_INPUT_WEIGHTS_IDX] = input_weights_name
        formatted_input_names[self.IR_HIDDEN_STATE_WEIGHTS_IDX] = rec_weights_name
        formatted_input_names[self.IR_GATE_BIASES_IDX] = bias_name
        formatted_input_names[self.IR_NORM_WEIGHTS_IDX] = OPTIONAL_INPUTS.norm_weights
        formatted_input_names[self.IR_CELL_STATE_WEIGHTS_IDX] = OPTIONAL_INPUTS.cell_state_weights
        formatted_input_names[self.IR_PROJ_WEIGHTS_IDX] = OPTIONAL_INPUTS.proj_weights
        formatted_input_names[self.IR_PROJ_BIAS_IDX] = OPTIONAL_INPUTS.proj_bias

        return formatted_input_names

    def convert_params_to_snpe(self, weights, rec_weights, bias, hidden_size):

        no_of_gates = self.no_of_gates

        if bias is None:
            bias = np.zeros((no_of_gates, hidden_size), dtype=np.float32)
        else:
            # for probably vendor specific reasons, ONNX defines LSTM bias to be
            # separated into forward and recurrent parts, that are always added
            # together So we will always combine.  We need to reshape bias which
            # is in (2*no_of_gates*hidden_size) into (2, no_of_gates *
            # hidden_size).
            bias = np.reshape(bias, (2, no_of_gates * hidden_size))
            new_bias = np.empty((no_of_gates * hidden_size), dtype=np.float32)
            # Elements are stored in [weights, rec_weights] where each column
            # represents the gate and the number of rows is the hidden size
            np.add(bias[0, :], bias[1, :], out=new_bias[:])
            bias = new_bias.reshape(no_of_gates, hidden_size)

        # weights and rec_weights are also laid out as (no_of_gates*hidden_size, input_size)
        # and (no_of_gates*hidden_size, hidden_size)respectively. We need to reshape
        # to SNPE format depending on the rnn type.
        weights = np.reshape(weights, (no_of_gates, hidden_size, weights.shape[-1]))
        rec_weights = np.reshape(rec_weights, (no_of_gates, hidden_size, hidden_size))

        return weights, rec_weights, bias

    def add_op(self, src_op, converter_context):
        graph = converter_context.ir_graph
        self.extract_parameters(src_op, converter_context)
        self.add_src_op_info(src_op.name, src_op, graph)
        return self.create_rnn(src_op, converter_context, self.create_unidirectional_lstm,
                               self.create_bidirectional_lstm, rnn_type='LSTM')

    def create_unidirectional_lstm(self, graph, op_name, input_name_list, **kwargs):
        direction = kwargs['direction'] if kwargs else self.direction
        h_0_input_name = kwargs['h_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_h
        c_0_input_name = kwargs['c_0_input_name'] if kwargs else OPTIONAL_INPUTS.initial_c

        # For constant weights, they could be shared across multiple RolledLstm nodes, so
        # add a check here to avoid post-processing the weights again
        def input_already_prepared(name):
            if name:
                return (graph.has_buffer(name) and
                        isinstance(graph.get_producer_op(name), op_adapter.ConstantOp))
            else:
                return True
        skip_prepare_weights = all(map(input_already_prepared,
                                       input_name_list[self.IR_INPUT_WEIGHTS_IDX:self.IR_PROJ_BIAS_IDX+1]))
        if not skip_prepare_weights:
            if kwargs:
                [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(kwargs['weights'],
                                                                                          kwargs['rec_weights'],
                                                                                          kwargs['bias'],
                                                                                          kwargs['hidden_size'])
                src_input_list = [(op_name, TraceType.OP), (op_name, TraceType.OP), (op_name, TraceType.OP)]
            else:
                [gate_weights, gate_rec_weights, gate_bias] = self.convert_params_to_snpe(self.weights,
                                                                                          self.rec_weights,
                                                                                          self.bias,
                                                                                          self.params.hidden_size)
                src_input_list = [self.weights_src_info, self.rec_weights_src_info, self.bias_src_info]

            # transform from iofc to ifoc
            if self.no_of_gates == 4:
                self.params.hidden_size = int(gate_bias.size / self.no_of_gates)

                new_gate_bias = np.empty(gate_bias.shape, dtype=np.float32)
                for new, old in enumerate([0, 2, 1, 3]):
                    new_gate_bias[new, :] = gate_bias[old, :]
                gate_bias = new_gate_bias

                new_gate_rec_weights = np.empty(gate_rec_weights.shape, dtype=np.float32)
                for new, old in enumerate([0, 2, 1, 3]):
                    new_gate_rec_weights[new, :, :] = gate_rec_weights[old, :, :]
                gate_rec_weights = new_gate_rec_weights

                new_gate_weights = np.empty(gate_weights.shape, dtype=np.float32)
                for new, old in enumerate([0, 2, 1, 3]):
                    new_gate_weights[new, :, :] = gate_weights[old, :, :]
                gate_weights = new_gate_weights

            # LSTM specific organization into gate format
            gate_bias = gate_bias.reshape(-1, )
            gate_rec_weights = gate_rec_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)
            gate_weights = gate_weights.reshape(self.no_of_gates * self.params.hidden_size, -1)

            input_weights_name = input_name_list[self.IR_INPUT_WEIGHTS_IDX]
            rec_weights_name = input_name_list[self.IR_HIDDEN_STATE_WEIGHTS_IDX]
            bias_name = input_name_list[self.IR_GATE_BIASES_IDX]
            cell_state_weights_name = input_name_list[self.IR_CELL_STATE_WEIGHTS_IDX]

            params_dict = {}
            params_dict[input_weights_name] = gate_weights
            params_dict[rec_weights_name] = gate_rec_weights
            params_dict[bias_name] = gate_bias
            # cell state weights is the only optional input that can be captured in Onnx lstm
            if cell_state_weights_name:
                params_dict[cell_state_weights_name] = self.cell_state_weights

            self.src_input_dic[input_weights_name] = src_input_list[0]
            self.src_input_dic[rec_weights_name] = src_input_list[1]
            self.src_input_dic[bias_name] = src_input_list[2]

            self.prepare_params_as_constants(graph, params_dict)

        return op_adapter.RolledLstmOp(op_name,
                                       direction=direction,
                                       time_major=True,
                                       c_0_input_name=c_0_input_name,
                                       h_0_input_name=h_0_input_name,
                                       # if c_0 and h_0 exist, reset_state_at_time_step_0 will be False
                                       reset_state_at_time_step_0=False if h_0_input_name and c_0_input_name else True,
                                       hidden_size=self.params.hidden_size)

    def create_bidirectional_lstm(self, src_op, converter_context):
        return self.create_bidirectional_module(src_op, converter_context, self.weights, self.rec_weights,
                                                self.bias,
                                                self.params, self.create_unidirectional_lstm)

    @staticmethod
    def validate_attribute_values(src_op, attr_name, attr_value):
        if attr_name == 'input_forget':
            OpSchemaBase.validate_attribute_values(src_op, attr_name, attr_value)
        elif attr_name == "activations":
            lstm_supported_activations = ["Sigmoid", "Tanh", "Tanh"]
            if len(attr_value) == 6:
                # bidirectional case has double the number of activations
                lstm_supported_activations = lstm_supported_activations*2
            log_assert(attr_value == lstm_supported_activations,
                       "Received activations list: {} differs from supported activations list: {}",
                       attr_value, lstm_supported_activations)


OnnxTranslations.register_translation(OnnxLSTMTranslation(),
                                      converter_type('LSTM', 'onnx'),
                                      op_adapter.LstmOp.TRANSLATION_KEY)
