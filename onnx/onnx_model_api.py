# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Text, Union

import onnx
import qti.aisw.converters.onnx.util as Utils
from onnx import GraphProto, ModelProto, NodeProto, TensorProto, helper
from qti.aisw.converters.onnx.optimizations.einsum import ONNXEinsumPatternOptimizer
from qti.aisw.converters.onnx.optimizations.const_fold import ONNXConstantFoldingOptimization
from qti.aisw.converters.onnx.optimizations.sequence_construct import ONNXSequenceConstructPatternOptimizer
from packaging import version
from qti.aisw.converters.common.framework_model_api import FrameworkModelAPI
from qti.aisw.converters.common.utils.converter_utils import (
    log_debug,
    log_debug1,
    log_error,
    log_warning,
)
from qti.aisw.converters.common.utils.framework_utils import spawn_process_and_exec
from qti.aisw.converters.onnx.optimizations.const_fold import (
    ONNXConstantFoldingOptimization,
)
from qti.aisw.converters.onnx.util import (
    MAX_INT_32,
    SYMBOLIC_SHAPE_INFER_MIN_OPSET_VER,
    ModelWrapper,
    cleanup,
    get_graph_by_name,
    get_graphs,
    get_initializer_value,
    get_type_dims_info,
)


class ONNXModelUtils(FrameworkModelAPI):
    def __init__(self, loader):
        self.loader = loader
        self.assign_names_to_empty_nodes()
        self.einsum_opt = ONNXEinsumPatternOptimizer(self.loader)
        self.const_fold_opt = ONNXConstantFoldingOptimization(self.loader)
        self.sequence_construct_opt = ONNXSequenceConstructPatternOptimizer(self.loader)

    def remove_initializer_from_input(self):
        """
        Function to remove initializers from graph inputs

        :return ONNXModelUtils: Self instance of the class.
        """
        if self.loader.model.ir_version < 4:
            log_warning(
                "Can't remove initializer from inputs for models having ir version < 4"
            )
            return self

        init_dict = Utils.get_initializer_mappings(self.loader.model)
        input_dict = {i.name: i for i in self.loader.model.graph.input}
        for ip_name in input_dict:
            if ip_name in init_dict:
                input_ = input_dict[ip_name]
                self.loader.model.graph.input.remove(input_)
        return self

    def _optimize_by_ort_helper(
        self, opt_level, disabled_optimizers
    ) -> Union[None, ModelWrapper]:
        """
        Function to optimize the model using the Onnx-Runtime Optimizers

        :param Optional[Text] opt_level: Graph optimization level
            (Default is BASIC), available options ["BASIC", "EXTENDED", "ALL"]
        :param Optional[List] disabled_optimizers: A list of names of
            disabled optimizers., defaults to []
        :return Union[None, ModelWrapper]: ModelWrapper object if operation is
            successful else None.
        """
        SUPPORTED_OPT = ["BASIC", "EXTENDED", "ALL"]
        try:
            import onnxruntime
            from onnxruntime.GraphOptimizationLevel import (
                ORT_ENABLE_ALL,
                ORT_ENABLE_BASIC,
                ORT_ENABLE_EXTENDED,
            )

        except ImportError:
            log_warning(
                "Onnxruntime package not found in current "
                "environment. Optimization using onnxruntime will be skipped."
            )
            return None

        if opt_level not in SUPPORTED_OPT:
            log_warning(
                f"Provided optimization level '{opt_level}' for "
                + "onnxruntime optimizations is not supported. "
                + "Optimizing model using BASIC optimizations."
            )
            opt_level = "BASIC"
        if opt_level == "BASIC":
            graph_optimization_level = ORT_ENABLE_BASIC
        elif opt_level == "EXTENDED":
            graph_optimization_level = ORT_ENABLE_EXTENDED
        else:
            graph_optimization_level = ORT_ENABLE_ALL
        kwargs = {}

        if disabled_optimizers:
            kwargs["disabled_optimizers"] = disabled_optimizers

        optimized_model = None
        model_path = self.loader.model_wrapper.get_model_path()
        status, _, optimized_model = Utils.create_ort_session(
            model_path,
            optimize=True,
            graph_optimization_level=graph_optimization_level,
            **kwargs,
        )
        if not status:
            return None

        return ModelWrapper(optimized_model)

    def optimize_by_ort(
        self,
        opt_level: Optional[Text] = "BASIC",
        disabled_optimizers: Optional[List] = [],
    ):
        """
        Function to optimize the model using the Onnx-Runtime Optimizers

        :param Optional[Text] opt_level: Graph optimization level (Default is BASIC), available options ["BASIC", "EXTENDED", "ALL"]
        :param Optional[List] disabled_optimizers: A list of names of disabled optimizers., defaults to []
        :return ONNXModelUtils: Self instance of the class.
        """

        log_debug1("Performing Optimization using ORT")
        status, model_wrapper = spawn_process_and_exec(
            self._optimize_by_ort_helper,
            opt_level,
            disabled_optimizers,
            process_name="ORT Optimizer",
        )
        if not status:
            log_warning("Optimization with ORT failed.")
            return self
        self.loader.update_model_wrapper(model_wrapper)
        return self

    def _optimize_by_simplifier_helper(
        self, static_input_shapes, skip_optimization=False
    ) -> ModelWrapper:
        """
        Function to optimize the model inplace using the Onnx Simplifier Tools

        :param Optional[Dict] static_input_shapes: Dict with mapping of
            input names to their corresponding static shapes., defaults to None
        :return ModelWrapper: Instance of onnx model in the form of ModelWrapper.
        """
        simplified_model = None
        try:
            from onnxsim import simplify

            # FIXME: Add a fix in onnx-simplifier repo where we can either pass
            # full model (<2gb) or model path (>2gb). For model with >2gb if we
            # pass path then it should not read the model and again dump to the
            # disk.
            model_w_weights = self.loader.model_wrapper.get_full_model()
            if static_input_shapes:
                # Covered the simplifier call inside contextlib to suppress the logs incase
                # the outputs are not matched.
                simplified_model, check = simplify(
                    model_w_weights,
                    input_shapes=static_input_shapes,
                    perform_optimization=not skip_optimization,
                )
            else:
                simplified_model, check = simplify(
                    model_w_weights, perform_optimization=not skip_optimization
                )

        except Exception as e:
            log_warning(f"Onnx model simplification failed due to: {e}")
            check = False

        if check == False:
            log_warning("Simplified model validation failed")
            return None
        else:
            log_debug1("Simplified model validation is successful")
            return ModelWrapper(simplified_model)

    def optimize_by_simplifier(self, static_input_shapes: Optional[Dict] = None):
        """
        Function to optimize the model inplace using the Onnx Simplifier Tools

        :param Optional[Dict] static_input_shapes: Dict with mapping of
            input names to their corresponding static shapes., defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        status, model_wrapper = spawn_process_and_exec(
            self._optimize_by_simplifier_helper,
            static_input_shapes,
            process_name="ONNX Simplifier Optimizer",
        )
        clean_model_req = False
        if not status:
            clean_model_req = True
            log_debug1("Running Onnx Simplifier Optimizer without any optimizations.")
            status, model_wrapper = spawn_process_and_exec(
                self._optimize_by_simplifier_helper,
                static_input_shapes,
                skip_optimization=True,
                process_name="ONNX Simplifier Optimizer",
            )
            if not status:
                return self

        log_debug1("Optimization using Onnx Simplifier applied successfully.")
        self.loader.update_model_wrapper(model_wrapper)

        # Since, onnx-simplifier fuses/replaces some nodes, the new nodes are
        # not having any names. Hence calling below line.
        self.assign_names_to_empty_nodes()

        if clean_model_req:
            # Need to call clean_model API as for few models it is observed that
            # when onnxsim is executed without any optimization, at that time
            # only constant folding and shape inference is executed. In that
            # there are some redundant initializers still present in the model
            # because these were expected to be removed due to
            # eliminate_unused_initializer pass in onnx-optimizer. As these are
            # not removed by simplifier these actually gets translated to IR.
            # This we need to prevent by clean_model() call.
            self.loader.utils.clean_model()
        return self

    def optimize(self, **kwargs):
        """
        Optimize the onnx graph inplace with all the default optimization.

        :kwargs static_input_shapes (List) : Optional Input shapes.
        :kwargs skip_optimization (bool) : Flag to skip onnx runtime and simplifier optimizations.
        :kwargs opt_level (Text) : Graph optimization level (Default is BASIC), available options ["BASIC", "EXTENDED", "ALL"]

        :return ONNXModelUtils: Self instance of the class.
        """
        # FIXME: Divide this function in 2 parts so that weight load unload can be minimized.
        #       - Weight dependent optimizations    (simplifier)
        #       - Weight independent optimizations (einsum)

        static_input_shapes = kwargs.get("static_input_shapes", None)
        skip_optimization = kwargs.get("skip_optimization", False)

        if not skip_optimization:
            self.optimize_by_simplifier(static_input_shapes)

            if not self.loader.has_custom_op:
                opt_level = kwargs.get("opt_level", "BASIC")
                disabled_optimizers = kwargs.get("disabled_optimizers", [])
                # FIXME: Some models are failing due to onnx runtime optimization at IR
                # level but working with onnx runtime.
                # self.optimize_by_ort(opt_level, disabled_optimizers)
        self.einsum_opt.optimize(skip_optimization=skip_optimization)
        self.sequence_construct_opt.optimize(skip_optimization=skip_optimization)

        # if skip_optimization:
        #     # Enable only if simplifier is skipped.
        # self.const_fold_opt.optimize()

        return self

    def update_input_node(self, input_names: List[Text], input_dims: List[Text]):
        """
        Updates input nodes dimensions if node is present as input else delete
        the existing graph inputs and add new inputs with provided shape.

        :param List[Text] input_names: List of input names.
        :param List[Text] input_dims: List of input dims to be used while updating.
        :raises ValueError: If the update input node is not supported.
        :raises ValueError: If the input name is one of the initializer.
        :raises ValueError: input_dims command not for any input.
        :raises ValueError: If the any of the input can't be updated.
        :return ONNXModelUtils: Self instance of the class.
        """

        # FIXME: Instead of taking input_names and input_dims as 2 different
        #        lists we shall take one dict which has mapping from name to dim.

        graph = self.loader.model.graph
        if version.parse(onnx.version.version) < version.parse("1.6.0"):
            raise ValueError(
                "update input node does not supported with ONNX versions < 1.6.0"
            )

        input_names = list(input_names)
        input_dims = list(input_dims)
        initializers = [init.name for init in graph.initializer]
        original_inputs = {model_input.name: model_input for model_input in graph.input}
        original_types = {
            model_input.name: model_input.type.tensor_type.elem_type
            for model_input in graph.input
        }
        new_inputs = {name: dim for name, dim in zip(input_names, input_dims)}

        # Step 1: remove original graph inputs
        for node_name in original_inputs:
            if node_name not in initializers:
                graph.input.remove(original_inputs[node_name])

        # Step 2: If input specified is part of graph inputs, update its dimensions
        for name in new_inputs:
            if name in initializers:
                raise ValueError(
                    "--input_dim command not supported with initializer " + name
                )
            elif name in original_inputs:
                dim = new_inputs[name]
                dims = tuple(map(int, dim.split(",")))
                type_new = original_types[name]
                input_new = onnx.helper.make_tensor_value_info(name, type_new, dims)
                graph.input.append(input_new)
                input_names.remove(name)
                input_dims.remove(dim)
            else:
                continue

        # Check if all inputs are accounted for, if Yes nothing more to be done. Return
        if len(input_names) == 0 and len(input_dims) == 0:
            return self

        # Get the type of each model input
        input_types = {}
        for input_name in input_names:
            input_found, input_type, _ = get_type_dims_info(
                self.loader.model.graph.input, input_name
            )
            input_types[input_name] = (
                input_type if input_found else onnx.TensorProto.FLOAT
            )

        # Step 3: If input specified is intermittent graph output,
        #         a.  Add this buffer to a list for removal later
        #         b.  Create input TensorProto with this name and dimension
        bufs_to_remove = set()
        for i, src_op in enumerate(graph.node):
            for output_buf_name in src_op.output:
                if output_buf_name in input_names:
                    position = input_names.index(output_buf_name)
                    dim = input_dims[position]
                    dims = tuple(map(int, dim.split(",")))
                    input_new = onnx.helper.make_tensor_value_info(
                        output_buf_name, input_types[output_buf_name], dims
                    )
                    graph.input.append(input_new)
                    bufs_to_remove.add(output_buf_name)
                    input_names.remove(output_buf_name)
                    input_dims.remove(dim)

        # Check if all inputs specified are accounted for
        if len(input_names) != 0 and len(input_dims) != 0:
            invalid_names = ", ".join(input_names)
            raise ValueError(
                "input_dim command input name(s) not found: {}".format(invalid_names)
            )

        # Step 4: Find all nodes to be removed from the graph. These include:
        #   a. Nodes that produce the buffers cached for removal
        #   b. All nodes that precede them in the graph
        nodes_to_remove = []
        while bufs_to_remove:
            buf_name = bufs_to_remove.pop()
            if buf_name in original_inputs or buf_name in initializers:
                # This was already removed or does not need to be handled
                continue

            # Find node that produces the buffer or is named after the buffer
            node_list = [node for node in graph.node if buf_name in node.output]
            if not node_list:
                raise KeyError("Node that produces {} not found".format(buf_name))
            elif len(node_list) != 1:
                raise KeyError(
                    "Multiple nodes {} found for output buffer {}".format(
                        node_list, buf_name
                    )
                )

            node = node_list[0]
            # Add all inputs of this node as also to be removed
            bufs_to_remove.update(set(node.input))
            # Add this node to be removed if not already added
            if node not in nodes_to_remove:
                nodes_to_remove.append(node)

        # Step 5: Remove the nodes marked in Step 4
        # Check that all buffers in a slice were specified, if not Throw Error
        remaining_nodes = [node for node in graph.node if node not in nodes_to_remove]
        remaining_buffers = set()
        for remaining_node in remaining_nodes:
            remaining_buffers.update(remaining_node.input)
        for node in nodes_to_remove:
            for output in node.output:
                if output in remaining_buffers and output not in input_names:
                    raise ValueError(
                        "Cannot disconnect node with outputs: {} as output buffer"
                        ": {} is still in use and was not specified as input to the Model".format(
                            str(node.output), str(output)
                        )
                    )
            graph.node.remove(node)

        return self

    def update_output_names(self, output_names: List[Text]):
        """
        Update the model output names inplace.
        This will make the provided output_names as output and remove the rest
        of the outputs from the model.

        :param List[Text] output_names: List of output names to update.
        :return ONNXModelUtils: Self instance of the class.
        """
        # Nodes to delete is map with node.name as Key and node as value
        nodes_to_delete = {}
        queue = list(output_names)
        visited = set(queue)
        map_node_output = {}
        model = self.loader.model

        # traverse the graph once and build a map with (node, output name) for the graph
        for node in model.graph.node:
            # same node might appear for more than one output_name
            for output_name in node.output:
                map_node_output[output_name] = node
            # add all nodes in the graph to nodes_to_delete, later retained nodes will be deleted from it
            nodes_to_delete[node.name] = node
        log_debug("Total nodes in nodes_to_delete {}".format(len(nodes_to_delete)))
        log_debug("Total nodes in map_node_output {}".format(len(map_node_output)))

        while queue:
            name = queue.pop(0)
            # find the node who has same output name from the map
            if name not in map_node_output:
                log_debug("tensor name {} is an input or constant tensor".format(name))
                continue
            current_node = map_node_output.get(name)

            if current_node.name not in nodes_to_delete:
                # if node does not exist in nodes_to_delete, it might have appeared for more than one output
                log_debug(
                    "{} node is not present. Might have already removed from map".format(
                        current_node.name
                    )
                )
                continue
            nodes_to_delete.pop(current_node.name)
            # save the input tensor name in queue for finding producer node for the same
            for input_name in current_node.input:
                if input_name in visited:
                    continue
                queue.append(input_name)
                visited.add(input_name)

        log_debug(
            "Total nodes remained in nodes_to_delete {}".format(len(nodes_to_delete))
        )
        # Remove nodes that are not retained
        for node in nodes_to_delete.values():
            model.graph.node.remove(node)

        # Get the output dimensions of the new output nodes
        new_output_value_infos = []
        for output_name in output_names:
            # First check the graph outputs for info on outputs
            output_found, output_type, output_dims = get_type_dims_info(
                model.graph.output, output_name
            )

            # Fallback to using optional value_info field for info on new outputs
            if not output_found and model.graph.value_info:
                output_found, output_type, output_dims = get_type_dims_info(
                    model.graph.value_info, output_name
                )

            # Finally, fallback to using graph inputs for info on new outputs
            if not output_found:
                output_found, output_type, output_dims = get_type_dims_info(
                    model.graph.input, output_name
                )

            if output_found:
                output_value_info = onnx.helper.make_tensor_value_info(
                    output_name, output_type, output_dims
                )
            else:
                output_value_info = onnx.helper.ValueInfoProto()
                output_value_info.name = output_name

            new_output_value_infos.append(output_value_info)

        # Remove old output nodes
        for output_node in [_ for _ in model.graph.output]:
            model.graph.output.remove(output_node)

        # Add new output info
        model.graph.output.extend(new_output_value_infos)
        return self

    def update_onnx_define_symbols(
        self, define_symbols: Optional[Dict] = None, batch: Text = None
    ):
        """
        Update the onnx define symbols as well as batch in place.

        :param Optional[Dict] define_symbols: onnx define symbols dictionary, defaults to None
        :param Text batch: Value for batch symbol to be imputed, defaults to None
        :raises ValueError: In case of not being able to update the onnx defined symbols.
        :return ONNXModelUtils: Self instance of the class.
        """
        if not (define_symbols or batch):
            return self

        graph = self.loader.model.graph
        if version.parse(onnx.version.version) < version.parse("1.6.0"):
            raise ValueError(
                "Not able to add batch size and define symbols for ONNX versions < 1.6.0"
            )

        # Override any symbols present in the input shapes with values passed by the client
        original_inputs = {node.name: node for node in graph.input}
        new_inputs = Utils.get_all_dims_info(Utils.get_inputs(self.loader.model))
        for name, dtype, dims in new_inputs:
            log_debug1(
                "Proccessing overrides for input {} with dims {}".format(name, dims)
            )
            modified = False
            if define_symbols:
                for i, dim in enumerate(dims):
                    if isinstance(dim, str):
                        if dim in define_symbols:
                            log_debug1(
                                'Overriding "{}" with {}'.format(
                                    dim, int(define_symbols[dim])
                                )
                            )
                            dims[i] = int(define_symbols[dim])
                            modified = True
                        else:
                            log_warning(
                                f"For input: {name}, the onnx defined "
                                f"symbol {dim} is not provided."
                            )

            # Override the batch dimension of all inputs with the passed value
            # TODO At some point make this batch logic common for all converters
            if batch:
                log_debug1(
                    "Overriding batch dim of {} from {} to {}".format(
                        name, dims[0], int(batch)
                    )
                )
                dims[0] = int(batch)
                modified = True

            # Remove the original input and add the new updated input
            if modified:
                new_input = onnx.helper.make_tensor_value_info(name, dtype, dims)
                log_debug1("Generated new input {} with dims {}".format(name, dims))
                graph.input.remove(original_inputs[name])
                graph.input.append(new_input)

        return self

    def add_inputs(self, input_names: List[Text], infer_shape: bool = True):
        """
        Function to convert the given tensor name into graph inputs.

        :param List[Text] input_names: List of tensor names which needs to be
            converted into graph inputs.
        :param bool infer_shape: Perform shape inference before input creation,
            defaults to True
        :raises Exception: If the given tensor can't be made the graph input.
        :return ONNXModelUtils: Self instance of the class.
        """
        if infer_shape:
            self.native_shape_inference(delete_existing_shapes=True)

        val_info_dict = Utils.get_value_info_proto_mappings(self.loader.model)
        init_dict = Utils.get_initializer_mappings(self.loader.model)
        node_by_output_tensor_name = Utils.get_node_by_output_name(self.loader.model)

        model_inputs = self.loader.get_input_names()
        filtered_input_names = [
            name for name in input_names if name not in model_inputs
        ]

        if not filtered_input_names:
            # Means all the elements of input_names are already inputs of model.
            return self

        # First check whether all the given input_names are available in value info.
        for name in filtered_input_names:
            if name not in init_dict and name not in val_info_dict:
                raise Exception(
                    f"{name} can't be made an input as it is an intermediate"
                    "tensor and it is not present in value info."
                )

        for name in filtered_input_names:
            # Remove the connection of this tensor with its parent node.
            if name in node_by_output_tensor_name:
                node = node_by_output_tensor_name[name]
                node_op_to_remove = [
                    node_op for node_op in node.output if node_op == name
                ]
                for node_op in node_op_to_remove:
                    node.output.remove(node_op)

                # Add the value info of the intermediate tensor in the graph's
                # input.
                self.loader.model.graph.input.append(val_info_dict[name])

            # Remove the instance of the new input from initializer if any.
            if name in init_dict:
                initializer = init_dict[name]
                initializer_numpy_data = get_initializer_value(initializer)
                self.loader.model.graph.initializer.remove(initializer)

                # Create a new value info for the initializer tensor and add the
                # same to graph's input.
                new_input_val_info = helper.make_tensor_value_info(
                    name,
                    onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[initializer_numpy_data.dtype],
                    initializer_numpy_data.shape,
                )
                self.loader.model.graph.input.append(new_input_val_info)

        # Cleanup the model as after removing the connection with the parent
        # node the graph has dangling nodes.
        self.clean_model()
        return self

    def add_outputs(self, output_names: List[Text], infer_shape: bool = True):
        """
        Function to convert the given tensor name into graph outputs.

        :param List[Text] output_names: List of tensors names which needs to be
            converted into graph outputs.
        :param bool infer_shape: Perform shape inference before output creation,
            defaults to True
        :raises ValueError: If the given tensor can't be made the graph output.
        :return ONNXModelUtils: Self instance of the class.
        """
        if infer_shape:
            self.native_shape_inference(delete_existing_shapes=True)

        tensor_val_info_dict = Utils.get_value_info_proto_mappings(self.loader.model)
        input_val_info_dict = self.loader.get_inputs()
        output_val_info_dict = self.loader.get_outputs()
        all_val_info_dict = {
            **tensor_val_info_dict,
            **input_val_info_dict,
            **output_val_info_dict,
        }

        model_outputs = self.loader.get_output_names()

        filtered_output_names = [
            name for name in output_names if name not in model_outputs
        ]

        if not filtered_output_names:
            # Means all the elements of output_names are already output of model.
            return self

        # First check whether all the given output_names are available in value info.
        for name in filtered_output_names:
            if name not in all_val_info_dict:
                raise ValueError(
                    f"{name} can't be made an output as it is not present in value info."
                )

        # If all output_names are available in value info then make the change to self.loader.
        for name in filtered_output_names:
            self.loader.model.graph.output.append(all_val_info_dict[name])
        return self

    def remove_inputs(self, input_names_to_remove: List[str]):
        """
        Function to remove give inputs from the model.
        Note: This API should not be used to remove the initializer which are
        present in graph as inputs also. For such cases call
        remove_initializer_from_input() API.

        :param List[str] input_names_to_remove: List of input names to be
                removed from model's input.
        :return ONNXModelUtils: Self instance of the class.
        """
        get_node_by_input_tensor_name = Utils.get_node_by_input_name(self.loader.model)

        input_dict = self.loader.get_inputs()
        for input_name in input_names_to_remove:
            if input_name in input_dict:
                ip = input_dict[input_name]
                if input_name not in get_node_by_input_tensor_name:
                    # This means it is not used by any nodes. Means we are safe to
                    # remove this input.
                    self.loader.model.graph.input.remove(ip)
                else:
                    children_nodes = [
                        n.name for n in get_node_by_input_tensor_name[input_name]
                    ]
                    log_warning(
                        f"{input_name} is can't be removed from the "
                        f"graph as it is used by {children_nodes} nodes."
                    )
            else:
                log_debug1(
                    f"{input_name} can't be removed from model's inputs as it "
                    "is not a model's input."
                )
        return self

    def remove_outputs(self, output_names_to_remove: List[str]):
        """
        Function to remove given outputs from the model.

        :param List[str] output_names_to_remove: List of output names to be
                removed from model's outputs.
        :return ONNXModelUtils: Self instance of the class.
        """
        model_output_names = self.loader.get_output_names()
        output_dict = self.loader.get_outputs()
        for output_name in output_names_to_remove:
            if output_name in model_output_names:
                op = output_dict[output_name]
                self.loader.model.graph.output.remove(op)
            else:
                log_debug1(
                    f"{output_name} can't be removed from "
                    "model's outputs as it is not a model's output."
                )
        return self

    def _native_shape_inference_helper(
        self, delete_existing_shapes
    ) -> Union[None, ModelWrapper]:
        """
        Run the shape inference over the model and add shape information for all
        nodes and all outputs.

        :param bool delete_existing_shapes: Delete existing shape information
            before obtaining shapes, defaults to True
        :return Union[None, ModelWrapper]: ModelWrapper object if operation is
            successful else None.
        """
        try:
            # As a first step try to run symbolic shape inference. If this fails
            # then as a fallback mechanism use normal shape inference.
            model_wrapper = self.__symbolic_shape_inference()
            log_debug1("Symbolic Shape inference executed successfully.")
        except Exception as e:
            # Note: Symbolic shape inference will fail for CustomOps.
            #       So as a fall back we should be calling normal shape
            #       inference.
            log_warning(
                "Symbolic shape inference Failed. "
                f"Exception: {e}. Running normal shape inference."
            )

            try:
                model_wrapper = self.__shape_inference(delete_existing_shapes)
                log_debug1("Onnx Shape inference executed successfully.")
            except Exception as e:
                log_warning(f"Shape inference Failed With Exception: {e}.")
                return None

        return model_wrapper

    def native_shape_inference(self, delete_existing_shapes: bool = False):
        """
        Run the shape inference over the model and add shape information for all
        nodes and all outputs.

        :param bool delete_existing_shapes: Delete existing shape information
            before obtaining shapes, defaults to True
        :return ONNXModelUtils: Self instance of the class.
        """
        status, model_wrapper = spawn_process_and_exec(
            self._native_shape_inference_helper,
            delete_existing_shapes,
            process_name="Shape Inference",
        )
        if not status:
            return self

        log_debug1("Shape inference executed successfully.")
        self.loader.update_model_wrapper(model_wrapper)
        return self

    def __symbolic_shape_inference(self) -> ModelWrapper:
        """
        This method will add the symbolic shape info to the model file.

        :return ModelWrapper: Instance of onnx model in the form of ModelWrapper.
        """
        # Symbolic shape inference works for both ModelProtos < 2GB and > 2GB.
        try:
            from onnxruntime.tools.symbolic_shape_infer import (
                SymbolicShapeInference,
                get_opset,
            )

            # Symbolic Shape Inference doesn't need to explicitly remove
            # existing shapes.

            # Symbolic shape inference needs model with weights.
            model_w_weights = self.loader.model_wrapper.get_full_model()
            onnx_opset = get_opset(model_w_weights)
            if (not onnx_opset) or onnx_opset < SYMBOLIC_SHAPE_INFER_MIN_OPSET_VER:
                raise RuntimeError(
                    "Symbolic Shape Inference only supports "
                    "models of onnx opset 7 and above."
                )
            symbolic_shape_inference = SymbolicShapeInference(
                int_max=MAX_INT_32, auto_merge=False, guess_output_rank=False, verbose=0
            )
            all_shapes_inferred = False
            symbolic_shape_inference._preprocess(model_w_weights)
            while symbolic_shape_inference.run_:
                all_shapes_inferred = symbolic_shape_inference._infer_impl()
            symbolic_shape_inference._update_output_from_vi()
            if not all_shapes_inferred:
                raise RuntimeError("Incomplete symbolic shape inference.")
            return ModelWrapper(symbolic_shape_inference.out_mp_)
        except ImportError as e:
            raise ImportError(
                "Onnxruntime package not found in current "
                "environment. Symbolic Shape Inference will be skipped."
            )

    def __shape_inference(self, delete_existing_shapes: bool) -> ModelWrapper:
        """
        This method will add the shape info to the model file.

        :param bool delete_existing_shapes: Delete existing shape information
            before obtaining shapes, defaults to True
        :return ModelWrapper: Instance of onnx model in the form of ModelWrapper.
        """
        if delete_existing_shapes:
            self.remove_shapes()
        # onnx's shape inference works without weights, too. Hence using
        # self.loader.model which is not having weights in it.

        # FIXME: Here we are cloning existing wrapper which means for the cloned
        # version the remove_on_del is also True. Right now we are in child process
        # and for all child process remove_on_del should be false.
        model = onnx.shape_inference.infer_shapes(self.loader.model)
        # Note: Here we are changing self.loader.model_wrapper instead of creating
        # new one. Shape inference is executed in children process hence any change
        # to self variable won't be affected to parent process.
        self.loader.model_wrapper.update_model(model)
        return self.loader.model_wrapper

    def clean_model(self):
        """
        Clean the model from dangling or redundant nodes or input or outputs in
        the graph.

        :return ONNXModelUtils: Self instance of the class.
        """
        model = cleanup(self.loader.model)
        self.loader.update_model(model)
        return self

    def remove_node(self, node: NodeProto):
        """
        Function to remove the node from the graph

        :param NodeProto node: Node reference to be removed from graph.
        :return ONNXModelUtils: Self instance of the class.
        """
        for graph in get_graphs(self.loader.model):
            if node in graph.node:
                graph.node.remove(node)
        return self

    def remove_nodes(self, nodes_to_remove: NodeProto):
        """
        Function to remove the nodes from the graph

        :param NodeProto nodes_to_remove: List of nodes to remove from the graph
        :return ONNXModelUtils: Self instance of the class.
        """
        for node in nodes_to_remove:
            self.remove_node(node)
        return self

    def add_initializer(self, tensor: TensorProto, graph_name: Text = None):
        """
        Function to add single initializer to the given graph.

        :param TensorProto tensor: Onnx TensorProto instance.
        :param Text graph_name:  Name of the graph in which node will be added,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if graph_name is None or graph_name == self.loader.model.graph.name:
            self.loader.model.graph.initializer.extend([tensor])
        else:
            graph = get_graph_by_name(self.loader.model, graph_name)
            graph.initializer.extend([tensor])
        return self

    def __get_topological_insert_id(self, graph: GraphProto, outputs: List[str]) -> int:
        """
        Function to get the topological insert id of the node.

        :param GraphProto graph: Onnx graph reference.
        :param List[str] outputs: Node output names for given node.
        :return int: Insert id of the given node.
        """
        for idx, node in enumerate(graph.node):
            for input in node.input:
                if input in outputs:
                    return idx
        return len(graph.node)

    def add_node(self, node: NodeProto, graph_name: Text = None):
        """
        Function to Add the single node to the given graph name.

        :param NodeProto node: Onnx node reference.
        :param Text graph_name: Name of the graph in which node will be added,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if graph_name is None or graph_name == self.loader.model.graph.name:
            graph_name = self.loader.model.graph.name

        graph = get_graph_by_name(self.loader.model, graph_name)
        insert_idx = self.__get_topological_insert_id(graph, node.output)
        graph.node.insert(insert_idx, node)
        return self

    def add_nodes(
        self,
        nodes_to_add: List[NodeProto],
        node_name_to_graph_name: Dict[Text, Text] = None,
    ):
        """
        Function to Add the nodes to the graph

        :param List[NodeProto] nodes_to_add: List of node to add to the graph
        :param Dict[Text, Text] node_name_to_graph_name: Node name to graph name
            dict which represents which node to be added in which graph,
            defaults to None
        :return ONNXModelUtils: Self instance of the class.
        """
        if node_name_to_graph_name is None:
            for node in nodes_to_add:
                self.add_node(node)
        else:
            for node in nodes_to_add:
                graph_name = node_name_to_graph_name[node.name]
                self.add_node(node, graph_name)
        return self

    def __graph_topological_sort(self, graph: GraphProto):
        """
        Function to get the topologically sorted graph from Graph proto

        :param GraphProto graph: GraphProto of the model.
        :return ONNXModelUtils: Self instance of the class.
        """
        deps_count = [0] * len(graph.node)  # dependency count of each node
        deps_to_nodes = {}  # input to node indices
        sorted_nodes = []  # initialize sorted_nodes

        for node_idx, node in enumerate(graph.node):
            # CANNOT use len(node.input) directly because input can be optional
            deps_count[node_idx] = sum(1 for _ in node.input if _)
            if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
                sorted_nodes.append(graph.node[node_idx])
                continue
            for input_name in node.input:
                if input_name not in deps_to_nodes:
                    deps_to_nodes[input_name] = [node_idx]
                else:
                    deps_to_nodes[input_name].append(node_idx)

        # Note: this logic only applies to top level graph since a sub graph could use intializer from parent graph
        initializer_names = [init.name for init in graph.initializer]
        graph_input_names = [input.name for input in graph.input]
        input_names = initializer_names + graph_input_names

        intermediate_output_tensor_names = set()
        for n in graph.node:
            for n_op in n.output:
                if n_op not in intermediate_output_tensor_names:
                    intermediate_output_tensor_names.add(n_op)

        for n in graph.node:
            for n_ip in n.input:
                if n_ip == "":
                    # Skip the blank named input as that input is an attribute
                    # of node but its value is not present. Assuming this as an
                    # optional attribute of the node.
                    continue
                if (
                    n_ip not in initializer_names
                    and n_ip not in graph_input_names
                    and n_ip not in intermediate_output_tensor_names
                ):
                    # If a node's input is not output of any node in the present graph then
                    # add that name in input_names with an assumption that the name is part of
                    # parent graph's some node's output or the node has no dependency on input
                    # and it is a constant node.
                    input_names.append(n_ip)

        input_names.sort()
        prev_input_name = None
        for input_name in input_names:
            if prev_input_name == input_name:
                continue
            prev_input_name = input_name
            if input_name in deps_to_nodes:
                for node_idx in deps_to_nodes[input_name]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(graph.node[node_idx])

        start = 0
        end = len(sorted_nodes)
        while start < end:
            for output in sorted_nodes[start].output:
                if output in deps_to_nodes:
                    for node_idx in deps_to_nodes[output]:
                        deps_count[node_idx] = deps_count[node_idx] - 1
                        if deps_count[node_idx] == 0:
                            sorted_nodes.append(graph.node[node_idx])
                            end = end + 1
            start = start + 1

        assert end == len(graph.node), "Graph is not a DAG"
        graph.ClearField("node")
        graph.node.extend(sorted_nodes)
        return self

    def topological_sort(self):
        """
        Function to get the topologically sorted graphs.

        :return Self instance of the class.
        """
        for graph in get_graphs(self.loader.model):
            self.__graph_topological_sort(graph)
        return self

    def assign_names_to_empty_nodes(self):
        """
        Add name to all the nodes whos name is "".

        :return ONNXModelUtils: Self instance of the class.
        """
        model = Utils.assign_names_to_empty_nodes(self.loader.model)
        self.loader.update_model(model)
        return self

    def remove_shapes(self):
        """
        Remove tensor shape information from model.

        :return ONNXModelUtils: Self instance of the class.
        """
        model = Utils.remove_shapes(self.loader.model)
        self.loader.update_model(model)
        return self
