# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import os
from typing import Any, Dict, List, Optional, Text, Union

import onnx
import qti.aisw.converters.onnx.util as Utils
from onnx import ModelProto, NodeProto
from qti.aisw.converters.common.custom_ops.op_factory import CustomOpFactory
from qti.aisw.converters.common.loader_base import FrameworkModelLoader
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning
from qti.aisw.converters.common.utils.framework_utils import (
    FrameworkSummary,
    TensorInfo,
)
from qti.aisw.converters.onnx import model_evaluator
from qti.aisw.converters.onnx.onnx_model_api import ONNXModelUtils
from qti.aisw.converters.onnx.util import ModelWrapper


class ONNXLoader(FrameworkModelLoader):
    def __init__(
        self,
        args: Any,
        custom_op_factory: Optional[CustomOpFactory],
    ) -> None:
        """
        Creates the onnx loader instance.
        :param custom_op_factory: CustomOpFactory Instance, defaults to None
        """
        self.init(args.input_network, args.defer_loading)
        super(ONNXLoader, self).__init__(
            converter_type="onnx", custom_op_factory=custom_op_factory, args=args
        )

    def init(
        self,
        path_or_proto: Union[Text, ModelProto],
        defer_loading: bool=False,
    ):
        """
        Initialize the ONNX Loader class.

        :param Union[Text,ModelProto] path_or_proto: Path of onnx model or model proto.
        :param bool defer_loading: If False, the model will be loaded eagerly. If True, the model will be loaded lazily.
        """
        self.model_wrapper = ModelWrapper(model_or_path = path_or_proto, include_attributes = True, defer_loading = defer_loading)
        self.utils = ONNXModelUtils(self)

    def clone_loader(self):
        """
        Clone the onnx loader safely

        :return ONNXLoader: Returns the deep copied ONNXLoader instance.
        """
        temp_loader = copy.deepcopy(self)
        model_w_weights = self.model_wrapper.get_full_model()
        temp_loader.init(model_w_weights)
        return temp_loader

    def update_model(self, model: ModelProto) -> None:
        """
        Update the current object's ModelProto with given ModelProto.

        :param ModelProto model: ModelProto instance to be copied.
        """
        self.model_wrapper.update_model(model)

    def update_model_wrapper(self, model_wrapper: ModelWrapper) -> None:
        """
        Update the current object's wrapper with given wrapper.

        :param ModelWrapper model: ModelWrapper instance to be copied.
        """
        self.model_wrapper.move_wrapper(model_wrapper)

    @property
    def model(self) -> ModelProto:
        """
        Get the ModelProto instance from loader.

        :return ModelProto: Onnx ModelProto instance.
        """
        return self.model_wrapper.model

    def get_inputs(self) -> Dict[str, onnx.ValueInfoProto]:
        """
        Get the ONNX Model input tensors dict.

        :return Dict[str, onnx.ValueInfoProto]: Dict with input tensor name as
            key and input tensor as value.
        """
        return {inp.name: inp for inp in Utils.get_inputs(self.model)}

    def get_outputs(self) -> Dict[str, onnx.ValueInfoProto]:
        """
        Get the ONNX Model output tensors dict.

        :return Dict[str, onnx.ValueInfoProto]: Dict with output tensor name as
            key and output tensor as value.
        """
        return {inp.name: inp for inp in Utils.get_outputs(self.model)}

    def get_input_names(self) -> List[str]:
        """
        Get the ONNX Model input names.

        :returns:List[str]: list of input names.
        """
        return [inp.name for inp in Utils.get_inputs(self.model)]

    def get_output_names(self) -> List[str]:
        """
        Get the Onnx Model output names.

        :returns:List[str]: list of output names.
        """
        return [out.name for out in Utils.get_outputs(self.model)]

    def get_nodes(self, include_subgraphs=True) -> List[NodeProto]:
        """
        Get all the nodes from underlying onnx model.

        :param bool include_subgraphs: If True, will include nodes from subgraphs also, default: True
        :returns:Underlying onnx nodes.
        """
        return Utils.get_nodes(self.model, include_subgraphs=include_subgraphs)

    def get_input_info(self) -> Dict[Text, TensorInfo]:
        """
        Get input name to TensorInfo Mappings. e.g. shape, dtype, layout etc.

        :return Dict[Text, TensorInfo]: TensorInfo mapping for inputs.
        """
        return Utils.get_input_info(self.model)

    def get_output_info(self) -> Dict[Text, TensorInfo]:
        """
        Get output name to TensorInfo Mappings. e.g. shape, dtype, layout etc.

        :return Dict[Text, TensorInfo]: TensorInfo mapping for outputs.
        """
        return Utils.get_output_info(self.model)

    def native_checker(self, dry_run=None) -> bool:
        """
        This method will return the result of onnx model checker as well as evaluate the model.
        :return: Boolean indicating the success/failure of the Native Onnx checker
        """
        if not self.model_wrapper.weights_loaded:
            log_warning("Skipping Native checker in Lazy Weight Loading")
            return False

        success = True
        # Calling graph checker for sanity checking about the graph's node names,
        # initializer names etc.
        graph_check_status = Utils.graph_checker(self.model)
        if not graph_check_status:
            log_warning("Duplicate naming found in the graph.")

        try:
            # Checker needs weights to be present in the model.
            if self.model_wrapper.has_external_data:
                model_path = self.model_wrapper.get_model_path()
                onnx.checker.check_model(model_path)
            else:
                onnx.checker.check_model(self.model)
        except RuntimeError as e:
            # If get_model_path is not able to save the model on disk.
            log_warning(f"Onnx checker failed due to exception: {e}")
            return False
        except Exception as e:
            log_warning("The model is invalid: %s" % e)
            return False

        if dry_run:
            log_info(
                "Proceeding with model evaluation in dry run mode...................................\n"
            )
            model_evaluator.setup_dry_run(self.model, dry_run)
            log_info(
                "Exiting conversion process in dry run mode...................................\n"
            )
            sys.exit(0)

        return success

    def save_model(self, path: Text) -> None:
        """
        Save The ONNX model on to the disc.

        :param Text path: Path where the model is to be saved.
        """
        if not path.endswith(".onnx"):
            prepared_name = self.model_wrapper.model_name
            path = os.path.join(path, prepared_name)
        model_w_weights = self.model_wrapper.get_full_model()
        Utils.save_model(model_w_weights, path, restore_back=False)

    def summarize_model(self, stage_info: Dict = {}) -> FrameworkSummary:
        """
        Populates summary of the onnx model.

        :param Dict stage_info: Dict: Stages information of the onnx model.
        :return FrameworkSummary: Returns the framework summary object.
        """
        summary = FrameworkSummary()
        summary.total_parameters = Utils.get_model_params(self.model)
        model_proto = self.model

        summary.ops_counter = Utils.get_unique_ops(model_proto)
        summary.ir_version = model_proto.ir_version

        # opset_details = []
        # for opi in model_proto.opset_import:
        #     opset_details.append(str(opi.version) + " " + opi.domain)
        # summary.opset_version = ", ".join(opset_details)

        summary.producer_name = ""
        # TODO ADD QNN Version here.

        summary.model_name = self.model_wrapper.model_name

        inp_specs = self.get_input_info()
        out_specs = self.get_output_info()

        summary.inp_specs = {
            k: (v.shape, "input", v.dtype) for k, v in inp_specs.items()
        }
        summary.out_specs = {
            k: (v.shape, "output", v.dtype) for k, v in out_specs.items()
        }
        summary.stage_info = stage_info
        return super().summarize_model(summary)
