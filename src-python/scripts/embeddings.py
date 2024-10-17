from pathlib import Path

import numpy as np
import shutil
from openvino.runtime import Core, Type

import openvino as ov
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops

class ReplaceTensor(MatcherPass):
    def __init__(self, packed_layername_tensor_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Multiply")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            for y in packed_layername_tensor_dict_list:
                root_name = root.get_friendly_name()
                if root_name.find(y["name"]) != -1:
                    max_fp16 = np.array([[[[-np.finfo(np.float16).max]]]]).astype(np.float32)
                    new_tenser = ops.constant(max_fp16, Type.f32, name="Constant_4431")

                    root.set_arguments([root.input_value(0).node, new_tenser])
                    packed_layername_tensor_dict_list.remove(y)

            return True

        self.register_matcher(Matcher(param, "ReplaceTensor"), callback)

def optimize_bge_embedding(model_path, output_model_path):
    """
    optimize_bge_embedding used to optimize BGE model for NPU device

    Arguments:
        model_path {str} -- original BGE IR model path
        output_model_path {str} -- Converted BGE IR model path
    """
    packed_layername_tensor_dict_list = [{"name": "aten::mul/Multiply"}]
    core = Core()
    ov_model = core.read_model(model_path)
    manager = Manager()
    manager.register_pass(ReplaceTensor(packed_layername_tensor_dict_list))
    manager.run_passes(ov_model)
    ov.save_model(ov_model, output_model_path, compress_to_fp16=False)

# optimum-cli export openvino --model BAAI/bge-m3 --task feature-extraction <output-dir>
# 이 명령어로 BGE 모델을 OpenVINO IR로 변환해두어야 함 
device = "NPU"
USING_NPU = device == "NPU"
embedding_model_id = "BAAI/bge-m3"
embedding_model_dir = "D:\\Intel\\ov_bge-m3"
npu_embedding_dir = embedding_model_dir + "-npu"
npu_embedding_path = Path(npu_embedding_dir) / "openvino_model.xml"

if USING_NPU and not Path(npu_embedding_path).exists():
    shutil.copytree(embedding_model_dir, npu_embedding_dir)
    optimize_bge_embedding(Path(embedding_model_dir) / "openvino_model.xml", npu_embedding_path)

embedding_model_name = npu_embedding_dir if USING_NPU else embedding_model_id
batch_size = 1 if USING_NPU else 4
embedding_model_kwargs = {"device": device, "compile": False}
encode_kwargs = {
    "mean_pooling": False,
    "normalize_embeddings": True,
    "batch_size": batch_size,
}