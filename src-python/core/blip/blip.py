# Reference: https://docs.openvino.ai/2024/notebooks/blip-visual-language-processing-with-output.html
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
import openvino as ov

# 1. Processor와 모델을 로드합니다.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

VISION_MODEL_OV = Path("./models/blip_vision_model.xml")
vision_model = model.vision_model
vision_model.eval()

TEXT_DECODER_OV = Path("./models/blip_text_decoder_with_past.xml")
text_decoder = model.text_decoder
text_decoder.eval()

core = ov.Core()
read_vision_model = core.read_model(VISION_MODEL_OV)
vision_input_layer = read_vision_model.input(0)
read_vision_model.reshape({ vision_input_layer.any_name: ov.PartialShape([1, 3, 384, 384]) })

read_text_decoder = core.read_model(TEXT_DECODER_OV)

# load models on device
ov_vision_model = core.compile_model(read_vision_model, "NPU")
ov_text_decoder_with_past = core.compile_model(read_text_decoder, "GPU")

from functools import partial
from blip_model import text_decoder_forward, OVBlipModel

text_decoder.forward = partial(text_decoder_forward, ov_text_decoder_with_past=ov_text_decoder_with_past)

ov_model = OVBlipModel(model.config, ov_vision_model, text_decoder)

def captioning(path):
    # 2. 이미지를 불러옵니다.
    image = Image.open(path)

    # 3. 이미지를 처리하고 캡션을 생성합니다.
    inputs = processor(image, return_tensors="pt")

    out = ov_model.generate_caption(inputs["pixel_values"], max_length=20)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption