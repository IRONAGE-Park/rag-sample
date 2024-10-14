import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from pathlib import Path
import openvino as ov

# 1. Processor와 모델을 로드합니다.
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 2. 이미지를 불러옵니다.
img_url = 'https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png'
image = Image.open(requests.get(img_url, stream=True).raw)

# 3. 이미지를 처리하고 캡션을 생성합니다.
inputs = processor(image, return_tensors="pt")

VISION_MODEL_OV = Path("./python/blip/models/blip_vision_model.xml")
vision_model = model.vision_model
vision_model.eval()

# check that model works and save it outputs for reusage as text encoder input
with torch.no_grad():
    vision_outputs = vision_model(inputs["pixel_values"])

# if openvino model does not exist, convert it to IR
if not VISION_MODEL_OV.exists():
    # export pytorch model to ov.Model
    with torch.no_grad():
        ov_vision_model = ov.convert_model(vision_model, example_input=inputs["pixel_values"])
    # save model on disk for next usages
    ov.save_model(ov_vision_model, VISION_MODEL_OV)
    print(f"Vision model successfuly converted and saved to {VISION_MODEL_OV}")
else:
    print(f"Vision model will be loaded from {VISION_MODEL_OV}")

text_decoder = model.text_decoder
text_decoder.eval()

TEXT_DECODER_OV = Path("./python/blip/models/blip_text_decoder_with_past.xml")

# prepare example inputs
input_ids = torch.tensor([[30522]])  # begin of sequence token id
attention_mask = torch.tensor([[1]])  # attention mask for input_ids
encoder_hidden_states = torch.rand((1, 10, 768))  # encoder last hidden state from text_encoder
encoder_attention_mask = torch.ones((1, 10), dtype=torch.long)  # attention mask for encoder hidden states

input_dict = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "encoder_hidden_states": encoder_hidden_states,
    "encoder_attention_mask": encoder_attention_mask,
}
text_decoder_outs = text_decoder(**input_dict)
# extend input dictionary with hidden states from previous step
input_dict["past_key_values"] = text_decoder_outs["past_key_values"]

text_decoder.config.torchscript = True
if not TEXT_DECODER_OV.exists():
    # export PyTorch model
    with torch.no_grad():
        ov_text_decoder = ov.convert_model(text_decoder, example_input=input_dict)
    # save model on disk for next usages
    ov.save_model(ov_text_decoder, TEXT_DECODER_OV)
    print(f"Text decoder successfuly converted and saved to {TEXT_DECODER_OV}")
else:
    print(f"Text decoder will be loaded from {TEXT_DECODER_OV}")

core = ov.Core()

print(core.available_devices)
device = "NPU"

read_vision_model = core.read_model(VISION_MODEL_OV)
vision_input_layer = read_vision_model.input(0)
vision_output_layer = read_vision_model.output(0)
read_vision_model.reshape({ vision_input_layer.any_name: ov.PartialShape([1, 3, 384, 384]) })

read_text_decoder = core.read_model(TEXT_DECODER_OV)
batch_size = 1
seq_len = 40
hidden_size = 768
num_heads = 12
head_size = 64
num_layers = 12
new_shapes = {
    "input_ids": ov.PartialShape([batch_size, seq_len]),
    "attention_mask": ov.PartialShape([batch_size, seq_len]),
    "encoder_hidden_states": ov.PartialShape([batch_size, seq_len, hidden_size]),
    "encoder_attention_mask": ov.PartialShape([batch_size, seq_len]),
    # "past_key_values": ov.PartialShape([2, batch_size, num_heads, seq_len, head_size]),
}
decoder_input_layer = read_text_decoder.input(0)
decoder_output_layer = read_text_decoder.output(0)
read_text_decoder.reshape(new_shapes)

print("Text Decoder Model Inputs:")
for input in read_text_decoder.inputs:
    print(f"Name: {input.any_name}, Shape: {input.partial_shape}, Type: {input.element_type}")

# load models on device
ov_vision_model = core.compile_model(read_vision_model, device)
ov_text_decoder_with_past = core.compile_model(read_text_decoder, "GPU")

from blip_model import OVBlipModel

ov_model = OVBlipModel(model.config, ov_vision_model, text_decoder)
out = ov_model.generate_caption(inputs["pixel_values"], max_length=20)
caption = processor.decode(out[0], skip_special_tokens=True)

print(f"Generated Caption: {caption}")