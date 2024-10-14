import torch
import numpy as np
import openvino as ov
from typing import List, Dict
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def init_past_inputs(model_inputs: List):
    """
    Helper function for initialization of past inputs on first inference step
    Parameters:
      model_inputs (List): list of model inputs
    Returns:
      pkv (List[ov.Tensor]): list of filled past key values
    """
    pkv = []
    for input_tensor in model_inputs[4:]:
        partial_shape = input_tensor.partial_shape
        partial_shape[0] = 1
        partial_shape[2] = 0
        pkv.append(ov.Tensor(ov.Type.f32, partial_shape.get_shape()))
    return pkv


def postprocess_text_decoder_outputs(output: Dict):
    """
    Helper function for rearranging model outputs and wrapping to CausalLMOutputWithCrossAttentions
    Parameters:
      output (Dict): dictionary with model output
    Returns
      wrapped_outputs (CausalLMOutputWithCrossAttentions): outputs wrapped to CausalLMOutputWithCrossAttentions format
    """
    logits = torch.from_numpy(output[0])
    past_kv = list(output.values())[1:]
    return CausalLMOutputWithCrossAttentions(
        loss=None,
        logits=logits,
        past_key_values=past_kv,
        hidden_states=None,
        attentions=None,
        cross_attentions=None,
    )


def text_decoder_forward(
    ov_text_decoder_with_past: ov.CompiledModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: List[ov.Tensor],
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    **kwargs
):
    """
    Inference function for text_decoder in one generation step
    Parameters:
      input_ids (torch.Tensor): input token ids
      attention_mask (torch.Tensor): attention mask for input token ids
      past_key_values (List[ov.Tensor] list of cached decoder hidden states from previous step
      encoder_hidden_states (torch.Tensor): encoder (vision or text) hidden states
      encoder_attention_mask (torch.Tensor): attnetion mask for encoder hidden states
    Returns
      model outputs (CausalLMOutputWithCrossAttentions): model prediction wrapped to CausalLMOutputWithCrossAttentions class including predicted logits and hidden states for caching
    """
    inputs = [input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask]
    if past_key_values is None:
        inputs.extend(init_past_inputs(ov_text_decoder_with_past.inputs))
    else:
        inputs.extend(past_key_values)
    outputs = ov_text_decoder_with_past(inputs)
    return postprocess_text_decoder_outputs(outputs)


class OVBlipModel:
    """
    Model class for inference BLIP model with OpenVINO
    """

    def __init__(
        self,
        config,
        vision_model,
        text_decoder,
    ):
        """
        Initialization class parameters
        """
        self.vision_model = vision_model
        self.vision_model_out = vision_model.output(0)
        self.text_decoder = text_decoder
        self.config = config
        self.decoder_input_ids = config.text_config.bos_token_id

    def generate_caption(self, pixel_values: torch.Tensor, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, **generate_kwargs):
        """
        Image Captioning prediction
        Parameters:
          pixel_values (torch.Tensor): preprocessed image pixel values
          input_ids (torch.Tensor, *optional*, None): pregenerated caption token ids after tokenization, if provided caption generation continue provided text
          attention_mask (torch.Tensor): attention mask for caption tokens, used only if input_ids provided
        Retruns:
          generation output (torch.Tensor): tensor which represents sequence of generated caption token ids
        """
        batch_size = pixel_values.shape[0]

        image_embeds = self.vision_model(pixel_values.detach().numpy())[self.vision_model_out]

        image_attention_mask = torch.ones(image_embeds.shape[:-1], dtype=torch.long)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = torch.LongTensor(
                [
                    [
                        self.config.text_config.bos_token_id,
                        self.config.text_config.eos_token_id,
                    ]
                ]
            ).repeat(batch_size, 1)
        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=torch.from_numpy(image_embeds),
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs