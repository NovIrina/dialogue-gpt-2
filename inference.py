"""A module for the inference of the GPT-2 model."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_current_torch_device() -> str:
    """
    Get the current torch device.

    :return: The current torch device.
    """
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_model(model_checkpoint: str | Path, torch_dtype: torch.dtype = torch.float32,
               lora_checkpoint: str | Path = None) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Load a GPT-2 model from a checkpoint, with optional LoRa adaptation.

    :param model_checkpoint: The path to the model checkpoint.
    :param torch_dtype: The precision in which to load the model.
    :param lora_checkpoint: The path to the LoRa checkpoint, if any.
    :return: The model and the tokenizer.
    """
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint, torch_dtype=torch_dtype)
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)

    tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
    tokenizer.add_tokens(['<|USER|>', '<|ASSISTANT|>'])

    model.resize_token_embeddings(len(tokenizer))

    if lora_checkpoint:
        model = PeftModel.from_pretrained(model, lora_checkpoint)

    model.to(get_current_torch_device())

    return model, tokenizer


def run_inference(model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, prompt: str) -> str:
    """
    Run inference.

    :param model: The model.
    :param tokenizer: The tokenizer.
    :param prompt: The input prompt.
    :return: The generated text.
    """
    full_prompt = '<|USER|>: ' + prompt + '<|ASSISTANT|>: '

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(model.device)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=100
    )

    prediction = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return prediction[len(full_prompt) + 2:]


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Interactive GPT-2 inference.")
    parser.add_argument("model_checkpoint", type=str, help="The path to the model checkpoint.")
    parser.add_argument("-l", "--lora-checkpoint", type=str,
                        help="The path to the LoRa checkpoint, if using LoRa.")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model_checkpoint, lora_checkpoint=args.lora_checkpoint)

    while True:
        prompt = input("Enter a prompt: ")
        if not prompt:
            print("Exiting...")
            break

        output_text = run_inference(model, tokenizer, prompt)
        print(output_text, end="\n\n")


if __name__ == "__main__":
    main()
