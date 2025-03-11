# tokenized_prompt = self.tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True, return_tensors="pt", tokenize=False
# )


"""
phi_prompter.py

Defines a PromptBuilder for building Phi-2 Input/Output Prompts --> recommended pattern used by HF / Microsoft.
Also handles Phi special case BOS token additions.

Reference: https://huggingface.co/microsoft/phi-2#qa-format
"""

from typing import Optional

from robovlms.data.prompting.base_prompter import PromptBuilder

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_IMAGE_START_TOKEN = "<|vision_start|>"
DEFAULT_IMAGE_END_TOKEN = "<|vision_end|>"


class Qwen25CoTrainPromptBuilder(PromptBuilder):
    def __init__(
            self, model_family: str, system_prompt: Optional[str] = None, eos=None, bos=None
    ) -> None:
        super().__init__(model_family, system_prompt, eos, bos)

        # Note =>> Phi Tokenizer is an instance of `CodeGenTokenizer(Fast)`
        #      =>> By default, does *not* append <BOS> / <EOS> tokens --> we handle that here (IMPORTANT)!
        if self.bos is None:
            self.bos = DEFAULT_IM_START_TOKEN
        if self.eos is None:
            self.eos = DEFAULT_IM_END_TOKEN

        self.default_image_token = DEFAULT_IMAGE_TOKEN
        #
        # if self.bos is None and self.eos is None:
        #     self.bos, self.eos = "<|endoftext|>", "<|endoftext|>"

        # Get role-specific "wrap" functions
        #   =>> Note that placement of <bos>/<eos> were based on experiments generating from Phi-2 in Input/Output mode
        self.wrap_human = lambda msg: f"{DEFAULT_IM_START_TOKEN}user\n{msg}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        # self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}\n{self.eos}"
        self.wrap_gpt = lambda msg: f"{msg}{DEFAULT_IM_END_TOKEN}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        # message = message.replace("<image>", "").strip()

        if self.turn_count == 0:
            # check DEFAULT_IMAGE_TOKEN
            if DEFAULT_IMAGE_TOKEN not in message:
                message = f"{DEFAULT_IMAGE_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IMAGE_END_TOKEN}" + message
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        elif (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human's) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        return self.prompt
