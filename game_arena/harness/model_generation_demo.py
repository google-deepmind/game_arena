# Copyright 2025 The game_arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo of model generation implementations."""

import concurrent.futures
import dataclasses
import datetime
import importlib.resources
import os
import pathlib
import time
import traceback
from typing import Any
from absl import app
from absl import flags
from game_arena.harness import model_generation
from game_arena.harness import model_generation_http
from game_arena.harness import model_generation_sdk
from game_arena.harness import model_registry

# Default values will run everything.

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "model_names",
    [
        # keep-sorted start
        "anthropic",
        "anthropic_streaming",
        "anthropic_sync",
        "deepseek",
        "gemini",
        "kimi",
        "openai_streaming",
        "openai_sync",
        "qwen3_thinking_parallel",
        "xai_grok-3-mini-fast_streaming",
        "xai_grok-3-mini-fast_sync",
        "xai_grok-4_streaming",
        # keep-sorted end

    ],
    "Comma-separated list of model names to use (e.g.,"
    " 'gemini,xai_grok-4_streaming'). Defaults to all available models.",
)
flags.DEFINE_integer(
    "num_calls", 1, "Number of times to call the specified model."
)
flags.DEFINE_list(
    "use_modality",
    ["text", "image"],
    "Comma-separated list of modalities to use (e.g., 'text,image').",
)


@dataclasses.dataclass
class FutureInfo:
  name: str
  start_time: float
  input_data: Any
  model_name: str


def write_to_file(
    output_folder: pathlib.Path,
    model_input: Any,
    response: model_generation.GenerateReturn | str,
    run_name: str,
    model_name: str,
    response_time: datetime.timedelta,
):
  """Writes model input and output to a file, relative to the current directory."""

  output_folder.mkdir(parents=True, exist_ok=True)
  filename = (
      f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{run_name}_output.txt"
  )
  filepath = output_folder / filename
  with open(filepath, "w") as f:
    f.write(f"--- Model: {model_name} ({run_name}) ---\n")
    f.write("--- Model Input ---\n")
    f.write(f"prompt_text: {getattr(model_input, 'prompt_text', 'N/A')}\n")
    f.write(
        "system_instruction:"
        f" {getattr(model_input, 'system_instruction', 'N/A')}\n"
    )
    if hasattr(model_input, "prompt_image_bytes"):
      f.write("prompt_image_bytes: <present>\n")
      f.write(
          "prompt_image_mime_type:"
          f" {getattr(model_input, 'prompt_image_mime_type', 'N/A')}\n"
      )
    f.write("\n--- Model Output ---\n")
    f.write(f"response_time: {response_time.total_seconds():.0f}s\n")

    if isinstance(response, str):
      f.write(response)
    else:
      for field, value in dataclasses.asdict(response).items():
        f.write(f"{field}: {value}\n")

  print(f"📝 Wrote output to {filepath}")


def main(_) -> None:
  selected_modalities = FLAGS.use_modality

  if "text" in selected_modalities:
    text_input = model_generation.ModelTextInput(
        prompt_text="""Let's play chess. The current game state in Forsyth-Edwards Notation (FEN) notation is:
rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
The moves played so far are:
1. e4 c5 2.
You are playing as player White.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in standard algebraic notation (SAN).""",
        system_instruction=None,
    )
    print("Text-only prompt: ", text_input)
  else:
    text_input = None

  if "image" in selected_modalities:
    image_path = importlib.resources.files("game_arena").joinpath(
        "harness", "chess_puzzle_image.png"
    )
    with open(str(image_path), "rb") as image_file:
      image_bytes = image_file.read()
    image_input = model_generation.ModelImageTextInput(
        prompt_text=(
            "This is a chess puzzle. What is the best next move for white?"
        ),
        prompt_image_bytes=image_bytes,
        prompt_image_mime_type="image/png",
        prompt_text_preceding_image="We are playing chess.",
        prompt_text_following_image="To play: ",
        system_instruction=None,
    )
    print("Image-text prompt: ", image_input.prompt_text)
  else:
    image_input = None

  models = {
      # keep-sorted start
      "anthropic": model_generation_sdk.AnthropicModel(
          model_name="claude-sonnet-4-20250514",
      ),
      "anthropic_streaming": model_generation_sdk.AnthropicModel(
          model_name="claude-sonnet-4-20250514",
          api_options={"stream": True},
          model_options={
              "max_tokens": 64000,
              "thinking": {"type": "enabled", "budget_tokens": 32000},
          },
      ),
      "anthropic_sync": model_generation_sdk.AnthropicModel(
          model_name="claude-sonnet-4-20250514",
          api_options={"stream": False},
          model_options={
              "thinking": {"type": "enabled", "budget_tokens": 8192}
          },
      ),
      "deepseek": model_generation_http.TogetherAIModel(
          model_name="deepseek-ai/DeepSeek-R1-0528-tput",
      ),
      "gemini": model_generation_sdk.AIStudioModel(
          model_name="models/gemini-2.5-flash",
          api_options={"include_thoughts": True},
      ),
      "kimi": model_generation_http.TogetherAIModel(
          model_name="moonshotai/Kimi-K2-Instruct",
      ),
      "openai_streaming": model_generation_sdk.OpenAIChatCompletionsModel(
          model_name="o3",
          api_options={
              "timeout": 60 * 10,  # 10 minutes
              "stream": True,
          },
      ),
      "openai_sync": model_generation_sdk.OpenAIChatCompletionsModel(
          model_name="o3",
          api_options={"timeout": 60 * 10},  # 10 minutes
      ),
      "qwen3_thinking_parallel": model_generation_http.TogetherAIModel(
          model_name="Qwen/Qwen3-235B-A22B-Thinking-2507",
          api_options={"parallel_attempts": 3, "timeout": 45},
      ),
      "xai_grok-3-mini-fast_streaming": model_generation_http.XAIModel(
          model_name="grok-3-mini-fast",
          api_options={"stream": True},
      ),
      "xai_grok-3-mini-fast_sync": model_generation_http.XAIModel(
          model_name="grok-3-mini-fast",
          api_options={"stream": False},
      ),
      "xai_grok-4_streaming": model_generation_http.XAIModel(
          model_name="grok-4",
          api_options={"stream": True},
      ),
      # keep-sorted end
  }

  overall_start_time = time.monotonic()

  with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_info: dict[
        concurrent.futures.Future[model_generation.GenerateReturn], FutureInfo
    ] = {}

    def submit_and_track(
        func,
        *args,
        name_suffix,
        input_data_for_file_write,
        model_name,
    ):
      future = executor.submit(func, *args)
      full_name = f"{name_suffix}"
      future_to_info[future] = FutureInfo(
          name=full_name,
          start_time=time.monotonic(),
          input_data=input_data_for_file_write,
          model_name=model_name,
      )
      return future

    # Determine which model and input to use based on flags
    selected_model_names = FLAGS.model_names
    num_calls = FLAGS.num_calls

    if not selected_model_names:
      raise ValueError("Flag --model_names cannot be empty.")
    if not selected_modalities:
      raise ValueError("Flag --use_modality cannot be empty.")

    for selected_model_name in selected_model_names:
      if selected_model_name not in models:
        print(f"Error: Model '{selected_model_name}' not found. Skipping.")
        continue

      model = models[selected_model_name]

      run_text = "text" in selected_modalities
      run_image = "image" in selected_modalities

      if run_text:
        assert text_input is not None
        print(
            f"Submitting {num_calls} text input tasks for {selected_model_name}"
        )
        for i in range(num_calls):
          submit_and_track(
              model.generate_with_text_input,
              text_input,
              name_suffix=f"{selected_model_name}_text_{i}",
              input_data_for_file_write=text_input,
              model_name=model.model_name,
          )

      if run_image:
        assert image_input is not None
        if not isinstance(model, model_generation.MultimodalModel):
          print(
              f"Error: Model '{selected_model_name}' does not support image"
              " input. Skipping image tasks."
          )
        else:
          print(
              f"Submitting {num_calls} image input tasks for"
              f" {selected_model_name}"
          )
          for i in range(num_calls):
            submit_and_track(
                model.generate_with_image_text_input,
                image_input,
                name_suffix=f"{selected_model_name}_image_{i}",
                input_data_for_file_write=image_input,
                model_name=model.model_name,
            )

    pending_futures = set(future_to_info.keys())
    last_status_time = time.monotonic()

    while pending_futures:
      done, pending_futures = concurrent.futures.wait(
          pending_futures, timeout=1
      )

      for future in done:
        info = future_to_info[future]
        name = info.name
        input_data = info.input_data
        model_name = info.model_name
        elapsed_future_time = datetime.timedelta(
            seconds=time.monotonic() - info.start_time
        )
        try:
          response = future.result()
          write_to_file(
              pathlib.Path(__file__).resolve().parent / "model_outputs",
              input_data,
              response,
              name,
              model_name,
              elapsed_future_time,
          )

        except Exception as exc:  # pylint: disable=broad-except
          print(
              f"{name} generated an exception (time:"
              f" {elapsed_future_time.total_seconds():.0f}s) Exception: {exc}\n"
              f" Traceback: {traceback.format_exc()}"
          )
          write_to_file(
              pathlib.Path(__file__).resolve().parent / "model_outputs_failed",
              input_data,
              f"Exception: {exc}",
              name,
              model_name,
              elapsed_future_time,
          )

      if time.monotonic() - last_status_time > 60:
        if pending_futures:
          pending_names = sorted(
              [future_to_info[f].name for f in pending_futures]
          )
          elapsed_total_time = time.monotonic() - overall_start_time
          print(
              "\n[Still"
              f" waiting on: {', '.join(pending_names)}] (Total elapsed:"
              f" {elapsed_total_time:.0f}s)\n"
          )
          last_status_time = time.monotonic()

  total_demo_time = time.monotonic() - overall_start_time
  print(f"\nAll models finished. Total demo time: {total_demo_time:.2f}s\n")


if __name__ == "__main__":
  app.run(main)
