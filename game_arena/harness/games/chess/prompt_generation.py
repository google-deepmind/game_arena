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

"""Chess specific multimodal prompt generation."""

from collections.abc import Mapping
from typing import Any
from game_arena.harness import base_game
from game_arena.harness import model_generation
from game_arena.harness import prompt_generation
from game_arena.harness import prompt_templates
from game_arena.harness import prompts
from game_arena.harness.games.chess import chess_state_renderer


class PromptGeneratorImageText(prompt_generation.PromptGeneratorImageText):
  """Chess specific multimodal prompt generator."""

  def generate_prompt_with_image_text(
      self,
      prompt_template: str,
      game_adapter: base_game.BaseGameEnvAdapter,
      prompt_configuration: Mapping[str, Any] | None = None,
      **prompt_substitutions,
  ) -> model_generation.ModelImageTextInput:
    if prompt_configuration is None:
      prompt_configuration = {}

    # If None, the piece set or color are randomly chosen:
    piece_set = prompt_configuration.get("piece_set", None)
    color = prompt_configuration.get("color", None)

    if prompt_template not in prompts.IMAGE_TEXT_PROMPTS:
      raise ValueError(
          "generate_prompt_with_image_text should only be called for multimodal"
          " prompt templates."
      )

    if game_adapter.native_state is None:
      raise ValueError("Environment adapter state is None.")

    prompt_substitutions["game_short_name"] = game_adapter.game_short_name
    prompt_substitutions["readable_state_str"] = (
        game_adapter.get_readable_state()
    )
    debug_string = game_adapter.native_state.debug_string()
    # Extract the part of the debug string starting with "To Play"
    start_index = debug_string.index("To play: ")
    newlines = []
    for line in debug_string[start_index:].splitlines():
      if "rook" in line:
        continue
      if "(king-side)" in line or "(queen-side)" in line:
        newlines.append(line.replace("1", "true").replace("0", "false"))
      else:
        newlines.append(line)
    prompt_substitutions["additional_state_info_text"] = "\n".join(newlines)
    if not self.is_valid_prompt_template(prompt_template):
      raise ValueError(f"Unsupported prompt template: {prompt_template}")

    generator = chess_state_renderer.ChessImageGenerator()
    prefix = prompt_templates.NO_LEGAL_ACTIONS_PREFIX.format(
        **prompt_substitutions
    )
    if prompt_template == "WITH_BOARD_IMAGE_RETHINK_APPENDED":
      suffix = prompt_templates.NO_LEGAL_ACTIONS_SUFFIX_RETHINK_APPENDED.format(
          **prompt_substitutions
      )
    else:
      suffix = prompt_templates.NO_LEGAL_ACTIONS_SUFFIX.format(
          **prompt_substitutions
      )
    image_bytes = generator.generate_image(
        game_adapter.get_readable_state(), color=color, piece_set=piece_set
    )
    return model_generation.ModelImageTextInput(
        prompt_text="",
        prompt_image_bytes=image_bytes,
        prompt_image_mime_type="image/png",
        prompt_text_preceding_image=prefix,
        prompt_text_following_image=suffix,
    )
