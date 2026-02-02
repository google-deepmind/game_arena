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

"""Library for generating prompts using templates."""

from typing import Any, Generic, Mapping, Protocol, runtime_checkable
from game_arena.harness import base_game
from game_arena.harness import model_generation
from game_arena.harness import prompt_templates
from game_arena.harness import prompts


@runtime_checkable
class PromptGeneratorSupportsImageText(
    Generic[model_generation.ModelImageTextInputT], Protocol
):
  """Generator of prompts containing text and e.g. a board image."""

  def is_valid_prompt_template(self, prompt_template: str) -> bool:
    ...

  # Keep dependency on game state as we may use prompt generators with
  # renderers that work with PySpiel or other environment for rendering.
  def generate_prompt_with_image_text(
      self,
      prompt_template: str,
      game_adapter: base_game.BaseGameEnvAdapter,
      prompt_configuration: Mapping[str, Any] | None = None,
      **prompt_substitutions,
  ) -> model_generation.ModelImageTextInputT:
    ...


@runtime_checkable
class PromptGeneratorSupportsText(
    Generic[model_generation.ModelTextInputT], Protocol
):
  """Generator of prompts containing text only."""

  def generate_prompt_with_text_only(
      self,
      prompt_template: str,
      game_short_name: str,
      **prompt_substitutions,
  ) -> model_generation.ModelTextInputT:
    ...


class PromptGeneratorText(PromptGeneratorSupportsText):
  """Generator of prompts containing text only."""

  def generate_prompt_with_text_only(
      self,
      prompt_template: str,
      game_short_name: str,
      **prompt_substitutions,
  ) -> model_generation.ModelTextInput:
    prompt_substitutions["game_short_name"] = game_short_name
    return model_generation.ModelTextInput(
        prompt_text=prompt_template.format(**prompt_substitutions)
    )


class PromptGeneratorImageText(PromptGeneratorSupportsImageText):
  """Generator of prompts containing text and e.g. a board image."""

  def is_valid_prompt_template(self, prompt_template: str) -> bool:
    return prompt_template in (
        prompt_templates.WITH_BOARD_IMAGE,
        prompt_templates.WITH_BOARD_IMAGE_RETHINK_APPENDED,
    )

  def generate_prompt_with_image_text(
      self,
      prompt_template: str,
      game_adapter: base_game.BaseGameEnvAdapter,
      prompt_configuration: Mapping[str, Any] | None = None,
      **prompt_substitutions,
  ) -> model_generation.ModelImageTextInput:

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
    prompt_substitutions["additional_state_info_text"] = ""
    if not self.is_valid_prompt_template(prompt_template):
      raise ValueError(f"Unsupported prompt template: {prompt_template}")

    raise NotImplementedError(
        "generate_prompt_with_image_text is not supported for"
        f" {game_adapter.game_short_name}"
    )
