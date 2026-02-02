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

"""Connect Four specific agent classes for Kaggle Game Arena."""

from collections.abc import Mapping
from typing import Any

from game_arena.harness import base_agents
from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import rethink
from game_arena.harness.games.connect_four import game as connect_four_game
from game_arena.harness.games.connect_four import parsers as connect_four_parsers
from game_arena.harness.games.connect_four import prompt_templates


class ConnectFourRethinkAgent(base_agents.RethinkAgent):
  """Rethink agent for Connect Four."""

  game_adapter: connect_four_game.ConnectFourGameAdapter

  def _get_prompt_substitutions(self) -> Mapping[str, Any]:
    """Returns the prompt substitutions for the current game state."""
    params = self.game_adapter.native_state.get_game().get_parameters()
    columns = int(params.get("columns", 7))
    return {
        "rows": int(params.get("rows", 6)),
        "columns": columns,
        "in_a_row": int(params.get("x_in_row", 4)),
        "max_column_index": columns - 1,
        "visual_board_state": self.game_adapter.native_state.to_string(),
        "player_name": self.game_adapter.game_notation.player_map[
            self.game_adapter.native_state.current_player()
        ],
    }


class ConnectFourRethinkSampler(rethink.RethinkSampler):
  """Rethink sampler for Connect Four that allows CONNECT_X_PROMPT."""

  def is_rethink_template(self, prompt_template: str | None) -> bool:
    """It's a rethink template if it's CONNECT_X_PROMPT or in default list."""
    return (
        prompt_template == prompt_templates.CONNECT_X_PROMPT
        or super().is_rethink_template(prompt_template)
    )


def build_default_rethink_agent(
    model: model_generation.Model,
    use_image: bool = False,
) -> ConnectFourRethinkAgent:
  """Builds a rethink agent with default settings for Connect Four."""
  # Multimodal Connect Four is not supported.
  if use_image:
    raise ValueError("Multimodal Connect Four is not supported.")

  prompt_generator = prompt_generation.PromptGeneratorText()
  prompt_template = prompt_templates.CONNECT_X_PROMPT

  c4_adapter = connect_four_game.ConnectFourGameAdapter()
  sampler = ConnectFourRethinkSampler(
      model=model,
      strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
      num_max_rethinks=1,
      move_parser=parsers.RuleBasedMoveParser(),
      legality_parser=connect_four_parsers.ConnectFourSoftParser(),
      prompt_generator=prompt_generator,
      rethink_template=prompt_templates.CONNECTX_RETHINK,
      game_adapter=c4_adapter,
  )
  agent = ConnectFourRethinkAgent(
      sampler=sampler,
      prompt_template=prompt_template,
      game_adapter=c4_adapter,
  )
  return agent
