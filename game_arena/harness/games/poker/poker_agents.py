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

"""Poker specific agent class for submitting to Kaggle Game Arena Simulation Environments."""

from collections.abc import Mapping
import json
from typing import Any

from game_arena.harness import base_agents
from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness.games.poker import game as poker_game
from game_arena.harness.games.poker import hand_history_utils as hh_utils
from game_arena.harness.games.poker import parsers as poker_parsers
from game_arena.harness.games.poker import prompt_templates as poker_prompt_templates
from game_arena.harness.games.poker import rethink as poker_rethink


class RepeatedPokerRethinkAgent(base_agents.RethinkAgent):
  """Rethink agent for repeated poker."""

  def _get_prompt_substitutions(self) -> Mapping[str, Any]:
    """Returns the prompt substitutions for the current game state."""
    state_dict = json.loads(str(self.game_adapter.native_state))
    pyspiel_state = self.game_adapter.native_state
    past_hhs = []
    for i, acpc_hh in enumerate(pyspiel_state.acpc_hand_histories()):
      hh, _ = hh_utils.parse_acpc_line(
          acpc_hh,
          cfg=hh_utils.Config(
              seats=pyspiel_state.num_players(),
              # TODO(John Schultz): Extract these from the state.
              small_blind=1,
              big_blind=2,
              starting_stacks=[200, 200],
          ),
          policy=hh_utils.ButtonPolicy(),
          button_index=(i % 2) + 1,
      )
      readable_hh = hh_utils.render_pokersite(
          hand=hh, observer_id=None, sitename=""  # Shows all cards.
      )
      past_hhs.append(readable_hh)
    if len(past_hhs) != state_dict["hand_number"]:
      raise ValueError(
          f"Number of past hands {len(past_hhs)} does not match number of hands"
          f"in state (current hand={state_dict['hand_number']})."
      )
    past_hhs_str = "\n\n".join(past_hhs)

    # Current hand must be handled separately.
    players = [f"Player{i}" for i in range(pyspiel_state.num_players())]
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"]
    acpc_state_str = acpc_state_str.split("\n")[0]
    if not acpc_state_str.startswith("STATE:"):
      raise ValueError(
          f"Expected ACPC state to start with STATE:, got {acpc_state_str}"
      )
    # Pluribus style:
    acpc_state_str = acpc_state_str + "::" + "|".join(players)
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=pyspiel_state.num_players(),
            # TODO(John Schultz): Extract these from the state.
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=(state_dict["hand_number"] % 2) + 1,
        hand_id_override=str(state_dict["hand_number"]),
    )
    observer_id = f"Player{pyspiel_state.current_player()}"
    readable_state_str = hh_utils.render_pokersite(
        hand=hh, observer_id=observer_id, sitename=""
    )
    readable_state_str = (
        "You are"
        f" Player{pyspiel_state.current_player()}.\n\n{readable_state_str}"
    )
    readable_state_str = (
        f"You are Player{pyspiel_state.current_player()}.\n\n"
        + "Previously played hands this session:\n\n"
        + past_hhs_str
        + "\n\n"
        + "Current hand:\n\n"
        + readable_state_str
    )

    prompt_substitutions = {"readable_state_str": readable_state_str}
    return prompt_substitutions


def build_repeated_poker_rethink_agent(
    model: model_generation.Model,
    **unused_kwargs,
) -> RepeatedPokerRethinkAgent:
  """Builds a rethink agent with default settings for a given model."""
  game_adapter = poker_game.PokerGameAdapter()
  sampler = poker_rethink.RethinkSampler(
      model=model,
      strategy=poker_rethink.PokerRethinkStrategy.RETHINK_REPEATED_POKER,
      num_max_rethinks=1,
      # Note: We don't want to include "is" as an additional parsing tag because
      # it conflicts with "raise".
      move_parser=parsers.RuleBasedMoveParser(additional_tags=[]),
      legality_parser=poker_parsers.PokerSoftParser(),
      game_adapter=game_adapter,
      prompt_generator=prompt_generation.PromptGeneratorText(),
      rethink_template=None,
  )
  agent = RepeatedPokerRethinkAgent(
      sampler=sampler,
      prompt_template=poker_prompt_templates.REPEATED_POKER,
      game_adapter=game_adapter,
  )
  return agent
