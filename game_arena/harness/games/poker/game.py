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

"""Poker game notation."""

import ast
import json
import re
from typing import Any

from game_arena.harness import base_game

import pyspiel


class PokerGameAdapter(base_game.OpenSpielGameAdapter):
  """Poker game adapter."""

  game_short_name = "Poker"
  game_notation = base_game.BaseGameNotation(
      state="""# BettingAbstraction: FULLGAME
  # P0 Cards: 5s
  # BoardCards
  # Node type?: Player node for player 0
  # ]
  # Round: 0
  # Spent: [P0: 100  P1: 100  ]
  #
  # Action Sequence: dd
  """,
      action="player=0 move=Call; player=0 move=Bet200; player=0 move=Bet201;",
      player_map={x: str(x) for x in range(9)},
      move_notation=(
          "Call, Fold or BetX where X is the number of chips you are betting"
      ),
      state_notation="text",
  )

  def get_readable_state(self, representation_type: str | int | None = None):

    lines = str(self.native_state).split("\n")
    new_lines = []
    for line in lines:
      if line.startswith("ACPC State:"):
        continue
      elif (
          line.startswith("P0 Cards:")
          and self.native_state.current_player() == 1
      ):
        continue
      elif (
          line.startswith("P1 Cards:")
          and self.native_state.current_player() == 0
      ):
        continue
      else:
        new_lines.append(line)
    return "\n".join(new_lines)

  def get_action_string_history(self):
    tmp_state = self.native_state.get_game().new_initial_state()
    move_list = []
    for action in self.native_state.history():
      action_str = tmp_state.action_to_string(
          tmp_state.current_player(), action
      )
      move_list.append(action_str)
      tmp_state.apply_action(action)

    # Use 'None' for empty history, matching C++ handler's initial state output.
    move_history = ", ".join(move_list) if move_list else "None"
    return move_history


_STREET_NAMES = {
    0: "pre-flop",
    1: "flop",
    2: "turn",
    3: "river",
}


# TODO(John Schultz): Add observation JSON to pokerkit directly.
def _convert_pokerkit_obs_to_json(observation_string: str) -> str:
  """Converts a formatted multi-line observation string into a JSON string.

  The function parses key-value pairs, converts keys to lowercase snake_case,
  and automatically detects the data type of the values.

  Args:
      observation_string: The input string in the specified obs format.

  Returns:
      A JSON formatted string with an indent of 2 for readability.

  Raises:
      ValueError: If the input string format is not as expected.
  """

  def to_snake_case(text: str) -> str:
    """Converts a string to lowercase snake_case."""
    # Handle special cases like "(s)" at the end of a word
    text = text.replace("(s)", "s")
    # Remove characters that are not letters, numbers, or spaces
    text = re.sub(r"[()']", "", text)
    # Convert to lowercase
    text = text.lower()
    # Replace spaces, hyphens, and slashes with a single underscore
    text = re.sub(r"[ /-]+", "_", text)
    return text.strip()

  def parse_value(value_str: str) -> Any:
    """Parses a string value into its correct Python type."""
    value_str = value_str.strip()
    try:
      # ast.literal_eval safely evaluates a string containing a
      # Python literal (e.g., string, number, tuple, list, bool, None)
      return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
      # If it's not a valid literal (e.g., a plain word),
      # return it as a string.
      return value_str

  # 2. Split the content into lines
  lines = observation_string.strip().split("\n")

  # 3. Process each line to build a dictionary
  data_dict = {}
  for line in lines:
    # Remove leading "||" characters and extra whitespace
    cleaned_line = line.lstrip("|").strip()
    if not cleaned_line:
      continue

    # Split the line into a key and a value at the first colon
    if ":" in cleaned_line:
      key_str, value_str = cleaned_line.split(":", 1)

      # Convert key to snake_case and parse the value
      snake_key = to_snake_case(key_str)
      parsed_value = parse_value(value_str)

      data_dict[snake_key] = parsed_value

  # 4. Convert the dictionary to a formatted JSON string
  return json.dumps(data_dict, indent=2)


# pylint: disable=protected-access
def _get_readable_pokerkit_observation(state: pyspiel.State) -> str:
  """Formats a pokerkit observation string."""
  if "pokerkit" not in str(state.get_game()):
    raise ValueError("This function should only be used for pokerkit games.")
  if state.num_players() != 2:
    raise ValueError("Currently only two player poker is supported.")
  observation_dict = json.loads(
      _convert_pokerkit_obs_to_json(state.observation_string())
  )
  del observation_dict["current_street_discard_draw_performed"]
  del observation_dict["bets"]
  observation_dict["your_seat_id(0-indexed)"] = state._player_to_seat[
      observation_dict.pop("player")
  ]
  observation_dict["current_street"] = _STREET_NAMES[
      observation_dict["current_street"]
  ]
  observation_dict["next_player_to_act"] = (
      f"seat{state._player_to_seat[observation_dict['next_player_to_act']]}"
  )
  game_params = state.get_game().get_parameters()
  observation_dict["blinds"] = game_params["pokerkit_game_params"]["blinds"]

  replay_state = state.get_game().new_initial_state()
  betting_history = {street: [] for street in _STREET_NAMES.values()}
  first_non_chance_node_found = False
  for action in state.history():
    if replay_state._hand_number != state._hand_number:
      replay_state.apply_action(action)
      continue
    if not replay_state.is_chance_node():
      if not first_non_chance_node_found:
        first_non_chance_node_found = True
        dealer_seat = replay_state._player_to_seat[
            replay_state.current_player()
        ]
        observation_dict["dealer(0-indexed)"] = f"seat{dealer_seat}"
        betting_history[_STREET_NAMES[0]].extend([
            (
                f"seat{dealer_seat}:post small blind"
                f" {min(observation_dict['blinds'])}"
            ),
            (
                f"seat{(dealer_seat+1)%state.num_players()}:post big blind"
                f" {max(observation_dict['blinds'])}"
            ),
        ])
      action_str = replay_state.action_to_string(action)
      # Remove "Player N:" prefix because model should only see seat ID. Player
      # ID remains fixed across hands but seat ID can change. Seeing both
      # confuses the model.
      action_str = action_str.split(":")[-1]
      bet_string = f"seat{replay_state._player_to_seat[replay_state.current_player()]}:{action_str}"
      replay_obs_dict = json.loads(
          _convert_pokerkit_obs_to_json(replay_state.observation_string())
      )
      street_name = _STREET_NAMES[replay_obs_dict["current_street"]]
      betting_history[street_name].append(bet_string)
    replay_state.apply_action(action)
  if not first_non_chance_node_found:
    observation_dict["dealer(0-indexed)"] = (
        f"seat{replay_state._player_to_seat[replay_state.current_player()]}"
    )
  observation_dict["betting_history"] = betting_history
  observation_dict["hand_number"] = state._hand_number
  return json.dumps(observation_dict, indent=2)


# pylint: enable=protected-access


class PokerkitGameAdapter(PokerGameAdapter):
  """Pokerkit game adapter."""

  game_short_name = "pokerkit"

  def get_readable_state(self, representation_type: str | int | None = None):
    return _get_readable_pokerkit_observation(self.native_state)

  def convert_pokerkit_obs_to_json(self, observation_string: str) -> str:
    return _convert_pokerkit_obs_to_json(observation_string)
