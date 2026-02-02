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

"""Go game notation."""

import json
import random

from game_arena.harness import base_game

import pyspiel


def _player_string(player: int) -> str:
  if player < 0:
    return pyspiel.PlayerId(player).name.lower()
  elif player == 0:
    return "B"
  elif player == 1:
    return "W"
  else:
    raise ValueError(f"Invalid player: {player}")


def _grid_from_board_string(board_string: str) -> list[list[dict[str, str]]]:
  """Converts a board string to a dictionary."""
  lines = board_string.strip().splitlines()
  if len(lines) < 3:
    raise ValueError("Input string is too short to be a valid board.")
  # The last line contains the column labels (e.g., "ABC...")
  column_labels = lines[-1].strip()
  board_rows = lines[2:-1]
  board_size = len(column_labels)
  if len(board_rows) != board_size:
    raise ValueError(
        f"Board dimension mismatch: {len(column_labels)} columns "
        f"but {len(board_rows)} rows."
    )
  grid = []
  symbol_from_stone = {"+": ".", "X": "B", "O": "W"}
  for i, row_line in enumerate(board_rows):
    row_number = board_size - i
    try:
      board_content = row_line.split(maxsplit=1)[1]
    except Exception as e:  # pylint: disable=broad-exception-caught
      raise ValueError(f"Malformed board row: '{row_line}'") from e
    current_row_list = []
    for col_number, stone_char in enumerate(board_content):
      col_letter = column_labels[col_number]
      coordinate = f"{col_letter}{row_number}"
      point_dict = {coordinate: symbol_from_stone[stone_char]}
      current_row_list.append(point_dict)
    grid.append(current_row_list)
  return grid


def format_state(state: pyspiel.State) -> str:
  """Converts a Go state to a JSON string."""
  clone_state = state.get_game().new_initial_state()
  action_strs = []
  for action in state.history():
    action_strs.append(clone_state.action_to_string(action))
    clone_state.apply_action(action)
  prev_move = None if not action_strs else action_strs[-1]
  board_dict = {
      "board_size": state.get_game().get_parameters()["board_size"],
      "komi": state.get_game().get_parameters()["komi"],
      "current_player_to_move": _player_string(state.current_player()),
      "move_number": len(state.history()) + 1,
      "previous_move_a1": prev_move,
      "board_grid": _grid_from_board_string(str(state)),
  }
  return json.dumps(board_dict)


def convert_state(state_str: str, swap_probability: float = 0.5) -> str:
  """Adds whitespace to go ascii grid."""
  state_str = state_str.strip()
  grid = []
  swap_dot = random.random() < swap_probability
  swap_player = random.random() < swap_probability
  lines = state_str.split("\n")[1:]  # skip the header line
  num_lines = len(lines)

  for i, line in enumerate(lines):
    if not line:
      continue
    row_number, content = line.rsplit(" ", 1)
    if swap_dot:
      content = content.replace("+", ".")

    # Apply swap_player only if it's not the last row
    if swap_player and i < num_lines - 1:
      content = content.replace("X", "B").replace("O", "W")
    grid.append(f'{row_number} {" ".join(list(content))}')
  return "\n".join(grid)


class GoGameAdapter(base_game.OpenSpielGameAdapter):
  """Go game adapter."""

  game_short_name = "Go"
  game_notation = base_game.BaseGameNotation(
      state="""#  7 +++++++
#  6 +++++++
#  5 +++++++
#  4 +++++++
#  3 +++++++
#  2 +++++++
#  1 +++++++
#    ABCDEFG
""",
      action="B a1; B b1; B c1; B d1; B e1",
      player_map={0: "B", 1: "W"},
      move_notation="col,row using coordinates in the grid",
      state_notation="ASCII notation",
  )

  def get_readable_state(self, representation_type: str | int | None = None):
    return convert_state(str(self.native_state))

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
