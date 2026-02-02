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

"""Connect X prompt generation."""

from . import prompt_templates


def generate_prompt(
    rows: int,
    columns: int,
    in_a_row: int,
    visual_board_state: str,
    player_name: str,
) -> str:
  """Generates a prompt for the Connect X game.

  Args:
    rows: The number of rows in the board.
    columns: The number of columns in the board.
    in_a_row: The number of pieces in a row to win.
    visual_board_state: A string representing the visual state of the board.
    player_name: The current player's name (e.g., "x" or "o").

  Returns:
    A formatted prompt string.
  """
  return prompt_templates.CONNECT_X_PROMPT.format(
      rows=rows,
      columns=columns,
      in_a_row=in_a_row,
      visual_board_state=visual_board_state,
      player_name=player_name,
      max_column_index=columns - 1,
      rethink_prompt="",
  )
