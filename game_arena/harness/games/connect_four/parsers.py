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

"""Connect Four specific parsers."""

import re
from typing import Sequence
from game_arena.harness import parsers


def connect_four_soft_parser(
    selected_action: str, spiel_legal_moves: Sequence[str] | None
) -> str | None:
  """Connect Four parser that matches against legal moves.

  Connect Four moves are column numbers 0-6. The legal moves from OpenSpiel
  include player prefixes like 'x0', 'x1', 'o3', etc. This parser extracts
  a digit from the model output and finds the matching legal move.

  Args:
    selected_action: The action string selected by the model.
    spiel_legal_moves: A sequence of legal move strings from OpenSpiel.

  Returns:
    The legal move string from `spiel_legal_moves` that matches the extracted
    column, or None if no match is found or inputs are invalid.
  """
  if selected_action is None:
    return None

  selected_action = selected_action.strip()
  if not selected_action:
    return None

  if spiel_legal_moves is None:
    return None

  match = re.search(r"(\d+)", selected_action)
  if match:
    column = match.group(1)
    # Legal moves are prefixed with player (e.g., 'x3', 'o3')
    # Find the legal move that ends with the column number
    for legal_move in spiel_legal_moves:
      if legal_move.endswith(column):
        return legal_move

  return None


class ConnectFourSoftParser(parsers.SoftMoveParser):
  """Connect Four soft parser."""

  def _parse_selected_action(
      self, parser_input: parsers.TextParserInput
  ) -> str | None:
    return connect_four_soft_parser(
        selected_action=parser_input.text,
        spiel_legal_moves=parser_input.legal_moves,
    )
