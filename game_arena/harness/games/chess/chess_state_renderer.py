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

"""Image generation for chess games."""

import random

from game_arena.harness import game_state_renderer
from game_arena.harness.games.chess import sprite_utils


class ChessImageGenerator(game_state_renderer.ImageGenerator):
  """Generates a Chess board image."""

  def __init__(self):
    super().__init__()
    self._colors = sprite_utils.COLORS
    self._piece_sets = sprite_utils.PIECE_SETS

  def generate_image(
      self,
      state_str: str,
      player_perspective: str = "white",
      color: str | None = None,
      piece_set: str | None = None,
      show_labels: bool = True,
  ) -> bytes:
    """Generates a Chess board image."""
    if player_perspective not in ["white", "black"]:
      raise ValueError(f"Invalid player perspective: {player_perspective}")
    color = color or random.choice(self._colors)
    piece_set = piece_set or random.choice(self._piece_sets)
    board_image = sprite_utils.fen_to_board(
        fen=state_str,
        color=color,
        piece_set=piece_set,
        player_perspective=player_perspective,
        show_labels=show_labels,
    )
    return board_image
