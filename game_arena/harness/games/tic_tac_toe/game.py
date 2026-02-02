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

"""Tic Tac Toe game notation."""

from game_arena.harness import base_game


class TicTacToeGameAdapter(base_game.OpenSpielGameAdapter):
  """Tic Tac Toe game adapter."""

  game_short_name = "tic_tac_toe"
  game_notation = game_notation = base_game.BaseGameNotation(
      state="""x..
...
...
""",
      action="o(0,1); o(2,0); x(2,1). The game begins with player x.",
      player_map={0: "x", 1: "o"},
      move_notation="zero-based (row,col)",
      state_notation="ASCII notation",
  )
