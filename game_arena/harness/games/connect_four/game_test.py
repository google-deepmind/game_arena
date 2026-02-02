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

"""Tests for Connect Four game."""

from game_arena.harness.games.connect_four import game as connect_four_game
from absl.testing import absltest
import pyspiel


class ConnectFourGameTest(absltest.TestCase):

  def test_get_prompt(self):
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    state.apply_action(2)
    state.apply_action(3)
    game_adapter = connect_four_game.ConnectFourGameAdapter()
    game_adapter.native_state = state
    prompt = game_adapter.get_prompt()
    self.assertIn(
        "D. Win Condition: The first player to get 4 of their pieces in a row",
        prompt,
    )
    self.assertIn(
        "F. Column Indices: Columns are 0-indexed from left to right (0 to 6).",
        prompt,
    )
    self.assertIn("Board Size: The board has 6 rows and 7 columns.", prompt)
    self.assertIn("Action is on you (Player x).", prompt)
    self.assertIn(
        """.......
.......
.......
.......
.......
..xo...""",
        prompt,
    )


if __name__ == "__main__":
  absltest.main()
