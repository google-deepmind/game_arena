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

import json

from game_arena.harness.games.go import game as go_formatter

from absl.testing import absltest
import pyspiel


# pylint: disable=line-too-long
_EXPECTED_STATE_DICT = {
    'board_size': 9,
    'komi': 7.5,
    'current_player_to_move': 'W',
    'move_number': 6,
    'previous_move_a1': 'B d5',
    'board_grid': [
        [{'A9': 'W'}, {'B9': '.'}, {'C9': '.'}, {'D9': '.'}, {'E9': '.'}, {'F9': '.'}, {'G9': '.'}, {'H9': '.'}, {'J9': 'B'}],
        [{'A8': '.'}, {'B8': '.'}, {'C8': '.'}, {'D8': '.'}, {'E8': '.'}, {'F8': '.'}, {'G8': '.'}, {'H8': '.'}, {'J8': '.'}],
        [{'A7': '.'}, {'B7': '.'}, {'C7': '.'}, {'D7': '.'}, {'E7': '.'}, {'F7': '.'}, {'G7': '.'}, {'H7': '.'}, {'J7': '.'}],
        [{'A6': '.'}, {'B6': '.'}, {'C6': '.'}, {'D6': '.'}, {'E6': '.'}, {'F6': '.'}, {'G6': '.'}, {'H6': '.'}, {'J6': '.'}],
        [{'A5': '.'}, {'B5': '.'}, {'C5': '.'}, {'D5': 'B'}, {'E5': '.'}, {'F5': '.'}, {'G5': '.'}, {'H5': '.'}, {'J5': '.'}],
        [{'A4': '.'}, {'B4': '.'}, {'C4': '.'}, {'D4': '.'}, {'E4': '.'}, {'F4': '.'}, {'G4': '.'}, {'H4': '.'}, {'J4': '.'}],
        [{'A3': '.'}, {'B3': '.'}, {'C3': '.'}, {'D3': '.'}, {'E3': '.'}, {'F3': '.'}, {'G3': '.'}, {'H3': '.'}, {'J3': '.'}],
        [{'A2': '.'}, {'B2': '.'}, {'C2': '.'}, {'D2': '.'}, {'E2': '.'}, {'F2': '.'}, {'G2': '.'}, {'H2': '.'}, {'J2': '.'}],
        [{'A1': 'B'}, {'B1': '.'}, {'C1': '.'}, {'D1': '.'}, {'E1': '.'}, {'F1': '.'}, {'G1': '.'}, {'H1': '.'}, {'J1': 'W'}],
    ],
}
# pylint: enable=line-too-long


_STATE_STR = """GoState(komi=7.5, to_play=W, history.size()=19)

19 +++O++++++O+++O++++
18 ++++O+++++X++++++++
17 ++++++++++++++++X++
16 +++++++++++++++++++
15 ++++++++X++++++++++
14 +O+X+++++++++++++++
13 +++++++++++++++++++
12 +++++++++++++++++++
11 ++++++++O++++++++++
10 +++++++++++++++++++
 9 ++++++++++++X++++++
 8 +++++++++++++++++++
 7 +++++X+++++++++++++
 6 +++++++++++++++++++
 5 ++++++++++++X+++++X
 4 ++O++++++++++++++++
 3 ++++++++++++++++O++
 2 +++++++++O+++++++++
 1 ++++XX+++++++++++++
   ABCDEFGHJKLMNOPQRST
"""

_EXPECTED_CONVERTED_STATE_STR = """19 . . . W . . . . . . W . . . W . . . .
18 . . . . W . . . . . B . . . . . . . .
17 . . . . . . . . . . . . . . . . B . .
16 . . . . . . . . . . . . . . . . . . .
15 . . . . . . . . B . . . . . . . . . .
14 . W . B . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . W . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . B . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . B . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . B . . . . . B
 4 . . W . . . . . . . . . . . . . . . .
 3 . . . . . . . . . . . . . . . . W . .
 2 . . . . . . . . . W . . . . . . . . .
 1 . . . . B B . . . . . . . . . . . . .
   A B C D E F G H J K L M N O P Q R S T"""


class GoTest(absltest.TestCase):

  def test_basic(self):
    game = pyspiel.load_game('go(board_size=9)')
    state = game.new_initial_state()
    state.apply_action(state.string_to_action('B a1'))
    state.apply_action(state.string_to_action('W a9'))
    state.apply_action(state.string_to_action('B j9'))
    state.apply_action(state.string_to_action('W j1'))
    state.apply_action(state.string_to_action('B d5'))
    state_str = go_formatter.format_state(state)
    state_dict = json.loads(state_str)
    self.assertEqual(state_dict, _EXPECTED_STATE_DICT)

  def test_convert_state(self):
    converted_state_str = go_formatter.convert_state(
        _STATE_STR, swap_probability=1.0
    )
    self.assertEqual(converted_state_str, _EXPECTED_CONVERTED_STATE_STR)


if __name__ == '__main__':
  absltest.main()
