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

from game_arena.harness.games.chess import sprite_utils
from absl.testing import absltest
from absl.testing import parameterized


class SpriteUtilsTest(parameterized.TestCase):
  @parameterized.parameters([
      (
          'blue',
          'celtic',
          'dark_k',
      ),
  ])

  def test_get_sprites(self, color, piece_set, piece):
    sprites_dict = sprite_utils.get_sprites()
    self.assertIn(f'{color}-{piece_set}', sprites_dict.keys())
    self.assertIn(piece, sprites_dict[f'{color}-{piece_set}'].keys())

  @parameterized.parameters([
      (
          'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR',
          b'\x89PNG'
      ),
  ])
  def test_fen_to_board(self, fen, png_prefix):
    generated_board = sprite_utils.fen_to_board(fen)
    self.assertIn(png_prefix, generated_board)


if __name__ == '__main__':
  absltest.main()
