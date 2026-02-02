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

"""Test for rendering game state as image."""

from unittest import mock

from absl.testing import absltest
from game_arena.harness.games.chess import chess_state_renderer


class ChessImageGeneratorTest(absltest.TestCase):

  @mock.patch('game_arena.harness.games.chess.sprite_utils.fen_to_board')
  def test_generate_image_calls_fen_to_board_correctly(
      self, mock_fen_to_board
  ):
    mock_fen_to_board.return_value = b'test_image'
    generator = chess_state_renderer.ChessImageGenerator()
    state_str = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    player_perspective = 'white'

    image_bytes = generator.generate_image(
        state_str, player_perspective=player_perspective
    )

    self.assertEqual(image_bytes, b'test_image')
    mock_fen_to_board.assert_called_once()
    _, call_kwargs = mock_fen_to_board.call_args
    self.assertEqual(call_kwargs['fen'], state_str)
    self.assertIn(call_kwargs['color'], generator._colors)
    self.assertIn(call_kwargs['piece_set'], generator._piece_sets)
    self.assertEqual(call_kwargs['player_perspective'], player_perspective)


if __name__ == '__main__':
  absltest.main()
