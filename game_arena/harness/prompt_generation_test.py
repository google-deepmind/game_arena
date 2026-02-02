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

"""Tests for prompt generation."""

from unittest import mock

from absl.testing import absltest
from game_arena.harness import prompt_generation
from game_arena.harness import prompt_templates
from game_arena.harness.games.chess import game as chess_game
from game_arena.harness.games.chess import prompt_generation as chess_prompt_generation
from game_arena.harness.games.poker import game as poker_game

import pyspiel


class PromptGeneratorImageTextTest(absltest.TestCase):

  def test_generate_prompt_with_image_text_raises_error_for_non_image_template(
      self,
  ):
    generator = prompt_generation.PromptGeneratorImageText()
    with self.assertRaises(ValueError):
      generator.generate_prompt_with_image_text(
          prompt_templates.NO_LEGAL_ACTIONS, game_adapter=None
      )

  def test_generate_prompt_with_image_text_raises_error_for_none_state(self):
    generator = prompt_generation.PromptGeneratorImageText()
    with self.assertRaises(AttributeError):
      generator.generate_prompt_with_image_text(
          prompt_templates.WITH_BOARD_IMAGE, game_adapter=None
      )

  def test_generate_prompt_with_image_text_raises_error_for_unsupported_game(
      self,
  ):
    generator = prompt_generation.PromptGeneratorImageText()
    game = pyspiel.load_game('universal_poker')
    state = game.new_initial_state()
    game_adapter = poker_game.PokerGameAdapter()
    game_adapter.native_state = state
    with self.assertRaises(NotImplementedError):
      generator.generate_prompt_with_image_text(
          prompt_templates.WITH_BOARD_IMAGE,
          game_adapter=game_adapter,
      )

  @mock.patch(
      'game_arena.harness.games.chess.chess_state_renderer.ChessImageGenerator'
  )
  def test_generate_prompt_with_image_text_chess_success(
      self, mock_image_generator
  ):
    mock_generator_instance = mock_image_generator.return_value
    mock_generator_instance.generate_image.return_value = b'test_image_bytes'

    generator = chess_prompt_generation.PromptGeneratorImageText()
    game = pyspiel.load_game('chess')
    state = game.new_initial_state()
    game_adapter = chess_game.ChessGameAdapter()
    game_adapter.native_state = state
    prompt_substitutions = {
        'player_name': 'Player 1',
        'move_notation': 'SAN',
    }
    result = generator.generate_prompt_with_image_text(
        prompt_templates.WITH_BOARD_IMAGE,
        game_adapter=game_adapter,
        **prompt_substitutions,
    )

    self.assertEqual(result.prompt_text, '')
    self.assertEqual(result.prompt_image_bytes, b'test_image_bytes')
    self.assertEqual(result.prompt_image_mime_type, 'image/png')
    self.assertIn("Let's play chess.", result.prompt_text_preceding_image)
    self.assertIn(
        'The current game board is shown as:',
        result.prompt_text_preceding_image,
    )
    self.assertIn(
        'You are playing as player Player 1',
        result.prompt_text_following_image,
    )
    self.assertIn('your chosen move in SAN', result.prompt_text_following_image)

  @mock.patch(
      'game_arena.harness.games.chess.chess_state_renderer.ChessImageGenerator'
  )
  def test_generate_prompt_with_image_text_chess_rethink_success(
      self, mock_image_generator
  ):
    mock_generator_instance = mock_image_generator.return_value
    mock_generator_instance.generate_image.return_value = b'test_image_bytes'

    generator = chess_prompt_generation.PromptGeneratorImageText()
    game = pyspiel.load_game('chess')
    state = game.new_initial_state()
    game_adapter = chess_game.ChessGameAdapter()
    game_adapter.native_state = state
    prompt_substitutions = {
        'player_name': 'Player 2',
        'move_notation': 'UCI',
        'rethink_prompt': 'Your last move was invalid.',
    }
    result = generator.generate_prompt_with_image_text(
        prompt_templates.WITH_BOARD_IMAGE_RETHINK_APPENDED,
        game_adapter=game_adapter,
        **prompt_substitutions,
    )

    self.assertEqual(result.prompt_text, '')
    self.assertEqual(result.prompt_image_bytes, b'test_image_bytes')
    self.assertEqual(result.prompt_image_mime_type, 'image/png')
    self.assertIn("Let's play chess.", result.prompt_text_preceding_image)
    self.assertIn(
        'The current game board is shown as:',
        result.prompt_text_preceding_image,
    )
    self.assertIn(
        'You are playing as player Player 2',
        result.prompt_text_following_image,
    )
    self.assertIn('your chosen move in UCI', result.prompt_text_following_image)
    self.assertIn(
        'Your last move was invalid.', result.prompt_text_following_image
    )


if __name__ == '__main__':
  absltest.main()
