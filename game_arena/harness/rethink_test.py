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

"""Tests for rethinking sampler."""

from unittest import mock

from absl.testing import absltest
from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import prompt_templates
from game_arena.harness import rethink
from game_arena.harness.games.chess import game as chess_game
from game_arena.harness.games.chess import rethink as chess_rethink
from game_arena.harness.games.go import game as go_game

import pyspiel


def _create_mock_adapter():
  mock_state = mock.create_autospec(pyspiel.State, instance=True)
  mock_state.to_string.return_value = 'fen_string'

  mock_adapter = mock.create_autospec(
      chess_game.ChessGameAdapter, instance=True
  )
  mock_adapter.raw_state = mock_state
  mock_adapter.legal_actions = ['e4']
  mock_state.current_player.return_value = 0
  mock_adapter.player_number = 0
  return mock_adapter


def _fake_legality_parser_parse(parser_input):
  if parser_input.text in parser_input.legal_moves:
    return parser_input.text
  return None


class _DummyPromptGeneratorSupportsText(
    prompt_generation.PromptGeneratorSupportsText
):

  def generate_prompt_with_text_only(self, **kwargs):
    pass


class _DummyPromptGeneratorSupportsImageText(
    prompt_generation.PromptGeneratorSupportsImageText
):

  def generate_prompt_with_image_text(self, **kwargs):
    pass

  def generate_prompt_with_text_only(self, **kwargs):
    pass


class RethinkSamplerTest(absltest.TestCase):

  maxDiff = None  # Print out the full difference if a test fails.

  def test_sample_action_text_only_success_first_try(self):
    """Tests the case where the first action is legal."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = chess_rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    generate_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.return_value = generate_return
    mock_prompt_generator.generate_prompt_with_text_only.return_value = 'prompt'
    mock_move_parser.parse.return_value = 'e4'
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, 'e4')
    self.assertEqual(output.generate_returns, [generate_return])
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            'parsed_action_attempt_0': 'e4',
            'maybe_legal_action_attempt_0': 'e4',
            'rethink_prompt_attempt_0': '',
        },
    )
    mock_generate_fn.assert_called_once_with('prompt')
    mock_move_parser.parse.assert_called_once()
    mock_legality_parser.parse.assert_called_once()

  def test_sample_action_text_only_single_rethink(self):
    """Tests the case where the first action is illegal and the second is legal."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return = model_generation.GenerateReturn(
        main_response='illegal_move', main_response_and_thoughts='illegal_move'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = ['illegal_move', 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, 'e4')
    self.assertEqual(output.generate_returns, [illegal_return, legal_return])
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            'parsed_action_attempt_0': 'illegal_move',
            'maybe_legal_action_attempt_0': None,
            'rethink_prompt_attempt_0': '',
            'parsed_action_attempt_1': 'e4',
            'maybe_legal_action_attempt_1': 'e4',
            'rethink_prompt_attempt_1': (
                prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                    last_move='illegal_move'
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 2)
    self.assertEqual(mock_move_parser.parse.call_count, 2)
    self.assertEqual(mock_legality_parser.parse.call_count, 2)

  def test_sample_action_text_only_unparsable_then_legal(self):
    """Tests the case where the first action is unparsable and the second is legal."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    unparsable_return = model_generation.GenerateReturn(
        main_response='unparsable', main_response_and_thoughts='unparsable'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [unparsable_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = [None, 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, 'e4')
    self.assertEqual(output.generate_returns, [unparsable_return, legal_return])
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            'parsed_action_attempt_0': None,
            'maybe_legal_action_attempt_0': None,
            'rethink_prompt_attempt_0': '',
            'parsed_action_attempt_1': 'e4',
            'maybe_legal_action_attempt_1': 'e4',
            'rethink_prompt_attempt_1': (
                prompt_templates.RETHINK_WITH_ENV_UNPARSABLE.format(
                    generation='unparsable'
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 2)
    self.assertEqual(mock_move_parser.parse.call_count, 2)
    self.assertEqual(mock_legality_parser.parse.call_count, 2)

  def test_sample_action_text_only_fail_after_max_rethinks(self):
    """Tests the case where all generated actions are illegal."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return1 = model_generation.GenerateReturn(
        main_response='illegal1', main_response_and_thoughts='illegal1'
    )
    illegal_return2 = model_generation.GenerateReturn(
        main_response='illegal2', main_response_and_thoughts='illegal2'
    )
    mock_generate_fn.side_effect = [illegal_return1, illegal_return2]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = ['illegal1', 'illegal2']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, 'illegal2')
    self.assertEqual(
        output.generate_returns, [illegal_return1, illegal_return2]
    )
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            'parsed_action_attempt_0': 'illegal1',
            'maybe_legal_action_attempt_0': None,
            'rethink_prompt_attempt_0': '',
            'parsed_action_attempt_1': 'illegal2',
            'maybe_legal_action_attempt_1': None,
            'rethink_prompt_attempt_1': (
                prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                    last_move='illegal1'
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 2)
    self.assertEqual(mock_move_parser.parse.call_count, 2)
    self.assertEqual(mock_legality_parser.parse.call_count, 2)

  @absltest.skip('TODO(Sohier Dane): fix this test.')
  def test_rethink_strategy(self):
    """Tests the RETHINK strategy."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template=None,
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return = model_generation.GenerateReturn(
        main_response='illegal_move', main_response_and_thoughts='illegal_move'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = ['illegal_move', 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
        foo='bar',
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[0],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt='',
            foo='bar',
        ),
    )
    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt='',
            foo='bar',
        ),
    )

  @absltest.skip('TODO(Sohier Dane): fix this test.')
  @mock.patch(
      'game_arena.harness.games.chess.rethink.rule_explain_illegal_move'
  )
  def test_rethink_with_env_rule_strategy(self, mock_rule_explain_illegal_move):
    """Tests the RETHINK_WITH_ENV_RULE strategy."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    mock_rule_explain_illegal_move.return_value = 'some reason'
    sampler = chess_rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV_RULE,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return = model_generation.GenerateReturn(
        main_response='illegal_move', main_response_and_thoughts='illegal_move'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = ['illegal_move', 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt=prompt_templates.RETHINK_WITH_ENV_RULE.format(
                last_move='illegal_move',
                reason='some reason',
            ),
        ),
    )
    mock_rule_explain_illegal_move.assert_called_once_with(
        fen='fen_string', move_str='illegal_move'
    )

  @absltest.skip('TODO(Sohier Dane): fix this test.')
  def test_rethink_with_env_strategy(self):
    """Tests the RETHINK_WITH_ENV strategy."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = chess_rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return = model_generation.GenerateReturn(
        main_response='illegal_move', main_response_and_thoughts='illegal_move'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = ['illegal_move', 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt=prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                last_move='illegal_move'
            ),
        ),
    )

  @absltest.skip('TODO(Sohier Dane): fix this test.')
  def test_rethink_with_env_strategy_unparsable(self):
    """Tests the RETHINK_WITH_ENV strategy with unparsable response."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = chess_rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    unparsable_return = model_generation.GenerateReturn(
        main_response='unparsable', main_response_and_thoughts='unparsable'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [unparsable_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
    ]
    mock_move_parser.parse.side_effect = [None, 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt=prompt_templates.RETHINK_WITH_ENV_UNPARSABLE.format(
                generation='unparsable'
            ),
        ),
    )

  # TODO(Sohier Dane): this patch should not be needed, but the current state
  # accurately reflects the older code.
  @mock.patch.object(
      rethink.prompts,
      'IMAGE_TEXT_PROMPTS',
      [prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED],
  )
  def test_image_based_workflow(self, *_):
    """Tests the image-based (multimodal) workflow."""
    mock_model = mock.create_autospec(
        model_generation.MultimodalModel, instance=True
    )
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsImageText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    generate_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.return_value = generate_return
    mock_prompt_generator.generate_prompt_with_image_text.return_value = (
        'prompt'
    )
    mock_move_parser.parse.return_value = 'e4'
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    mock_prompt_generator.generate_prompt_with_image_text.assert_called_once()

  def test_rethink_with_env_rule_wrong_game_error(self):
    """Tests ValueError for RETHINK_WITH_ENV_RULE with a non-chess game."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = go_game.GoGameAdapter()
    game_adapter.legal_actions = ['e4']
    game_short_name = game_adapter.game_short_name
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV_RULE,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return = model_generation.GenerateReturn(
        main_response='illegal_move', main_response_and_thoughts='illegal_move'
    )
    mock_generate_fn.return_value = illegal_return
    mock_prompt_generator.generate_prompt_with_text_only.return_value = 'prompt'
    mock_move_parser.parse.return_value = 'illegal_move'
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    with self.assertRaisesRegex(
        NotImplementedError,
        f'explain_illegal_move is not implemented for {game_short_name}',
    ):
      sampler._sample_action(
          model_generate_fn=mock_generate_fn,
          game_adapter=game_adapter,
          prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
      )

  def test_unsupported_strategy_error(self):
    """Tests ValueError for an unsupported strategy."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )

    with self.assertRaisesRegex(ValueError, 'Unsupported strategy'):
      rethink.RethinkSampler(
          model=mock_model,
          strategy='unsupported_strategy',
          num_max_rethinks=1,
          move_parser=mock_move_parser,
          legality_parser=mock_legality_parser,
          prompt_generator=mock_prompt_generator,
          rethink_template='',
          game_adapter=chess_game.ChessGameAdapter(),
      )

  @absltest.skip('TODO(Sohier Dane): fix this test.')
  def test_rethink_with_env_illegal_history_strategy(self):
    """Tests the RETHINK_WITH_ENV_ILLEGAL_HISTORY strategy."""
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    game_adapter = _create_mock_adapter()
    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=rethink.RethinkStrategy.RETHINK_WITH_ENV_ILLEGAL_HISTORY,
        num_max_rethinks=2,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        prompt_generator=mock_prompt_generator,
        rethink_template='',
        game_adapter=game_adapter,
    )

    mock_generate_fn = mock.Mock()
    illegal_return1 = model_generation.GenerateReturn(
        main_response='illegal1', main_response_and_thoughts='illegal1'
    )
    illegal_return2 = model_generation.GenerateReturn(
        main_response='illegal2', main_response_and_thoughts='illegal2'
    )
    legal_return = model_generation.GenerateReturn(
        main_response='e4', main_response_and_thoughts='e4'
    )
    mock_generate_fn.side_effect = [
        illegal_return1,
        illegal_return2,
        legal_return,
    ]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        'prompt1',
        'prompt2',
        'prompt3',
    ]
    mock_move_parser.parse.side_effect = ['illegal1', 'illegal2', 'e4']
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        game_adapter=game_adapter,
        prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, 'e4')
    self.assertEqual(
        output.generate_returns,
        [illegal_return1, illegal_return2, legal_return],
    )
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            'parsed_action_attempt_0': 'illegal1',
            'maybe_legal_action_attempt_0': None,
            'rethink_prompt_attempt_0': '',
            'parsed_action_attempt_1': 'illegal2',
            'maybe_legal_action_attempt_1': None,
            'rethink_prompt_attempt_1': (
                prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                    last_move='illegal1'
                )
            ),
            'parsed_action_attempt_2': 'e4',
            'maybe_legal_action_attempt_2': 'e4',
            'rethink_prompt_attempt_2': (
                prompt_templates.RETHINK_WITH_ENV_ILLEGAL_HISTORY.format(
                    illegal_history='illegal1, illegal2'
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 3)
    self.assertEqual(mock_move_parser.parse.call_count, 3)
    self.assertEqual(mock_legality_parser.parse.call_count, 3)

    calls = mock_prompt_generator.generate_prompt_with_text_only.call_args_list
    self.assertLen(calls, 3)
    self.assertEqual(
        calls[0],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt='',
        ),
    )
    self.assertEqual(
        calls[1],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt=prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                last_move='illegal1'
            ),
        ),
    )
    self.assertEqual(
        calls[2],
        mock.call(
            prompt_template=prompt_templates.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name='chess',
            rethink_prompt=prompt_templates.RETHINK_WITH_ENV_ILLEGAL_HISTORY.format(
                illegal_history='illegal1, illegal2'
            ),
        ),
    )


if __name__ == '__main__':
  absltest.main()
