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

"""Tests for different model samplers such as majority voting."""

import random
from unittest import mock

from absl.testing import absltest
from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import samplers


class MajorityVoteSamplerTest(absltest.TestCase):

  def test_sample_action_with_clear_majority(self):
    mock_model = mock.Mock(spec=model_generation.Model)
    mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response='A', main_response_and_thoughts=''
        ),
        model_generation.GenerateReturn(
            main_response='B', main_response_and_thoughts=''
        ),
        model_generation.GenerateReturn(
            main_response='A', main_response_and_thoughts=''
        ),
    ]
    mock_parser = mock.Mock(spec=parsers.TextParser)
    mock_parser.parse.side_effect = ['A', 'B', 'A']

    sampler = samplers.MajorityVoteSampler(
        model=mock_model, num_samples=3, parser=mock_parser
    )
    model_input = model_generation.ModelTextInput(
        prompt_text='prompt',
    )
    sampler_output = sampler.sample_action_with_text_input(model_input)

    self.assertEqual(sampler_output.action, 'A')
    self.assertEqual(
        sampler_output.generate_returns,
        [
            model_generation.GenerateReturn(
                main_response='A', main_response_and_thoughts=''
            ),
            model_generation.GenerateReturn(
                main_response='B', main_response_and_thoughts=''
            ),
            model_generation.GenerateReturn(
                main_response='A', main_response_and_thoughts=''
            ),
        ],
    )
    self.assertDictEqual(
        sampler_output.auxiliary_outputs, {'frequencies': {'A': 2, 'B': 1}}
    )
    self.assertEqual(mock_model.generate_with_text_input.call_count, 3)
    self.assertEqual(mock_parser.parse.call_count, 3)

  def test_sample_action_with_tie(self):
    mock_model = mock.Mock(spec=model_generation.Model)
    mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response='A', main_response_and_thoughts=''
        ),
        model_generation.GenerateReturn(
            main_response='B', main_response_and_thoughts=''
        ),
    ]
    mock_parser = mock.Mock(spec=parsers.TextParser)
    mock_parser.parse.side_effect = ['A', 'B']

    sampler = samplers.MajorityVoteSampler(
        model=mock_model, num_samples=2, parser=mock_parser
    )
    model_input = model_generation.ModelTextInput(
        prompt_text='prompt',
    )
    sampler_output = sampler.sample_action_with_text_input(model_input)

    self.assertEqual(
        sampler_output.action, random.Random(42).choice(['A', 'B'])
    )
    self.assertEqual(
        sampler_output.generate_returns,
        [
            model_generation.GenerateReturn(
                main_response='A', main_response_and_thoughts=''
            ),
            model_generation.GenerateReturn(
                main_response='B', main_response_and_thoughts=''
            ),
        ],
    )
    self.assertDictEqual(
        sampler_output.auxiliary_outputs, {'frequencies': {'A': 1, 'B': 1}}
    )
    self.assertEqual(mock_model.generate_with_text_input.call_count, 2)
    self.assertEqual(mock_parser.parse.call_count, 2)

  def test_sample_action_with_no_valid_actions(self):
    mock_model = mock.Mock(spec=model_generation.Model)
    mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response='', main_response_and_thoughts=''
        ),
        model_generation.GenerateReturn(
            main_response='', main_response_and_thoughts=''
        ),
    ]
    mock_parser = mock.Mock(spec=parsers.TextParser)
    mock_parser.parse.side_effect = [None, None]

    sampler = samplers.MajorityVoteSampler(
        model=mock_model, num_samples=2, parser=mock_parser
    )
    model_input = model_generation.ModelTextInput(
        prompt_text='prompt',
    )
    sampler_output = sampler.sample_action_with_text_input(model_input)

    self.assertIsNone(sampler_output.action)
    self.assertEqual(
        sampler_output.generate_returns,
        [
            model_generation.GenerateReturn(
                main_response='', main_response_and_thoughts=''
            ),
            model_generation.GenerateReturn(
                main_response='', main_response_and_thoughts=''
            ),
        ],
    )
    self.assertEqual(mock_model.generate_with_text_input.call_count, 2)
    self.assertEqual(mock_parser.parse.call_count, 2)

  def test_sample_action_with_image_input(self):
    mock_model = mock.Mock(spec=model_generation.MultimodalModel)
    mock_model.generate_with_image_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response='A', main_response_and_thoughts=''
        ),
        model_generation.GenerateReturn(
            main_response='B', main_response_and_thoughts=''
        ),
        model_generation.GenerateReturn(
            main_response='A', main_response_and_thoughts=''
        ),
    ]
    mock_parser = mock.Mock(spec=parsers.TextParser)
    mock_parser.parse.side_effect = ['A', 'B', 'A']

    sampler = samplers.MajorityVoteMultimodalSampler(
        model=mock_model, num_samples=3, parser=mock_parser
    )
    model_input = model_generation.ModelImageTextInput(
        prompt_text='prompt',
        prompt_image_bytes=b'image_bytes',
        prompt_image_mime_type='image/png',
    )
    sampler_output = sampler.sample_action_with_image_text_input(model_input)

    self.assertEqual(sampler_output.action, 'A')
    self.assertEqual(
        sampler_output.generate_returns,
        [
            model_generation.GenerateReturn(
                main_response='A', main_response_and_thoughts=''
            ),
            model_generation.GenerateReturn(
                main_response='B', main_response_and_thoughts=''
            ),
            model_generation.GenerateReturn(
                main_response='A', main_response_and_thoughts=''
            ),
        ],
    )
    self.assertDictEqual(
        sampler_output.auxiliary_outputs, {'frequencies': {'A': 2, 'B': 1}}
    )
    self.assertEqual(mock_model.generate_with_image_text_input.call_count, 3)
    self.assertEqual(mock_parser.parse.call_count, 3)


if __name__ == '__main__':
  absltest.main()
