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

"""Tests for model generation with client SDKs."""

import io
from unittest import mock

mock.patch(
    'game_arena.harness.model_generation._timing_decorator', new=lambda x: x
).start()

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from anthropic import types as anthropic_types
from game_arena.harness import model_generation
from game_arena.harness import model_generation_sdk
from google.genai import types as google_genai_types
import openai
from openai.types import completion_usage as openai_completion_usage
from openai.types.chat import chat_completion as openai_chat_completion_types

from openai.types.chat import chat_completion_chunk


class AIStudioModelTest(absltest.TestCase):
  maxDiff = None

  @mock.patch(
      'game_arena.harness.model_generation_sdk.google_genai.Client', spec=True
  )
  def test_generate_with_text_input(self, mock_client_constructor):
    mock_generative_model = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.models = mock_generative_model

    mock_part = mock.Mock()
    mock_part.text = 'response text'
    mock_part.thought = False

    mock_response = mock.Mock()
    mock_response.candidates = [mock.Mock()]
    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.usage_metadata = mock.Mock()
    mock_response.usage_metadata.prompt_token_count = 1
    mock_response.usage_metadata.candidates_token_count = 2
    mock_response.usage_metadata.thoughts_token_count = 3
    mock_response.to_json_dict.return_value = {'response_key': 'response_value'}

    mock_generative_model.generate_content.return_value = mock_response

    model = model_generation_sdk.AIStudioModel(
        model_name='gemini-pro',
        model_options={'temperature': 0.42},
    )

    response = model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
        )
    )

    self.assertDictEqual(
        response.request_for_logging,
        {
            'model': 'gemini-pro',
            'contents': ['prompt text'],
            'config': {
                'http_options': {'timeout': 1200000},
                'temperature': 0.42,
                'thinking_config': {},
            },
        },
    )
    self.assertDictEqual(
        response.response_for_logging, {'response_key': 'response_value'}
    )
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=5,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gemini-pro',
                'contents': ['prompt text'],
                'config': {
                    'http_options': {'timeout': 1200000},
                    'temperature': 0.42,
                    'thinking_config': {},
                },
            },
            response_for_logging={'response_key': 'response_value'},
        ),
    )
    mock_generative_model.generate_content.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.google_genai.Client', spec=True
  )
  def test_generate_with_text_input_with_thoughts(
      self, mock_client_constructor
  ):
    mock_generative_model = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.models = mock_generative_model

    mock_part_thought = mock.Mock()
    mock_part_thought.text = 'thought text'
    mock_part_thought.thought = True

    mock_part_text = mock.Mock()
    mock_part_text.text = 'response text'
    mock_part_text.thought = False

    mock_response = mock.Mock()
    mock_response.candidates = [mock.Mock()]
    mock_response.candidates[0].content.parts = [
        mock_part_thought,
        mock_part_text,
    ]
    mock_response.usage_metadata = mock.Mock()
    mock_response.usage_metadata.prompt_token_count = 1
    mock_response.usage_metadata.candidates_token_count = 2
    mock_response.usage_metadata.thoughts_token_count = 3
    mock_response.to_json_dict.return_value = {'response_key': 'response_value'}

    mock_generative_model.generate_content.return_value = mock_response

    model = model_generation_sdk.AIStudioModel(
        model_name='gemini-pro',
        model_options={'temperature': 0.42, 'thinking_budget': 99},
        api_options={'include_thoughts': True},
    )

    response = model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
        )
    )

    self.assertDictEqual(
        response.request_for_logging,
        {
            'model': 'gemini-pro',
            'contents': ['prompt text'],
            'config': {
                'http_options': {'timeout': 1200000},
                'temperature': 0.42,
                'thinking_config': {
                    'include_thoughts': True,
                    'thinking_budget': 99,
                },
            },
        },
    )
    self.assertDictEqual(
        response.response_for_logging, {'response_key': 'response_value'}
    )
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='thought text',
            prompt_tokens=1,
            generation_tokens=5,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gemini-pro',
                'contents': ['prompt text'],
                'config': {
                    'http_options': {'timeout': 1200000},
                    'temperature': 0.42,
                    'thinking_config': {
                        'include_thoughts': True,
                        'thinking_budget': 99,
                    },
                },
            },
            response_for_logging={'response_key': 'response_value'},
        ),
    )
    mock_generative_model.generate_content.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.google_genai.Client', spec=True
  )
  def test_generate_with_image_text_input(self, mock_client_constructor):
    mock_generative_model = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.models = mock_generative_model

    mock_part = mock.Mock()
    mock_part.text = 'response text'
    mock_part.thought = False

    mock_response = mock.Mock()
    mock_response.candidates = [mock.Mock()]
    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.usage_metadata = mock.Mock()
    mock_response.usage_metadata.prompt_token_count = 1
    mock_response.usage_metadata.candidates_token_count = 2
    mock_response.usage_metadata.thoughts_token_count = 3
    mock_response.to_json_dict.return_value = {'response_key': 'response_value'}

    mock_generative_model.generate_content.return_value = mock_response

    model = model_generation_sdk.AIStudioModel(
        model_name='gemini-pro',
        model_options={'temperature': 0.42},
    )

    image_bytes = b'image bytes'
    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='prompt text',
            prompt_image_bytes=image_bytes,
            prompt_image_mime_type='image/png',
        )
    )
    contents = [
        'prompt text',
        google_genai_types.Part.from_bytes(
            data=image_bytes, mime_type='image/png'
        ),
    ]
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=5,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gemini-pro',
                'contents': contents,
                'config': {
                    'http_options': {'timeout': 1200000},
                    'temperature': 0.42,
                    'thinking_config': {},
                },
            },
            response_for_logging={'response_key': 'response_value'},
        ),
    )
    mock_generative_model.generate_content.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.google_genai.Client', spec=True
  )
  def test_generate_with_image_text_input_split_prompt(
      self, mock_client_constructor
  ):
    mock_generative_model = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.models = mock_generative_model

    mock_part = mock.Mock()
    mock_part.text = 'response text'
    mock_part.thought = False

    mock_response = mock.Mock()
    mock_response.candidates = [mock.Mock()]
    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.usage_metadata = mock.Mock()
    mock_response.usage_metadata.prompt_token_count = 1
    mock_response.usage_metadata.candidates_token_count = 2
    mock_response.usage_metadata.thoughts_token_count = 3
    mock_response.to_json_dict.return_value = {'response_key': 'response_value'}

    mock_generative_model.generate_content.return_value = mock_response

    model = model_generation_sdk.AIStudioModel(
        model_name='gemini-pro',
        model_options={'temperature': 0.42},
    )

    image_bytes = b'image bytes'
    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='',
            prompt_text_preceding_image='prompt text before',
            prompt_text_following_image='prompt text after',
            prompt_image_bytes=image_bytes,
            prompt_image_mime_type='image/png',
        )
    )
    contents = [
        'prompt text before',
        google_genai_types.Part.from_bytes(
            data=image_bytes, mime_type='image/png'
        ),
        'prompt text after',
    ]
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=5,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gemini-pro',
                'contents': contents,
                'config': {
                    'http_options': {'timeout': 1200000},
                    'temperature': 0.42,
                    'thinking_config': {},
                },
            },
            response_for_logging={'response_key': 'response_value'},
        ),
    )
    mock_generative_model.generate_content.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.google_genai.Client', spec=True
  )
  def test_thinking_budget_is_set(self, mock_client_constructor):
    mock_generative_model = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.models = mock_generative_model

    mock_part = mock.Mock()
    mock_part.text = 'response text'
    mock_part.thought = False

    mock_response = mock.Mock()
    mock_response.candidates = [mock.Mock()]
    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.usage_metadata = mock.Mock()
    mock_response.usage_metadata.prompt_token_count = 1
    mock_response.usage_metadata.candidates_token_count = 2
    mock_response.usage_metadata.thoughts_token_count = 3
    mock_response.to_json_dict.return_value = {'response_key': 'response_value'}

    mock_generative_model.generate_content.return_value = mock_response

    model = model_generation_sdk.AIStudioModel(
        model_name='gemini-pro',
        model_options={'thinking_budget': 42},
    )

    model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
        )
    )

    mock_generative_model.generate_content.assert_called_once()
    _, kwargs = mock_generative_model.generate_content.call_args
    self.assertEqual(
        kwargs['config'].thinking_config.thinking_budget,
        42,
    )


class OpenAIChatCompletionsModelTest(absltest.TestCase):

  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_text_input_streaming(self, mock_client_constructor):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    stream = [
        chat_completion_chunk.ChatCompletionChunk(
            id='test_id',
            choices=[
                chat_completion_chunk.Choice(
                    delta={'content': 'response '},
                    finish_reason='stop',
                    index=0,
                )
            ],
            created=123,
            model='gpt',
            object='chat.completion.chunk',
        ),
        chat_completion_chunk.ChatCompletionChunk(
            id='test_id',
            choices=[
                chat_completion_chunk.Choice(
                    delta={'content': 'text'},
                    finish_reason='stop',
                    index=0,
                )
            ],
            created=123,
            model='gpt',
            object='chat.completion.chunk',
            usage=openai_completion_usage.CompletionUsage(
                prompt_tokens=1,
                completion_tokens=2,
                completion_tokens_details=openai_completion_usage.CompletionTokensDetails(
                    reasoning_tokens=3
                ),
                total_tokens=6,
            ),
        ),
    ]
    mock_completions.create.return_value = stream

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': True},
    )

    response = model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
        )
    )
    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt text'}]}
    ]
    config = {
        'temperature': mock.ANY,
        'top_p': mock.ANY,
        'max_completion_tokens': mock.ANY,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=2,
            reasoning_tokens=3,
            total_tokens=6,
            request_for_logging={
                'model': 'gpt',
                'messages': messages,
                'config': config,
                'stream_options': {'include_usage': True},
            },
            response_for_logging={
                'response_chunks': [
                    c.to_dict() for c in stream if c.usage is not None
                ]
            },
        ),
    )
    mock_completions.create.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_image_text_input_streaming(
      self,
      mock_client_constructor,
  ):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    stream = [
        chat_completion_chunk.ChatCompletionChunk(
            id='test_id',
            choices=[
                chat_completion_chunk.Choice(
                    delta={'content': 'response '},
                    finish_reason='stop',
                    index=0,
                )
            ],
            created=123,
            model='gpt',
            object='chat.completion.chunk',
        ),
        chat_completion_chunk.ChatCompletionChunk(
            id='test_id',
            choices=[
                chat_completion_chunk.Choice(
                    delta={'content': 'text'},
                    finish_reason='stop',
                    index=0,
                )
            ],
            created=123,
            model='gpt',
            object='chat.completion.chunk',
            usage=openai_completion_usage.CompletionUsage(
                prompt_tokens=1,
                completion_tokens=2,
                completion_tokens_details=openai_completion_usage.CompletionTokensDetails(
                    reasoning_tokens=3
                ),
                total_tokens=6,
            ),
        ),
    ]

    mock_completions.create.return_value = stream

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': True},
    )

    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='prompt text',
            prompt_image_bytes=b'image bytes',
            prompt_image_mime_type='image/png',
        )
    )
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'prompt text'},
            {
                'type': 'image_url',
                'image_url': {'url': 'data:image/png;base64,aW1hZ2UgYnl0ZXM='},
            },
        ],
    }]
    config = {
        'temperature': mock.ANY,
        'top_p': mock.ANY,
        'max_completion_tokens': mock.ANY,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=2,
            reasoning_tokens=3,
            total_tokens=6,
            request_for_logging={
                'model': 'gpt',
                'messages': messages,
                'config': config,
                'stream_options': {'include_usage': True},
            },
            response_for_logging={
                'response_chunks': [
                    c.to_dict() for c in stream if c.usage is not None
                ]
            },
        ),
    )
    mock_completions.create.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_text_input_non_streaming(
      self, mock_client_constructor
  ):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    completion = openai_chat_completion_types.ChatCompletion(
        id='test_id',
        choices=[
            openai_chat_completion_types.Choice(
                finish_reason='stop',
                index=0,
                message=openai_chat_completion_types.ChatCompletionMessage(
                    content='response text', role='assistant'
                ),
            )
        ],
        created=123,
        model='gpt',
        object='chat.completion',
        usage=openai_chat_completion_types.CompletionUsage(
            prompt_tokens=1,
            completion_tokens=2,
            completion_tokens_details=openai_completion_usage.CompletionTokensDetails(
                reasoning_tokens=3
            ),
            total_tokens=6,
        ),
    )
    mock_completions.create.return_value = completion

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': False},
    )

    response = model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
        )
    )
    messages = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt text'}]}
    ]
    config = {
        'temperature': mock.ANY,
        'top_p': mock.ANY,
        'max_completion_tokens': mock.ANY,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=2,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gpt',
                'messages': messages,
                'config': config,
            },
            response_for_logging=completion.to_dict(),
        ),
    )
    mock_completions.create.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_image_text_input_non_streaming(
      self, mock_client_constructor
  ):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    completion = openai_chat_completion_types.ChatCompletion(
        id='test_id',
        choices=[
            openai_chat_completion_types.Choice(
                finish_reason='stop',
                index=0,
                message=openai_chat_completion_types.ChatCompletionMessage(
                    content='response text', role='assistant'
                ),
            )
        ],
        created=123,
        model='gpt',
        object='chat.completion',
        usage=openai_chat_completion_types.CompletionUsage(
            prompt_tokens=1,
            completion_tokens=2,
            completion_tokens_details=openai_completion_usage.CompletionTokensDetails(
                reasoning_tokens=3
            ),
            total_tokens=6,
        ),
    )
    mock_completions.create.return_value = completion

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': False},
    )

    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='prompt text',
            prompt_image_bytes=b'image bytes',
            prompt_image_mime_type='image/png',
        )
    )
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'prompt text'},
            {
                'type': 'image_url',
                'image_url': {'url': 'data:image/png;base64,aW1hZ2UgYnl0ZXM='},
            },
        ],
    }]
    config = {
        'temperature': mock.ANY,
        'top_p': mock.ANY,
        'max_completion_tokens': mock.ANY,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=2,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gpt',
                'messages': messages,
                'config': config,
            },
            response_for_logging=completion.to_dict(),
        ),
    )
    mock_completions.create.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_image_text_input_split_prompt_streaming(
      self,
      mock_client_constructor,
  ):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    stream = [
        chat_completion_chunk.ChatCompletionChunk(
            id='test_id',
            choices=[
                chat_completion_chunk.Choice(
                    delta={'content': 'response '},
                    finish_reason='stop',
                    index=0,
                )
            ],
            created=123,
            model='gpt',
            object='chat.completion.chunk',
        ),
        chat_completion_chunk.ChatCompletionChunk(
            id='test_id',
            choices=[
                chat_completion_chunk.Choice(
                    delta={'content': 'text'},
                    finish_reason='stop',
                    index=0,
                )
            ],
            created=123,
            model='gpt',
            object='chat.completion.chunk',
            usage=openai_completion_usage.CompletionUsage(
                prompt_tokens=1,
                completion_tokens=2,
                completion_tokens_details=openai_completion_usage.CompletionTokensDetails(
                    reasoning_tokens=3
                ),
                total_tokens=6,
            ),
        ),
    ]

    mock_completions.create.return_value = stream

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': True},
    )

    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='',
            prompt_text_preceding_image='prompt text before',
            prompt_text_following_image='prompt text after',
            prompt_image_bytes=b'image bytes',
            prompt_image_mime_type='image/png',
        )
    )
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'prompt text before'},
            {
                'type': 'image_url',
                'image_url': {'url': 'data:image/png;base64,aW1hZ2UgYnl0ZXM='},
            },
            {'type': 'text', 'text': 'prompt text after'},
        ],
    }]
    config = {
        'temperature': mock.ANY,
        'top_p': mock.ANY,
        'max_completion_tokens': mock.ANY,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=2,
            reasoning_tokens=3,
            total_tokens=6,
            request_for_logging={
                'model': 'gpt',
                'messages': messages,
                'config': config,
                'stream_options': {'include_usage': True},
            },
            response_for_logging={
                'response_chunks': [
                    c.to_dict() for c in stream if c.usage is not None
                ]
            },
        ),
    )
    mock_completions.create.assert_called_once()

  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_image_text_input_split_prompt_non_streaming(
      self, mock_client_constructor
  ):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    completion = openai_chat_completion_types.ChatCompletion(
        id='test_id',
        choices=[
            openai_chat_completion_types.Choice(
                finish_reason='stop',
                index=0,
                message=openai_chat_completion_types.ChatCompletionMessage(
                    content='response text', role='assistant'
                ),
            )
        ],
        created=123,
        model='gpt',
        object='chat.completion',
        usage=openai_chat_completion_types.CompletionUsage(
            prompt_tokens=1,
            completion_tokens=2,
            completion_tokens_details=openai_completion_usage.CompletionTokensDetails(
                reasoning_tokens=3
            ),
            total_tokens=6,
        ),
    )
    mock_completions.create.return_value = completion

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': False},
    )

    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='',
            prompt_text_preceding_image='prompt text before',
            prompt_text_following_image='prompt text after',
            prompt_image_bytes=b'image bytes',
            prompt_image_mime_type='image/png',
        )
    )
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'prompt text before'},
            {
                'type': 'image_url',
                'image_url': {'url': 'data:image/png;base64,aW1hZ2UgYnl0ZXM='},
            },
            {'type': 'text', 'text': 'prompt text after'},
        ],
    }]
    config = {
        'temperature': mock.ANY,
        'top_p': mock.ANY,
        'max_completion_tokens': mock.ANY,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='',
            prompt_tokens=1,
            generation_tokens=2,
            reasoning_tokens=3,
            request_for_logging={
                'model': 'gpt',
                'messages': messages,
                'config': config,
            },
            response_for_logging=completion.to_dict(),
        ),
    )
    mock_completions.create.assert_called_once()

  @mock.patch('sys.stdout', new_callable=io.StringIO)
  @mock.patch(
      'game_arena.harness.model_generation_sdk.openai.OpenAI', spec=True
  )
  def test_generate_with_text_input_stream_error(
      self, mock_client_constructor, mock_stdout
  ):
    mock_chat = mock.Mock()
    mock_completions = mock.Mock()
    mock_chat.completions = mock_completions
    mock_client = mock_client_constructor.return_value
    mock_client.chat = mock_chat

    mock_response = mock.Mock()
    mock_response.status_code = 404
    mock_response.request = mock.Mock()
    mock_response.headers.get.return_value = None
    mock_completions.create.side_effect = openai.NotFoundError(
        'test error', response=mock_response, body=None
    )

    model = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name='gpt',
        api_options={'stream': True},
    )
    with self.assertRaises(model_generation.DoNotRetryError):
      model.generate_with_text_input(
          model_generation.ModelTextInput(
              prompt_text='prompt text',
          )
      )
    self.assertEqual(mock_stdout.getvalue(), '')


class AnthropicModelTest(absltest.TestCase):

  @mock.patch.dict(model_generation_sdk._ANTHROPIC_MAX_TOKENS, {'claude': 4096})
  @mock.patch(
      'game_arena.harness.model_generation_sdk.anthropic.Anthropic', spec=True
  )
  def test_generate_with_text_input(self, mock_client_constructor):
    mock_messages = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.messages = mock_messages

    message = anthropic_types.Message(
        id='test_id',
        content=[anthropic_types.TextBlock(text='response text', type='text')],
        model='claude',
        role='assistant',
        type='message',
        usage=anthropic_types.Usage(input_tokens=1, output_tokens=2),
    )
    mock_messages.create.return_value = message

    model = model_generation_sdk.AnthropicModel(
        model_name='claude',
        api_options={'timeout': 42},
    )

    response = model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
            system_instruction='some_system_instruction',
        )
    )
    messages = [{'role': 'user', 'content': 'prompt text'}]
    config = {
        'system': 'some_system_instruction',
        'max_tokens': 4096,
        'temperature': mock.ANY,
        'top_k': mock.ANY,
        'top_p': mock.ANY,
        'thinking': mock.ANY,
        'timeout': 42,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='response text',
            prompt_tokens=1,
            generation_tokens=2,
            request_for_logging={
                'model': 'claude',
                'messages': messages,
                'config': config,
            },
            response_for_logging=message.to_dict(),
        ),
    )
    mock_messages.create.assert_called_once()

  @mock.patch.dict(model_generation_sdk._ANTHROPIC_MAX_TOKENS, {'claude': 4096})
  @mock.patch(
      'game_arena.harness.model_generation_sdk.anthropic.Anthropic', spec=True
  )
  def test_generate_with_text_input_with_thoughts(
      self, mock_client_constructor
  ):
    mock_messages = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.messages = mock_messages

    message = anthropic_types.Message(
        id='test_id',
        content=[
            anthropic_types.ThinkingBlock(
                signature='xxx', thinking='thinking text', type='thinking'
            ),
            anthropic_types.TextBlock(text='response text', type='text'),
        ],
        model='claude',
        role='assistant',
        type='message',
        usage=anthropic_types.Usage(input_tokens=1, output_tokens=2),
    )
    mock_messages.create.return_value = message

    model = model_generation_sdk.AnthropicModel(
        model_name='claude',
        api_options={'timeout': 42},
    )

    response = model.generate_with_text_input(
        model_generation.ModelTextInput(
            prompt_text='prompt text',
        )
    )
    messages = [{'role': 'user', 'content': 'prompt text'}]
    config = {
        'system': mock.ANY,
        'max_tokens': 4096,
        'temperature': mock.ANY,
        'top_k': mock.ANY,
        'top_p': mock.ANY,
        'thinking': mock.ANY,
        'timeout': 42,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='thinking textresponse text',
            prompt_tokens=1,
            generation_tokens=2,
            request_for_logging={
                'model': 'claude',
                'messages': messages,
                'config': config,
            },
            response_for_logging=message.to_dict(),
        ),
    )
    mock_messages.create.assert_called_once()

  @mock.patch.dict(model_generation_sdk._ANTHROPIC_MAX_TOKENS, {'claude': 4096})
  @mock.patch(
      'game_arena.harness.model_generation_sdk.anthropic.Anthropic', spec=True
  )
  def test_generate_with_image_text_input(self, mock_client_constructor):
    mock_messages = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.messages = mock_messages

    message = anthropic_types.Message(
        id='test_id',
        content=[anthropic_types.TextBlock(text='response text', type='text')],
        model='claude',
        role='assistant',
        type='message',
        usage=anthropic_types.Usage(input_tokens=1, output_tokens=2),
    )
    mock_messages.create.return_value = message

    model = model_generation_sdk.AnthropicModel(
        model_name='claude',
        api_options={'timeout': 42},
    )

    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='prompt text',
            prompt_image_bytes=b'image bytes',
            prompt_image_mime_type='image/png',
        )
    )
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'prompt text'},
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': 'aW1hZ2UgYnl0ZXM=',
                },
            },
        ],
    }]
    config = {
        'system': mock.ANY,
        'max_tokens': 4096,
        'temperature': mock.ANY,
        'top_k': mock.ANY,
        'top_p': mock.ANY,
        'thinking': mock.ANY,
        'timeout': 42,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='response text',
            prompt_tokens=1,
            generation_tokens=2,
            request_for_logging={
                'model': 'claude',
                'messages': messages,
                'config': config,
            },
            response_for_logging=message.to_dict(),
        ),
    )
    mock_messages.create.assert_called_once()

  @mock.patch.dict(model_generation_sdk._ANTHROPIC_MAX_TOKENS, {'claude': 4096})
  @mock.patch(
      'game_arena.harness.model_generation_sdk.anthropic.Anthropic', spec=True
  )
  def test_generate_with_image_text_input_split_prompt(
      self, mock_client_constructor
  ):
    mock_messages = mock.Mock()
    mock_client = mock_client_constructor.return_value
    mock_client.messages = mock_messages

    message = anthropic_types.Message(
        id='test_id',
        content=[anthropic_types.TextBlock(text='response text', type='text')],
        model='claude',
        role='assistant',
        type='message',
        usage=anthropic_types.Usage(input_tokens=1, output_tokens=2),
    )
    mock_messages.create.return_value = message

    model = model_generation_sdk.AnthropicModel(
        model_name='claude',
        api_options={'timeout': 42},
    )

    response = model.generate_with_image_text_input(
        model_generation.ModelImageTextInput(
            prompt_text='',
            prompt_text_preceding_image='prompt text before',
            prompt_text_following_image='prompt text after',
            prompt_image_bytes=b'image bytes',
            prompt_image_mime_type='image/png',
        )
    )
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'prompt text before'},
            {
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': 'aW1hZ2UgYnl0ZXM=',
                },
            },
            {'type': 'text', 'text': 'prompt text after'},
        ],
    }]
    config = {
        'system': mock.ANY,
        'max_tokens': 4096,
        'temperature': mock.ANY,
        'top_k': mock.ANY,
        'top_p': mock.ANY,
        'thinking': mock.ANY,
        'timeout': 42,
    }
    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='response text',
            main_response_and_thoughts='response text',
            prompt_tokens=1,
            generation_tokens=2,
            request_for_logging={
                'model': 'claude',
                'messages': messages,
                'config': config,
            },
            response_for_logging=message.to_dict(),
        ),
    )
    mock_messages.create.assert_called_once()


class ProcessAnthropicStreamTest(absltest.TestCase):

  def test_process_anthropic_stream(self):
    mock_stream = [
        anthropic_types.RawMessageStartEvent(
            message=anthropic_types.Message(
                id='test_id',
                content=[],
                model='claude',
                role='assistant',
                type='message',
                usage=anthropic_types.Usage(input_tokens=1, output_tokens=2),
            ),
            type='message_start',
        ),
        anthropic_types.RawContentBlockDeltaEvent(
            delta=anthropic_types.TextDelta(
                text='response ', type='text_delta'
            ),
            type='content_block_delta',
            index=0,
        ),
        anthropic_types.RawContentBlockDeltaEvent(
            delta=anthropic_types.TextDelta(text='text', type='text_delta'),
            type='content_block_delta',
            index=0,
        ),
        anthropic_types.RawMessageDeltaEvent(
            delta=anthropic_types.raw_message_delta_event.Delta(
                stop_reason='end_turn'
            ),
            usage=anthropic_types.MessageDeltaUsage(output_tokens=3),
            type='message_delta',
        ),
    ]

    anthropic_return, response_chunks = (
        model_generation_sdk._process_anthropic_stream(mock_stream)
    )
    self.assertEqual(
        anthropic_return,
        model_generation_sdk.AnthropicReturn(
            main_response='response text',
            main_response_and_thoughts='response text',
            prompt_tokens=1,
            generation_tokens=5,
        ),
    )
    self.assertEqual(response_chunks, mock_stream)

  def test_process_anthropic_stream_with_thoughts(self):
    mock_stream = [
        anthropic_types.RawMessageStartEvent(
            message=anthropic_types.Message(
                id='test_id',
                content=[],
                model='claude',
                role='assistant',
                type='message',
                usage=anthropic_types.Usage(input_tokens=1, output_tokens=2),
            ),
            type='message_start',
        ),
        anthropic_types.RawContentBlockDeltaEvent(
            delta=anthropic_types.ThinkingDelta(
                thinking='thinking ', type='thinking_delta'
            ),
            type='content_block_delta',
            index=0,
        ),
        anthropic_types.RawContentBlockDeltaEvent(
            delta=anthropic_types.ThinkingDelta(
                thinking='text', type='thinking_delta'
            ),
            type='content_block_delta',
            index=0,
        ),
        anthropic_types.RawContentBlockDeltaEvent(
            delta=anthropic_types.TextDelta(
                text='response ', type='text_delta'
            ),
            type='content_block_delta',
            index=0,
        ),
        anthropic_types.RawContentBlockDeltaEvent(
            delta=anthropic_types.TextDelta(text='text', type='text_delta'),
            type='content_block_delta',
            index=0,
        ),
        anthropic_types.RawMessageDeltaEvent(
            delta=anthropic_types.raw_message_delta_event.Delta(
                stop_reason='end_turn'
            ),
            usage=anthropic_types.MessageDeltaUsage(output_tokens=3),
            type='message_delta',
        ),
    ]

    anthropic_return, response_chunks = (
        model_generation_sdk._process_anthropic_stream(mock_stream)
    )
    self.assertEqual(
        anthropic_return,
        model_generation_sdk.AnthropicReturn(
            main_response='response text',
            main_response_and_thoughts='thinking textresponse text',
            prompt_tokens=1,
            generation_tokens=5,
        ),
    )
    self.assertEqual(response_chunks, mock_stream)


if __name__ == '__main__':
  absltest.main()
