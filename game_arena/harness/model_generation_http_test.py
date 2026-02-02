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

"""Tests for model generation via HTTP POST APIs."""

import asyncio
import datetime
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import aiohttp
from game_arena.harness import model_generation
from game_arena.harness import model_generation_http


class ModelGenerationHttpTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'no_tags',
          'just some text',
          None,
      ),
      (
          'only_opening_tag',
          'just<think>some text',
          None,
      ),
      (
          'only_closing_tag',
          'just</think>some text',
          None,
      ),
      (
          'base_case',
          '<think>thought</think>postscript',
          ('postscript', 'thought'),
      ),
      (
          'empty_thought',
          '<think></think>postscript',
          ('postscript', ''),
      ),
      (
          'empty_main_response',
          '<think>thought</think>',
          ('', 'thought'),
      ),
      (
          'empty_string',
          '',
          None,
      ),
  )
  def test_deepseek_separate_main_response_and_thoughts(
      self, response, expected
  ):
    self.assertEqual(
        model_generation.separate_main_response_and_thoughts(
            response
        ),
        expected,
    )

  def test_create_image_text_content(self):
    model_input = model_generation.ModelImageTextInput(
        prompt_text='My prompt',
        prompt_image_bytes=b'someimagedata',
        prompt_image_mime_type='image/png',
    )
    expected_content = [
        {'type': 'text', 'text': 'My prompt'},
        {
            'type': 'image_url',
            'image_url': {'url': 'data:image/png;base64,c29tZWltYWdlZGF0YQ=='},
        },
    ]
    self.assertEqual(
        model_generation_http._create_image_text_content(model_input),
        expected_content,
    )

  def test_create_image_text_content_split_prompt(self):
    model_input = model_generation.ModelImageTextInput(
        prompt_text='',
        prompt_text_preceding_image='My prompt before',
        prompt_text_following_image='My prompt after',
        prompt_image_bytes=b'someimagedata',
        prompt_image_mime_type='image/png',
    )
    expected_content = [
        {'type': 'text', 'text': 'My prompt before'},
        {
            'type': 'image_url',
            'image_url': {'url': 'data:image/png;base64,c29tZWltYWdlZGF0YQ=='},
        },
        {'type': 'text', 'text': 'My prompt after'},
    ]
    self.assertEqual(
        model_generation_http._create_image_text_content(model_input),
        expected_content,
    )


class XAIModelTest(absltest.TestCase):

  @mock.patch('requests.post', spec=True)
  def test_generate_with_reasoning(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{
            'message': {
                'content': 'main response',
                'reasoning_content': 'reasoning',
            }
        }]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.XAIModel(
        model_name='grok', api_key='fake_key'
    )
    response = model._generate(
        messages=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt'}]}
        ]
    )

    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='main response',
            main_response_and_thoughts='main response\n\nreasoning',
            request_for_logging={
                'model': 'grok',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'prompt'}],
                }],
                'stream': False,
                'stream_options': {'include_usage': True},
            },
            response_for_logging={
                'choices': [{
                    'message': {
                        'content': 'main response',
                        'reasoning_content': 'reasoning',
                    }
                }]
            },
        ),
    )
    mock_post.assert_called_once()

  @mock.patch('requests.post', spec=True)
  def test_generate_without_reasoning(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'main response'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.XAIModel(
        model_name='grok', api_key='fake_key'
    )
    response = model._generate(
        messages=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt'}]}
        ]
    )

    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='main response',
            main_response_and_thoughts='',
            request_for_logging={
                'model': 'grok',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'prompt'}],
                }],
                'stream': False,
                'stream_options': {'include_usage': True},
            },
            response_for_logging={
                'choices': [{'message': {'content': 'main response'}}]
            },
        ),
    )
    mock_post.assert_called_once()

  @mock.patch('requests.post', spec=True)
  def test_generate_with_model_options(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'main response'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.XAIModel(
        model_name='grok',
        api_key='fake_key',
        model_options={'temperature': 0.5, 'top_p': 0.8},
    )
    model._generate(
        messages=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt'}]}
        ],
    )

    mock_post.assert_called_once()
    called_json = mock_post.call_args.kwargs['json']
    self.assertEqual(called_json['temperature'], 0.5)
    self.assertEqual(called_json['top_p'], 0.8)

  @mock.patch('requests.post', spec=True)
  def test_generate_with_system_instruction(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'main response'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.XAIModel(
        model_name='grok', api_key='fake_key'
    )
    model._generate(
        messages=[
            {'role': 'system', 'content': 'system instruction'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt'}]},
        ]
    )

    mock_post.assert_called_once()
    called_json = mock_post.call_args.kwargs['json']
    self.assertEqual(
        called_json['messages'][0],
        {'role': 'system', 'content': 'system instruction'},
    )

  @mock.patch('requests.post', spec=True)
  def test_generate_streaming(self, mock_post):
    mock_response = mock.Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_lines.return_value = [
        (
            b'data: {"choices":[{"delta":{"content":"main "}}],'
            b' "usage":{"completion_tokens":1, "prompt_tokens":1}}'
        ),
        (
            b'data: {"choices":[{"delta":{"reasoning_content":"reasoning "}}],'
            b' "usage":{"completion_tokens":1, "prompt_tokens":0,'
            b' "completion_tokens_details":{"reasoning_tokens":1}}}'
        ),
        (
            b'data: {"choices":[{"delta":{"content":"response"}}],'
            b' "usage":{"completion_tokens":1, "prompt_tokens":0}}'
        ),
        (
            b'data: {"choices":[{"delta":{"reasoning_content":"thoughts"}}],'
            b' "usage":{"completion_tokens":0, "prompt_tokens":0,'
            b' "completion_tokens_details":{"reasoning_tokens":1}}}'
        ),
        b'data: [DONE]',
    ]
    mock_post.return_value = mock_response

    model = model_generation_http.XAIModel(
        model_name='grok-streaming',
        api_key='fake_key',
        api_options={'stream': True},
    )
    response = model._generate_streaming(
        messages=[
            {'role': 'user', 'content': [{'type': 'text', 'text': 'prompt'}]}
        ]
    )

    self.assertEqual(response.main_response, 'main response')
    self.assertEqual(
        response.main_response_and_thoughts,
        'main response\n\nreasoning thoughts',
    )
    self.assertEqual(response.generation_tokens, 3)
    self.assertEqual(response.prompt_tokens, 1)
    self.assertEqual(response.reasoning_tokens, 2)
    self.assertEqual(
        response.request_for_logging,
        {
            'model': 'grok-streaming',
            'messages': [{
                'role': 'user',
                'content': [{'type': 'text', 'text': 'prompt'}],
            }],
            'stream': True,
            'stream_options': {'include_usage': True},
        },
    )
    # Check elapsed_time separately as it's non-deterministic.
    self.assertIsInstance(
        response.response_for_logging['filtered_chunks_list'][0][
            'elapsed_time'
        ],
        float,
    )
    del response.response_for_logging['filtered_chunks_list'][0]['elapsed_time']
    self.assertEqual(
        response.response_for_logging,
        {
            'filtered_chunks_list': [
                {
                    'chunk': (
                        '{"choices":[{"delta":{"content":"main "}}],'
                        ' "usage":{"completion_tokens":1,'
                        ' "prompt_tokens":1}}'
                    ),
                    'full_content': 'main ',
                },
                {
                    'usage_chunk': {
                        'choices': [{'delta': {'content': 'main '}}],
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 1,
                        },
                    }
                },
                {
                    'usage_chunk': {
                        'choices': [
                            {'delta': {'reasoning_content': 'reasoning '}}
                        ],
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 0,
                            'completion_tokens_details': {
                                'reasoning_tokens': 1
                            },
                        },
                    }
                },
                {
                    'usage_chunk': {
                        'choices': [{'delta': {'content': 'response'}}],
                        'usage': {
                            'completion_tokens': 1,
                            'prompt_tokens': 0,
                        },
                    }
                },
                {
                    'usage_chunk': {
                        'choices': [
                            {'delta': {'reasoning_content': 'thoughts'}}
                        ],
                        'usage': {
                            'completion_tokens': 0,
                            'prompt_tokens': 0,
                            'completion_tokens_details': {
                                'reasoning_tokens': 1
                            },
                        },
                    }
                },
            ]
        },
    )
    mock_post.assert_called_once()


class TogetherAIModelTest(absltest.TestCase):

  @mock.patch('requests.post', spec=True)
  def test_generate_standard_model(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'main response'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.TogetherAIModel(
        model_name='not-deepseek', api_key='fake_key'
    )
    response = model._generate(
        content=[{'type': 'text', 'text': 'prompt'}], system_instruction=None
    )

    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='main response',
            main_response_and_thoughts='',
            request_for_logging={
                'model': 'not-deepseek',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'prompt'}],
                }],
                'max_tokens': 16384,
            },
            response_for_logging={
                'choices': [{'message': {'content': 'main response'}}]
            },
        ),
    )
    mock_post.assert_called_once()

  @mock.patch('requests.post', spec=True)
  def test_generate_deepseek_with_thoughts(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': '<think>thought</think>main'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.TogetherAIModel(
        model_name='deepseek-model', api_key='fake_key'
    )
    response = model._generate(
        content=[{'type': 'text', 'text': 'prompt'}], system_instruction=None
    )

    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='main',
            main_response_and_thoughts='<think>thought</think>main',
            request_for_logging={
                'model': 'deepseek-model',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'prompt'}],
                }],
                'max_tokens': 16384,
            },
            response_for_logging={
                'choices': [
                    {'message': {'content': '<think>thought</think>main'}}
                ]
            },
        ),
    )
    mock_post.assert_called_once()

  @mock.patch('requests.post', spec=True)
  def test_generate_deepseek_without_thoughts(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'main response'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.TogetherAIModel(
        model_name='deepseek-model', api_key='fake_key'
    )
    response = model._generate(
        content=[{'type': 'text', 'text': 'prompt'}], system_instruction=None
    )

    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='main response',
            main_response_and_thoughts='',
            request_for_logging={
                'model': 'deepseek-model',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'prompt'}],
                }],
                'max_tokens': 16384,
            },
            response_for_logging={
                'choices': [{'message': {'content': 'main response'}}]
            },
        ),
    )
    mock_post.assert_called_once()

  @mock.patch('requests.post', spec=True)
  def test_generate_with_model_options(self, mock_post):
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'main response'}}]
    }
    mock_post.return_value = mock_response

    model = model_generation_http.TogetherAIModel(
        model_name='some-model',
        api_key='fake_key',
        model_options={'max_tokens': 500},
    )
    response = model._generate(
        content=[{'type': 'text', 'text': 'prompt'}], system_instruction=None
    )

    self.assertEqual(
        response,
        model_generation.GenerateReturn(
            main_response='main response',
            main_response_and_thoughts='',
            request_for_logging={
                'model': 'some-model',
                'messages': [{
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'prompt'}],
                }],
                'max_tokens': 500,
            },
            response_for_logging={
                'choices': [{'message': {'content': 'main response'}}]
            },
        ),
    )
    mock_post.assert_called_once()
    called_json = mock_post.call_args.kwargs['json']
    self.assertEqual(called_json['max_tokens'], 500)


class PostRequestAsyncTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch('aiohttp.ClientSession.post')
  async def test_post_request_async_success(self, mock_post):
    mock_response = mock.AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {'key': 'value'}
    mock_post.return_value.__aenter__.return_value = mock_response

    session = mock.AsyncMock()
    session.post = mock_post

    result = await model_generation_http._post_request_async(
        session=session,
        name='test_request',
        url='http://test.com',
        payload={'data': 'test'},
        timeout=datetime.timedelta(seconds=10),
    )

    self.assertEqual(result.name, 'test_request')
    self.assertEqual(result.status, 200)
    self.assertEqual(result.json_response, {'key': 'value'})
    session.post.assert_called_once_with(
        'http://test.com', json={'data': 'test'}, headers=None, timeout=10
    )

  @mock.patch('aiohttp.ClientSession.post')
  async def test_post_request_async_failure(self, mock_post):
    mock_post.side_effect = aiohttp.ClientError('Test error')

    session = mock.AsyncMock()
    session.post = mock_post

    with self.assertRaises(aiohttp.ClientError):
      await model_generation_http._post_request_async(
          session=session,
          name='test_request',
          url='http://test.com',
          payload={'data': 'test'},
          timeout=datetime.timedelta(seconds=10),
      )


class PostRequestsAsyncReturnFirstSuccessTest(unittest.IsolatedAsyncioTestCase):

  @mock.patch(
      'game_arena.harness.model_generation_http._post_request_async',
      autospec=True,
  )
  async def test_first_request_succeeds(self, mock_post_async):
    # Arrange
    async def side_effect(*, name, **kwargs):
      del kwargs  # Unused
      if name == 'fast':
        await asyncio.sleep(0.01)
        return model_generation_http.PostRequestResult(
            name='fast', status=200, json_response={'result': 'fast_ok'}
        )
      if name == 'slow':
        await asyncio.sleep(0.1)
        return model_generation_http.PostRequestResult(
            name='slow', status=200, json_response={'result': 'slow_ok'}
        )
      return None

    mock_post_async.side_effect = side_effect

    requests = {
        'fast': {
            'url': 'http://fast.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=1),
        },
        'slow': {
            'url': 'http://slow.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=1),
        },
    }

    # Act
    successful_result, error_infos = (
        await model_generation_http._post_requests_async_return_first_success(
            requests
        )
    )

    # Assert
    self.assertIsNotNone(successful_result)
    self.assertEqual(successful_result.name, 'fast')
    self.assertEqual(successful_result.status, 200)
    self.assertEqual(successful_result.json_response, {'result': 'fast_ok'})
    self.assertEqual(len(error_infos), 0)

  @mock.patch(
      'game_arena.harness.model_generation_http._post_request_async',
      autospec=True,
  )
  async def test_first_request_fails_second_succeeds(self, mock_post_async):
    # Arrange
    async def side_effect(*, name, **kwargs):
      del kwargs  # Unused
      if name == 'fail':
        await asyncio.sleep(0.01)
        raise aiohttp.ClientError('Failed request')
      if name == 'succeed':
        await asyncio.sleep(0.02)
        return model_generation_http.PostRequestResult(
            name='succeed', status=200, json_response={'result': 'ok'}
        )
      return None

    mock_post_async.side_effect = side_effect

    requests = {
        'fail': {
            'url': 'http://fail.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=1),
        },
        'succeed': {
            'url': 'http://succeed.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=1),
        },
    }

    # Act
    successful_result, error_infos = (
        await model_generation_http._post_requests_async_return_first_success(
            requests
        )
    )

    # Assert
    self.assertIsNotNone(successful_result)
    self.assertEqual(successful_result.name, 'succeed')
    self.assertEqual(successful_result.status, 200)
    self.assertEqual(successful_result.json_response, {'result': 'ok'})
    self.assertEqual(len(error_infos), 1)
    self.assertEqual(error_infos[0]['request_name'], 'fail')
    self.assertEqual(error_infos[0]['error'], 'ClientError')

  @mock.patch(
      'game_arena.harness.model_generation_http._post_request_async',
      autospec=True,
  )
  async def test_all_requests_fail(self, mock_post_async):
    # Arrange
    async def side_effect(*, name, **kwargs):
      del name, kwargs  # Unused
      await asyncio.sleep(0.01)
      raise aiohttp.ClientError('Failed request')

    mock_post_async.side_effect = side_effect

    requests = {
        'fail1': {
            'url': 'http://fail1.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=1),
        },
        'fail2': {
            'url': 'http://fail2.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=1),
        },
    }

    # Act
    successful_result, error_infos = (
        await model_generation_http._post_requests_async_return_first_success(
            requests
        )
    )

    # Assert
    self.assertIsNone(successful_result)
    self.assertEqual(len(error_infos), 2)
    self.assertCountEqual(
        [e['request_name'] for e in error_infos], ['fail1', 'fail2']
    )

  async def test_no_requests(self):
    # Act
    successful_result, error_infos = (
        await model_generation_http._post_requests_async_return_first_success(
            {}
        )
    )

    # Assert
    self.assertIsNone(successful_result)
    self.assertEqual(len(error_infos), 0)

  @mock.patch(
      'game_arena.harness.model_generation_http._post_request_async',
      autospec=True,
  )
  async def test_cancellation_does_not_wait_for_long_task(
      self, mock_post_async
  ):
    # Arrange
    async def side_effect(*, name, **kwargs):
      del kwargs  # Unused
      if name == 'fast':
        await asyncio.sleep(0.1)
        return model_generation_http.PostRequestResult(
            name='fast', status=200, json_response={'result': 'fast_ok'}
        )
      if name == 'slow':
        # This task will sleep for a long time, but it should be cancelled
        # before it completes.
        await asyncio.sleep(10)
        return model_generation_http.PostRequestResult(
            name='slow', status=200, json_response={'result': 'slow_ok'}
        )
      return None

    mock_post_async.side_effect = side_effect

    requests = {
        'fast': {
            'url': 'http://fast.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=60),
        },
        'slow': {
            'url': 'http://slow.com',
            'payload': {},
            'timeout': datetime.timedelta(seconds=60),
        },
    }

    # Act & Assert
    # We expect this to complete quickly, much faster than the 10s sleep
    # in the 'slow' task.
    try:
      successful_result, error_infos = await asyncio.wait_for(
          model_generation_http._post_requests_async_return_first_success(
              requests
          ),
          timeout=1.0,
      )
    except asyncio.TimeoutError:
      self.fail(
          '_post_requests_async_return_first_success took too long to complete.'
      )

    self.assertIsNotNone(successful_result)
    self.assertEqual(successful_result.name, 'fast')
    self.assertEqual(len(error_infos), 0)


if __name__ == '__main__':
  absltest.main()
