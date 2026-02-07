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

"""Model generation implementations using HTTP POST APIs."""

import asyncio
import base64
import dataclasses
import datetime
import json
import multiprocessing
import os
import threading
import time
from typing import Any, Mapping, Sequence
from absl import logging
import aiohttp
from game_arena.harness import model_generation
import requests


def _create_image_text_content(
    model_input: model_generation.ModelImageTextInput,
):
  """Creates content payload with text followed by image."""
  base64_image = base64.b64encode(model_input.prompt_image_bytes).decode(
      "utf-8"
  )
  image_content = {
      "type": "image_url",
      "image_url": {
          "url": (
              f"data:{model_input.prompt_image_mime_type};base64,{base64_image}"
          )
      },
  }
  if (
      model_input.prompt_text_preceding_image is not None
      and model_input.prompt_text_following_image is not None
  ):
    assert not model_input.prompt_text
    content = [
        {"type": "text", "text": model_input.prompt_text_preceding_image},
        image_content,
        {"type": "text", "text": model_input.prompt_text_following_image},
    ]
  else:
    content = [
        {"type": "text", "text": model_input.prompt_text},
        image_content,
    ]
  return content


class TogetherAIModel(model_generation.MultimodalModel):
  """Wrapper for access to models served by Together.AI."""

  def __init__(
      self,
      model_name: str,
      *,
      model_options: Mapping[str, Any] | None = None,
      api_options: Mapping[str, Any] | None = None,
      api_key: str | None = None,
  ):
    super().__init__(
        model_name, model_options=model_options, api_options=api_options
    )
    # If API key is None, defaults to TOGETHER_API_KEY in environment.
    if api_key is None:
      try:
        api_key = os.environ["TOGETHER_API_KEY"]
      except KeyError as e:
        logging.error(
            "TOGETHER_API_KEY environment variable not set. Please set it to"
            " use %s.",
            self._model_name,
        )
        raise e
    self._headers = {"Authorization": f"Bearer {api_key}"}

  # TODO(google-deepmind): Add error handling.
  def _generate(
      self,
      content: Sequence[Mapping[str, Any]],
      system_instruction: str | None = None,
  ) -> model_generation.GenerateReturn:
    messages = []
    if system_instruction is not None:
      messages.append({"role": "system", "content": system_instruction})
    messages.append({
        "role": "user",
        "content": content,
    })

    if self._model_options is None:
      self._model_options = {}
    if self._api_options is None:
      self._api_options = {}

    request_options = {}

    if "max_tokens" in self._model_options:
      max_sampling_length = self._model_options["max_tokens"]
    else:
      match self._model_name:
        case "deepseek-ai/DeepSeek-R1-0528-tput":
          max_context_length = 163839
        case "deepseek-ai/DeepSeek-R1":
          max_context_length = 163839
        case "deepseek-ai/DeepSeek-V3":
          max_context_length = 163839
        case "moonshotai/Kimi-K2-Instruct":
          max_context_length = 128000
        case "Qwen/Qwen3-235B-A22B-Thinking-2507":
          max_context_length = 262144
        case "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":
          max_context_length = 262144
        case "Qwen/Qwen3-235B-A22B-fp8-tput":
          max_context_length = 40960
        case _:
          logging.warning(
              "max_context_length unknown for model %s. Defaulting to 32768.",
              self._model_name,
          )
          max_context_length = 32768
      max_sampling_length = max_context_length - self._model_options.get(
          "max_prompt_length", 16384
      )

    request_options["max_tokens"] = max_sampling_length

    for supported_model_option in [
        "temperature",
        "top_k",
        "top_p",
    ]:
      if supported_model_option in self._model_options:
        request_options[supported_model_option] = self._model_options[
            supported_model_option
        ]
    request = {
        "model": self._model_name,
        "messages": messages,
    } | request_options

    parallel_attempts = self._api_options.get("parallel_attempts", 1)
    if parallel_attempts > 1:
      payload_details = {
          "url": "https://api.together.xyz/v1/chat/completions",
          "payload": request,
          "headers": self._headers,
          "timeout": datetime.timedelta(
              seconds=self._api_options.get("timeout", 3600)
          ),
      }
      name_to_payload_details = {
          f"{self._model_name}_attempt_{i}": payload_details
          for i in range(parallel_attempts)
      }
      maybe_successful_result, error_infos = asyncio.run(
          _post_requests_async_return_first_success(name_to_payload_details)
      )
      if maybe_successful_result is None:
        raise RuntimeError(
            f"All {parallel_attempts} attempts failed: {error_infos}"
        )
      completion = maybe_successful_result.json_response
    else:
      # Splitting the request into a thread to be able to inspect the status of
      # the request and print the progress.
      q = multiprocessing.Queue()

      def do_request():
        try:
          assert self._api_options is not None
          response = requests.post(
              "https://api.together.xyz/v1/chat/completions",
              json=request,
              headers=self._headers,
              timeout=self._api_options.get("timeout", 3600),
          )
          response.raise_for_status()
          q.put({"status_code": response.status_code, "json": response.json()})
        except requests.exceptions.RequestException as e:
          logging.exception("Request to Together.AI failed: %s", e)
          q.put(e)

      thread = threading.Thread(target=do_request)
      thread.start()

      last_wait_message_time = time.monotonic()
      while thread.is_alive():
        thread.join(timeout=1.0)
        current_time = time.monotonic()
        if current_time - last_wait_message_time >= 120:  # 2 minutes
          print(f"Still waiting on {self._model_name}")
          last_wait_message_time = current_time

      result = q.get()
      if isinstance(result, Exception):
        raise result

      assert result["status_code"] == 200

      completion = result["json"]

    content = completion["choices"][0]["message"]["content"]
    if content is None:
      logging.warning(
          "Together.AI Completion return content is None. Returning empty"
          " string. Request: %s",
          request,
      )
      content = ""

    main_response = content
    main_response_and_thoughts = ""

    if "deepseek" in self._model_name.lower():
      maybe_separated = model_generation.separate_main_response_and_thoughts(
          content
      )
      if maybe_separated is not None:
        main_response = maybe_separated[0]
        main_response_and_thoughts = content

    generation_tokens = None
    prompt_tokens = None
    if "usage" in completion:
      usage = completion["usage"]
      generation_tokens = usage["completion_tokens"]
      prompt_tokens = usage["prompt_tokens"]

    return model_generation.GenerateReturn(
        main_response=main_response,
        main_response_and_thoughts=main_response_and_thoughts,
        request_for_logging=request,
        response_for_logging=completion,
        generation_tokens=generation_tokens,
        prompt_tokens=prompt_tokens,
    )

  def generate_with_text_input(
      self, model_input: model_generation.ModelTextInput
  ) -> model_generation.GenerateReturn:
    content = [{"type": "text", "text": model_input.prompt_text}]
    return self._generate(content, model_input.system_instruction)

  def generate_with_image_text_input(
      self, model_input: model_generation.ModelImageTextInput
  ) -> model_generation.GenerateReturn:
    content = _create_image_text_content(model_input)
    return self._generate(content, model_input.system_instruction)


class XAIModel(model_generation.MultimodalModel):
  """Wrapper to call the xAI API. Supports both streaming and non-streaming."""

  def __init__(
      self,
      model_name: str,
      *,
      model_options: Mapping[str, Any] | None = None,
      api_options: Mapping[str, Any] | None = None,
      api_key: str | None = None,
      debug: bool = False,
  ):
    super().__init__(
        model_name, model_options=model_options, api_options=api_options
    )
    if self._model_options is None:
      self._model_options = {}
    if self._api_options is None:
      self._api_options = {}

    if api_key is None:
      api_key = os.environ["XAI_API_KEY"]

    if api_key is None:
      raise ValueError("XAI API key not found.")

    self._headers = {"Authorization": f"Bearer {api_key}"}
    self._debug = debug
    self._stream = self._api_options.get("stream", True)

  # TODO(google-deepmind): Add error handling.
  def _post_request(self, request: Mapping[str, Any], stream: bool):
    """Sends a POST request to the xAI API and handles errors."""
    assert self._api_options is not None
    timeout = self._api_options.get("timeout", 20 * 60)
    try:
      response = requests.post(
          "https://api.x.ai/v1/chat/completions",
          json=request,
          headers=self._headers,
          stream=stream,
          timeout=timeout,
      )
      response.raise_for_status()

      return response
    except requests.exceptions.HTTPError as e:
      if e.response is not None:
        print("Error during XAI request: ", flush=True)
        print(f"Status code: {e.response.status_code}", flush=True)
        print(f"Reason: {e.response.reason}", flush=True)
        print(f"Headers: {e.response.headers}", flush=True)
        print(f"Response text: {e.response.text}", flush=True)
        print(request)
        if e.response.status_code == 400:
          raise model_generation.DoNotRetryError(
              str(e),
              info={"request": request, "response": e.response},
          ) from e
      raise

  def _generate(
      self,
      messages: Sequence[Mapping[str, Any]],
  ) -> model_generation.GenerateReturn:
    request = {
        "model": self._model_name,
        "messages": messages,
        "stream": False,
        "stream_options": {"include_usage": True},
    }
    for option in ["temperature", "top_p", "top_k"]:
      if option in self._model_options:
        request[option] = self._model_options[option]

    response = self._post_request(request, stream=False)
    completion = response.json()
    full_content = completion["choices"][0]["message"]["content"]

    if full_content is None:
      logging.warning(
          "xAI Completion return content is None. Returning empty string."
          " Request: %s",
          request,
      )
      full_content = ""

    full_reasoning_content = ""
    if "reasoning_content" in completion["choices"][0]["message"]:
      full_reasoning_content = completion["choices"][0]["message"][
          "reasoning_content"
      ]
    total_generation_tokens = None
    total_prompt_tokens = None
    total_reasoning_tokens = None
    if "usage" in completion:
      usage = completion["usage"]
      if "completion_tokens" in usage:
        total_generation_tokens = usage["completion_tokens"]
      if "prompt_tokens" in usage:
        total_prompt_tokens = usage["prompt_tokens"]
      if (
          "completion_tokens_details" in usage
          and "reasoning_tokens" in usage["completion_tokens_details"]
      ):
        total_reasoning_tokens = usage["completion_tokens_details"][
            "reasoning_tokens"
        ]
    main_response_and_thoughts = (
        full_content + ("\n\n" + full_reasoning_content)
        if full_reasoning_content
        else ""
    )
    return model_generation.GenerateReturn(
        main_response=full_content,
        main_response_and_thoughts=main_response_and_thoughts,
        request_for_logging=request,
        response_for_logging=completion,
        generation_tokens=total_generation_tokens,
        prompt_tokens=total_prompt_tokens,
        reasoning_tokens=total_reasoning_tokens,
    )

  def _print_debug(self, message: str):
    if self._debug:
      print(message)

  def _log_chunk_received(
      self,
      start_time: float,
      elapsed_time: float,
      full_content: str,
      delta_content: str,
      is_thinking_chunk: bool,
      last_full_content_print_time: float,
      current_time: float,
  ):
    """Logs information about received chunks."""
    time_since_last_full_print = current_time - last_full_content_print_time
    if (
        last_full_content_print_time == start_time
        or time_since_last_full_print >= 120
    ):
      if is_thinking_chunk:
        self._print_debug(
            f"[{self._model_name}] Received thinking chunk, elapsed:"
            f" {elapsed_time:.2f}s, content_len:"
            f" {len(full_content)}, full_content: {full_content}"
        )
      else:
        self._print_debug(
            f"[{self._model_name}] Received chunk, elapsed:"
            f" {elapsed_time:.2f}s, content_len:"
            f" {len(full_content)}, full_content: {full_content}"
        )
      return True
    else:
      if is_thinking_chunk:
        self._print_debug(
            f"[{self._model_name}] Received thinking chunk, elapsed:"
            f" {elapsed_time:.2f}s, chunk: {delta_content}"
        )
      else:
        self._print_debug(
            f"[{self._model_name}] Received chunk, elapsed:"
            f" {elapsed_time:.2f}s, chunk: {delta_content}"
        )
      return False

  def _generate_streaming(
      self,
      messages: Sequence[Mapping[str, Any]],
  ) -> model_generation.GenerateReturn:
    request = {
        "model": self._model_name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    full_content = ""
    full_reasoning_content = ""
    total_generation_tokens = 0
    total_prompt_tokens = 0
    total_reasoning_tokens = 0
    response_for_logging = []
    start_time = time.monotonic()
    last_full_content_print_time = start_time
    self._print_debug(f"[{self._model_name}] Starting streaming completion.")

    response = self._post_request(request, stream=True)
    try:
      for chunk in response.iter_lines():
        if chunk:
          current_time = time.monotonic()
          elapsed_time = current_time - start_time
          if chunk.startswith(b"data: "):
            chunk_data = chunk[len(b"data: ") :].decode("utf-8")
            if chunk_data == "[DONE]":
              self._print_debug(
                  f"[{self._model_name}] Streaming finished, elapsed:"
                  f" {elapsed_time:.2f}s"
              )
              break
            try:
              json_data = json.loads(chunk_data)
              if "choices" in json_data and json_data["choices"]:
                delta = json_data["choices"][0].get("delta", {})
                if "content" in delta:
                  delta_content = delta["content"]
                  full_content += delta_content
                  is_full_content_print = self._log_chunk_received(
                      start_time=start_time,
                      elapsed_time=elapsed_time,
                      full_content=full_content,
                      delta_content=delta_content,
                      is_thinking_chunk=False,
                      last_full_content_print_time=last_full_content_print_time,
                      current_time=current_time,
                  )
                  if is_full_content_print:
                    # Log a chunk only every 2 minutes.
                    response_for_logging.append({
                        "chunk": chunk_data,
                        "elapsed_time": elapsed_time,
                        "full_content": full_content,
                    })
                    last_full_content_print_time = current_time
                if "reasoning_content" in delta:
                  delta_reasoning_content = delta["reasoning_content"]
                  full_reasoning_content += delta_reasoning_content
                  is_full_content_print = self._log_chunk_received(
                      start_time=start_time,
                      elapsed_time=elapsed_time,
                      full_content=full_reasoning_content,
                      delta_content=delta_reasoning_content,
                      is_thinking_chunk=True,
                      last_full_content_print_time=last_full_content_print_time,
                      current_time=current_time,
                  )
                  if is_full_content_print:
                    response_for_logging.append({
                        "chunk": chunk_data,
                        "elapsed_time": elapsed_time,
                        "full_content": full_reasoning_content,
                    })
                    last_full_content_print_time = current_time

              if "usage" in json_data:
                usage = json_data["usage"]
                response_for_logging.append({"usage_chunk": json_data})
                if "completion_tokens" in usage:
                  total_generation_tokens += usage["completion_tokens"]
                if "prompt_tokens" in usage:
                  total_prompt_tokens += usage["prompt_tokens"]
                if (
                    "completion_tokens_details" in usage
                    and "reasoning_tokens" in usage["completion_tokens_details"]
                ):
                  total_reasoning_tokens += usage["completion_tokens_details"][
                      "reasoning_tokens"
                  ]

            except json.JSONDecodeError as e:
              logging.warning(
                  "[%s] JSON decode error for chunk: %s, elapsed: %.2fs",
                  self._model_name,
                  chunk_data,
                  elapsed_time,
              )
              raise e
    finally:
      response.close()

    main_response_and_thoughts = (
        full_content + ("\n\n" + full_reasoning_content)
        if full_reasoning_content
        else ""
    )
    return model_generation.GenerateReturn(
        main_response=full_content,
        main_response_and_thoughts=main_response_and_thoughts,
        request_for_logging=request,
        response_for_logging={"filtered_chunks_list": response_for_logging},
        generation_tokens=total_generation_tokens,
        prompt_tokens=total_prompt_tokens,
        reasoning_tokens=total_reasoning_tokens,
    )

  def generate_with_text_input(
      self, model_input: model_generation.ModelTextInput
  ) -> model_generation.GenerateReturn:
    messages = []
    if model_input.system_instruction is not None:
      messages.append(
          {"role": "system", "content": model_input.system_instruction}
      )

    messages.append({
        "role": "user",
        "content": model_input.prompt_text,
    })

    if self._stream:
      return self._generate_streaming(messages)
    else:
      return self._generate(messages)

  def generate_with_image_text_input(
      self, model_input: model_generation.ModelImageTextInput
  ) -> model_generation.GenerateReturn:
    if "grok-3" in self._model_name:
      raise model_generation.UnsupportedCapabilityError(
          f"Model {self._model_name} does not support image input."
      )
    messages = []
    if model_input.system_instruction is not None:
      messages.append(
          {"role": "system", "content": model_input.system_instruction}
      )
    content = _create_image_text_content(model_input)
    messages.append({
        "role": "user",
        "content": content,
    })
    if self._stream:
      return self._generate_streaming(messages)
    else:
      return self._generate(messages)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PostRequestResult:
  name: str
  status: int
  json_response: Mapping[str, Any]


async def _post_request_async(
    *,
    session: aiohttp.ClientSession,
    name: str,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: datetime.timedelta,
) -> PostRequestResult:
  """Posts a request asynchronously and returns the result."""
  print(f"Starting POST for '{name}' to {url}")
  try:
    async with session.post(
        url, json=payload, headers=headers, timeout=timeout.total_seconds()
    ) as response:
      print(f"Finished POST for '{name}' with status: {response.status}")
      # Raise an exception for bad status codes (4xx or 5xx)
      response.raise_for_status()

      json_response = await response.json()
      return PostRequestResult(
          name=name, status=response.status, json_response=json_response
      )

  except asyncio.CancelledError:
    print(f"POST for '{name}' was cancelled.")
    raise
  except Exception as e:
    print(f"POST for '{name}' failed: {type(e).__name__}")
    # Re-raise the exception so the main loop knows it failed
    raise


async def _post_requests_async_return_first_success(
    name_to_payload_details: Mapping[str, Mapping[str, Any]],
) -> tuple[PostRequestResult | None, Sequence[Mapping[str, Any]]]:
  """Posts requests asynchronously and returns the first successful result."""
  error_infos = []
  successful_result = None

  async def _wrapper(
      *,
      session: aiohttp.ClientSession,
      name: str,
      url: str,
      payload: Mapping[str, Any],
      headers: Mapping[str, str] | None = None,
      timeout: datetime.timedelta,
  ) -> tuple[str, PostRequestResult | None, Exception | None]:
    """Wraps the async post request to return name and exception."""
    try:
      result = await _post_request_async(
          session=session,
          name=name,
          url=url,
          payload=payload,
          headers=headers,
          timeout=timeout,
      )
      return name, result, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      return name, None, e

  async with aiohttp.ClientSession() as session:
    if not name_to_payload_details:
      return None, []

    tasks = [
        asyncio.create_task(
            _wrapper(session=session, name=name, **payload_details)
        )
        for name, payload_details in name_to_payload_details.items()
    ]

    for future in asyncio.as_completed(tasks):
      name, result, exc = await future
      if exc is None:
        successful_result = result
        print(f"\n--- First SUCCESSFUL POST received from '{name}' ---")
        break  # We have a winner
      else:
        print(f"Ignoring failed task '{name}'. Waiting for the next one...")
        error_infos.append({
            "request_name": name,
            "error": type(exc).__name__,
            "message": str(exc),
        })
    else:
      print("\n--- All POST requests failed. No successful result found. ---\n")

    # Cancel remaining tasks
    for task in tasks:
      if not task.done():
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    return successful_result, error_infos


class OpenAIGenericAPIModel(model_generation.MultimodalModel):
  """Wrapper to call the generic OpenAI API."""

  def __init__(
      self,
      model_name: str,
      api_endpoint: str = None,
      local_model: bool = False,
      *,
      model_options: Mapping[str, Any] | None = None,
      api_options: Mapping[str, Any] | None = None,
      api_key: str | None = None,
      debug: bool = False,
  ):
    super().__init__(
        model_name, model_options=model_options, api_options=api_options
    )
    if self._model_options is None:
      self._model_options = {}
    if self._api_options is None:
      self._api_options = {}

    if api_endpoint is None:
      raise ValueError("api_endpoint url is not set.")

    if not api_endpoint.startswith(('http://', 'https://')):
      raise ValueError("api_endpoint must be a valid HTTP/HTTPS URL")

    if api_key is None:
      if local_model:
        api_key = ""
      else:
        raise ValueError("API key is required for non-local models.")

    self.api_endpoint = api_endpoint
    if api_key:
      self._headers = {"Authorization": f"Bearer {api_key}"}
    else:
      self._headers = {}

    self._debug = debug
    self._stream = self._api_options.get("stream", True)
    self._image_support = self._api_options.get("image_support", False)

  # TODO(google-deepmind): Add error handling.
  def _post_request(self, request: Mapping[str, Any], stream: bool):
    """Sends a POST request to the API and handles errors."""
    timeout = self._api_options.get("timeout", 20 * 60)
    try:
      response = requests.post(
          self.api_endpoint,
          json=request,
          headers=self._headers,
          stream=stream,
          timeout=timeout,
      )
      response.raise_for_status()

      return response
    except requests.exceptions.HTTPError as e:
      if e.response is not None:
        print(f"Error during {self._model_name}  request: ", flush=True)
        print(f"Status code: {e.response.status_code}", flush=True)
        print(f"Reason: {e.response.reason}", flush=True)
        print(f"Headers: {e.response.headers}", flush=True)
        print(f"Response text: {e.response.text}", flush=True)
        print(request)
        if e.response.status_code == 400:
          raise model_generation.DoNotRetryError(
              str(e),
              info={"request": request, "response": e.response},
          ) from e
      raise

  def _generate(
      self,
      messages: Sequence[Mapping[str, Any]],
  ) -> tournament_util.GenerateReturn:
    request = {
        "model": self._model_name,
        "messages": messages,
        "stream": False,
        "stream_options": {"include_usage": True},
    }
    for option in ["temperature", "top_p", "top_k", "max_tokens"]:
      if option in self._model_options:
        request[option] = self._model_options[option]

    response = self._post_request(request, stream=False)
    completion = response.json()
    full_content = completion["choices"][0]["message"]["content"]

    if full_content is None:
      logging.warning(
          f"{self._model_name} Completion return content is None. Returning empty string."
          " Request: %s",
          request,
      )
      full_content = ""

    full_reasoning_content = ""
    if "reasoning_content" in completion["choices"][0]["message"]:
      full_reasoning_content = completion["choices"][0]["message"][
          "reasoning_content"
      ]
    total_generation_tokens = None
    total_prompt_tokens = None
    total_reasoning_tokens = None
    if "usage" in completion:
      usage = completion["usage"]
      if "completion_tokens" in usage:
        total_generation_tokens = usage["completion_tokens"]
      if "prompt_tokens" in usage:
        total_prompt_tokens = usage["prompt_tokens"]
      if (
          "completion_tokens_details" in usage
          and "reasoning_tokens" in usage["completion_tokens_details"]
      ):
        total_reasoning_tokens = usage["completion_tokens_details"][
            "reasoning_tokens"
        ]
    main_response_and_thoughts = (
        full_content + ("\n\n" + full_reasoning_content)
        if full_reasoning_content
        else ""
    )
    return tournament_util.GenerateReturn(
        main_response=full_content,
        main_response_and_thoughts=main_response_and_thoughts,
        request_for_logging=request,
        response_for_logging=completion,
        generation_tokens=total_generation_tokens,
        prompt_tokens=total_prompt_tokens,
        reasoning_tokens=total_reasoning_tokens,
    )

  def _print_debug(self, message: str):
    if self._debug:
      print(message)

  def _log_chunk_received(
      self,
      start_time: float,
      elapsed_time: float,
      full_content: str,
      delta_content: str,
      is_thinking_chunk: bool,
      last_full_content_print_time: float,
      current_time: float,
  ):
    """Logs information about received chunks."""
    time_since_last_full_print = current_time - last_full_content_print_time
    if (
        last_full_content_print_time == start_time
        or time_since_last_full_print >= 120
    ):
      if is_thinking_chunk:
        self._print_debug(
            f"[{self._model_name}] Received thinking chunk, elapsed:"
            f" {elapsed_time:.2f}s, content_len:"
            f" {len(full_content)}, full_content: {full_content}"
        )
      else:
        self._print_debug(
            f"[{self._model_name}] Received chunk, elapsed:"
            f" {elapsed_time:.2f}s, content_len:"
            f" {len(full_content)}, full_content: {full_content}"
        )
      return True
    else:
      if is_thinking_chunk:
        self._print_debug(
            f"[{self._model_name}] Received thinking chunk, elapsed:"
            f" {elapsed_time:.2f}s, chunk: {delta_content}"
        )
      else:
        self._print_debug(
            f"[{self._model_name}] Received chunk, elapsed:"
            f" {elapsed_time:.2f}s, chunk: {delta_content}"
        )
      return False

  def _generate_streaming(
      self,
      messages: Sequence[Mapping[str, Any]],
  ) -> tournament_util.GenerateReturn:
    request = {
        "model": self._model_name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for option in ["temperature", "top_p", "top_k", "max_tokens"]:
      if option in self._model_options:
        request[option] = self._model_options[option]

    full_content = ""
    full_reasoning_content = ""
    total_generation_tokens = 0
    total_prompt_tokens = 0
    total_reasoning_tokens = 0
    response_for_logging = []
    start_time = time.monotonic()
    last_full_content_print_time = start_time
    self._print_debug(f"[{self._model_name}] Starting streaming completion.")

    response = self._post_request(request, stream=True)
    try:
      for chunk in response.iter_lines():
        if chunk:
          current_time = time.monotonic()
          elapsed_time = current_time - start_time
          if chunk.startswith(b"data: "):
            chunk_data = chunk[len(b"data: ") :].decode("utf-8")
            if chunk_data == "[DONE]":
              self._print_debug(
                  f"[{self._model_name}] Streaming finished, elapsed:"
                  f" {elapsed_time:.2f}s"
              )
              break
            try:
              json_data = json.loads(chunk_data)
              if "choices" in json_data and json_data["choices"]:
                delta = json_data["choices"][0].get("delta", {})
                if "content" in delta:
                  delta_content = delta["content"]
                  full_content += delta_content
                  is_full_content_print = self._log_chunk_received(
                      start_time=start_time,
                      elapsed_time=elapsed_time,
                      full_content=full_content,
                      delta_content=delta_content,
                      is_thinking_chunk=False,
                      last_full_content_print_time=last_full_content_print_time,
                      current_time=current_time,
                  )
                  if is_full_content_print:
                    # Log a chunk only every 2 minutes.
                    response_for_logging.append({
                        "chunk": chunk_data,
                        "elapsed_time": elapsed_time,
                        "full_content": full_content,
                    })
                    last_full_content_print_time = current_time
                if "reasoning_content" in delta:
                  delta_reasoning_content = delta["reasoning_content"]
                  full_reasoning_content += delta_reasoning_content
                  is_full_content_print = self._log_chunk_received(
                      start_time=start_time,
                      elapsed_time=elapsed_time,
                      full_content=full_reasoning_content,
                      delta_content=delta_reasoning_content,
                      is_thinking_chunk=True,
                      last_full_content_print_time=last_full_content_print_time,
                      current_time=current_time,
                  )
                  if is_full_content_print:
                    response_for_logging.append({
                        "chunk": chunk_data,
                        "elapsed_time": elapsed_time,
                        "full_content": full_reasoning_content,
                    })
                    last_full_content_print_time = current_time

              if "usage" in json_data:
                usage = json_data["usage"]
                response_for_logging.append({"usage_chunk": json_data})
                if "completion_tokens" in usage:
                  total_generation_tokens += usage["completion_tokens"]
                if "prompt_tokens" in usage:
                  total_prompt_tokens += usage["prompt_tokens"]
                if (
                    "completion_tokens_details" in usage
                    and "reasoning_tokens" in usage["completion_tokens_details"]
                ):
                  total_reasoning_tokens += usage["completion_tokens_details"][
                      "reasoning_tokens"
                  ]

            except json.JSONDecodeError as e:
              logging.warning(
                  "[%s] JSON decode error for chunk: %s, elapsed: %.2fs",
                  self._model_name,
                  chunk_data,
                  elapsed_time,
              )
              raise e
    finally:
      response.close()

    main_response_and_thoughts = (
        full_content + ("\n\n" + full_reasoning_content)
        if full_reasoning_content
        else ""
    )
    return tournament_util.GenerateReturn(
        main_response=full_content,
        main_response_and_thoughts=main_response_and_thoughts,
        request_for_logging=request,
        response_for_logging={"filtered_chunks_list": response_for_logging},
        generation_tokens=total_generation_tokens,
        prompt_tokens=total_prompt_tokens,
        reasoning_tokens=total_reasoning_tokens,
    )

  def generate_with_text_input(
      self, model_input: tournament_util.ModelTextInput
  ) -> tournament_util.GenerateReturn:
    messages = []
    if model_input.system_instruction is not None:
      messages.append(
          {"role": "system", "content": model_input.system_instruction}
      )

    messages.append({
        "role": "user",
        "content": model_input.prompt_text,
    })

    if self._stream:
      return self._generate_streaming(messages)
    else:
      return self._generate(messages)

  def generate_with_image_text_input(
      self, model_input: tournament_util.ModelImageTextInput
  ) -> tournament_util.GenerateReturn:
    if not self._image_support:
      raise model_generation.UnsupportedCapabilityError(
          f"Model {self._model_name} does not support image input."
      )
    if not model_input.prompt_image_bytes:
      raise ValueError("Image input is empty")
    if not hasattr(model_input, 'prompt_image_mime_type'):
      raise ValueError("Image MIME type is required")
    messages = []
    if model_input.system_instruction is not None:
      messages.append(
          {"role": "system", "content": model_input.system_instruction}
      )
    content = _create_image_text_content(model_input)
    messages.append({
        "role": "user",
        "content": content,
    })
    if self._stream:
      return self._generate_streaming(messages)
    else:
      return self._generate(messages)
