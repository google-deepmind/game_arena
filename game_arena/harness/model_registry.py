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

"""Model registry for Kaggle Game Arena."""

import enum

from game_arena.harness import model_generation_http
from game_arena.harness import model_generation_sdk


class ModelRegistry(enum.Enum):
  """Model registry for Kaggle Game Arena."""

  # keep-sorted start
  ANTHROPIC_CLAUDE_OPUS_4 = "claude-opus-4-20250514"
  ANTHROPIC_CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
  DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1-0528"
  GEMINI_2_5_FLASH = "gemini-2.5-flash"
  GEMINI_2_5_PRO = "gemini-2.5-pro"
  KIMI_K2 = "moonshotai/Kimi-K2-Instruct"
  OPENAI_GPT_4_1 = "gpt-4.1-2025-04-14"
  OPENAI_GPT_5 = "gpt-5-2025-08-07"
  OPENAI_O3 = "o3-2025-04-16"
  OPENAI_O4_MINI = "o4-mini-2025-04-16"
  QWEN_3 = "Qwen/Qwen3-235B-A22B-Thinking-2507"
  QWEN_3_PARALLEL_THREE = "Qwen/Qwen3-235B-A22B-Thinking-2507"
  XAI_GROK_4 = "grok-4-0709"
  # keep-sorted end

  def build(self, api_key: str, **kwargs):
    match self:
      case ModelRegistry.ANTHROPIC_CLAUDE_SONNET_4:
        default_kwargs = {
            "api_options": {"stream": True},
            "model_options": {
                "max_tokens": 64000,
                "thinking": {"type": "enabled", "budget_tokens": 32000},
            },
        }
        kwargs = default_kwargs | kwargs
        return model_generation_sdk.AnthropicModel(
            model_name=self.value,
            api_key=api_key,
            **kwargs,
        )
      case ModelRegistry.ANTHROPIC_CLAUDE_OPUS_4:
        default_kwargs = {
            "api_options": {"stream": True},
            "model_options": {
                "max_tokens": 32000,
                "thinking": {"type": "enabled", "budget_tokens": 24000},
            },
        }
        kwargs = default_kwargs | kwargs
        return model_generation_sdk.AnthropicModel(
            model_name=self.value,
            api_key=api_key,
            **kwargs,
        )
      case (
          # keep-sorted start
          ModelRegistry.DEEPSEEK_R1
          | ModelRegistry.KIMI_K2
          | ModelRegistry.QWEN_3
          # keep-sorted end
      ):
        return model_generation_http.TogetherAIModel(
            model_name=self.value,
            api_key=api_key,
            **kwargs,
        )
      case ModelRegistry.QWEN_3_PARALLEL_THREE:
        return model_generation_http.TogetherAIModel(
            model_name=self.value,
            api_key=api_key,
            api_options={"parallel_attempts": 3, "timeout": 3600},
            **kwargs,
        )
      case ModelRegistry.GEMINI_2_5_FLASH | ModelRegistry.GEMINI_2_5_PRO:
        default_kwargs = {
            "api_options": {"include_thoughts": True},
        }
        kwargs = default_kwargs | kwargs
        return model_generation_sdk.AIStudioModel(
            model_name=self.value,
            api_key=api_key,
            **kwargs,
        )
      case (
          # keep-sorted start
          ModelRegistry.OPENAI_GPT_4_1
          | ModelRegistry.OPENAI_GPT_5
          | ModelRegistry.OPENAI_O3
          | ModelRegistry.OPENAI_O4_MINI
          # keep-sorted end
      ):
        return model_generation_sdk.OpenAIChatCompletionsModel(
            model_name=self.value,
            api_key=api_key,
            **kwargs,
        )
      case ModelRegistry.XAI_GROK_4:
        default_kwargs = {
            "api_options": {"stream": True},
        }
        kwargs = default_kwargs | kwargs
        return model_generation_http.XAIModel(
            model_name=self.value,
            api_key=api_key,
            **kwargs,
        )
      case _:
        raise ValueError(f"Unsupported model: {self}")
