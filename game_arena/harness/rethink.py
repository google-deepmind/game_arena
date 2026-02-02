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

"""Rethinking sampler that prompts again if an illegal action is generated."""

import enum
from typing import Any, Callable, Generic, Mapping, TypeVar

from absl import logging
from game_arena.harness import base_game
from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import prompt_templates
from game_arena.harness import prompts
from game_arena.harness import samplers
from game_arena.harness import telemetry


_TELEMETRY = telemetry.get_logger(__name__)


def _sample_parse_return(
    model_input: Any,
    generate_fn: Callable[[Any], model_generation.GenerateReturn],
    parser: parsers.TextParser,
) -> tuple[str | None, model_generation.GenerateReturn]:
  generate_return = generate_fn(model_input)
  parsed_action = parser.parse(
      parsers.TextParserInput(text=generate_return.main_response)
  )
  return parsed_action, generate_return


class RethinkStrategy(enum.Enum):
  RETHINK = "rethink"
  RETHINK_WITH_ENV = "rethink_with_env"
  RETHINK_WITH_ENV_ILLEGAL_HISTORY = "rethink_with_env_illegal_history"
  RETHINK_WITH_ENV_RULE = "rethink_with_env_rule"


StrategyT = TypeVar("StrategyT", bound=enum.Enum)


class RethinkSampler(samplers.Sampler, Generic[StrategyT]):
  """Samples action by prompting again if an illegal action is generated."""

  def _is_rethink_strategy(self, strategy: StrategyT):
    return strategy in (
        RethinkStrategy.RETHINK,
        RethinkStrategy.RETHINK_WITH_ENV,
        RethinkStrategy.RETHINK_WITH_ENV_ILLEGAL_HISTORY,
        RethinkStrategy.RETHINK_WITH_ENV_RULE,
    )

  def __init__(
      self,
      model: model_generation.Model,
      # TODO(google-deepmind): put these into a config class:
      strategy: StrategyT,
      num_max_rethinks: int,
      move_parser: parsers.TextParser,
      legality_parser: parsers.TextParser,
      prompt_generator: prompt_generation.PromptGeneratorSupportsText,
      rethink_template: str | None,
      game_adapter: base_game.BaseGameEnvAdapter,
  ):
    if not self._is_rethink_strategy(strategy):
      raise ValueError(f"Unsupported strategy: {strategy}")
    super().__init__(model)
    self._strategy: StrategyT = strategy
    self._num_max_rethinks = num_max_rethinks
    self._move_parser = move_parser
    self._legality_parser = legality_parser
    self._prompt_generator = prompt_generator
    self._rethink_template = rethink_template
    self._game_adapter = game_adapter

  def is_rethink_template(self, prompt_template: str | None) -> bool:
    """Separate method to allow for per-game overrides."""
    return prompt_template in prompts.RETHINK_PROMPTS

  def explain_illegal_move(self, move_str):
    raise NotImplementedError(
        "explain_illegal_move is not implemented for"
        f" {self._game_adapter.game_short_name}"
    )

  def sample_action_with_text_and_state_input(
      self,
      game_adapter: base_game.BaseGameEnvAdapter,
      prompt_template: str | None = None,
      **prompt_substitutions,
  ) -> samplers.SamplerOutput:
    return self._sample_action(
        self._model.generate_with_text_input,
        game_adapter,
        prompt_template,
        **prompt_substitutions,
    )

  def sample_action_with_image_text_and_state_input(
      self,
      game_adapter: base_game.BaseGameEnvAdapter,
      prompt_template: str | None = None,
      prompt_configuration: Mapping[str, Any] | None = None,
      **prompt_substitutions,
  ) -> samplers.SamplerOutput:
    # TODO(google-deepmind): flatten MultimodalModel to just Model so we don't have
    # to do this separate dispatch and isinstance check.
    if isinstance(self._model, model_generation.MultimodalModel):
      return self._sample_action(
          self._model.generate_with_image_text_input,
          game_adapter,
          prompt_template,
          prompt_configuration=prompt_configuration,
          **prompt_substitutions,
      )
    else:
      raise ValueError("Multimodal model expected but not provided.")

  def _sample_action(
      self,
      model_generate_fn: Callable[[Any], model_generation.GenerateReturn],
      game_adapter: base_game.BaseGameEnvAdapter,
      prompt_template: str | None = None,
      prompt_configuration: Mapping[str, Any] | None = None,
      **prompt_substitutions,
  ) -> samplers.SamplerOutput:
    if not self.is_rethink_template(prompt_template):
      raise ValueError("Unsupported prompt template for rethinking.")

    # The zeroth prompt does not have any rethinking text substituted into it:
    prompt_substitutions["rethink_prompt"] = ""

    if prompt_template in prompts.IMAGE_TEXT_PROMPTS:
      if isinstance(
          self._prompt_generator,
          prompt_generation.PromptGeneratorSupportsImageText,
      ):
        prompt_fn = self._prompt_generator.generate_prompt_with_image_text
        prompt_args = {
            "prompt_template": prompt_template,
            "prompt_configuration": prompt_configuration,
            "game_short_name": game_adapter.game_short_name,
            **prompt_substitutions,
        }
      else:
        raise ValueError(
            "A prompt that expects image text input was provided, but the"
            " prompt generator does not support it."
        )
    else:
      if isinstance(
          self._prompt_generator,
          prompt_generation.PromptGeneratorSupportsText,
      ):
        prompt_fn = self._prompt_generator.generate_prompt_with_text_only
        prompt_args = {
            "prompt_template": prompt_template,
            "game_short_name": self._game_adapter.game_short_name,
            **prompt_substitutions,
        }
      else:
        raise ValueError(
            "A prompt that expects text input was provided, but the"
            " prompt generator does not support it."
        )

    parsed_action = None
    parsed_action_history: list[str] = []
    maybe_legal_action = None
    num_attempts = 0
    generate_returns = []
    auxiliary_outputs = {}

    # Zeroth attempt is not counted against the max number of rethinks:
    while (
        parsed_action is None or maybe_legal_action is None
    ) and num_attempts < self._num_max_rethinks + 1:
      if num_attempts > 0:
        logging.info(
            "Rethinking attempt %d with rethink prompt substitution: %s",
            num_attempts,
            prompt_args["rethink_prompt"],
        )
        _TELEMETRY(rethinking_attempt={"number": num_attempts})
      else:
        logging.info("Initial attempt before rethinking.")
        _TELEMETRY(initial_attempt=True)
      complete_prompt = prompt_fn(**prompt_args)
      parsed_action, generate_return = _sample_parse_return(
          complete_prompt,
          model_generate_fn,
          self._move_parser,
      )
      generate_returns.append(generate_return)
      auxiliary_outputs[f"parsed_action_attempt_{num_attempts}"] = parsed_action
      auxiliary_outputs[f"rethink_prompt_attempt_{num_attempts}"] = prompt_args[
          "rethink_prompt"
      ]
      parsed_action_history.append(str(parsed_action))
      # TextParserInput does not accept None as input so we convert it to an
      # empty string if it is None. This enables the legality parser to return
      # default legal actions if the parsed action is None.
      maybe_legal_action = self._legality_parser.parse(
          parsers.TextParserInput(
              text="" if parsed_action is None else parsed_action,
              state_str=game_adapter.get_readable_state(),
              legal_moves=self._game_adapter.legal_actions,
              player_number=game_adapter.player_number,
          )
      )
      auxiliary_outputs[f"maybe_legal_action_attempt_{num_attempts}"] = (
          maybe_legal_action
      )

      if maybe_legal_action is not None:
        break  # We have a legal action and can return now.
      else:
        auxiliary_outputs[f"maybe_legal_action_attempt_{num_attempts}"] = None

      # Prepare for the next attempt by generating the rethinking part of the
      # prompt:
      num_attempts += 1
      match self._strategy:
        case RethinkStrategy.RETHINK:
          if self._rethink_template is None:
            rethink_prompt = ""
          else:
            raise ValueError("Rethink template should be initialized as None.")
        case RethinkStrategy.RETHINK_WITH_ENV:
          if self._rethink_template:
            rethink_prompt = self._rethink_template.format(
                generation=generate_returns[-1].main_response
            )
          elif parsed_action is None:
            rethink_prompt = (
                prompt_templates.RETHINK_WITH_ENV_UNPARSABLE.format(
                    generation=generate_returns[-1].main_response
                )
            )
          else:
            rethink_prompt = prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                last_move=parsed_action
            )
        case RethinkStrategy.RETHINK_WITH_ENV_ILLEGAL_HISTORY:
          if parsed_action is None:
            rethink_prompt = (
                prompt_templates.RETHINK_WITH_ENV_UNPARSABLE.format(
                    generation=generate_returns[-1].main_response
                )
            )
          else:
            if num_attempts > 1:
              rethink_prompt = (
                  prompt_templates.RETHINK_WITH_ENV_ILLEGAL_HISTORY.format(
                      illegal_history=", ".join(parsed_action_history)
                  )
              )
            else:
              rethink_prompt = prompt_templates.RETHINK_WITH_ENV_ILLEGAL.format(
                  last_move=parsed_action
              )
        case RethinkStrategy.RETHINK_WITH_ENV_RULE:
          if parsed_action is None:
            rethink_prompt = (
                prompt_templates.RETHINK_WITH_ENV_UNPARSABLE.format(
                    generation=generate_returns[-1].main_response
                )
            )
          else:
            # TODO(google-deepmind): the rule-parsed string may have extraneous
            # characters that do not fit the regex used in formulating the
            # illegality explanation (derived from python-chess). So it may
            # benefit to clean up the string before trying to get an
            # explanation.
            reason = self.explain_illegal_move(move_str=parsed_action)
            rethink_prompt = prompt_templates.RETHINK_WITH_ENV_RULE.format(
                last_move=parsed_action, reason=reason
            )
        case _:
          raise ValueError(f"Unsupported strategy: {self._strategy}")

      prompt_args["rethink_prompt"] = rethink_prompt

    if maybe_legal_action is not None:
      action = maybe_legal_action
      move_type = samplers.MoveType.LEGAL
    else:
      action = parsed_action
      move_type = samplers.MoveType.ILLEGAL

    return samplers.SamplerOutput(
        # The `action` returned here may be illegal!
        action=action,
        extracted_action=parsed_action,
        matched_action=maybe_legal_action,
        generate_returns=generate_returns,
        auxiliary_outputs=auxiliary_outputs,
        move_type=move_type,
    )
