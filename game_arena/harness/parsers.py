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

"""Move parsers."""

import dataclasses
import re
from typing import Protocol, Sequence
from absl import logging


def parse_move_from_response(
    response: str,
    action_tag: str = "Final Answer:",
    additional_tags: Sequence[str] = (":", "is"),
) -> dict[str, str] | None:
  """Extracts move, and the text before the move from response."""
  if response is None:
    return None
  try:
    last_index = -1
    final_split_token = ""
    for split_token in [action_tag] + list(additional_tags):
      tmp_index = response.rfind(split_token)
      if tmp_index > last_index:
        last_index = tmp_index
        final_split_token = split_token
    if last_index == -1:
      return None
    # Split the string
    suffix = response[last_index + len(final_split_token) :]
    if suffix is None:
      return None
    move_str = (
        suffix.strip(" .")
        .replace("$", "")
        .replace("\\boxed{", "")
        .replace("\\text{", "")
        .replace("\boxed{", "")
        .replace("\text{", "")
        .replace("}", "")
        .replace("*", "")
        .replace(" ", "")
        .replace("`", "")
        .replace("\n", "")
    )
    clean = re.compile("<.*?>")
    move_str = re.sub(clean, "", move_str)
    if not move_str:
      return None
    prefix = response[:last_index]
    return {
        "move": move_str,
        "prefix": "" if prefix is None else prefix.strip(),
    }
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.error(
        "parse_move_from_response failed with input response %s", response
    )
    raise e


# TODO(google-deepmind): divide this into TextParserInput and
# TextParserWithStateInput:
# TODO(John Schultz): Consider adding serialized game and state. Simplifies things
# as the legal moves and player number can be accessed from the state. Also,
# information needed for parsing may be more easily accessible from the
# deserialized pyspiel state object than from the state string.
@dataclasses.dataclass(frozen=True, kw_only=True)
class TextParserInput:
  text: str
  state_str: str | None = None
  legal_moves: Sequence[str] | None = None
  player_number: int | None = None


class TextParser(Protocol):
  """Generic text parser."""

  # TODO(google-deepmind): should we define a parser output type?
  def parse(self, parser_input: TextParserInput) -> str | None:
    ...


class ChainedMoveParser(TextParser):
  """Parses text into a move by successively applying parsers.

  For example, parsing with a LLM followed by parsing of the structured output
  with a rule-based parser or with a soft-parser against legal moves.
  """

  def __init__(self, parsers: Sequence[TextParser]):
    self._parsers = parsers

  def parse(self, parser_input: TextParserInput) -> str | None:
    maybe_text_output = None
    for parser in self._parsers:
      maybe_text_output = parser.parse(parser_input)
      if maybe_text_output is None:
        return None
      parser_input = dataclasses.replace(parser_input, text=maybe_text_output)
    return maybe_text_output


class RuleBasedMoveParser(TextParser):
  """Parses text into a move with string processing built into Python."""

  def __init__(
      self,
      action_tag: str = "Final Answer: ",
      additional_tags: Sequence[str] = (":", "is"),
  ):
    self._action_tag = action_tag
    self._additional_tags = additional_tags

  def parse(self, parser_input: TextParserInput) -> str | None:
    maybe_move_and_prefix = parse_move_from_response(
        parser_input.text,
        action_tag=self._action_tag,
        additional_tags=self._additional_tags,
    )
    if maybe_move_and_prefix is None:
      return None
    return maybe_move_and_prefix["move"]


class SoftMoveParser(TextParser):
  """Parses text into a move by soft matching against legal moves."""

  def _parse_selected_action(self, parser_input: TextParserInput) -> str | None:
    """Parses text into a move by soft matching against legal moves."""
    selected_action = parser_input.text

    if not selected_action:
      return None
    raise ValueError("SoftMoveParser is not supported for this game.")

  def _validate_parser_input(self, parser_input: TextParserInput) -> None:
    """Validates the parser input for soft move parsing."""
    if (
        parser_input.state_str is None
        or parser_input.legal_moves is None
        or parser_input.player_number is None
    ):
      raise ValueError(
          "State as a string, legal moves, and player number are required for"
          " soft move parsing."
      )

  def parse(self, parser_input: TextParserInput) -> str | None:
    self._validate_parser_input(parser_input)
    if not parser_input.text:
      return None
    selected_action = self._parse_selected_action(parser_input)
    if selected_action not in parser_input.legal_moves:
      return None
    return selected_action
