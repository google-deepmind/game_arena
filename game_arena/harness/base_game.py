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

"""Base classes for game environment adapters.

Adapters that span multiple games belong here, e.g. OpenSpielGameAdapter,
while game-specific adapters belong in games/*.
"""

import abc
from collections.abc import Sequence
import dataclasses
from typing import Any, Mapping
import pyspiel


@dataclasses.dataclass(frozen=True, kw_only=True)
class BaseGameNotation:
  """Base game notation."""

  state: str
  action: str
  player_map: dict[int, str]
  move_notation: str
  state_notation: str


class BaseGameEnvAdapter(abc.ABC):
  """Base game environment adapter class.

  All methods that need game state should access it through an instance of this
  class.

  Parses and converts between the game's native state and notations suitable for
  LLM consumption. Provides utility functions for common operations.

  Every game should have an implementation of this class. Even games that can
  use the default methods for a given environment should still provide the game
  notation.
  """

  def __init__(self):
    # TODO(Sohier Dane): Add str player name? Can default to str(player_number).
    self.player_number: int | None = None
    self.legal_actions: str | Sequence[str] | None = None
    # TODO(Sohier Dane): ensure the native vs str versions are used correctly.
    self.native_legal_actions: int | Sequence[int] | None = None
    self.native_state: Any = None
    self._observation = None
    self._configuration = None

  @property
  @abc.abstractmethod
  def game_short_name(self) -> str:
    """Returns the game short name."""
    ...

  @property
  @abc.abstractmethod
  def game_notation(self) -> BaseGameNotation:
    """Returns the game notation for the game."""
    ...

  @abc.abstractmethod
  def update_game_state(self, observation: Mapping[str, Any]) -> None:
    """Updates the game state, typically after a game step is completed."""
    ...

  @abc.abstractmethod
  def string_to_action(self, action_str: str) -> Any:
    """Convert an LLM action response into a valid action for the game."""
    ...

  @abc.abstractmethod
  def action_to_string(self, action: Any) -> str:
    """Convert an action to a string for the game."""
    ...

  @abc.abstractmethod
  def get_readable_state(
      self, representation_type: str | int | None = None, **kwargs
  ) -> str:
    """Get a string representation of the game state suitable for LLM consumption.

    Args:
      representation_type: Key for the desired representation type for games
        that have multiple notations.
      **kwargs: Additional arguments for games that have variable formatting
        needs or multiple representation types.

    Returns:
      A string representation of the game state.
    """
    ...


class OpenSpielGameAdapter(BaseGameEnvAdapter, abc.ABC):
  """Base openspiel game class."""

  def update_game_state(self, observation: Mapping[str, Any]) -> None:
    self._observation = observation
    serialized_game_and_state = self._observation.get("serializedGameAndState")
    _, pyspiel_state = pyspiel.deserialize_game_and_state(
        serialized_game_and_state
    )
    self.native_state = pyspiel_state
    if not self.native_state:
      self.legal_actions = []
    else:
      self.native_legal_actions = self.native_state.legal_actions()
      self.legal_actions = [
          self.native_state.action_to_string(
              self.native_state.current_player(), action_int
          )
          for action_int in self.native_legal_actions
      ]
    self.player_number = self.native_state.current_player()
    assert self.game_notation is not None

    self.player_name: str = self.game_notation.player_map[self.player_number]

  def action_to_string(self, action: int) -> str:
    """Convert an action int to a string."""
    assert self.native_state is not None
    return self.native_state.action_to_string(action)

  def string_to_action(self, action_str: str) -> int:
    """Convert an LLM action response into a valid action for the game."""
    assert self.native_state is not None
    return self.native_state.string_to_action(action_str)

  def get_action_string_history(self) -> str:
    """Returns a string history of actions."""

    tmp_state = self.native_state.get_game().new_initial_state()
    move_list = []
    for action in self.native_state.history():
      action_str = tmp_state.action_to_string(
          tmp_state.current_player(), action
      )
      move_list.append(action_str)
      tmp_state.apply_action(action)

    move_history = " ".join(move_list) if move_list else "None"
    return move_history

  def get_readable_state(
      self, representation_type: str | int | None = None
  ) -> str:
    """Get a string representation of the game state suitable for LLM consumption."""
    assert self.native_state is not None
    return "\n".join([
        " ".join(list(line.strip()))
        for line in str(self.native_state).split("\n")
    ])
