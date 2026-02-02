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

"""Poker specific parsers."""

import json
import re
from typing import Sequence
from game_arena.harness import parsers
from game_arena.harness.games.poker import hand_history_utils as hh_utils


def repeated_poker_soft_parser(
    state_str: str,
    selected_action: str,
    legal_moves: Sequence[str],
    player_number: int,
) -> str | None:
  """Parser for Open Spiel repeated_poker."""
  if player_number >= 2:
    raise ValueError("More than 2 players not currently supported.")
  state_dict = json.loads(state_str)
  up_state_dict = json.loads(state_dict["current_universal_poker_json"])
  acpc_state_str = up_state_dict["acpc_state"]
  acpc_state_str = acpc_state_str.split("\n")[0]
  if not acpc_state_str.startswith("STATE:"):
    raise ValueError(
        f"Expected ACPC state to start with STATE:, got {acpc_state_str}"
    )
  starting_stacks = up_state_dict.get("starting_stacks", [])
  num_players = len(starting_stacks)
  if not num_players:
    raise ValueError(f"No starting stacks found in {state_dict}.")
  players = [f"Player{i}" for i in range(num_players)]
  acpc_state_str_full = acpc_state_str + "::" + "|".join(players)
  cfg = hh_utils.Config(
      seats=num_players,
      small_blind=state_dict["small_blind"],
      big_blind=state_dict["big_blind"],
      starting_stacks=starting_stacks,
  )
  hand, parse_state = hh_utils.parse_acpc_line(
      acpc_state_str_full,
      cfg=cfg,
      policy=hh_utils.ButtonPolicy(),
      button_index=state_dict["hand_number"] % 2 + 1,
  )
  most_recent_cur_player_event_this_street = None
  if parse_state.street == hh_utils.Street.PREFLOP:
    # Need to include blinds as well
    for event in hand.events:
      if event.actor == player_number:
        most_recent_cur_player_event_this_street = event
  else:
    for event in hand.events:
      if event.street != parse_state.street:
        continue
      elif event.actor == player_number:
        most_recent_cur_player_event_this_street = event
  if most_recent_cur_player_event_this_street is None:
    contrib_street = 0
  else:
    contrib_street = most_recent_cur_player_event_this_street.to_amount
  if contrib_street is None:
    contrib_street = 0
  contrib_total = parse_state.contrib_total[player_number]
  contrib_prev = contrib_total - contrib_street

  number_match = re.findall(r"\d+", selected_action)
  if "fold" in selected_action.lower():
    poker_move = "Fold"
  elif "check" in selected_action.lower():
    poker_move = "Call"
  elif "call" in selected_action.lower():
    poker_move = "Call"
  elif (
      "all in" in selected_action.lower() or "all-in" in selected_action.lower()
  ):
    return legal_moves[-1]
  elif number_match:
    # The parsed amount represents the total amount player is committing on the
    # current street. This needs to map to the ACPC convention of the total
    # amount the player is committing across all streets.
    parsed_amount = int(number_match[-1])
    if parsed_amount <= 0:
      return selected_action  # Illegal move.
    bet_size = parsed_amount + contrib_prev

    legal_bet_moves = [a for a in legal_moves if "Bet" in a]
    legal_bet_sizes = [int(a.split("Bet")[1]) for a in legal_bet_moves]
    if bet_size in legal_bet_sizes:
      poker_move = f"Bet{bet_size}"
    else:
      # To get here, a player must have declared a number without saying "fold",
      # "check", or "call", so they're committed to some bet amount. If the
      # bet/raise is too small, the action is mapped to the smallest legal bet
      # size. If the bet/raise is too large, the action is mapped to the largest
      # legal bet size, or a call if there are no legal bets/raises.
      if not legal_bet_moves:
        poker_move = "Call"
      else:
        poker_move = f"Bet{max(legal_bet_sizes)}"
        for legal_bet_size in legal_bet_sizes:
          if legal_bet_size >= bet_size:
            poker_move = f"Bet{legal_bet_size}"
            break

  else:
    # Illegal move.
    return None
  selected_action = f"player={player_number} move={poker_move}"
  if selected_action not in legal_moves:
    return None
  else:
    return selected_action


class PokerSoftParser(parsers.SoftMoveParser):
  """Poker soft parser."""

  def _parse_selected_action(
      self, parser_input: parsers.TextParserInput
  ) -> str | None:

    return repeated_poker_soft_parser(
        parser_input.state_str,
        parser_input.text,
        parser_input.legal_moves,
        parser_input.player_number,
    )

  def parse(self, parser_input: parsers.TextParserInput) -> str | None:
    self._validate_parser_input(parser_input)
    if not parser_input.text:
      return None
    selected_action = self._parse_selected_action(parser_input)
    return selected_action
