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

"""Chess specific parsers."""

import random
import re
from typing import Sequence
import chess
from game_arena.harness import parsers


def chess_soft_parser_v1(
    state_str: str, selected_action: str, spiel_legal_moves: Sequence[str]
) -> str | None:
  """Chess parser that matches against legal moves."""
  if selected_action is None:
    return None

  selected_action = selected_action.strip()

  if not selected_action:
    return None

  if not selected_action.startswith("0-0") and selected_action[0].isdigit():
    # Castling may be indicated with leading Zero-Zero.
    # \d+ is the first capturing group, matching one or more digits.
    # \.|\.\.\. is the second capturing group, matching one or three dots.
    # e.g. 1. for first white move, 2... for second black move.
    # .* captures remaining characters.
    match = re.search(r"(\d+)(\.{1,3})(.*)", selected_action)
    if match is not None:
      _, _, selected_action = match.groups()
    else:
      return None
  # There might be whitespace between the dot(s) and the move:
  selected_action = selected_action.lstrip()
  # python-chess uses a regex that expects the move to end with the destination
  # square, followed optionally by pawn promotion and/or + or # for checkmate.
  # The following characters should definitely be removed if they are at the end
  # and they also do not appear or are not differentiating according to chess
  # notation:
  for char_to_remove in [
      ":",  # Indicates capture but not expected by python-chess.
      ".",  # Only used in the move number.
      "*",
      ",",
      "&",
      "^",
      "\\",
      "<",
      ">",
      "{",
      "}",
      "[",
      "]",
      "?",  # Move quality comment.
      "!",  # Move quality comment.
  ]:
    selected_action = selected_action.replace(char_to_remove, "")

  # En passant annotation (which is e.p. or e.p), but we removed the dots.
  selected_action = selected_action.removesuffix("ep")
  # N.B. python-chess also considers castling indicated with zeros instead of
  # capital letter Os. Zeroes are non-standard notation.

  board = chess.Board(state_str)
  maybe_legal_move_san = selected_action
  # Match between python-chess and OpenSpiel with UCI standard, which uses
  # ambiguity-free pure algebraic coordinate notation. It is slightly different
  # from LAN: https://www.chessprogramming.org/Algebraic_Chess_Notation#UCI
  selected_uci = None
  try:
    selected_uci = board.parse_san(maybe_legal_move_san).uci()
  except ValueError as e:
    error_str = str(e)
    # TODO(google-deepmind): handle other ambiguous cases.
    if "ambiguous" in error_str:
      possible_moves_uci = []
      for legal_move in board.legal_moves:
        legal_move_san = board.san(legal_move)
        legal_move_san_short = legal_move_san.rstrip("+#")
        maybe_legal_move_san_short = maybe_legal_move_san.rstrip("+#")
        # Examples: Rad1 and Rhd1, R2d5 and R8d5, Ngf3 and Nef3
        if legal_move_san_short.startswith(
            maybe_legal_move_san_short[0]
        ) and legal_move_san_short.endswith(maybe_legal_move_san_short[-2:]):
          possible_moves_uci.append(legal_move.uci())
      if not possible_moves_uci:
        return None
      rng = random.Random(42)
      selected_uci = rng.choice(possible_moves_uci)
  if selected_uci is None:
    return None
  else:
    legal_move = board.parse_uci(selected_uci)
    # Translate the move to a SAN string.
    selected_action = board.san(legal_move)
  # Match exactly with OpenSpiel legal (SAN) moves:
  if selected_action not in spiel_legal_moves:
    return None
  return selected_action


class ChessSoftParser(parsers.SoftMoveParser):
  """Chess soft parser."""

  def _parse_selected_action(
      self, parser_input: parsers.TextParserInput
  ) -> str | None:
    return chess_soft_parser_v1(
        state_str=parser_input.state_str,
        selected_action=parser_input.text,
        spiel_legal_moves=parser_input.legal_moves,
    )
