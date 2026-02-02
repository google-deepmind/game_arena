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

"""Rethinking rule functions to be used with the rethinking sampler."""

import chess
from game_arena.harness import rethink
import re


def rule_explain_illegal_move(*, fen: str, move_str: str):
  """Analyzes an illegal chess move and provides a detailed, rule-based reason for why it's not allowed.

  Args:
      fen (str): The FEN string representing the board state.
      move_str (str): The move in UCI (e.g., 'e2e4') or SAN (e.g., 'Nf3')
        format.

  Returns:
      str: A detailed string explaining why the move is illegal. If the move
           is legal, it returns a confirmation message.
  """
  board = chess.Board(fen)

  # --- Step 1: Parse the move string ---
  try:
    # We try to parse UCI first as it's unambiguous for illegal moves.
    move = chess.Move.from_uci(move_str)
  except ValueError:
    # This handles malformed UCI notation. We now try to parse SAN.
    try:
      move = board.parse_san(move_str)
    except ValueError as parse_san_error:
      # --- Enhanced SAN Error Analysis ---
      error_msg = str(parse_san_error)
      if "ambiguous" in error_msg:
        # Find the pieces causing the ambiguity.
        try:
          # Extract piece and destination square from SAN
          p_match = re.match(
              r"([NBRQK])?([a-h]?[1-8]?)?x?([a-h][1-8])", move_str
          )
          if p_match is None:
            raise ValueError(f"Invalid SAN: {move_str}") from parse_san_error
          else:
            piece_char, _, to_square_name = p_match.groups()

          piece_type = (
              chess.PIECE_SYMBOLS.index(piece_char.lower())
              if piece_char
              else chess.PAWN
          )
          to_square = chess.SQUARE_NAMES.index(to_square_name)

          ambiguous_pieces = []
          for legal_move in board.legal_moves:
            piece = board.piece_at(legal_move.from_square)
            if (
                legal_move.to_square == to_square
                and piece is not None
                and piece.piece_type == piece_type
            ):
              ambiguous_pieces.append(chess.square_name(legal_move.from_square))

          return (
              f"Invalid SAN: The move '{move_str}' is ambiguous. "
              f"The following pieces can also move to {to_square_name}: "
              f"{', '.join(ambiguous_pieces)}. "
              "Please specify the starting file or rank (e.g., 'Rha1')."
          )
        except (ValueError, AttributeError, TypeError):
          return (
              f"Invalid SAN: The move '{move_str}' is ambiguous. "
              "Please be more specific."
          )

      elif "illegal" in error_msg:
        # Check if the SAN is missing a piece indicator.
        # A move like 'e5' is parsed as a pawn move. If it's illegal,
        # it might be because the user intended a piece move but
        # forgot the piece letter.
        if move_str[0] in "abcdefgh":
          return (
              f"Invalid SAN: The move '{move_str}' is interpreted as a pawn"
              " move, which is illegal in this position. If you intended to"
              f" move a piece to {move_str[-2:]}, you must specify which piece"
              f" (e.g., 'N{move_str[-2:]}', 'B{move_str[-2:]}')."
          )
        else:
          return (
              f"Invalid SAN: The move '{move_str}' is illegal. The specified"
              " piece cannot legally make that move in this position."
          )
      else:
        return f"The move '{move_str}' is invalid. Reason: {parse_san_error}"

  # --- Step 2: Check if the move is actually legal ---
  if move in board.legal_moves:
    return f"The move {move.uci()} is legal."

  # --- Step 3: Begin detailed rule analysis for illegal moves ---
  from_square = move.from_square
  to_square = move.to_square
  piece = board.piece_at(from_square)

  # Rule: Is there a piece to move?
  if piece is None:
    return (
        "Illegal Move: There is no piece on the starting square "
        f"({chess.square_name(from_square)})."
    )

  # Rule: Is it the correct player's turn?
  if piece.color != board.turn:
    turn_color = "White" if board.turn == chess.WHITE else "Black"
    return (
        f"Illegal Move: It is {turn_color}'s turn to move, but the piece on "
        f"{chess.square_name(from_square)} is not theirs."
    )

  # Rule: Cannot capture your own pieces.
  target_piece = board.piece_at(to_square)
  if target_piece and target_piece.color == piece.color:
    return (
        f"Illegal Move: The square {chess.square_name(to_square)} is occupied"
        " by a friendly piece."
    )

  # Rule: King is in check.
  # This is the most common reason for a move being "pseudo-legal"
  # but not fully legal.
  if move in board.pseudo_legal_moves:
    return "Illegal Move: This move would leave the king in check."

  # --- Step 4: Analyze piece-specific movement and capture rules ---
  piece_type = piece.piece_type

  # King-specific rules
  if piece_type == chess.KING:
    if board.is_castling(move):
      if board.is_check():
        return "Illegal Castling: You cannot castle while in check."
      # Check path for castling
      for sq in chess.SquareSet(chess.between(from_square, to_square)):
        if board.is_attacked_by(not board.turn, sq):
          return (
              "Illegal Castling: The king cannot pass through an attacked"
              f" square ({chess.square_name(sq)})."
          )
    return (
        "Illegal King Move: The king can only move to adjacent squares and"
        " cannot move into check."
    )

  # Knight-specific rules
  if piece_type == chess.KNIGHT:
    return (
        "Illegal Knight Move: A knight moves in an 'L' shape (two squares in"
        " one direction, then one perpendicularly)."
    )

  # Pawn-specific rules
  if piece_type == chess.PAWN:
    # Pawns can't capture forward
    if target_piece is None and chess.square_file(
        from_square
    ) != chess.square_file(to_square):
      if not board.is_en_passant(move):
        return "Illegal Pawn Move: Pawns can only capture diagonally."
    # Pawns can't move diagonally without capturing
    if target_piece is not None and chess.square_file(
        from_square
    ) == chess.square_file(to_square):
      return (
          "Illegal Pawn Move: Pawns cannot capture by moving straight forward."
      )
    return (
        "Illegal Pawn Move: Pawns have specific rules for moving forward and"
        " capturing."
    )

  # Sliding pieces (Rook, Bishop, Queen)
  if piece_type in (chess.ROOK, chess.BISHOP, chess.QUEEN):
    # Check if path is blocked
    for sq in chess.SquareSet(chess.between(from_square, to_square)):
      if board.piece_at(sq) is not None:
        blocker_sq_name = chess.square_name(sq)
        return (
            "Illegal Move: The path is blocked by a piece on"
            f" {blocker_sq_name}."
        )

    # If path is not blocked, the move itself is geometrically wrong
    if piece_type == chess.ROOK:
      return (
          "Illegal Rook Move: A rook can only move horizontally or vertically."
      )
    if piece_type == chess.BISHOP:
      return "Illegal Bishop Move: A bishop can only move diagonally."
    if piece_type == chess.QUEEN:
      return (
          "Illegal Queen Move: The queen's path is clear, but the move is not a"
          " valid straight or diagonal line."
      )

  return (
      "Illegal Move: The move violates the rules of chess for an unknown"
      " reason."
  )


class RethinkSampler(rethink.RethinkSampler):
  """Rethink sampler for chess."""

  def explain_illegal_move(self, move_str):
    return rule_explain_illegal_move(
        fen=self._game_adapter.get_readable_state(representation_type="fen"),
        move_str=move_str,
    )
