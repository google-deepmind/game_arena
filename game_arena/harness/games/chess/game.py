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

"""Chess game notation."""

from typing import Mapping

import chess
import chess.pgn
from game_arena.harness import base_game

import pyspiel


def format_chess_movetext(
    game: chess.pgn.Game,
    numbering_scheme: str,
    use_lan: bool,
    add_current_fen: bool,
) -> str:
  """Formats the movetext for a given chess game.

  See (https://en.wikipedia.org/wiki/Portable_Game_Notation#Movetext).

  Args:
    game: Chess game.
    numbering_scheme: Move number formatting.
    use_lan: Whether to use long algebraic notation (defaults to SAN).
    add_current_fen: Whether to add the current FEN at the end of the movetext.

  Returns:
    The formatted movetext for the given game.
  """
  rv = []
  preceding_comment = False
  preceding_board = game.board()
  nodes = list(game.mainline())
  for i in range(len(nodes) + 1):
    if add_current_fen and i == len(nodes):
      rv.append(f"{{ [%FEN {preceding_board.fen()}] }}")
    fullmove_number = f"{(i + 2) // 2}"
    if i % 2 == 0:
      fullmove_number += "."
    else:
      fullmove_number += "..."
    if numbering_scheme == "all":
      rv.append(fullmove_number)
    elif numbering_scheme == "default":
      if i % 2 == 0 or preceding_comment:
        rv.append(fullmove_number)
    elif numbering_scheme == "none":
      pass
    else:
      raise ValueError(f"Unknown numbering_scheme: {numbering_scheme}")
    if i == len(nodes):
      break
    else:
      n = nodes[i]
    if use_lan:
      rv.append(str(n.move))
    else:
      rv.append(str(preceding_board.san(n.move)))
    if n.comment:
      preceding_comment = True
      rv.append(f"{{ {n.comment} }}")
    else:
      preceding_comment = False
    preceding_board = n.board()
  return " ".join(rv)


def get_pgn(target_state, player_names=None) -> chess.pgn.Game:
  """Creates a PGN game from a target state."""
  if player_names is None:
    player_names = ["Black", "White"]
  game = pyspiel.load_game("chess")
  state = game.new_initial_state()
  moves = []
  for action in target_state.history():
    # Append move in UCI notation.
    moves.append(pyspiel.chess.action_to_move(action, state.board()).to_lan())
    state.apply_action(action)

  # Create a new game.
  pgn_game = chess.pgn.Game()
  pgn_game.headers["Event"] = "Chess Game"
  pgn_game.headers["White"] = player_names[1]
  pgn_game.headers["Black"] = player_names[0]
  # Add results header.
  if target_state.is_terminal():
    score = {
        -1: "0",
        0: "1/2",
        1: "1",
    }
    int_returns = [score[x] for x in target_state.returns()]
    # Note: Results are 'white-black', while returns are 'black, white'.
    result = "-".join(reversed(int_returns))
  else:
    result = "*"
  pgn_game.headers["Result"] = result

  # Add moves to the game.
  pgn_game.add_line([chess.Move.from_uci(move) for move in moves])
  return pgn_game


def get_chess_piece_positions(
    fen: str, include_empty_squares: bool = False
) -> Mapping[str, str]:
  """Returns a dictionary of piece positions and types for a given FEN string.

  Args:
    fen: The FEN (Forsyth-Edwards Notation) string representing the board state.
    include_empty_squares: Include empty squares in the dict with value "empty".

  Returns:
    A dictionary where the keys are the square notation (e.g., "e4") and
    the values are the piece names (e.g., "white pawn").
  """
  board = chess.Board(fen)

  piece_map = {}
  for square in chess.SQUARES:
    piece = board.piece_at(square)
    square_name = chess.square_name(square)
    if piece:
      piece_color = "white" if piece.color == chess.WHITE else "black"
      piece_type_name = chess.piece_name(piece.piece_type)
      piece_map[square_name] = f"{piece_color} {piece_type_name}"
    elif include_empty_squares:
      piece_map[square_name] = "empty"
  return piece_map


def get_chess_ascii_board(fen: str, caption: bool = False) -> str:
  """Returns an ASCII chess board representation given the FEN string."""
  ascii_lines = str(chess.Board(fen)).split("\n")
  ascii_lines_with_numbers = []
  for i, line in enumerate(ascii_lines):
    ascii_lines_with_numbers.append(line + " " + str(8 - i))
  ascii_lines_with_numbers_and_letters = ascii_lines_with_numbers + [
      "a b c d e f g h  "
  ]
  ascii_board = "\n".join(ascii_lines_with_numbers_and_letters)
  if caption:
    return (
        ascii_board
        + "\n"
        + "Uppercase letters represent white pieces and lowercase letters"
        " represent black pieces. The numbers in the right-most column denote"
        " the ranks and the letters in the bottom row denote the files."
    )
  return ascii_board


def get_fen(state: pyspiel.State) -> str:
  """Returns the FEN string for a given pyspiel state."""
  return str(state.board().to_fen())


class ChessGameAdapter(base_game.OpenSpielGameAdapter):
  """Chess game adapter."""

  game_short_name = "chess"
  game_notation = base_game.BaseGameNotation(
      state="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      action="a3; a4; Na3; Nc3",
      player_map={0: "Black", 1: "White"},
      move_notation="standard algebraic notation (SAN)",
      state_notation="Forsyth-Edwards Notation (FEN) notation",
  )

  def get_readable_state(self, representation_type: str | int | None = "fen"):
    assert self.native_state is not None
    if representation_type == "fen":
      return get_fen(self.native_state)
    elif representation_type == "ascii":
      return get_chess_ascii_board(self.native_state.board().fen, caption=False)
    else:
      raise NotImplementedError(
          f"Unsupported representation type: {representation_type}"
      )

  def get_action_string_history(self):
    assert self.native_state is not None
    game = get_pgn(self.native_state)
    move_history = format_chess_movetext(
        game,
        numbering_scheme="default",
        use_lan=False,
        add_current_fen=False,
    )
    return move_history
