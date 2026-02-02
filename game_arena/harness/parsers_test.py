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

"""Tests for move parsers."""

from game_arena.harness import parsers
from game_arena.harness.games.chess import parsers as chess_parsers
from absl.testing import absltest
from absl.testing import parameterized


class MoveParsingTest(parameterized.TestCase):

  def test_trailing_newline(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: e5
"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  @parameterized.named_parameters(
      ('one_whitespace', 1),
      ('two_whitespace', 2),
  )
  def test_trailing_whitespace(self, num_trailing_whitespace):
    raw_response_with_no_trailing_whitespace = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: e5"""
    with_trailing_whitespace = (
        raw_response_with_no_trailing_whitespace + ' ' * num_trailing_whitespace
    )
    self.assertEqual(
        parsers.parse_move_from_response(with_trailing_whitespace)['move'],
        'e5',
    )

  def test_single_backslash_boxed_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \boxed{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_double_backslash_boxed_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \\boxed{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_single_backslash_text_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \text{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_double_backslash_text_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \\text{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_latex_dollar_sign_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: $e5$"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_space_and_period_at_end(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: e5 ."""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_final_answer_tag(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

The final answer is e5
"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_html_tags(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: <answer>e5</answer>"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_poker(self):
    raw_response = """
[Assumption]: I am assuming the villain is an unknown, reasonably competent player. In the absence of specific reads suggesting he is weak or prone to spewy plays, I should default to a GTO-based, fundamentally sound strategy. This means respecting his polarized shove and assuming he has a mathematically sound ratio of value to bluffs, against which my specific hand is not profitable enough to call.
Key Variables:
Pot Odds: The requirement of 37.2% equity sets a high bar for a drawing hand.
Opponent's Range Construction: The entire decision rests on whether Villain's range is a) GTO-polarized (making this a fold) or b) weak/wide and includes many one-pair hands (making this a marginal call).
Stack Preservation: I have a commanding stack lead (192 vs 35). Making a high-variance, marginal call that could significantly reduce this lead is less desirable than making a tight fold and continuing to apply pressure with my stack advantage in future hands.
Future Plan:
N/A. This decision ends the hand for me, one way or the other.
Final Answer: CALL 22
""".strip()
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'CALL22',
    )


class ChessSoftParserV1Test(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'simple_move',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'e4',
          ['e4'],
          'e4',
      ),
      (
          'with_move_number',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          '1. e4',
          ['e4'],
          'e4',
      ),
      (
          'with_dots',
          'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
          '2...Nf3',
          ['Nf3'],
          'Nf3',
      ),
      (
          'with_dots_and_space',
          'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
          '2... Nf3',
          ['Nf3'],
          'Nf3',
      ),

      (
          'with_extra_chars',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'e4.',
          ['e4'],
          'e4',
      ),
      (
          'ambiguous_move_resolvable',
          (
              'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq'
              ' - 0 2'
          ),
          'Nxf7',
          ['Nxf7'],
          'Nxf7',
      ),
      (
          'invalid_move',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'e5',
          ['e4'],
          None,
      ),
      (
          'not_in_legal_moves',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'd4',
          ['e4'],
          None,
      ),
      (
          'en_passant',
          'r1b2rk1/1p4pp/p1n1p3/4Pp2/4R3/2NB4/PPP3PP/2K4R w - f6 0 17',
          'exf6e.p',
          ['exf6'],
          'exf6',
      ),
      (
          'kingside_castling_with_numeric_0',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQK2R w KQkq - 1 6',
          '0-0',
          ['O-O'],
          'O-O',
      ),
      (
          'kingside_castling_with_numeric_0_with_step_number',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQK2R w KQkq - 1 6',
          '6. 0-0',
          ['O-O'],
          'O-O',
      ),
      (
          'queenside_castling_with_numeric_0',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQ1RK1 b kq - 2 6',
          '0-0-0',
          ['O-O-O'],
          'O-O-O',
      ),
      (
          'queenside_castling_with_numeric_0_with_step_number',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQ1RK1 b kq - 2 6',
          '6... 0-0-0',
          ['O-O-O'],
          'O-O-O',
      ),
      (
          'kingside_castling_with_letter_O',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQK2R w KQkq - 1 6',
          'O-O',
          ['O-O'],
          'O-O',
      ),
      (
          'kingside_castling_with_letter_O_with_step_number',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQK2R w KQkq - 1 6',
          '6. O-O',
          ['O-O'],
          'O-O',
      ),
      (
          'queenside_castling_with_letter_O',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQ1RK1 b kq - 2 6',
          'O-O-O',
          ['O-O-O'],
          'O-O-O',
      ),
      (
          'queenside_castling_with_letter_O_with_step_number',
          'r3kbnr/pppn1ppp/4P3/1B4q1/8/5N1b/PPPP1PPP/RNBQ1RK1 b kq - 2 6',
          '6... O-O-O',
          ['O-O-O'],
          'O-O-O',
      ),
  )
  def test_chess_soft_parser(
      self, state_str, selected_action, spiel_legal_moves, expected_move
  ):
    self.assertEqual(
        chess_parsers.chess_soft_parser_v1(
            state_str, selected_action, spiel_legal_moves
        ),
        expected_move,
    )


if __name__ == '__main__':
  absltest.main()
