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

"""Utilities for converting FEN into images using Lichess sprites."""

from collections.abc import Mapping
import importlib.resources
import io
from typing import Any

import chess
import chess.pgn
import frozendict
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageSequence

COLORS = (
    'blue',
    'brown',
    'green',
    'ic',
    'pink',
    'purple',
)

# Piece set and license description. Source of truth for licenses:
# https://github.com/lichess-org/lila/blob/master/COPYING.md
PIECE_SETS = (
    'caliente',  # CC BY-NC-SA 4.0
    'california',  # CC BY-NC-SA 4.0
    'cardinal',  # CC BY-NC-SA 4.0
    'cburnett',  # GPLv2+
    'celtic',  # MIT
    'chessnut',  # Apache 2.0
    'companion',  # Freeware https://www.enpassant.dk/chess/fonteng.htm
    'dubrovny',  # # CC BY-NC-SA 4.0
    'fantasy',  # MIT https://github.com/maurimo/chess-art
    'fresca',  # CC BY-NC-SA 4.0
    'gioco',  # CC BY-NC-SA 4.0
    'icpieces',  # CC BY-NC-SA 4.0
    'leipzig',  # Freeware https://www.enpassant.dk/chess/fonteng.htm
    'libra',  # CC BY-NC-SA 4.0
    'maestro',  # CC BY-NC-SA 4.0
    'merida',  # GPLv2+
    'mpchess',  # GPL3v3+
    'pirouetti',  # AGPLv3+
    'spatial',  # MIT https://github.com/maurimo/chess-art
    'staunty',  # CC BY-NC-SA 4.0
    'tatiana',  # CC BY-NC-SA 4.0
)

IMG_WIDTH = 90
IMG_HEIGHT = 90
BORDER_SIZE = 30
FONT_SIZE = 20
# Dark border, light text.
BORDER_COLOR = (49, 46, 43)
TEXT_COLOR = (255, 255, 255)


def extract_sprite(frame: Image.Image, row: int, col: int) -> Image.Image:
  x = row * IMG_WIDTH
  y = col * IMG_HEIGHT
  return frame.crop((x, y, x + IMG_WIDTH, y + IMG_HEIGHT))


def extract_all_sprites(img: Image.Image) -> dict[str, Any]:
  """Extract all sprites from an image."""
  sprite_dict = {}
  sprite_dict['light_empty'] = extract_sprite(img, 0, 0)
  sprite_dict['dark_empty'] = extract_sprite(img, 1, 0)
  sprite_dict['light_p'] = extract_sprite(img, 0, 1)
  sprite_dict['light_n'] = extract_sprite(img, 0, 2)
  sprite_dict['light_b'] = extract_sprite(img, 0, 3)
  sprite_dict['light_r'] = extract_sprite(img, 0, 4)
  sprite_dict['light_q'] = extract_sprite(img, 0, 5)
  sprite_dict['light_k'] = extract_sprite(img, 0, 6)
  sprite_dict['dark_p'] = extract_sprite(img, 1, 1)
  sprite_dict['dark_n'] = extract_sprite(img, 1, 2)
  sprite_dict['dark_b'] = extract_sprite(img, 1, 3)
  sprite_dict['dark_r'] = extract_sprite(img, 1, 4)
  sprite_dict['dark_q'] = extract_sprite(img, 1, 5)
  sprite_dict['dark_k'] = extract_sprite(img, 1, 6)
  sprite_dict['light_P'] = extract_sprite(img, 4, 1)
  sprite_dict['light_N'] = extract_sprite(img, 4, 2)
  sprite_dict['light_B'] = extract_sprite(img, 4, 3)
  sprite_dict['light_R'] = extract_sprite(img, 4, 4)
  sprite_dict['light_Q'] = extract_sprite(img, 4, 5)
  sprite_dict['light_K'] = extract_sprite(img, 4, 6)
  sprite_dict['dark_P'] = extract_sprite(img, 5, 1)
  sprite_dict['dark_N'] = extract_sprite(img, 5, 2)
  sprite_dict['dark_B'] = extract_sprite(img, 5, 3)
  sprite_dict['dark_R'] = extract_sprite(img, 5, 4)
  sprite_dict['dark_Q'] = extract_sprite(img, 5, 5)
  sprite_dict['dark_K'] = extract_sprite(img, 5, 6)
  return sprite_dict


def get_shade(square: chess.Square) -> str:
  square_rank = chess.square_rank(square)
  i = square + (square_rank % 2)
  shade = 'light' if i % 2 == 1 else 'dark'
  return shade


def get_sprites() -> dict[str, dict[str, Any]]:
  """Get sprites images from files."""
  sprites = {}
  for color in COLORS:
    for piece_set in PIECE_SETS:
      # importlib.resources API changed between Python 3.12 and 3.13:
      sprite_bytes = importlib.resources.read_binary(
          'game_arena.harness.games.chess.sprites', f'{color}-{piece_set}.gif'
      )
      img = Image.open(io.BytesIO(sprite_bytes), 'r')
      frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
      sprites[f'{color}-{piece_set}'] = frames[0]
  sprite_dicts = {}
  for sprite_name, sprite_img in sprites.items():
    sprite_dicts[sprite_name] = extract_all_sprites(sprite_img)
  return sprite_dicts


SPRITE_DICTS = frozendict.frozendict(get_sprites())


def fen_to_board(
    fen: str,
    color: str = 'brown',
    piece_set: str = 'cburnett',
    player_perspective: str = 'white',
    show_labels: bool = False,
    sprite_dicts: Mapping[str, Mapping[str, Any]] = SPRITE_DICTS,
) -> bytes:
  """Convert a FEN to a board image.

  Args:
    fen: FEN board str to render.
    color: color patterns for the board.
    piece_set: style of pieces.
    player_perspective: view board from white/black player's perspective.
    show_labels: If True, add a border with rank and file labels.
    sprite_dicts: dictionary that maps a {color}-{piece_set} into all possible
      square images for that style.

  Returns:
    png: PNG image of the board in bytes format.
  """
  if f'{color}-{piece_set}' not in sprite_dicts:
    raise ValueError(f'Invalid color/piece_set combination:{color}-{piece_set}')
  sprite_dict = sprite_dicts[f'{color}-{piece_set}']
  # Create a blank canvas
  board = chess.Board(fen)
  canvas_width = 8 * IMG_WIDTH
  canvas_height = 8 * IMG_HEIGHT
  canvas = Image.new('RGB', (canvas_width, canvas_height))

  for i in range(64):
    p = board.piece_at(i)
    if p is None:
      p = 'empty'
    else:
      p = p.symbol()

    square_rank = chess.square_rank(i)
    square_file = chess.square_file(i)
    shade = get_shade(i)
    img = sprite_dict[f'{shade}_{p}']
    # Calculate grid position
    if player_perspective == 'white':
      x = square_file * IMG_WIDTH
      y = (7 - square_rank) * IMG_HEIGHT
    elif player_perspective == 'black':
      x = (7 - square_file) * IMG_WIDTH
      y = square_rank * IMG_HEIGHT
    else:
      raise ValueError(f'Invalid player perspective: {player_perspective}')
    canvas.paste(img, (x, y))

  if show_labels:
    bordered_canvas = Image.new(
        'RGB',
        (canvas_width + 2 * BORDER_SIZE, canvas_height + 2 * BORDER_SIZE),
        color=BORDER_COLOR,
    )
    bordered_canvas.paste(canvas, (BORDER_SIZE, BORDER_SIZE))
    draw = ImageDraw.Draw(bordered_canvas)

    try:
      font = ImageFont.truetype('arialbd.ttf', FONT_SIZE)
    except IOError:
      font = ImageFont.load_default(FONT_SIZE)

    files = 'abcdefgh'
    ranks = '87654321'
    if player_perspective == 'black':
      files = files[::-1]
      ranks = ranks[::-1]

    # Draw file labels (letters)
    for i, letter in enumerate(files):
      x = BORDER_SIZE + i * IMG_WIDTH + IMG_WIDTH / 2
      # Top
      draw.text(
          (x, BORDER_SIZE / 2),
          letter,
          fill=TEXT_COLOR,
          font=font,
          anchor='mm',
      )
      # Bottom
      draw.text(
          (x, canvas_height + BORDER_SIZE + BORDER_SIZE / 2),
          letter,
          fill=TEXT_COLOR,
          font=font,
          anchor='mm',
      )

    # Draw rank labels (numbers)
    for i, number in enumerate(ranks):
      y = BORDER_SIZE + i * IMG_HEIGHT + IMG_HEIGHT / 2
      # Left
      draw.text(
          (BORDER_SIZE / 2, y),
          number,
          fill=TEXT_COLOR,
          font=font,
          anchor='mm',
      )
      # Right
      draw.text(
          (canvas_width + BORDER_SIZE + BORDER_SIZE / 2, y),
          number,
          fill=TEXT_COLOR,
          font=font,
          anchor='mm',
      )
    canvas = bordered_canvas

  buffer = io.BytesIO()
  canvas.save(buffer, format='PNG')
  png = buffer.getvalue()
  return png
