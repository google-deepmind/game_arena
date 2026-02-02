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

"""Image generation base class for games."""

import abc
import io
from PIL import Image
from PIL import ImageDraw


class ImageGenerator(abc.ABC):
  """Base class for generating game board images."""

  def __init__(
      self,
      board_size: int = 768,
      background_color: tuple[int, int, int] = (255, 255, 255),
  ):
    """Initializes the image generator.

    Args:
        board_size: The width and height of the board in pixels.
        background_color: The background color of the board.
    """
    self._board_size = board_size
    self._background_color = background_color
    self._img = Image.new(
        "RGB", (self._board_size, self._board_size), self._background_color
    )
    self._draw = ImageDraw.Draw(self._img)

  def _get_image_bytes(self) -> bytes:
    """Converts the current image to PNG bytes."""
    buffer = io.BytesIO()
    self._img.save(buffer, format="PNG")
    return buffer.getvalue()

  @abc.abstractmethod
  def generate_image(self, state_str: str) -> bytes:
    """Abstract method to generate the image from the board as a string.

    Args:
        state_str: String representation of the board e.g. str(state) or
          state.to_string().
    """
    raise NotImplementedError("Subclasses must implement this method.")
