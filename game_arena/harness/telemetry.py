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

"""Defines telemetry for Game Arena harness."""

from typing import Callable, Protocol


class FailedTelemetryCallbackError(Exception):
  """Error raised when telemetry callback fails."""


class TelemetrySendFn(Protocol):
  """Callback that sends a key-value pair of telemetry data."""

  def __call__(self, module: str, **kwargs) -> None:
    ...

SEND: TelemetrySendFn = lambda module, **kwargs: None


def get_logger(module: str) -> Callable[..., None]:
  """Returns a module-labelled logger that sends key-value pairs to telemetry."""
  # Late binding to the implementation of `SEND`:
  def logger(**kwargs):
    return SEND(module=module, **kwargs)
  return logger


def set_exporter(send_fn: TelemetrySendFn) -> None:
  """Helper method to set the global telemetry exporter."""
  global SEND
  SEND = send_fn
