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

"""Tests for basic telemetry functionality. Does not cover implementations."""

from unittest import mock

from absl.testing import absltest
from game_arena.harness import telemetry


class TelemetryTest(absltest.TestCase):

  def test_get_logger(self):
    mock_send = mock.Mock()
    with mock.patch.object(telemetry, 'SEND', mock_send):
      logger = telemetry.get_logger('test_module')
      logger(key1='value1', key2=42)

    mock_send.assert_has_calls([
        mock.call(module='test_module', key1='value1', key2=42),
    ])


if __name__ == '__main__':
  absltest.main()
