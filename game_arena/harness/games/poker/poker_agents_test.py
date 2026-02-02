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

import json
import unittest.mock

from game_arena.harness import model_generation
from game_arena.harness.games.poker import poker_agents
import kaggle_environments
from kaggle_environments.envs.open_spiel_env import open_spiel_env

from absl.testing import absltest


DEFAULT_UNIVERSAL_POKER_GAME_STRING = (
    "universal_poker("
    "betting=nolimit,"
    "bettingAbstraction=fullgame,"
    "blind=2 1,"
    "firstPlayer=2 1 1 1,"
    "numBoardCards=0 3 1 1,"
    "numHoleCards=2,"
    "numPlayers=2,"
    "numRanks=13,"
    "numRounds=4,"
    "numSuits=4,"
    "stack=200 200)"
)

DEFAULT_REPEATED_POKER_GAME_STRING = (
    "repeated_poker("
    "max_num_hands=100,"
    "reset_stacks=True,"
    "rotate_dealer=True,"
    f"universal_poker_game_string={DEFAULT_UNIVERSAL_POKER_GAME_STRING})"
)


class PokerAgentsTest(absltest.TestCase):

  def test_stepping_repeated_poker_with_rethink_agent_in_env(self):
    p0_mock_model = unittest.mock.MagicMock()
    p0_mock_responses = [
        "Thinking......\nFinal Answer: bet20",
    ]
    p0_mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response=response,
            main_response_and_thoughts=response,
        )
        for response in p0_mock_responses
    ]
    p0_agent = poker_agents.build_repeated_poker_rethink_agent(p0_mock_model)
    p1_mock_model = unittest.mock.MagicMock()
    p1_mock_responses = [
        "Thinking......\nFinal Answer: bet10",
        "Thinking......\nFinal Answer: call",
    ]
    p1_mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response=response,
            main_response_and_thoughts=response,
        )
        for response in p1_mock_responses
    ]
    p1_agent = poker_agents.build_repeated_poker_rethink_agent(p1_mock_model)
    agents = [p0_agent, p1_agent]

    game_id = "repeated_poker"
    envs = open_spiel_env._register_game_envs(
        [DEFAULT_REPEATED_POKER_GAME_STRING]
    )
    env_name = envs[f"open_spiel_{game_id}"]
    env = kaggle_environments.make(env_name, debug=True)

    env.reset(len(agents))
    for _ in range(4):
      obs = env.state
      actions = [
          agent(obs[i]["observation"], env.configuration)
          for i, agent in enumerate(agents)
      ]
      env.step(actions)
    self.assertEqual(env.os_state.current_player(), 0)
    state_dict = json.loads(str(env.os_state))
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"]
    self.assertStartsWith(acpc_state_str, "STATE:0:r10r20c")

  def test_rethink_agent_produces_legal_action(self):
    p0_mock_model = unittest.mock.MagicMock()
    p0_agent = poker_agents.build_repeated_poker_rethink_agent(p0_mock_model)

    p1_mock_model = unittest.mock.MagicMock()
    # First, an illegal move, then a legal one.
    p1_mock_responses = [
        "Thinking... I will play an ILLEGAL move. Final Answer: ILLEGAL",
        "Thinking... That was illegal. I'll bet 10 now. Final Answer: bet10",
    ]
    p1_mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response=response,
            main_response_and_thoughts=response,
        )
        for response in p1_mock_responses
    ]
    p1_agent = poker_agents.build_repeated_poker_rethink_agent(p1_mock_model)

    agents = [p0_agent, p1_agent]

    game_id = "repeated_poker"
    envs = open_spiel_env._register_game_envs(
        [DEFAULT_REPEATED_POKER_GAME_STRING]
    )
    env_name = envs[f"open_spiel_{game_id}"]
    env = kaggle_environments.make(env_name, debug=True)

    env.reset(len(agents))
    env.step([{"submission": -1}, {"submission": -1}])
    # Let p1 make one move.
    obs = env.state
    print(obs)
    actions = [
        agent(obs[i]["observation"], env.configuration)
        for i, agent in enumerate(agents)
    ]
    env.step(actions)

    # Check that p1's model was called twice due to rethink.
    self.assertEqual(p1_mock_model.generate_with_text_input.call_count, 2)
    # Check that p0's model was not called because it was not its turn.
    self.assertEqual(p0_mock_model.generate_with_text_input.call_count, 0)

    # Check that the prompt for the second call contains the rethink text.
    calls = p1_mock_model.generate_with_text_input.call_args_list
    self.assertLen(calls, 2)
    prompt_text_1 = calls[0].args[0].prompt_text
    self.assertNotIn("A legal action could not be parsed", prompt_text_1)
    prompt_text_2 = calls[1].args[0].prompt_text
    self.assertIn("A legal action could not be parsed", prompt_text_2)
    self.assertIn(p1_mock_responses[0], prompt_text_2)

    # Check that the final state reflects p0 betting 10.
    state_dict = json.loads(str(env.os_state))
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"]
    # Action history should show a raise to 10.
    self.assertStartsWith(acpc_state_str, "STATE:0:r10")

  def test_rethink_agent_produces_legal_action_remapped(self):
    p0_mock_model = unittest.mock.MagicMock()
    p0_agent = poker_agents.build_repeated_poker_rethink_agent(p0_mock_model)

    p1_mock_model = unittest.mock.MagicMock()
    # First, an illegal move, then a legal one.
    p1_mock_responses = [
        "Thinking... I will play an ILLEGAL move. Final Answer: ILLEGAL",
        "Thinking... That was illegal. Now I forgot to answer.",
    ]
    p1_mock_model.generate_with_text_input.side_effect = [
        model_generation.GenerateReturn(
            main_response=response,
            main_response_and_thoughts=response,
        )
        for response in p1_mock_responses
    ]
    p1_agent = poker_agents.build_repeated_poker_rethink_agent(p1_mock_model)

    agents = [p0_agent, p1_agent]

    game_id = "repeated_poker"
    envs = open_spiel_env._register_game_envs(
        [DEFAULT_REPEATED_POKER_GAME_STRING]
    )
    env_name = envs[f"open_spiel_{game_id}"]
    env = kaggle_environments.make(env_name, debug=True)

    env.reset(len(agents))
    env.step([{"submission": -1}, {"submission": -1}])
    # Let p1 make one move.
    obs = env.state
    print(obs)
    actions = [
        agent(obs[i]["observation"], env.configuration)
        for i, agent in enumerate(agents)
    ]
    env.step(actions)

    # Check that p1's model was called twice due to rethink.
    self.assertEqual(p1_mock_model.generate_with_text_input.call_count, 2)
    # Check that p0's model was not called because it was not its turn.
    self.assertEqual(p0_mock_model.generate_with_text_input.call_count, 0)

    # Check that the prompt for the second call contains the rethink text.
    calls = p1_mock_model.generate_with_text_input.call_args_list
    self.assertLen(calls, 2)
    prompt_text_1 = calls[0].args[0].prompt_text
    self.assertNotIn("A legal action could not be parsed", prompt_text_1)
    prompt_text_2 = calls[1].args[0].prompt_text
    self.assertIn("A legal action could not be parsed", prompt_text_2)
    self.assertIn(p1_mock_responses[0], prompt_text_2)

    # Check that the final state reflects p0 being folded.
    state_dict = json.loads(str(env.os_state))
    up_state_dict = json.loads(state_dict["prev_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"]
    # Action history should show a raise to 10.
    self.assertStartsWith(acpc_state_str, "STATE:0:f")


if __name__ == "__main__":
  absltest.main()
