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

"""Demo of chess, prompt generation, model generation, and parser together."""

from absl import app
from absl import flags
from game_arena.harness import game_notation_examples, model_generation_http
from game_arena.harness import llm_parsers
from game_arena.harness import parsers
from game_arena.harness import prompt_generation
from game_arena.harness import prompts
from game_arena.harness import tournament_util
import termcolor

import pyspiel


colored = termcolor.colored

_NUM_MOVES = flags.DEFINE_integer(
    "num_moves",
    3,
    "Number of moves to play.",
)

_DEEP_SEEK_MODEL = flags.DEFINE_string(
    "deep_seek_model",
    "deepseek-llm-7b-chat.Q4_K_M",
    "Deepseek model to play as player one.",
)

_GPT_OSS_MODEL = flags.DEFINE_string(
    "gpt_oss_model",
    "gpt-oss-20b-Q3_K_M",
    "GPT-OSS model to play as player two.",
)

_OPENAI_MODEL = flags.DEFINE_string(
    "openai_model",
    "gpt-4.1",
    "Deepseek model to play as player two.",
)

_PARSER_CHOICE = flags.DEFINE_enum_class(
    "parser_choice",
    tournament_util.ParserChoice.RULE_THEN_SOFT,
    tournament_util.ParserChoice,
    "Move parser to use.",
)


def main(_) -> None:
  # Set up game:
  pyspiel_game = pyspiel.load_game("chess")
  pyspiel_state = pyspiel_game.new_initial_state()

  # Set up prompt generator:
  prompt_generator = prompt_generation.PromptGeneratorText()
  prompt_template = prompts.PromptTemplate.WITH_LEGAL_ACTIONS

  # Set up model generation:
  model_player_one = model_generation_http.OpenAIGenericAPIModel(
    model_name=_DEEP_SEEK_MODEL.value,
    api_endpoint="http://127.0.0.1:8081/v1/chat/completions",
    local_model=True,
    api_options={
      "stream": False
    },
    model_options={
      "image_support": False
    }
  )

  model_player_two = model_generation_http.OpenAIGenericAPIModel(
    model_name=_DEEP_SEEK_MODEL.value,
    api_endpoint="http://127.0.0.1:8081/v1/chat/completions",
    local_model=True,
    api_options={
      "stream": False
    },
    model_options={
      "image_support": False
    }
  )

  # Set up parser;
  # RULE_THEN_SOFT: rule-based (regex, replace, strip) then soft-matching
  # against legal moves
  # LLM_ONLY: feed the game-playing model's response to a separate LLM for
  # move parsing
  match _PARSER_CHOICE.value:
    case tournament_util.ParserChoice.RULE_THEN_SOFT:
      parser = parsers.ChainedMoveParser(
          [parsers.RuleBasedMoveParser(), parsers.SoftMoveParser("chess")]
      )
    case tournament_util.ParserChoice.LLM_ONLY:
      parser_model = model_generation_http.OpenAIGenericAPIModel(
        model_name=_DEEP_SEEK_MODEL.value,
        api_endpoint="http://127.0.0.1:8082/v1/chat/completions",
        local_model=True,
        api_options={
          "stream": False
        },
        model_options={
          "image_support": False
        }
      )
      parser = llm_parsers.LLMParser(
          model=parser_model,
          instruction_config=llm_parsers.OpenSpielChessInstructionConfig_V0,
      )
    case _:
      raise ValueError(f"Unsupported parser choice: {_PARSER_CHOICE.value}")

  for move_number in range(_NUM_MOVES.value):
    print(f"Pre-move debug string: {pyspiel_state.debug_string()}")
    if pyspiel_state.is_terminal():
      print(colored("Game is terminal, ending move loop.", "red"))
      break

    print(colored(f"Commencing move {move_number}...", "green"))

    # 1. Generate the prompt from the game state:
    prompt_substitutions = {
        "readable_state_str": tournament_util.convert_to_readable_state(
            game_short_name="chess",
            state_str=pyspiel_state.to_string(),
            current_player=pyspiel_state.current_player(),
        ),
        "move_history": (
            tournament_util.get_action_string_history(pyspiel_state) or "None"
        ),
        "player_name": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
            "player_map"
        ][pyspiel_state.current_player()],
        "move_notation": game_notation_examples.GAME_SPECIFIC_NOTATIONS[
            "chess"
        ]["move_notation"],
        "notation": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
            "state_notation"
        ],
        "legal_actions": parsers.get_legal_action_strings(pyspiel_state),
    }
    prompt = prompt_generator.generate_prompt_with_text_only(
        prompt_template=prompt_template,
        game_short_name="chess",
        **prompt_substitutions,
    )
    print(colored(f"Formatted prompt: {prompt.prompt_text}", "blue"))

    # 2. Call the model:
    if pyspiel_state.current_player() == 0:
      model = model_player_one
    else:
      model = model_player_two
    response = model.generate_with_text_input(prompt)
    print(
        colored(
            f"Model player {pyspiel_state.current_player()} main response:"
            f" {response.main_response}",
            "yellow",
        )
    )

    # 3. Parse the model response:
    parser_input = parsers.TextParserInput(
        text=response.main_response,
        # TODO(google-deepmind): raw state str and readable state str should be
        # differentiated in signatures.
        state_str=pyspiel_state.to_string(),
        legal_moves=parsers.get_legal_action_strings(pyspiel_state),
        player_number=pyspiel_state.current_player(),
    )
    parser_output = parser.parse(parser_input)
    if parser_output is None:
      print(colored("Parser output is None, ending game.", "red"))
      return
    else:
      print(colored(f"Parser output is {parser_output}.", "magenta"))

    # 4. Apply the move:
    pyspiel_state.apply_action(pyspiel_state.string_to_action(parser_output))

  print(colored("Game over.", "green"))


if __name__ == "__main__":
  app.run(main)
