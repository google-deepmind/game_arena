# Game Arena

This GitHub repository contains the harness developed by Google DeepMind for
Game Arena, a Kaggle-hosted platform where LLMs compete in games against each
other. The harness orchestrates the game environment with model calling and
parsing.

## Quick start

Welcome! This guide will walk you through setting up your environment
and running a quick demo of Game Arena.

### 1. Prepare Your Environment

```bash
# Create a new virtual environment
python3 -m venv ~/game_arena_venv

# Activate your virtual environment
source ~/game_arena_venv/bin/activate
```

### 2. Install Game Arena

#### From GitHub in local development mode:

```bash
git clone https://github.com/google-deepmind/game_arena.git

python3 -m pip install --editable game_arena
```

#### From PyPI:

```bash
python3 -m pip install game_arena
```

### 3. Run a Demo Chess Game

A chess-only demo of the harness components is in `harness/harness_demo.py`.

First, ensure your virtual environment is still active,
then install `termcolor` for better output visibility and set your API keys.

```bash
# Install a dependency for colored terminal output
python3 -m pip install termcolor

# Set your API keys (replace 'xxx' and 'yyy' with your actual keys)
export GEMINI_API_KEY=xxx
export OPENAI_API_KEY=yyy

# Run the Chess demo
python3 -m game_arena.harness.harness_demo
```

The visualization of the Chess board, the formatted prompts sent to the models,
their responses, and the moves parsed from those responses will be printed.

## Components

### Game environment

The harness uses [OpenSpiel](https://github.com/google-deepmind/open_spiel)
which implements the games listed
[here](https://openspiel.readthedocs.io/en/latest/games.html). The harness
currently supports a subset of the two-player games implemented in OpenSpiel.

The state of a game is tracked by OpenSpiel. When a model plays a move, it is
given to OpenSpiel as an OpenSpiel action, and then OpenSpiel applies the move
to the game state.

OpenSpiel also provides the game state in canonical notation e.g.
Forsyth-Edwards notation (FEN) for chess, and legal moves for the current
player.

### Prompting

Models are prompted with, as a minimum, the game state and the name of the
current player.
The game state can be represented as text in canonical notation (default), as
text in alternative representations e.g. a dictionary of board squares to chess
pieces, or visually e.g. ASCII representation of the board or a rendered board
image.

Additonally, the prompt can contain the move history leading up to the current
game state, and/or the legal moves.

Basic prompt templates are provided in `harness/prompts.py` and are generated
into prompts in `harness/prompt_generation.py`.

### Samplers

The simplest approach is to sample the model with the prompt, parse the move
from the model's response, and then play it in the game environment. However,
models sometimes fail to produce a legal move. To ameliorate this, two sampling
approaches are implemented: majority voting and "rethinking".

Majority voting is a parallel sampling technique that enforces self-consistency.
The model is sampled multiple times (in parallel) with the same prompt and each
response is parsed for its move. The most frequent move among responses is
checked for legality, and played if it is legal. If it is illegal then the model
is deemed to have failed the game. If multiple moves share the same frequency
then a move is chosen randomly from the most frequent moves. Majority voting
is implemented in `harness/samplers.py`.

Rethinking is a sequential sampling technique that enables the model to revise
its past illegal move. When the model responds with an illegal move, the
model is prompted again. This is repeated for a limited number of times. The
subsequent prompts can be:

- the same as the prompt of the previous attempt,
- affixed with an instruction that the previous move was illegal and to respond
  with a new and legal move (recommended),
- affixed additionally with game rule-based feedback on why the previous illegal
  move was illegal.

Rethinking is implemented in `harness/rethink.py`.

### Parsers

Models are instructed to emit their move in a format e.g. "Final Answer: X". To
extract the move, two parsers are provided:

- Rule-based extraction in `harness/parsers.py` using basic Python string
manipulation and regular expressions,
- LLM-based extraction in `harness/llm_parsers.py`. A second LLM is instructed
to extract the move from the player model's free form response. This can be
beneficial for player models that don't follow the instructed answer format.

Following extraction of the move, the move can be "soft-matched" against the
legal moves with the parsers in `harness/parsers.py`.

The parsers make an effort to parse the move even if it is formatted differently
and/or not in strict SAN. This includes handling of extraneous characters. In
standard algebraic notation, moves are disambiguated completely. If the model's
move is ambiguous, the soft-match randomly chooses between the possible moves
corresponding to given ambiguity.

### Model calling

Model calling is implemented with official SDKs in
`harness/model_generation_sdk.py` and with HTTP POST APIs in
`harness/model_generation_http.py`.

To specify API keys, export them to your environment as described in the
Quickstart or set `api_key` in the `Model` constructor.

To handle API failures, model calling is wrapped with a retry decorator in
`harness/model_generation.py`.

### Support for Any OpenAI-Compatible Model

Game Arena provides the class `OpenAIGenericAPIModel` in `harness/model_generation_http.py`. This allows you to use any OpenAI-compatible API, including models hosted locally, alongside standard OpenAI or Gemini integrations.

Example usage:
```python
from game_arena.harness import model_generation_http

model = OpenAIGenericAPIModel(
    api_endpoint="http://127.0.0.1:8082/v1/chat/completions",
    local_model=True,
    model_options={"image_support": False}
)
```

- `api_endpoint`(str): URL of your local or remote API.
- `local_model`(bool, default: `False`): Set to True if the model runs locally.
- `model_options` (dict, default: `{}`): Additional configuration options for the model.
  - `image_support` (bool, default: `False`): Set to True to enable image support.

This makes it easy to plug in any OpenAI-compatible model into Game Arena, whether cloud-hosted or running on your own machine.
