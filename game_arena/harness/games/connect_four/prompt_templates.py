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

"""Connect X prompt templates."""

CONNECT_X_PROMPT = """
You are a world-class Connect X AI. Your task is to analyze the current game state
and make the optimal move.

I. Game Rules & Configuration

A. Game Name: Connect X (Generalized Connect Four).
B. Board Size: The board has {rows} rows and {columns} columns.
C. Gravity: Disks fall to the lowest empty spot in a column.
D. Win Condition: The first player to get {in_a_row} of their pieces in a row (horizontally, vertically, or diagonally) wins.
E. Legal Moves: **You cannot put your piece in a column if the top row (Row 0) is occupied.**
F. Column Indices: Columns are 0-indexed from left to right (0 to {max_column_index}).
G. You are playing as player {player_name}.

II. Input Data Format

Current Board State:

{visual_board_state}

III. Required Final Answer Format

All responses MUST start with your **reasoning** and conclude with the final
answer.
The final answer MUST be on a single, final, new line.
The final answer line MUST be in the precise format:

Final Answer: <column_index>

Where <column_index> is a single integer representing the **zero-based column
index**.
Action is on you (Player {player_name}). Choose the optimal column.
{rethink_prompt}
""".strip()

CONNECTX_RETHINK = """
A legal action (a single integer column index) could not be parsed from your previous response.
Think carefully and respond with a legal, optimal column index.
Remember to include the final answer on the final line of your response.
It must EXACTLY follow the specified final answer format:
Final Answer: <column_index>

Your previous response concluded with:
{generation}
""".strip()
