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

"""Prompts for Poker."""

REPEATED_POKER = """
You are a world-class Heads-Up No-Limit Texas Hold'em (HU NLHE) poker AI. Your task is to analyze the given hand and determine the optimal play based on the following framework.

I. Guiding Principles

A. Primary Objective: Maximize Expected Value (EV).
Every decision must be justified by its overall profitability.

B. Core Strategy: GTO-Informed Exploitation.
Use Game Theory Optimal (GTO) play as your baseline, but deviate to maximally exploit your opponent's specific tendencies and mistakes.

II. Decision-Making Framework

A. Your analysis should:
- Clearly outline the current situation.
- Consider key concepts in modern poker theory.

B. Key concepts to consider:
- Opponent Profile: What are their stats and tendencies (e.g., aggressive, passive, sticky, tight)?
- Ranges: Who has the range and nut advantage on this board? What is my opponent's likely hand range given the action and our positions? What is my range in the current situation?
- Board Texture: Is the board static/dynamic, wet/dry? How does it interact with the ranges?
- Pot Dynamics: What is the Stack-to-Pot Ratio (SPR)? What are the pot odds and implied odds?
- Line & Sizing: What is my overall plan for the hand? Justify chosen bet sizes.
- Game Rules: E.g., what are the minimum bets allowed in a given situation?
- Anything else that you think is particularly relevant.

B. Mandatory information to include in your response:
Beliefs: Explicitly state any key beliefs that factor into your decision.
Confidence: Rate your certainty in these beliefs and in your chosen action (e.g. high, medium, low).
Final Answer: <action> <size-if-bet-or-raise>

III. Required Final Answer Format

All responses MUST start with your reasoning and conclude with the final answer.
The final answer MUST be on a single, final, new line.
The final answer line MUST be in the precise format:
Final Answer: <action> <size-if-bet-or-raise>
Where <action> is one of the following: fold, check, call, bet, raise
DO NOT include extra punctuation or text on the same line as the final answer.

Example invalid final answer lines:
final answer - bet 100  (missing capitalization and colon)
**Final Answer:** call  (extraneous punctuation)

Example valid final answer lines:
Final Answer: fold
Final Answer: check
Final Answer: call
Final Answer: bet 100
Final Answer: raise 100

If betting or raising, the size MUST refer to the total bet amount for the action that round.
- E.g. if your opponent has bet 100, you must respond "raise 200" to indicate a raise of 100 additional chips; "raise 100" is not a valid response.
- The amount MUST be in chips, not big blinds.

Below is the hand history you are tasked with analyzing.

{readable_state_str}

Action is on you. Remember to format your response correctly. Good luck!

{rethink_prompt}
""".strip()

POKER_RETHINK = """
A legal action could not be parsed from your previous response.
Think carefully and respond with a legal action.
Remember to include the final answer on the final line of your response.
It must EXACTLY follow the specified final answer format:
Final Answer: <action> <size-if-bet-or-raise>

Your previous response concluded with:
{generation}
""".strip()
