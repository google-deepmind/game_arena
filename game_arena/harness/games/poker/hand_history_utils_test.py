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

from game_arena.harness.games.poker import hand_history_utils as hh_utils
from absl.testing import absltest
import pyspiel


class HandHistoryUtilsTest(absltest.TestCase):

  def test_pluribus_hh(self):
    acpc_state_str = "STATE:102:ffr225cff/cr825f:KcJd|4dTc|8dTh|3h8s|8cQc|5h6h/As5cJs:-50|-100|0|0|-225|375:Budd|MrWhite|MrOrange|Hattori|MrBlue|Pluribus"
    cfg = hh_utils.Config(
        seats=6,
        small_blind=50,
        big_blind=100,
        starting_stacks=[10000] * 6,
    )
    hand, _ = hh_utils.parse_acpc_line(
        acpc_state_str, cfg=cfg, policy=hh_utils.LegacyACPCPolicy()
    )
    self.assertEqual(hand.hand_id, "102")
    self.assertLen(hand.players, 6)
    self.assertEqual(hand.profits, [-50, -100, 0, 0, -225, 375])
    self.assertEqual(hand.players[0].id, "Budd")
    self.assertEqual(hand.players[5].id, "Pluribus")
    self.assertEqual(
        [str(c) for c in hand.hole_cards[0]], ["Kc", "Jd"]
    )
    self.assertEqual(
        [str(c) for c in hand.hole_cards[5]], ["5h", "6h"]
    )
    self.assertLen(hand.community, 1)
    self.assertEqual(
        [str(c) for c in hand.community[0]], ["As", "5c", "Js"]
    )
    # 2 blind events, 6 preflop, 3 flop = 11 events
    self.assertLen(hand.events, 11)
    self.assertEqual(hand.events[-3].kind, hh_utils.ActionKind.CHECK)
    self.assertEqual(hand.events[-3].actor, 4)  # MrBlue
    self.assertEqual(hand.events[-2].kind, hh_utils.ActionKind.BET)
    self.assertEqual(hand.events[-2].actor, 5)  # Pluribus
    self.assertEqual(hand.events[-2].delta, 600)
    self.assertEqual(hand.events[-1].kind, hh_utils.ActionKind.FOLD)
    self.assertEqual(hand.events[-1].actor, 4)  # MrBlue
    result = """
<pokersite> Hand #102: Hold'em No Limit (50/100)
Table '' 6-max (USD) Seat #6 is the button
Seat 1: Budd (10000 in chips)
Seat 2: MrWhite (10000 in chips)
Seat 3: MrOrange (10000 in chips)
Seat 4: Hattori (10000 in chips)
Seat 5: MrBlue (10000 in chips)
Seat 6: Pluribus (10000 in chips)
Budd: posts small blind 50
MrWhite: posts big blind 100
*** HOLE CARDS ***
Dealt to Budd [Kc Jd]
Dealt to MrWhite [4d Tc]
Dealt to MrOrange [8d Th]
Dealt to Hattori [3h 8s]
Dealt to MrBlue [8c Qc]
Dealt to Pluribus [5h 6h]
MrOrange: folds
Hattori: folds
MrBlue: raises 125 to 225
Pluribus: calls 225
Budd: folds
MrWhite: folds
*** FLOP *** [As 5c Js]
MrBlue: checks
Pluribus: bets 600
MrBlue: folds
Uncalled bet (600) returned to Pluribus
Pluribus collected 600.0 from pot
*** SUMMARY ***
Total pot 600 | Rake 0
Board [As 5c Js]
Seat 1: Budd folded
Seat 2: MrWhite folded
Seat 3: MrOrange folded
Seat 4: Hattori folded
Seat 5: MrBlue folded
""".strip()
    self.assertEqual(
        hh_utils.render_pokersite(hand, sitename="<pokersite>"),
        result,
    )

  def test_repeated_poker_hh(self):
    game_str = (
        "repeated_poker(max_num_hands=3,reset_stacks=True,rotate_dealer=True,"
        "universal_poker_game_string=universal_poker(betting=nolimit,"
        "bettingAbstraction=fullgame,blind=2 1,firstPlayer=2 1 1 1,"
        "numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=2,numRanks=13,"
        "numRounds=4,numSuits=4,stack=200 200))"
    )
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()
    action_strs = [
        "player=-1 move=Deal Qc",
        "player=-1 move=Deal 6d",
        "player=-1 move=Deal Qd",
        "player=-1 move=Deal 3s",
        "player=1 move=Bet5",
        "player=0 move=Call",
        "player=-1 move=Deal 8s",
        "player=-1 move=Deal Kc",
        "player=-1 move=Deal Tc",
        "player=0 move=Bet8",
        "player=1 move=Fold",
    ]
    for action_str in action_strs:
      state.apply_action(state.string_to_action(action_str))
    hand_num = 0
    acpc_state_str = state.acpc_hand_histories()[hand_num]
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=state.num_players(),
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=hand_num % 2 + 1,
    )
    public_hh = hh_utils.render_pokersite(
        hh, observer_id=None, sitename="<pokersite>"
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 2-max (USD) Seat #2 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Player1: posts small blind 1
Player0: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [Qc 6d]
Dealt to Player1 [Qd 3s]
Player1: raises 3 to 5
Player0: calls 3
*** FLOP *** [8s Kc Tc]
Player0: bets 3
Player1: folds
Uncalled bet (3) returned to Player0
Player0 collected 10.0 from pot
*** SUMMARY ***
Total pot 10 | Rake 0
Board [8s Kc Tc]
Seat 2: Player1 folded
""".strip()
    self.assertEqual(public_hh, expected_result)
    p0_hh = hh_utils.render_pokersite(
        hh, observer_id="Player0", sitename="<pokersite>"
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 2-max (USD) Seat #2 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Player1: posts small blind 1
Player0: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [Qc 6d]
Dealt to Player1 [?? ??]
Player1: raises 3 to 5
Player0: calls 3
*** FLOP *** [8s Kc Tc]
Player0: bets 3
Player1: folds
Uncalled bet (3) returned to Player0
Player0 collected 10.0 from pot
*** SUMMARY ***
Total pot 10 | Rake 0
Board [8s Kc Tc]
Seat 2: Player1 folded
""".strip()
    self.assertEqual(p0_hh, expected_result)
    p1_hh = hh_utils.render_pokersite(
        hh, observer_id="Player1", sitename="<pokersite>"
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 2-max (USD) Seat #2 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Player1: posts small blind 1
Player0: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [?? ??]
Dealt to Player1 [Qd 3s]
Player1: raises 3 to 5
Player0: calls 3
*** FLOP *** [8s Kc Tc]
Player0: bets 3
Player1: folds
Uncalled bet (3) returned to Player0
Player0 collected 10.0 from pot
*** SUMMARY ***
Total pot 10 | Rake 0
Board [8s Kc Tc]
Seat 2: Player1 folded
""".strip()
    self.assertEqual(p1_hh, expected_result)

  def test_repeated_poker_3p_hh(self):
    game_str = (
        "repeated_poker(max_num_hands=3,reset_stacks=True,rotate_dealer=True,"
        "universal_poker_game_string=universal_poker(betting=nolimit,"
        "bettingAbstraction=fullgame,blind=1 2 0,firstPlayer=3 1 1 1,"
        "numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=3,numRanks=13,"
        "numRounds=4,numSuits=4,stack=200 200 200))"
    )
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()
    action_strs = [
        "player=-1 move=Deal 2c",
        "player=-1 move=Deal 2d",
        "player=-1 move=Deal 3d",
        "player=-1 move=Deal 3s",
        "player=-1 move=Deal 4d",
        "player=-1 move=Deal 4s",
        "player=2 move=Fold",
        "player=0 move=Bet5",
        "player=1 move=Call",
        "player=-1 move=Deal 8s",
        "player=-1 move=Deal Kc",
        "player=-1 move=Deal Tc",
        "player=0 move=Bet27",
        "player=1 move=Bet57",
        "player=0 move=Bet200",
        "player=1 move=Call",
        "player=-1 move=Deal As",
        "player=-1 move=Deal Ac",
    ]
    for action_str in action_strs:
      state.apply_action(state.string_to_action(action_str))
    acpc_state_str = state.acpc_hand_histories()[0]
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=state.num_players(),
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=2,
    )
    public_hh = hh_utils.render_pokersite(
        hh, observer_id=None, sitename="<pokersite>"
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 3-max (USD) Seat #3 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Seat 3: Player2 (200 in chips)
Player0: posts small blind 1
Player1: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [2c 2d]
Dealt to Player1 [3d 3s]
Dealt to Player2 [4d 4s]
Player2: folds
Player0: raises 3 to 5
Player1: calls 3
*** FLOP *** [8s Kc Tc]
Player0: bets 22
Player1: raises 30 to 52
Player0: raises 143 to 195 and is all-in
Player1: calls 143
*** TURN *** [8s Kc Tc] [As]
*** RIVER *** [8s Kc Tc] [As] [Ac]
Player1 collected 400.0 from pot
*** SUMMARY ***
Total pot 400 | Rake 0
Board [8s Kc Tc As Ac]
Seat 3: Player2 folded
""".strip()
    self.assertEqual(public_hh, expected_result)

  def test_repeated_poker_4p_split_pot_hh(self):
    game_str = (
        "repeated_poker(max_num_hands=3,reset_stacks=True,rotate_dealer=True,"
        "universal_poker_game_string=universal_poker(betting=nolimit,"
        "bettingAbstraction=fullgame,blind=1 2 0 0,firstPlayer=3 1 1 1,"
        "numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=4,numRanks=13,"
        "numRounds=4,numSuits=4,stack=200 200 200 200))"
    )
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()
    action_strs = [
        "player=-1 move=Deal 5c",
        "player=-1 move=Deal 5d",
        "player=-1 move=Deal 5h",
        "player=-1 move=Deal 5s",
        "player=-1 move=Deal 4d",
        "player=-1 move=Deal 4s",
        "player=-1 move=Deal 3d",
        "player=-1 move=Deal 3s",
        "player=2 move=Call",
        "player=3 move=Call",
        "player=0 move=Call",
        "player=1 move=Call",
        "player=-1 move=Deal 8s",
        "player=-1 move=Deal Kc",
        "player=-1 move=Deal Tc",
        "player=0 move=Bet50",
        "player=1 move=Call",
        "player=2 move=Call",
        "player=3 move=Bet100",
        "player=0 move=Call",
        "player=1 move=Bet200",
        "player=2 move=Call",
        "player=3 move=Call",
        "player=0 move=Call",
        "player=-1 move=Deal As",
        "player=-1 move=Deal Ac",
    ]
    for action_str in action_strs:
      state.apply_action(state.string_to_action(action_str))
    acpc_state_str = state.acpc_hand_histories()[0]
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=state.num_players(),
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200, 200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=3,
    )
    public_hh = hh_utils.render_pokersite(
        hh, observer_id=None, sitename="<pokersite>"
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 4-max (USD) Seat #4 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Seat 3: Player2 (200 in chips)
Seat 4: Player3 (200 in chips)
Player0: posts small blind 1
Player1: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [5c 5d]
Dealt to Player1 [5h 5s]
Dealt to Player2 [4d 4s]
Dealt to Player3 [3d 3s]
Player2: calls 2
Player3: calls 2
Player0: calls 1
Player1: checks
*** FLOP *** [8s Kc Tc]
Player0: bets 48
Player1: calls 48
Player2: calls 48
Player3: raises 50 to 98
Player0: calls 50
Player1: raises 100 to 198 and is all-in
Player2: calls 150
Player3: calls 100
Player0: calls 100
*** TURN *** [8s Kc Tc] [As]
*** RIVER *** [8s Kc Tc] [As] [Ac]
Player0 collected 400.0 from pot
Player1 collected 400.0 from pot
*** SUMMARY ***
Total pot 800 | Rake 0
Board [8s Kc Tc As Ac]
""".strip()
    self.assertEqual(public_hh, expected_result)

  def test_uncalled_flop_bet_hh(self):
    acpc_state_str = "STATE:0:r5c/cr10f:Th9c|9hAd/Kc4h5h:-5|5:Player0|Player1"
    cfg = hh_utils.Config(
        seats=2,
        small_blind=1,
        big_blind=2,
        starting_stacks=[200, 200],
    )
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=cfg,
        policy=hh_utils.ButtonPolicy(),
        button_index=1,
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 2-max (USD) Seat #2 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Player1: posts small blind 1
Player0: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [Th 9c]
Dealt to Player1 [9h Ad]
Player1: raises 3 to 5
Player0: calls 3
*** FLOP *** [Kc 4h 5h]
Player0: checks
Player1: bets 5
Player0: folds
Uncalled bet (5) returned to Player1
Player1 collected 10.0 from pot
*** SUMMARY ***
Total pot 10 | Rake 0
Board [Kc 4h 5h]
Seat 1: Player0 folded
""".strip()
    self.assertEqual(
        hh_utils.render_pokersite(hh, observer_id=None, sitename="<pokersite>"),
        expected_result,
    )

  def test_uncalled_preflop_bet_hh(self):
    game_str = (
        "repeated_poker(max_num_hands=3,reset_stacks=True,rotate_dealer=True,"
        "universal_poker_game_string=universal_poker(betting=nolimit,"
        "bettingAbstraction=fullgame,blind=2 1,firstPlayer=2 1 1 1,"
        "numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=2,numRanks=13,"
        "numRounds=4,numSuits=4,stack=200 200))"
    )
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()
    action_strs = [
        "player=-1 move=Deal Qc",
        "player=-1 move=Deal 6d",
        "player=-1 move=Deal Qd",
        "player=-1 move=Deal 3s",
        "player=1 move=Bet7",
        "player=0 move=Fold",
    ]
    for action_str in action_strs:
      state.apply_action(state.string_to_action(action_str))
    hand_num = 0
    acpc_state_str = state.acpc_hand_histories()[hand_num]
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=state.num_players(),
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=hand_num % 2 + 1,
    )
    public_hh = hh_utils.render_pokersite(
        hh, observer_id=None, sitename="<pokersite>",
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 2-max (USD) Seat #2 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Player1: posts small blind 1
Player0: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [Qc 6d]
Dealt to Player1 [Qd 3s]
Player1: raises 5 to 7
Player0: folds
Uncalled bet (5) returned to Player1
Player1 collected 4.0 from pot
*** SUMMARY ***
Total pot 4 | Rake 0
Seat 1: Player0 folded
""".strip()
    self.assertEqual(public_hh, expected_result)

  def test_repeated_poker_preflop_3p_fold_around(self):
    game_str = (
        "repeated_poker(max_num_hands=3,reset_stacks=True,rotate_dealer=True,"
        "universal_poker_game_string=universal_poker(betting=nolimit,"
        "bettingAbstraction=fullgame,blind=1 2 0 0,firstPlayer=3 1 1 1,"
        "numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=4,numRanks=13,"
        "numRounds=4,numSuits=4,stack=200 200 200 200))"
    )
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()
    action_strs = [
        "player=-1 move=Deal 2c",
        "player=-1 move=Deal 2d",
        "player=-1 move=Deal 3d",
        "player=-1 move=Deal 3s",
        "player=-1 move=Deal 4d",
        "player=-1 move=Deal 4s",
        "player=-1 move=Deal 5d",
        "player=-1 move=Deal 5s",
        "player=2 move=Fold",
        "player=3 move=Fold",
        "player=0 move=Fold",
    ]
    for action_str in action_strs:
      state.apply_action(state.string_to_action(action_str))
    acpc_state_str = state.acpc_hand_histories()[0]
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=state.num_players(),
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200, 200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=3,
    )
    public_hh = hh_utils.render_pokersite(
        hh, observer_id=None, sitename="<pokersite>",
    )
    expected_result = """
<pokersite> Hand #0: Hold'em No Limit (1/2)
Table '' 4-max (USD) Seat #4 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Seat 3: Player2 (200 in chips)
Seat 4: Player3 (200 in chips)
Player0: posts small blind 1
Player1: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [2c 2d]
Dealt to Player1 [3d 3s]
Dealt to Player2 [4d 4s]
Dealt to Player3 [5d 5s]
Player2: folds
Player3: folds
Player0: folds
Uncalled bet (1) returned to Player1
Player1 collected 2.0 from pot
*** SUMMARY ***
Total pot 2 | Rake 0
Seat 1: Player0 folded
Seat 3: Player2 folded
Seat 4: Player3 folded
""".strip()
    self.assertEqual(public_hh, expected_result)

  def test_in_progress_hand(self):
    # pylint: disable=line-too-long
    serialized_game_and_state = "# Automatically generated by OpenSpiel SerializeGameAndState\n[Meta]\nVersion: 1\n\n[Game]\nrepeated_poker(max_num_hands=100,reset_stacks=True,rotate_dealer=True,universal_poker_game_string=universal_poker(betting=nolimit,bettingAbstraction=fullgame,blind=2 1,calcOddsNumSims=1000000,firstPlayer=2 1 1 1,numBoardCards=0 3 1 1,numHoleCards=2,numPlayers=2,numRanks=13,numRounds=4,numSuits=4,stack=200 200))\n[State]\n37\n40\n2\n50\n5\n\n"
    # pylint: enable=line-too-long
    _, state = pyspiel.deserialize_game_and_state(serialized_game_and_state)
    players = [
        f"Player{i}" for i in range(state.num_players())
    ]
    state_dict = json.loads(str(state))
    up_state_dict = json.loads(state_dict["current_universal_poker_json"])
    acpc_state_str = up_state_dict["acpc_state"]
    acpc_state_str = acpc_state_str.split("\n")[0]
    if not acpc_state_str.startswith("STATE:"):
      raise ValueError(
          f"Expected ACPC state to start with STATE:, got {acpc_state_str}"
      )
    # Pluribus style:
    acpc_state_str = acpc_state_str + "::" + "|".join(players)
    hh, _ = hh_utils.parse_acpc_line(
        acpc_state_str,
        cfg=hh_utils.Config(
            seats=state.num_players(),
            small_blind=1,
            big_blind=2,
            starting_stacks=[200, 200],
        ),
        policy=hh_utils.ButtonPolicy(),
        button_index=(state_dict["hand_number"] % 2) + 1,
        hand_id_override=str(state_dict["hand_number"]),
    )
    observer_id = f"Player{state.current_player()}"
    readable_state_str = hh_utils.render_pokersite(
        hand=hh,
        observer_id=observer_id,
        sitename=""
    )
    expected_result = """
Hand #0: Hold'em No Limit (1/2)
Table '' 2-max (USD) Seat #2 is the button
Seat 1: Player0 (200 in chips)
Seat 2: Player1 (200 in chips)
Player1: posts small blind 1
Player0: posts big blind 2
*** HOLE CARDS ***
Dealt to Player0 [Jd Qc]
Dealt to Player1 [?? ??]
Player1: raises 3 to 5
""".strip()
    self.assertEqual(readable_state_str, expected_result)

if __name__ == "__main__":
  absltest.main()
