import numpy as np
import jax
import jax.numpy as jnp
from gym.spaces import MultiDiscrete
from jax import lax
import chex
from flax import struct
from typing import Tuple, Dict
from functools import partial
from gymnax.environments.spaces import Discrete
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

INDEX_COLOR_MAP_DICT = {0: jnp.array([0, 0, 0, 1]), 1: jnp.array([0, 0, 0, 1]), 2: jnp.array([0, 0, 0, 1]),
                        3: jnp.array([0, 0, 0, 1]), 4: jnp.array([0, 0, 1, 0]), 5: jnp.array([0, 0, 1, 0]),
                        6: jnp.array([0, 0, 1, 0]), 7: jnp.array([0, 0, 1, 0]),
                        8: jnp.array([0, 1, 0, 0]), 9: jnp.array([0, 1, 0, 0]), 10: jnp.array([0, 1, 0, 0]),
                        11: jnp.array([0, 1, 0, 0]), 12: jnp.array([1, 0, 0, 0]), 13: jnp.array([1, 0, 0, 0]),
                        14: jnp.array([1, 0, 0, 0]), 15: jnp.array([1, 0, 0, 0])}

INDEX_COLOR_MAP = jnp.array(
    [jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 0, 1]),
     jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 1, 0]), jnp.array([0, 0, 1, 0]),
     jnp.array([0, 0, 1, 0]), jnp.array([0, 0, 1, 0]),
     jnp.array([0, 1, 0, 0]), jnp.array([0, 1, 0, 0]), jnp.array([0, 1, 0, 0]),
     jnp.array([0, 1, 0, 0]), jnp.array([1, 0, 0, 0]), jnp.array([1, 0, 0, 0]),
     jnp.array([1, 0, 0, 0]), jnp.array([1, 0, 0, 0])]
)


@struct.dataclass
class State:

    op_area_width_height: int  # 16
    move_area_width_height: int  # 18
    index_color_map: chex.Array  # records the color of 16 yokai cards
    index_color_map_dict: dict  # records the color of 16 yokai cards
    yokai_card_config: chex.Array  # 16*16*2 array (1st layer: card presence, 2nd layer: card index)
    yokai_card_world: chex.Array  # 16*16*4 array, show all the color of yokai cards
    hint_card: chex.Array  # save all the hint cards color
    hint_card_revealed: chex.Array  # num of hint card *1, 0 means not revealed, 1 means revealed
    hint_card_placed: chex.Array  # num of hint card *1, 0 means not placed, 1 means placed
    unlocked_yokai_card: chex.Array  # 16*16 array, 0 means locked or no yokai card, 1 means there is a yokai card with unlocked state
    hint_card_terminal: bool  # indicate if one round of the game ends or not
    player_terminal: bool  # indicate if one of the player ends the game or not
    card_knowledge: chex.Array  # tracks what each player knows about the yokai: num_player * 16 * 16 * 9
    cur_player_idx: chex.Array  # num of players * 1 array, indicates the current activate player
    is_graph: bool  # if True the obs is a graph ,if False, the obs is a vector



class Yokai(MultiAgentEnv):

    def __init__(self, num_agents=2, num_colors=4, max_hint_tokens=14, max_hint_of_color_num=np.array([4, 6, 4]),
                 deck_size=16, move_size=18, num_hint_of_color_num=np.array([2, 3, 2]), total_card = None,agents=None,
                 action_spaces=None,
                 obs_size=None,obs_size_graph = None, num_terminate=None, num_moves_ob=None, num_moves_move=None, num_moves_hint=None,
                 hint_card_1=None,
                 hint_card_2=None, hint_card_3=None, num_hint_place=None, observation_spaces=None):
        super().__init__(num_agents)

        self.num_agents = num_agents
        self.agent_range = jnp.arange(num_agents)
        self.num_colors = num_colors
        self.max_hint_tokens = max_hint_tokens
        self.max_hint_of_color_num = max_hint_of_color_num
        self.num_hint_of_color_num = num_hint_of_color_num
        self.hint_size = jnp.sum(num_hint_of_color_num)  # total num of the chosen hint cards
        self.deck_size = deck_size  # numbers of total yokai cards
        self.move_size = move_size
        if hint_card_1 is None:
            total1 = jnp.array(
                [jnp.array([0, 0, 0, 1]), jnp.array([0, 0, 1, 0]), jnp.array([0, 1, 0, 0]), jnp.array([1, 0, 0, 0])])
            total1 = total1.astype(int)
            self.hint_card_total1 = total1
        if total_card is None:
            self.total_card = self.deck_size+self.hint_size
        if hint_card_2 is None:
            total2 = jnp.array([jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 0, 1]), jnp.array([1, 0, 0, 1]),
                                jnp.array([0, 1, 1, 0]), jnp.array([1, 0, 1, 0]), jnp.array([1, 1, 0, 0])])
            total2 = total2.astype(int)
            self.hint_card_total2 = total2
        if hint_card_3 is None:
            total3 = jnp.array(
                [jnp.array([0, 1, 1, 1]), jnp.array([1, 0, 1, 1]), jnp.array([1, 1, 0, 1]), jnp.array([1, 1, 1, 0])])
            total3 = total3.astype(int)
            self.hint_card_total3 = total3

        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert len(
                agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents
        if num_terminate is None:
            self.num_terminate = np.sum(np.array([
                # noop
                1,
                # not terminate
                1,
                # terminate
                1
            ])).squeeze()
        if num_moves_ob is None:
            self.num_moves_ob = np.sum(np.array([
                1,
                # observe one of the 16 yokai cards
                deck_size
            ])).squeeze()  # step1 17
        if num_moves_move is None:
            self.num_moves_move = np.sum(np.array([
                1,
                # move one of the 16 yokai cards next to the up,down, left, right position
                # of one of the 16 yokai cards
                deck_size ** 2 * 4
            ])).squeeze()
        if num_moves_hint is None:
            self.num_moves_hint = np.sum(np.array([
                1,
                # reveal one of the hint card
                self.hint_size,
                # place one of the hint card onto one of the yokai card
                self.hint_size * deck_size
            ])).squeeze()
        if num_hint_place is None:
            self.num_hint_place = np.sum(np.array([
                self.hint_size * deck_size
            ])).squeeze()
        if obs_size is None:
            self.obs_size = (
                    self.deck_size * self.deck_size * (self.num_colors + 2)
                    # current agent index
                    + self.num_agents
                    # current revealed hint card
                    + self.num_colors * self.hint_size
                    # current placed hint card
                    + self.hint_size * self.num_colors)  # 1594
        if obs_size_graph is None:
            self.obs_size_graph = (
                (self.deck_size + self.hint_size)**2
                    + (4+1+2)*(self.deck_size + self.hint_size)
                   )  # 690
        if action_spaces is None:
            self.action_spaces = {
                i: MultiDiscrete([self.num_moves_ob, self.num_moves_ob, self.num_moves_move, self.num_moves_hint]) for i
                in self.agents}
        if observation_spaces is None:
            self.observation_spaces = {i: Discrete(self.obs_size_graph) for i in self.agents}


    @partial(jax.jit, static_argnums=[0])
    def get_legal_moves(self, state: State) -> chex.Array:

        @partial(jax.vmap, in_axes=[0, None])
        def _legal_moves(aidx: int, state: State) -> chex.Array:
            '''
            0: no op
            1-16 observe card with index 0 - 15
            currently, it is legal to observe one card twice for one agent in a round.
            '''
            legal_moves_ob = jnp.zeros(self.num_moves_ob, dtype=jnp.int32)
            unlock_state = state.unlocked_yokai_card
            # select the index all not locked card as observable card
            yokai_config = state.yokai_card_config
            yokai_config_index = yokai_config[:, :, 1:].reshape(16, 16)
            unlock_index = unlock_state.reshape(self.deck_size, self.deck_size) * yokai_config_index.reshape(
                self.deck_size, self.deck_size) + unlock_state.reshape(self.deck_size, self.deck_size)
            unlock_index = unlock_index.flatten()
            legal_moves_ob = legal_moves_ob.at[unlock_index].set(1)
            # if not current player, no op is legal, otherwise, not legal
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            not_cur_player = (aidx != cur_player)
            legal_moves_ob -= legal_moves_ob * not_cur_player
            legal_moves_ob = legal_moves_ob.at[0].set(not_cur_player)

            '''
            1(no op) 16(16 card to be moved) * 16(move next to which card) * 4(the direction)
            order by
            no op
            the movable position for card from index 0 to 15
            '''
            unlock_state = state.unlocked_yokai_card
            # select the index all not locked card as observable card
            yokai_config = state.yokai_card_config
            yokai_config_index_1 = yokai_config[:, :, 1:].reshape(16, 16)
            yokai_config_index = jnp.concatenate((yokai_config[:, :, 0].reshape(self.deck_size, self.deck_size, 1),
                                                  yokai_config_index_1.reshape(self.deck_size, self.deck_size, 1)),
                                                 axis=2)
            unlock_index = unlock_state.reshape(self.deck_size, self.deck_size) * yokai_config_index_1.reshape(
                self.deck_size, self.deck_size) + unlock_state.reshape(self.deck_size, self.deck_size)
            unlock_index = unlock_index.flatten()
            all_candidates = jnp.zeros(self.deck_size ** 2 * 4, dtype=jnp.int32)

            def _body_large(i, all_candidates):
                card = jnp.arange(self.deck_size, dtype=jnp.int32)[i]
                '''
                get all possible movable position and the number
                '''

                def _get_targets(fifteen_cards, all_cards):
                    idx_x, idx_y = fifteen_cards.nonzero(size=self.deck_size - 1)
                    ones_array = jnp.zeros((self.move_size, self.move_size))
                    idx_x_p = idx_x + 1
                    idx_x_n = idx_x - 1
                    idx_y_p = idx_y + 1
                    idx_y_n = idx_y - 1
                    ones_array = ones_array.at[idx_x_p, idx_y].set(1)
                    ones_array = ones_array.at[idx_x_n, idx_y].set(1)
                    ones_array = ones_array.at[idx_x, idx_y_p].set(1)
                    ones_array = ones_array.at[idx_x, idx_y_n].set(1)
                    candidates = ones_array
                    target_pos = candidates * jnp.logical_not(
                        fifteen_cards)  ###if change to all cards, then means can not move to previous position
                    one_num = jnp.sum(target_pos)
                    return target_pos, one_num

                is_not_movable = jnp.logical_not(
                    jnp.isin(card + 1, unlock_index).any())  # if one card is lock, true, unlock, false
                relative_candidates = jnp.zeros(self.deck_size * 4, dtype=jnp.int32)
                all_cards = yokai_config[:, :, 0].reshape(self.deck_size, self.deck_size)
                all_cards_new = jnp.zeros((self.move_size, self.move_size))
                all_cards_new = all_cards_new.at[1:17, 1:17].set(
                    all_cards)  # 18*18 set the middle 16*16 as card presence
                index_layer = yokai_config_index[:, :, 1]
                index_layer_new = jnp.zeros((self.move_size, self.move_size))
                index_layer_new = index_layer_new.at[1:17, 1:17].set(
                    index_layer)  # 18*18 set the middle 16*16 as card index, if no card 0
                presence_layer = yokai_config_index[:, :, 0]
                presence_layer_new = jnp.zeros((self.move_size, self.move_size))
                presence_layer_new = presence_layer_new.at[1:17, 1:17].set(presence_layer)
                position = jnp.where((presence_layer_new == 1) & (index_layer_new == card), size=1)
                fifteen_cards = all_cards_new.at[position[0], position[1]].set(0)
                fifteen_cards_index = index_layer_new + presence_layer_new + presence_layer_new
                fifteen_cards_index = fifteen_cards_index.at[position[0], position[1]].set(0)  ## index from 2 to 17
                target_pos, one_num = _get_targets(fifteen_cards, all_cards)
                one_num = one_num.astype(jnp.int32)
                ones_indices = target_pos.nonzero(size=60)

                def _body(i, all_zeros):
                    flag = (i <= one_num)
                    all_zeros = all_zeros.at[i, ones_indices[0][i], ones_indices[1][i]].set(flag)
                    return all_zeros

                all_zeros = lax.fori_loop(0, one_num, _body, jnp.empty((60, self.move_size, self.move_size)))

                def _body_2(i, index_array):
                    flag = (i <= one_num)
                    index_array = index_array.at[i, 0].set(flag)
                    return index_array

                index_array = lax.fori_loop(0, one_num, _body_2, jnp.zeros((60, 1), dtype=jnp.int32))

                target_pos_one = all_zeros + fifteen_cards
                index_array = index_array[:, None]
                target_pos_one = target_pos_one * index_array
                target_pos = target_pos.astype(jnp.int32)

                def is_fully_connected(arr):
                    visited = jnp.zeros_like(arr)
                    indices = jnp.argwhere(arr == 1, size=16)
                    # Use the first '1' as a starting point for our connectivity check
                    start = indices[0]
                    visited = visited.at[start[0], start[1]].set(1)
                    visited = visited.astype(bool)
                    arr = arr.astype(bool)

                    def _body(i, visited):
                        neighbors = jnp.pad(visited, 1)[1:-1, 1:-1] | \
                                    jnp.pad(visited, 1)[2:, 1:-1] | jnp.pad(visited, 1)[:-2, 1:-1] | \
                                    jnp.pad(visited, 1)[1:-1, 2:] | jnp.pad(visited, 1)[1:-1, :-2]
                        visited |= (neighbors & arr)
                        return visited

                    visited = lax.fori_loop(0, self.deck_size, _body, visited)
                    fully_connected = jnp.all((arr == 0) | (visited == 1))
                    return fully_connected

                is_connected_initial = is_fully_connected(fifteen_cards)
                result = jnp.empty((60, self.move_size, self.move_size))

                def _true(target_pos_one, one_num, result):
                    return result

                def _false(target_pos_one, one_num, result):
                    def _body3(i, result):
                        is_connected = is_fully_connected(target_pos_one[i])
                        result = result.at[i, ones_indices[0][i], ones_indices[1][i]].set(is_connected)
                        return result

                    result = lax.fori_loop(0, one_num, _body3, jnp.empty((60, self.move_size, self.move_size)))
                    return result

                result = lax.cond(is_connected_initial, _true, _false, target_pos_one, one_num, result)
                result = jnp.sum(result, axis=0)
                target_pos = target_pos * is_connected_initial + (
                            1 - is_connected_initial) * result  # 18*18 for one card
                target_pos = target_pos - is_not_movable * target_pos
                target_pos_relative = target_pos + fifteen_cards_index

                def find_positions(arr):
                    # Define the target values and the value to search for
                    target_values = jnp.arange(2, 18)  # The numbers 2 to 17
                    search_value = 1
                    target_mask = (arr >= 2) & (arr <= 17)
                    search_mask = (arr == search_value)
                    up_mask = jnp.roll(search_mask, shift=1, axis=0)
                    down_mask = jnp.roll(search_mask, shift=-1, axis=0)
                    left_mask = jnp.roll(search_mask, shift=1, axis=1)
                    right_mask = jnp.roll(search_mask, shift=-1, axis=1)
                    up_positions = target_mask & up_mask
                    down_positions = target_mask & down_mask
                    left_positions = target_mask & left_mask
                    right_positions = target_mask & right_mask
                    positions = jnp.empty((60, 2))

                    def _body(i, positions):
                        value = target_values[i]
                        value_mask = (arr == value)
                        is_up = jnp.sum(up_positions & value_mask) > 0
                        is_down = jnp.sum(down_positions & value_mask) > 0
                        is_left = jnp.sum(left_positions & value_mask) > 0
                        is_right = jnp.sum(right_positions & value_mask) > 0

                        def _up_true(positions):
                            is_zero_pair = (positions[:, 0] == 0) & (positions[:, 1] == 0)
                            flipped_arr = jnp.flip(is_zero_pair)
                            flipped_index = jnp.argmax(flipped_arr == 0)
                            index = is_zero_pair.size - 1 - flipped_index
                            return positions.at[(index + 1) % 60, :].set(jnp.array([value - 2, 1]))

                        def _up_false(positions):
                            return positions

                        positions = lax.cond(is_up, _up_true, _up_false, positions)

                        def _down_true(positions):
                            is_zero_pair = (positions[:, 0] == 0) & (positions[:, 1] == 0)
                            flipped_arr = jnp.flip(is_zero_pair)
                            flipped_index = jnp.argmax(flipped_arr == 0)
                            index = is_zero_pair.size - 1 - flipped_index
                            return positions.at[(index + 1) % 60, :].set(jnp.array([value - 2, 2]))

                        def _down_false(positions):
                            return positions

                        positions = lax.cond(is_down, _down_true, _down_false, positions)

                        def _left_true(positions):
                            is_zero_pair = (positions[:, 0] == 0) & (positions[:, 1] == 0)
                            flipped_arr = jnp.flip(is_zero_pair)
                            flipped_index = jnp.argmax(flipped_arr == 0)
                            index = is_zero_pair.size - 1 - flipped_index
                            return positions.at[(index + 1) % 60, :].set(jnp.array([value - 2, 3]))

                        def _left_false(positions):
                            return positions

                        positions = lax.cond(is_left, _left_true, _left_false, positions)

                        def _right_true(positions):
                            is_zero_pair = (positions[:, 0] == 0) & (positions[:, 1] == 0)
                            flipped_arr = jnp.flip(is_zero_pair)
                            flipped_index = jnp.argmax(flipped_arr == 0)
                            index = is_zero_pair.size - 1 - flipped_index
                            return positions.at[(index + 1) % 60, :].set(jnp.array([value - 2, 4]))

                        def _right_false(positions):
                            return positions

                        positions = lax.cond(is_right, _right_true, _right_false, positions)
                        return positions

                    positions = lax.fori_loop(0, 16, _body, positions)
                    return positions

                target_pos_dir = find_positions(target_pos_relative)
                target_pos_dir = target_pos_dir.astype(jnp.int32)
                is_not_valid = (target_pos_dir[:, 0] == 0) & (target_pos_dir[:, 1] == 0)
                index_position = target_pos_dir[:, 0] * 4 + target_pos_dir[:, 1] - 1 + is_not_valid * 9999
                relative_candidates = relative_candidates.at[index_position].set(1)
                relative_candidates = relative_candidates - is_not_movable * relative_candidates

                all_candidates = lax.dynamic_update_slice(all_candidates, relative_candidates, (i * 64,))
                return all_candidates

            all_candidates = lax.fori_loop(0, self.deck_size, _body_large, all_candidates)
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            not_cur_player = (aidx != cur_player)
            all_candidates -= all_candidates * not_cur_player
            no_op = jnp.array([not_cur_player]).astype(jnp.int32)
            all_candidates = jnp.concatenate([no_op, all_candidates])
            '''
            should it be MultiDiscrete[2,2,num_hint,num_hint,num_yokai] or 1+num_hint+num_hint*num_yokai
            order by
            no op
            reveal hint card
            place 0 hint card on 0 yokai card - 15 yokai card
            place 1 hint card on 0 yokai card - 15 yokai card...
            '''
            legal_moves_hint = jnp.zeros(self.num_moves_hint, dtype=jnp.int32)
            # reveal not revealed hint card is legal
            hint_revealed = state.hint_card_revealed
            hint_not_revealed = 1 - hint_revealed
            hint_placed = state.hint_card_placed
            legal_moves_hint = lax.dynamic_update_slice(legal_moves_hint, hint_not_revealed.reshape(self.hint_size),
                                                        (1,))
            # place revealed and not placed hint card on not locked yokai card is legal
            hint_placeable = hint_revealed - hint_placed
            unlock_state = state.unlocked_yokai_card
            yokai_config = state.yokai_card_config
            yokai_config_index = yokai_config[:, :, 1:].reshape(16, 16)
            unlock_index = unlock_state.reshape(self.deck_size, self.deck_size) * yokai_config_index.reshape(
                self.deck_size, self.deck_size) + unlock_state.reshape(self.deck_size, self.deck_size)
            unlock_index = unlock_index.flatten()
            # the index array of all the placeable hint card
            indices = lax.iota(jnp.int32, self.hint_size) + 1
            hint_placeable_index = hint_placeable.reshape(self.hint_size) * indices * self.deck_size
            # Add a and b using broadcasting, resulting in a 2D array of all combinations of sums
            legal_moves_place = jnp.zeros(self.num_hint_place, dtype=jnp.int32)
            is_unlock = (unlock_index != 0)
            is_placeable = (hint_placeable_index != 0)
            is_placeable = is_placeable.reshape(self.hint_size, 1)
            is_unlock = is_unlock.reshape(1, self.deck_size ** 2)
            result_intermediate = jnp.logical_and(is_placeable, is_unlock)
            result_flat = result_intermediate.flatten()
            unlock_index = unlock_index.reshape(1, self.deck_size ** 2)
            hint_placeable_index = hint_placeable_index.reshape(self.hint_size, 1)
            positions = ((hint_placeable_index - self.deck_size) + (unlock_index - 1)).flatten()
            positions = positions * result_flat + (1 - result_flat) * 9999
            legal_moves_place = legal_moves_place.at[positions].set(result_flat)
            legal_moves_hint = lax.dynamic_update_slice(legal_moves_hint, legal_moves_place, ((1 + self.hint_size),))
            # if not current player, no op is legal, otherwise, not legal
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            not_cur_player = (aidx != cur_player)
            legal_moves_hint -= legal_moves_hint * not_cur_player
            legal_moves_hint = legal_moves_hint.at[0].set(not_cur_player)
            legal_moves = (legal_moves_ob, legal_moves_ob, all_candidates, legal_moves_hint)
            return legal_moves

        legal_moves = _legal_moves(self.agent_range, state)
        dict_result = {}
        for i, a in enumerate(self.agents):
            dict_result[a] = [legal_moves[0][i], legal_moves[1][i], legal_moves[2][i], legal_moves[3][i]]
        return dict_result
        # return {a: legal_moves[i] for i, a in enumerate(self.agents)}

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict, State]:
        """Reset the environment"""

        ### reset the yokai_card_config as a 16*16*5 array, and the middle 4*4 array is set to 1 in the first layer of the third dim
        ### and assign random colors in the 2-5 layers of the third dim
        # step 0: set constant values
        op_area_width_height = self.deck_size  # 16
        move_area_width_height = self.deck_size + 2  # 18

        # Step 1: Initialize the card_config as 16*16*5 zero array
        card = jnp.zeros((self.deck_size, self.deck_size, 2))
        # Step 2: Set the middle 4x4 region of the first dimension to 1 to indicate that the cards are located in the middle
        start_initial = int(self.deck_size / 2 - self.deck_size ** 0.5 / 2)
        end_initial = int(self.deck_size / 2 + self.deck_size ** 0.5 / 2)
        middle_indices = (
            slice(start_initial, end_initial), slice(start_initial, end_initial),
            slice(1))  # Targeting the first dimension
        card = card.at[middle_indices].set(1)
        # Step 3 & 4: Generate and shuffle colors
        numbers = np.arange(16)
        # binary_representation = np.array([list(np.binary_repr(number, width=4)) for number in numbers]).astype(int)
        key, _key = jax.random.split(key)
        # shuffled_colors = jax.random.permutation(_key, all_colors, axis=0)
        shuffled_colors = jax.random.permutation(_key, numbers, axis=0)
        # Step 5: Assign shuffled colors to the middle 4x4 region for the last four dimensions
        shuffled_colors = shuffled_colors.reshape(int(self.deck_size ** 0.5), int(self.deck_size ** 0.5), 1)
        card = card.at[start_initial:end_initial, start_initial:end_initial, slice(1, 2)].set(shuffled_colors)
        card = card.astype(jnp.int32)

        ### reset the yokai_card_world:
        def _get_world(yokai_card_config):
            card = yokai_card_config
            card_presence = card[:, :, 0] == 1
            decimal_indices = card[:, :, 1:].reshape(16, 16)  # Extract binary indices
            color_encodings = jnp.zeros((16, 16, 4))  # 4 for the one-hot encoding of colors
            for card_index in range(self.deck_size):
                matches = (decimal_indices == card_index) & card_presence
                color_encoding = lax.dynamic_slice(INDEX_COLOR_MAP, start_indices=jnp.array([card_index, 0]),
                                                   slice_sizes=(1, 4))
                color_encodings = jnp.where(matches[..., None], color_encoding, color_encodings)
            color_encodings = color_encodings.astype(jnp.int32).reshape(16, 16, 4)
            return color_encodings

        yokai_card_world = _get_world(card)

        ### reset the hint_card state: randomly choose several hint cards from all hint cards w.r.t agent number
        def _draw_hint_per_color(color_num, hint_card_total, key: chex.PRNGKey, max_hint_of_color_num,
                                 num_hint_of_color_num):
            key, subkey = jax.random.split(key)
            num_hint = max_hint_of_color_num[color_num - 1]
            hint = lax.dynamic_slice(hint_card_total, (0, 0), (num_hint_of_color_num[color_num - 1], 4))
            return hint

        hint1 = _draw_hint_per_color(1, self.hint_card_total1, key, self.max_hint_of_color_num,
                                     self.num_hint_of_color_num)
        hint2 = _draw_hint_per_color(2, self.hint_card_total2, key, self.max_hint_of_color_num,
                                     self.num_hint_of_color_num)
        hint3 = _draw_hint_per_color(3, self.hint_card_total3, key, self.max_hint_of_color_num,
                                     self.num_hint_of_color_num)
        hint_card_choose = jnp.vstack((hint1, hint2, hint3)).astype(jnp.int32)

        ### reset hint_card_revealed
        hint_card_revealed = jnp.zeros((self.hint_size, 1), dtype=jnp.int32)

        ### reset hint_card_placed
        hint_card_placed = jnp.zeros((self.hint_size, 1), dtype=jnp.int32)

        ### reset locked_yokai_card locked: 0, unlocked: 1 16*16
        unlocked_yokai_card = jnp.zeros((self.deck_size, self.deck_size), dtype=jnp.int32)
        unlocked_yokai_card = unlocked_yokai_card.at[start_initial:end_initial, start_initial:end_initial].set(1)

        ### reset hint_card_terminal
        hint_card_terminal = False

        ### reset player_terminal
        player_terminal = False

        ### reset current_player_index
        cur_player_idx = jnp.zeros(self.num_agents).at[0].set(1)
        cur_player_idx = cur_player_idx.astype(jnp.int32)
        ### reset is_graph
        is_graph = False

        ### reset card knowledge set the color dimension(2-5 layers) to [1,1,1,1] because before check the card color, all colors are possible
        # and set the middle 4*4 of 1 layer of the fourth dim to 1
        # set the 6-9 layers of the fourth dim to [1,1,1,1] because initially no hint card is placed,all color is possible
        card_knowledge_per_agent = jnp.zeros((self.deck_size, self.deck_size, 5))
        card_knowledge_per_agent = card_knowledge_per_agent.at[middle_indices].set(1)
        card_knowledge_per_agent = card_knowledge_per_agent.at[start_initial:end_initial, start_initial:end_initial,
                                   1:5].set(
            jnp.ones((int(self.deck_size ** 0.5), int(self.deck_size ** 0.5), self.num_colors), dtype=jnp.int32))
        hint_color_knowledge = jnp.zeros((self.deck_size, self.deck_size, self.num_colors), dtype=jnp.int32)
        hint_color_knowledge = hint_color_knowledge.at[start_initial:end_initial, start_initial:end_initial, :].set(
            jnp.zeros((int(self.deck_size ** 0.5), int(self.deck_size ** 0.5), self.num_colors), dtype=jnp.int32)
        )
        # unlocked_yokai_card_reshape = unlocked_yokai_card.reshape(16,16,1)
        card_knowledge_per_agent = jnp.concatenate([card_knowledge_per_agent, hint_color_knowledge], axis=-1)
        card_knowledge = jnp.stack([card_knowledge_per_agent] * self.num_agents, axis=0)

        state = State(
            op_area_width_height=op_area_width_height,
            move_area_width_height=move_area_width_height,
            index_color_map=INDEX_COLOR_MAP,
            index_color_map_dict=INDEX_COLOR_MAP_DICT,
            yokai_card_config=card,
            yokai_card_world=yokai_card_world,
            hint_card=hint_card_choose,
            hint_card_revealed=hint_card_revealed,
            hint_card_placed=hint_card_placed,
            unlocked_yokai_card=unlocked_yokai_card,
            hint_card_terminal=hint_card_terminal,
            player_terminal=player_terminal,
            card_knowledge=card_knowledge,
            cur_player_idx=cur_player_idx,
            is_graph=is_graph,
        )
        return self.get_obs(state), state

    """
    Get all agents' observations
    depends on isGraph == True or False, get the graph observation or vector observation
    """
    # @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict:

        """
        16*16*6 array + 7*4 array  + 7*4array + num_agent*1array

        16*16*6 array:
        1 layer: card presence layer of yokai card
        2-5 layers: observed colors by agent itself or color of the placed hint card
        6 layer : unlock state layer, 1 means there is yokai card and this card is not unlocked, 0 means no yokai card or yokai card is locked
        7*4 array represent revealed hint card color
        7*4 array represents  placed hint cards color
        num_agent*1 array represent current active player
        """
        @partial(jax.vmap, in_axes=[0, None])
        def _observation_linearlize_grid(aidx: int, state: State) -> chex.Array:
            # 1-5th layers of 16*16*6 array
            knowledge = state.card_knowledge.at[aidx].get()
            arr_1 = knowledge[:, :, 0].reshape((self.deck_size, self.deck_size, 1))
            arr_1_4 = knowledge[:, :, 1:5]
            arr_5_8 = knowledge[:, :, 5:]
            condition_mask = (arr_5_8.sum(axis=-1) == 0) | (arr_5_8.sum(axis=-1) == 4)
            condition_mask = condition_mask[:, :, None]
            knowledge = condition_mask * arr_1_4 + (1 - condition_mask) * arr_5_8
            knowledge = jnp.concatenate([arr_1, knowledge], axis=2)
            knowledge = jnp.reshape(knowledge, (-1,))

            # knowledge = state.card_knowledge.at[aidx].get()
            # arr_1 = knowledge[:, :, 0].reshape((self.deck_size, self.deck_size, 1))
            # card = state.yokai_card_config
            # card_presence = card[:, :, 0] == 1
            # decimal_indices = card[:, :, 1:].reshape(16, 16)  # Extract binary indices
            # color_encodings = jnp.zeros((16, 16, 4))  # 4 for the one-hot encoding of colors
            # for card_index in range(self.deck_size):
            #     matches = (decimal_indices == card_index) & card_presence
            #     color_encoding = lax.dynamic_slice(state.index_color_map, start_indices=jnp.array([card_index, 0]),
            #                                        slice_sizes=(1, 4))
            #     color_encodings = jnp.where(matches[..., None], color_encoding, color_encodings)
            # color_encodings = color_encodings.astype(jnp.int32).reshape(16, 16, 4)
            # knowledge = jnp.concatenate([arr_1, color_encodings], axis=2)
            # knowledge = jnp.reshape(knowledge, (-1,))

            # 6th layer of 16*16*6 array
            locked_card = state.unlocked_yokai_card
            locked_card = jnp.reshape(locked_card, (-1,))

            # 7*4 array, if not revealed, [0,0,0,0]
            revealed_hint_card = state.hint_card_revealed
            hint_card = state.hint_card
            mask = revealed_hint_card.flatten() == 1
            revealed_hint_card = jnp.where(mask[:, None], hint_card, jnp.zeros_like(hint_card))
            revealed_hint_card = jnp.reshape(revealed_hint_card, (-1,))

            # 7*4 array, if not placed, [0,0,0,0]
            placed_hint_card = state.hint_card_placed
            mask = placed_hint_card.flatten() == 1
            placed_hint_card = jnp.where(mask[:, None], hint_card, jnp.zeros_like(hint_card))
            placed_hint_card = jnp.reshape(placed_hint_card, (-1,))

            # num_agent*1 array
            cur_agent = state.cur_player_idx
            cur_agent = jnp.reshape(cur_agent, (-1,))

            obs = jnp.concatenate([knowledge, locked_card, revealed_hint_card, placed_hint_card, cur_agent])
            return obs

        @partial(jax.vmap, in_axes=[0, None])
        def _observation_graph(aidx: int, state: State) -> chex.Array:
            obs_info1 = jnp.zeros((self.deck_size, 7), dtype=jnp.int32)
            obs_info2 = jnp.zeros((self.hint_size, 7), dtype=jnp.int32)
            obs_info = jnp.concatenate([obs_info1, obs_info2], axis=0)

            obs_connection = jnp.zeros((self.total_card, self.total_card), dtype=jnp.int32)
            card = state.yokai_card_config
            knowledge = state.card_knowledge.at[aidx].get()
            unlock_state = state.unlocked_yokai_card
            cur_player = state.cur_player_idx
            arr_1_4 = knowledge[:, :, 1:5]  # Layers 1-4
            arr_5_8 = knowledge[:, :, 5:]  # Layers 5-8
            condition_mask = (arr_5_8.sum(axis=-1) == 0) | (arr_5_8.sum(axis=-1) == 4)
            condition_mask = condition_mask[:, :, None]
            knowledge = condition_mask * arr_1_4 + (1 - condition_mask) * arr_5_8
            card_presence = card[:, :, 0]
            decimal_indices = card[:, :, 1:].reshape(self.deck_size, self.deck_size)  # Extract binary indices
            decimal_indices = card_presence + decimal_indices
            move_deck = jnp.zeros((self.move_size, self.move_size), dtype=jnp.int32)
            move_deck = move_deck.at[1:17, 1:17].set(decimal_indices)
            non_zero_positions = jnp.nonzero(move_deck > 0, size=self.deck_size)  # 找到所有非零元素的位
            card_presence = card[:, :, 0] == 1
            decimal_indices = card[:, :, 1:].reshape(16, 16)  # Extract binary indices
            color_encodings = jnp.zeros((16, 16, 4))  # 4 for the one-hot encoding of colors
            for card_index in range(self.deck_size):
                matches = (decimal_indices == card_index) & card_presence
                color_encoding = lax.dynamic_slice(state.index_color_map, start_indices=jnp.array([card_index, 0]),
                                                   slice_sizes=(1, 4))
                color_encodings = jnp.where(matches[..., None], color_encoding, color_encodings)
            color_encodings = color_encodings.astype(jnp.int32)
            card_new = jnp.zeros((self.deck_size, self.deck_size, 5), dtype=jnp.int32)
            card_new = card_new.at[:, :, 1:5].set(color_encodings.reshape(16, 16, 4))
            card_new = card_new.at[:, :, 0].set(card[:, :, 0])
            card_color = card_new[:, :, 1:5]
            def _body_graph(i,para):
                non_zero_positions,move_deck,obs_info,obs_connection = para
                i_x = non_zero_positions[0][i]
                i_y = non_zero_positions[1][i]
                cur_card = move_deck[i_x, i_y]
                up_card = move_deck[i_x - 1, i_y]
                down_card = move_deck[i_x + 1, i_y]
                left_card = move_deck[i_x, i_y - 1]
                right_card = move_deck[i_x, i_y + 1]
                is_valid_up = up_card > 0
                is_valid_down = down_card > 0
                is_valid_left = left_card > 0
                is_valid_right = right_card > 0
                up_card_valid = is_valid_up * up_card + (1 - is_valid_up) * 9999
                down_card_valid = is_valid_down * down_card + (1 - is_valid_down) * 9999
                left_card_valid = is_valid_left * left_card + (1 - is_valid_left) * 9999
                right_card_valid = is_valid_right * right_card + (1 - is_valid_right) * 9999
                obs_connection = obs_connection.at[cur_card - 1, up_card_valid - 1].set(1)
                obs_connection = obs_connection.at[cur_card - 1, down_card_valid - 1].set(1)
                obs_connection = obs_connection.at[cur_card - 1, left_card_valid - 1].set(1)
                obs_connection = obs_connection.at[cur_card - 1, right_card_valid - 1].set(1)
                i_x16 = i_x - 1
                i_y16 = i_y - 1
                knowledge_one = knowledge[i_x16, i_y16, :].reshape(4, )
                unlock_one = unlock_state[i_x16, i_y16]
                obs_info = obs_info.at[cur_card - 1, 0:4].set(knowledge_one)
                obs_info = obs_info.at[cur_card - 1, 4].set(unlock_one)
                obs_info = obs_info.at[cur_card - 1, 5:7].set(cur_player)
                return (non_zero_positions,move_deck,obs_info,obs_connection)
            result = lax.fori_loop(0, self.deck_size, _body_graph, (non_zero_positions,move_deck,obs_info,obs_connection))
            non_zero_positions, move_deck, obs_info, obs_connection=result
            obs_connection = jnp.fill_diagonal(obs_connection,1,inplace=False)

            revealed_hint_card = state.hint_card_revealed
            hint_card = state.hint_card
            mask = revealed_hint_card.flatten() == 1
            revealed_hint_card = jnp.where(mask[:, None], hint_card, jnp.zeros_like(hint_card))

            # 7*4 array
            placed_hint_card = state.hint_card_placed.reshape(self.hint_size, 1)
            obs_info = lax.dynamic_update_slice(obs_info, revealed_hint_card,
                                                jnp.array([16, 0]))

            # obs_info = obs_info.at[16:16+self.hint_size,0:4].set(revealed_hint_card)
            obs_info = lax.dynamic_update_slice(obs_info, placed_hint_card,
                                                jnp.array([16, 4]))
            # obs_info=obs_info.at[16:16+self.hint_size,4].set(placed_hint_card)
            cur_player = cur_player.reshape(2, )
            cur_player = jnp.tile(cur_player, (self.hint_size, 1))
            obs_info = lax.dynamic_update_slice(obs_info, cur_player,
                                                jnp.array([16, 5]))
            # obs_info = obs_info.at[16:16 + self.hint_size, 5:7].set(cur_player)
            obs_info = jnp.reshape(obs_info, (-1,))
            obs_connection = jnp.reshape(obs_connection, (-1,))
            obs = jnp.concatenate([obs_info, obs_connection])
            return obs

        obs = _observation_graph(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}


    @partial(jax.jit, static_argnums=[0])
    def step_env(
            self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        actions = jnp.array([actions[i] for i in self.agents])
        # get the current agent index
        aidx = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
        action = actions.at[aidx].get()
        state, reward = self.step_agent(key, state, aidx, action)
        done = self.terminal(state)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        shaped_reward, final_reward = self.calculate_reward(state, reward)
        rewards = {agent: final_reward for agent in self.agents}
        rewards["__all__"] = final_reward
        #rewards = {agent: reward1 for agent in self.agents}
        #rewards["__all__"] = reward1

        info = {
            'shaped_reward': {agent: shaped_reward for agent in self.agents}
        }
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            rewards,
            dones,
            info
        )

    def step_agent(self, key: chex.PRNGKey, state: State, aidx: int, action: chex.Array,
                   ) -> Tuple[State, int]:
        ob_1 = action[0] - 1
        ob_2 = action[1] - 1
        move = action[2]
        hint = action[3]
        """
        Execute the  action (observse two Yokai cards)
        action is from 0 to 15
        0 indicates to observe the yokai card with index 0
         """
        def _step_agent_obs_two(state: State, key: chex.PRNGKey, aidx: int, action: int, action2: int) -> State:
            is_op = (action >= 0)

            def _ob_card_two(state, action, action2):
                card_index = action
                card_index2 = action2
                # get the previous card knowledge for current agent
                knowledge_before = state.card_knowledge.at[aidx].get()
                mask = knowledge_before[:, :, 0] == 1
                knowledge_before = knowledge_before.at[:, :, 1:5].set(jnp.where(mask[:, :, None], jnp.array([1,1,1,1]), knowledge_before[:, :, 1:5]))
                index_array_new = jnp.concatenate(
                    [jnp.array([1], dtype=jnp.int32), jnp.array([card_index], dtype=jnp.int32)], axis=0)
                index_array_new2 = jnp.concatenate(
                    [jnp.array([1], dtype=jnp.int32), jnp.array([card_index2], dtype=jnp.int32)], axis=0)
                # get the colors of the observed cards
                color = state.index_color_map[card_index]
                color2 = state.index_color_map[card_index2]
                # get the position of the observed cards
                sliced_arr = state.yokai_card_config
                matches = jnp.all(sliced_arr == index_array_new, axis=2)
                matches2 = jnp.all(sliced_arr == index_array_new2, axis=2)
                # x_coord and y_coord of the position, x,y are arrays with only 1 elements
                matches_flat = matches.ravel()
                matches_flat2 = matches2.ravel()
                index_flat = jnp.argmax(matches_flat)
                index_flat2 = jnp.argmax(matches_flat2)
                x = index_flat // self.deck_size
                y = index_flat % self.deck_size
                x2 = index_flat2 // self.deck_size
                y2 = index_flat2 % self.deck_size
                # update the card_knowledge state of the current agent
                knowledge_after = knowledge_before.at[x, y, 1:5].set(color)
                knowledge_after = knowledge_after.at[x2, y2, 1:5].set(color2)
                card_knowledge_after = state.card_knowledge.at[aidx].set(knowledge_after)
                return state.replace(card_knowledge=card_knowledge_after)

            def _no_op(state, action, action2):
                return state

            state_new = lax.cond(is_op, _ob_card_two, _no_op, state, action, action2)
            return state_new



        """
        Execute the  action (move one yokai card to another position)
        """
        def _step_agent_move(key: chex.PRNGKey, state: State, aidx: int, action: int,
                             ) -> Tuple[State, int]:
            is_no_op = (action == 0)
            # get the coordinates based on yokai card index
            def _find_coord(card_index, state):
                index_arr = jnp.array([card_index], dtype=jnp.int32)
                index_arr_new = jnp.concatenate([jnp.array([1], dtype=jnp.int32), index_arr], axis=0)
                sliced_arr = state.yokai_card_config
                matches = jnp.all(sliced_arr == index_arr_new, axis=2)
                matches_flat = matches.ravel()
                index_flat = jnp.argmax(matches_flat)
                x = index_flat // self.deck_size
                y = index_flat % self.deck_size
                return x, y, index_arr

            # check if the movement will exceed the 16*16 grid boundary, if yes, shift cards, if no, not change anything

            def _legal_move(aidx: int, state: State, dir: int, x_next: int, y_next: int) -> Tuple[State, int, int]:
                def _not_shift_down(state, x_next, y_next):
                    return state, x_next - 1, y_next

                def _shift_down(state, x_next, y_next):
                    config_before = state.yokai_card_config
                    unlock_card_before = state.unlocked_yokai_card
                    card_knowledge_before = state.card_knowledge
                    new_arr_yokai_config = jnp.zeros_like(config_before)
                    new_arr_unlock = jnp.zeros_like(unlock_card_before)
                    new_arr_knowledge = jnp.zeros_like(card_knowledge_before)
                    new_arr_yokai_config = new_arr_yokai_config.at[1:16, :, :].set(config_before[0:15, :, :])
                    new_arr_unlock = new_arr_unlock.at[1:16, :].set(unlock_card_before[0:15, :])
                    new_arr_knowledge = new_arr_knowledge.at[:, 1:16, :, :].set(card_knowledge_before[:, 0:15, :, :])
                    return state.replace(
                        yokai_card_config=new_arr_yokai_config,
                        unlocked_yokai_card=new_arr_unlock,
                        card_knowledge=new_arr_knowledge

                    ), x_next, y_next

                def _not_shift_up(state, x_next, y_next):
                    return state, x_next + 1, y_next

                def _shift_up(state, x_next, y_next):
                    config_before = state.yokai_card_config
                    unlock_card_before = state.unlocked_yokai_card
                    card_knowledge_before = state.card_knowledge
                    new_arr_yokai_config = jnp.zeros_like(config_before)
                    new_arr_unlock = jnp.zeros_like(unlock_card_before)
                    new_arr_knowledge = jnp.zeros_like(card_knowledge_before)
                    new_arr_yokai_config = new_arr_yokai_config.at[0:15, :, :].set(config_before[1:16, :, :])
                    new_arr_unlock = new_arr_unlock.at[0:15, :].set(unlock_card_before[1:16, :])
                    new_arr_knowledge = new_arr_knowledge.at[:, 0:15, :, :].set(card_knowledge_before[:, 1:16, :, :])
                    return state.replace(
                        yokai_card_config=new_arr_yokai_config,
                        unlocked_yokai_card=new_arr_unlock,
                        card_knowledge=new_arr_knowledge

                    ), x_next, y_next

                def _not_shift_right(state, x_next, y_next):
                    return state, x_next, y_next - 1

                def _shift_right(state, x_next, y_next):
                    config_before = state.yokai_card_config
                    unlock_card_before = state.unlocked_yokai_card
                    card_knowledge_before = state.card_knowledge
                    new_arr_yokai_config = jnp.zeros_like(config_before)
                    new_arr_unlock = jnp.zeros_like(unlock_card_before)
                    new_arr_knowledge = jnp.zeros_like(card_knowledge_before)
                    new_arr_yokai_config = new_arr_yokai_config.at[:, 1:16, :].set(config_before[:, 0:15, :])
                    new_arr_unlock = new_arr_unlock.at[:, 1:16].set(unlock_card_before[:, 0:15])
                    new_arr_knowledge = new_arr_knowledge.at[:, :, 1:16, :].set(card_knowledge_before[:, :, 0:15, :])
                    return state.replace(
                        yokai_card_config=new_arr_yokai_config,
                        unlocked_yokai_card=new_arr_unlock,
                        card_knowledge=new_arr_knowledge

                    ), x_next, y_next

                def _not_shift_left(state, x_next, y_next):
                    return state, x_next, y_next + 1

                def _shift_left(state, x_next, y_next):
                    config_before = state.yokai_card_config
                    unlock_card_before = state.unlocked_yokai_card
                    card_knowledge_before = state.card_knowledge
                    new_arr_yokai_config = jnp.zeros_like(config_before)
                    new_arr_unlock = jnp.zeros_like(unlock_card_before)
                    new_arr_knowledge = jnp.zeros_like(card_knowledge_before)
                    new_arr_yokai_config = new_arr_yokai_config.at[:, 0:15, :].set(config_before[:, 1:16, :])
                    new_arr_unlock = new_arr_unlock.at[:, 0:15].set(unlock_card_before[:, 1:16])
                    new_arr_knowledge = new_arr_knowledge.at[:, :, 0:15, :].set(card_knowledge_before[:, :, 1:16, :])
                    return state.replace(
                        yokai_card_config=new_arr_yokai_config,
                        unlocked_yokai_card=new_arr_unlock,
                        card_knowledge=new_arr_knowledge

                    ), x_next, y_next

                def _branch0(state, x_next, y_next):
                    is_shift_down = (x_next == 0)
                    state, x_next, y_next = lax.cond(is_shift_down, _shift_down, _not_shift_down, state, x_next, y_next)
                    return state, x_next, y_next

                def _branch1(state, x_next, y_next):
                    is_shift_up = (x_next == 15)
                    state, x_next, y_next = lax.cond(is_shift_up, _shift_up, _not_shift_up, state, x_next, y_next)
                    return state, x_next, y_next

                def _branch2(state, x_next, y_next):
                    is_shift_right = (y_next == 0)
                    state, x_next, y_next = lax.cond(is_shift_right, _shift_right, _not_shift_right, state, x_next,
                                                     y_next)
                    return state, x_next, y_next

                def _branch3(state, x_next, y_next):
                    is_shift_left = (y_next == 15)
                    state, x_next, y_next = lax.cond(is_shift_left, _shift_left, _not_shift_left, state, x_next, y_next)
                    return state, x_next, y_next

                branches = [_branch0, _branch1, _branch2, _branch3]
                state, x_next, y_next = lax.switch(dir, branches, state, x_next, y_next)
                return state, x_next, y_next


            def _no_op(state, action, aidx):
                return state, 0.0

            def _move(state, action, aidx):

                # get the  colors of yokai_cards and saves the color in a 16*16 grid.
                def _get_decimal_card_color(state):
                    card = state.yokai_card_config
                    card_presence = card[:, :, 0] == 1
                    decimal_indices = card[:, :, 1:2].reshape(16, 16)  # Extract card indices
                    color_encodings = jnp.zeros((*card.shape[:2], 4))
                    for card_index in range(16):
                        matches = (decimal_indices == card_index) & card_presence
                        color_encoding = lax.dynamic_slice(state.index_color_map,
                                                           start_indices=jnp.array([card_index, 0]),
                                                           slice_sizes=(1, 4))
                        color_encodings = jnp.where(matches[..., None], color_encoding, color_encodings)
                    color_encodings = color_encodings.astype(jnp.int32)
                    decimal_card_color = color_encodings.dot(jnp.array([8, 4, 2, 1]))
                    return decimal_card_color

                # convert action to move_card_index,next_to_card_index and direction
                move_card_index = (action - 1) // (self.deck_size * 4)
                next_to_card_index = ((action - 1) % (self.deck_size * 4)) // (4)
                dir_index = ((action - 1) % (self.deck_size * 4)) % (4)

                # get the position in the grid
                x_move, y_move, move_decimal_index = _find_coord(move_card_index, state)
                x_next, y_next, next_decimal_index = _find_coord(next_to_card_index, state)



                # calculate the cards with same color before move
                decimal_card_color = _get_decimal_card_color(state)

                # get the card_knowledge of the moved card
                knowledge_move_card_all_agent = state.card_knowledge[:, x_move, y_move, :]
                yokai_card_config_before = state.yokai_card_config

                # set the third dim of the moved card position to [0,0,0,0,0]
                yokai_card_config_after = yokai_card_config_before.at[x_move, y_move, :].set(jnp.zeros(2, dtype=int))
                unlock_before = state.unlocked_yokai_card

                # unlock_yokai_card:set the moved card position to 0: means no card here
                unlock_after = unlock_before.at[x_move, y_move].set(0)

                # car_knowledge: set the moved_card position to 0 to all dimensions for all agents
                card_knowledge = state.card_knowledge
                for i in range(self.num_agents):
                    card_knowledge = card_knowledge.at[i, x_move, y_move, :].set(jnp.zeros(9, dtype=int))
                state_new = state.replace(
                    yokai_card_config=yokai_card_config_after,
                    unlocked_yokai_card=unlock_after,
                    card_knowledge=card_knowledge
                )
                state_shifted, x_target, y_target = _legal_move(aidx, state_new, dir_index, x_next, y_next)
                # set the new yokai_card_config,set the target position as [1,index]
                config_before = state_shifted.yokai_card_config
                config_new = config_before.at[x_target, y_target, 0].set(1)
                config_new = config_new.at[x_target, y_target, 1:2].set(move_decimal_index)
                # set the new unlocked_yokai_card state
                unlock_old = state_shifted.unlocked_yokai_card
                unlock_new = unlock_old.at[x_target, y_target].set(1)
                # set the new card_knowledge state
                card_old = state_shifted.card_knowledge
                card_new = card_old.at[:, x_target, y_target, :].set(knowledge_move_card_all_agent)
                state_new = state.replace(
                    yokai_card_config=config_new,
                    unlocked_yokai_card=unlock_new,
                    card_knowledge=card_new
                )

                return state_new, 0.0

            state, reward = lax.cond(is_no_op, _no_op, _move, state, action, aidx)
            return state, 0.0

        """
        Execute the  action (reveal one hint card or place one hint card)
        """
        def _step_agent_hint(key: chex.PRNGKey, state: State, aidx: int, action: int, reward: int
                             ) -> Tuple[State, int]:
            is_no_op = (action == 0)
            is_reveal = jnp.logical_and((action <= self.hint_size), (action >= 1))
            def _find_coord(card_index, state):
                index_array = jnp.array([card_index], dtype=jnp.int32)
                index_array_new = jnp.concatenate([jnp.array([1], dtype=jnp.int32), index_array], axis=0)
                sliced_arr = state.yokai_card_config
                matches = jnp.all(sliced_arr == index_array_new, axis=2)
                matches_flat = matches.ravel()
                index_flat = jnp.argmax(matches_flat)
                x = index_flat // self.deck_size
                y = index_flat % self.deck_size
                return x, y

            def _end(state, action, aidx,reward):
                return state,reward

            def _hint(state, action, aidx,reward):
                state,reward = lax.cond(is_reveal, _reveal_hint, _place_hint, state, action, aidx,reward)
                return state,reward

            # reveal one hint card
            def _reveal_hint(state, action, aidx,reward):
                reveal_hint_index = action - 1
                hint_revealed_before = state.hint_card_revealed
                hint_revealed_after = hint_revealed_before.at[reveal_hint_index, 0].set(1)
                aidx = (aidx + 1) % self.num_agents
                cur_player_idx = jnp.zeros(self.num_agents).astype(jnp.int32)
                cur_player_idx = cur_player_idx.at[aidx].set(1)
                return state.replace(
                    hint_card_revealed=hint_revealed_after,
                    cur_player_idx=cur_player_idx),reward

            # place one hint card
            def _place_hint(state, action, aidx,reward):
                place_hint_index = (action - 1 - self.hint_size) // (self.deck_size)
                placed_on_yokai = (action - 1 - self.hint_size) % (self.deck_size)
                hint_color = state.hint_card[place_hint_index]
                placed_on_yokai_color = state.index_color_map[placed_on_yokai]
                is_included = jnp.any((hint_color-placed_on_yokai_color)<0).astype(jnp.int32)
                reward = reward + is_included*0.001
                hint_place_before = state.hint_card_placed
                hint_place_after = hint_place_before.at[place_hint_index, 0].set(1)
                unlock_before = state.unlocked_yokai_card
                # get the position of the placed yokai card
                x_placed, y_placed = _find_coord(placed_on_yokai, state)
                unlock_after = unlock_before.at[x_placed, y_placed].set(0)
                knowledge_before = state.card_knowledge
                stack_color = jnp.tile(hint_color, (self.num_agents, 1))
                knowledge_after = knowledge_before.at[:, x_placed, y_placed, 5:9].set(stack_color)
                hint_terminal = (np.sum(hint_place_after) == self.hint_size)
                aidx = (aidx + 1) % self.num_agents
                cur_player_idx = jnp.zeros(self.num_agents).astype(jnp.int32)
                cur_player_idx = cur_player_idx.at[aidx].set(1)
                return state.replace(
                    hint_card_placed=hint_place_after,
                    unlocked_yokai_card=unlock_after,
                    card_knowledge=knowledge_after,
                    cur_player_idx=cur_player_idx,
                    hint_card_terminal=hint_terminal
                ),reward

            state,reward = lax.cond(is_no_op, _end, _hint, state, action, aidx,reward)
            return state, reward

        state_ob = _step_agent_obs_two(state, key, aidx, ob_1, ob_2)
        state_move, reward_move = _step_agent_move(key, state_ob, aidx, move)
        state_hint, reward_hint = _step_agent_hint(key, state_move, aidx, hint, reward_move)

        def _get_world(state):
            card = state.yokai_card_config
            card_presence = card[:, :, 0] == 1
            decimal_indices = card[:, :, 1:].reshape(16, 16)  # Extract binary indices
            color_encodings = jnp.zeros((16, 16, 4))  # 4 for the one-hot encoding of colors
            for card_index in range(self.deck_size):
                matches = (decimal_indices == card_index) & card_presence
                color_encoding = lax.dynamic_slice(INDEX_COLOR_MAP, start_indices=jnp.array([card_index, 0]),
                                                   slice_sizes=(1, 4))
                color_encodings = jnp.where(matches[..., None], color_encoding, color_encodings)
            color_encodings = color_encodings.astype(jnp.int32).reshape(16, 16, 4)
            return color_encodings

        yokai_card_world = _get_world(state_hint)
        state_hint = state_hint.replace(yokai_card_world=yokai_card_world)
        return state_hint, reward_hint

    @partial(jax.jit, static_argnums=[0])
    def calculate_reward(self, state: State, reward_step):
        is_end = self.terminal(state)
        card = state.yokai_card_config
        card_presence = card[:, :, 0] == 1
        decimal_indices = card[:, :, 1:].reshape(16, 16)  # Extract binary indices
        color_encodings = jnp.zeros((16, 16, 4))  # 4 for the one-hot encoding of colors
        for card_index in range(self.deck_size):
            matches = (decimal_indices == card_index) & card_presence
            color_encoding = lax.dynamic_slice(state.index_color_map, start_indices=jnp.array([card_index, 0]),
                                               slice_sizes=(1, 4))
            color_encodings = jnp.where(matches[..., None], color_encoding, color_encodings)
        color_encodings = color_encodings.astype(jnp.int32)
        card_new = jnp.zeros((self.deck_size, self.deck_size, 5), dtype=jnp.int32)
        card_new = card_new.at[:, :, 1:5].set(color_encodings.reshape(16, 16, 4))
        card_new = card_new.at[:, :, 0].set(card[:, :, 0])
        card_color = card_new[:, :, 1:5]
        decimal_card_color = card_color.dot(jnp.array([8, 4, 2, 1]))
        def _not_end(state, decimal_card_color):
            def _cal_distance(nonzeros):
                a = nonzeros[0]
                b = nonzeros[1]
                x_diff = a[:, jnp.newaxis] - a
                y_diff = b[:, jnp.newaxis] - b
                distances = jnp.sqrt(x_diff ** 2 + y_diff ** 2)
                total_distance = jnp.sum(jnp.triu(distances, 1))
                return total_distance
            color1 = jnp.where(decimal_card_color == 1, 1, 0)
            color2 = jnp.where(decimal_card_color == 2, 1, 0)
            color3 = jnp.where(decimal_card_color == 4, 1, 0)
            color4 = jnp.where(decimal_card_color == 8, 1, 0)
            non_zero1 = jnp.nonzero(color1, size=4)
            non_zero2 = jnp.nonzero(color2, size=4)
            non_zero3 = jnp.nonzero(color3, size=4)
            non_zero4 = jnp.nonzero(color4, size=4)
            dis1 = _cal_distance(non_zero1)
            dis2 = _cal_distance(non_zero2)
            dis3 = _cal_distance(non_zero3)
            dis4 = _cal_distance(non_zero4)
            max_dis = 64 + 32 * jnp.sqrt(2)
            min_dis = 4 + 2* jnp.sqrt(2)
            r1 = max_dis - dis1
            r2 = max_dis - dis2
            r3 = max_dis - dis3
            r4 = max_dis - dis4
            return (r1+r2+r3+r4) * 0.001+reward_step, 0.0

        # if the game does not ends, only get the intermediate rewards
        # def _not_end(state, decimal_card_color):
        #     def _count_cluster(arr):
        #         arr1 = jnp.where(arr == 1, 1, 0)
        #         visited = jnp.zeros_like(arr1)
        #         indices = jnp.argwhere(arr1 == 1, size=4)
        #         # Use the first '1' as a starting point for our connectivity check
        #         start = indices[0]
        #         visited1 = visited.at[start[0], start[1]].set(1)
        #         visited1 = visited1.astype(bool)
        #
        #         def _body(i, visited):
        #             neighbors = jnp.pad(visited, 1)[1:-1, 1:-1] | \
        #                         jnp.pad(visited, 1)[2:, 1:-1] | jnp.pad(visited, 1)[:-2, 1:-1] | \
        #                         jnp.pad(visited, 1)[1:-1, 2:] | jnp.pad(visited, 1)[1:-1, :-2]
        #             visited |= (neighbors & arr)
        #             visited = visited.astype(bool)
        #             return visited
        #
        #         visited1 = lax.fori_loop(0, 16, _body, visited1)
        #         visited1 = jnp.where(visited1 == 1, 1, 0)
        #         print(visited1)
        #         start2 = indices[1]
        #         visited2 = visited.at[start2[0], start2[1]].set(1)
        #         visited2 = visited2.astype(bool)
        #         visited2 = lax.fori_loop(0, 16, _body, visited2)
        #         visited2 = jnp.where(visited2 == 1, 2, 0)
        #         print(visited2)
        #         start3 = indices[2]
        #         visited3 = visited.at[start3[0], start3[1]].set(1)
        #         visited3 = visited3.astype(bool)
        #         visited3 = lax.fori_loop(0, 16, _body, visited3)
        #         visited3 = jnp.where(visited3 == 1, 3, 0)
        #         print(visited3)
        #
        #         start4 = indices[3]
        #         visited4 = visited.at[start4[0], start4[1]].set(1)
        #         visited4 = visited4.astype(bool)
        #         visited4 = lax.fori_loop(0, 16, _body, visited4)
        #         visited4 = jnp.where(visited4 == 1, 4, 0)
        #         print(visited4)
        #         all = jnp.maximum(jnp.maximum(jnp.maximum(visited4, visited3), visited2), visited1)
        #         i1 = jnp.any(all == 1).astype(jnp.int32)
        #         i2 = jnp.any(all == 2).astype(jnp.int32)
        #         i3 = jnp.any(all == 3).astype(jnp.int32)
        #         i4 = jnp.any(all == 4).astype(jnp.int32)
        #         return i1 + i2 + i3 + i4
        #
        #     c1 = _count_cluster(jnp.where(decimal_card_color == 1, 1, 0))
        #     c2 = _count_cluster(jnp.where(decimal_card_color == 2, 1, 0))
        #     c3 = _count_cluster(jnp.where(decimal_card_color == 4, 1, 0))
        #     c4 = _count_cluster(jnp.where(decimal_card_color == 8, 1, 0))
        #     step_process = jnp.sum(state.hint_card_revealed) + jnp.sum(state.hint_card_placed)
        #     step_process = step_process / (self.hint_size * 2)
        #     # step_process = 1
        #     return (16 - (c1 + c2 + c3 + c4)) * step_process * 0.001, 0.0

        # if the game ends, get the final rewards
        def _end(state, decimal_card_color):
            # check if the current yokai card configuration is fully connected
            def _is_fully_connected(arr, color):
                arr = jnp.where(arr == color, 1, 0)
                visited = jnp.zeros_like(arr)
                indices = jnp.argwhere(arr == 1, size=4)
                # Use the first '1' as a starting point for our connectivity check
                start = indices[0]
                visited = visited.at[start[0], start[1]].set(1)
                visited = visited.astype(bool)
                arr = arr.astype(bool)
                def _body(i, visited):
                    neighbors = jnp.pad(visited, 1)[1:-1, 1:-1] | \
                                jnp.pad(visited, 1)[2:, 1:-1] | jnp.pad(visited, 1)[:-2, 1:-1] | \
                                jnp.pad(visited, 1)[1:-1, 2:] | jnp.pad(visited, 1)[1:-1, :-2]
                    visited |= (neighbors & arr)
                    return visited
                visited = lax.fori_loop(0, self.deck_size, _body, visited)
                fully_connected = jnp.all((arr == 0) | (visited == 1))
                return fully_connected

            # check if in the current yokai_card_configuraton, cards with the same color are clusterd together
            def _is_color_clustered(grid):
                colors = jnp.array([1, 2, 4, 8])
                result = True
                for i in range(4):
                    color = colors[i]
                    indicator = 1 - _is_fully_connected(grid, color)
                    def _true(result):
                        result = False
                        return result
                    def _false(result):
                        return result
                    result = lax.cond(indicator, _true, _false, result)
                return result

            # if cards with the same color are clusterd together,
            # calculate rewards based on the game's rule
            # otherwise, rewards == -42
            is_cluster = _is_color_clustered(decimal_card_color)
            def _true_cluster(state):
                hint_reveal = state.hint_card_revealed
                hint_not_reveal_score = (len(hint_reveal) - jnp.sum(hint_reveal)) * 5
                hint_placed = state.hint_card_placed
                hint_not_place_score = (jnp.sum(hint_reveal) - jnp.sum(hint_placed)) * 2
                card_hint = state.card_knowledge[0, :, :, 5:9]
                comparison_array_ones = jnp.array([1, 1, 1, 1])
                comparison_array_zeros = jnp.array([0, 0, 0, 0])
                matches_ones = jnp.all(card_hint == comparison_array_ones, axis=2)
                matches_zeros = jnp.all(card_hint == comparison_array_zeros, axis=2)
                non_matches = ~(matches_ones | matches_zeros)
                x_positions, y_positions = jnp.nonzero(non_matches, size=len(hint_placed))
                positions = jnp.stack((x_positions, y_positions), axis=1)
                hint_values = card_hint[positions[:, 0], positions[:, 1], :]
                color_values = card_color[positions[:, 0], positions[:, 1], :]
                matches = (hint_values == 1) & (color_values == 1)
                rows_with_true = jnp.any(matches, axis=1)
                num_rows_with_true = jnp.sum(rows_with_true)
                rows_with_all_false = jnp.all(~matches, axis=1)
                num_rows_with_all_false = jnp.sum(rows_with_all_false)
                reward = hint_not_place_score + hint_not_reveal_score + num_rows_with_true - num_rows_with_all_false
                return (reward + 8).astype(jnp.float32)

            def _not_cluster(state):
                reward = -0.0
                return reward

            reward = lax.cond(is_cluster, _true_cluster, _not_cluster, state)
            return 0.0,reward

        reward_dense,reward_sparse = lax.cond(is_end, _end, _not_end, state, decimal_card_color)
        return reward_dense,reward_sparse

    """Check whether state is terminal."""
    def terminal(self, state: State) -> bool:
        is_terminal = jnp.logical_or(state.player_terminal, state.hint_card_terminal)
        return is_terminal


    """ Observation space for a given agent."""
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    """ Action space for a given agent."""
    def action_space(self, agent: str):
        return self.action_spaces[agent]