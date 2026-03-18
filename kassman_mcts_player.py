import random
import math
import typing
import time
from collections import deque

import numpy as np

from player import Player as PlayerParent

MAX_TIME = 59  # in seconds, a little lower than the actual 60s
UCB_CONST = math.sqrt(2)  # constant C in UCB1 metric


class MCTNode:
    def __init__(self, parent=None, player=1, move=None, valid_moves=None):
        assert isinstance(valid_moves, set)
        self.parent = parent
        self.player = player
        self.move = move
        self.playouts = 0
        self.wins = 0
        self.losses = 0
        self.children = deque()
        self.remaining_moves = valid_moves

        # Upper confidence bound applied to trees metric
        self.ucb1 = math.inf

    def create_child(self, move, valid_moves):
        new_node = MCTNode(
            parent=self, player=-1 * self.player, move=move, valid_moves=valid_moves
        )
        self.children.append(new_node)
        self.remaining_moves.remove(move)
        return new_node


class CylinderBoard:
    def __init__(self, board: np.ndarray, init_player, connect_number) -> None:
        self.rows, self.cols = board.shape
        self.connect_number = connect_number

        self.init_player = init_player  # player who can act on initial board
        self.history = deque()

        self.board_all = board != 0
        self.board_p1 = board > 0
        self.board_p2 = board < 0

        # stores row index of next empty space in each col
        self.empty_loc = self.rows - np.sum(self.board_all, axis=0) - 1

    def get_cur_player(self) -> int:
        assert self.init_player is not None
        return self.init_player if len(self.history) % 2 == 0 else -self.init_player

    def act(self, action) -> None:
        row = self.empty_loc[action]
        self.empty_loc[action] -= 1

        self.history.append(action)

        self.board_all[row, action] = True

        if self.get_cur_player() > 0:
            self.board_p1[row, action] = True
        else:
            self.board_p2[row, action] = True

    def undo_act(self) -> None:
        """Undo the last action taken."""

        assert self.init_player is not None

        last_col = self.history.pop()
        last_player = (
            self.init_player if (len(self.history) + 1) % 2 == 1 else -self.init_player
        )
        last_action = (self.empty_loc[last_col] + 1, last_col)

        self.board_all[last_action[0], last_action[1]] = False

        if last_player > 0:
            self.board_p1[last_action[0], last_action[1]] = False
        else:
            self.board_p2[last_action[0], last_action[1]] = False

        self.empty_loc[last_action[1]] += 1

    def undo_all(self) -> None:
        """Undo all recorded moves."""
        for _ in range(len(self.history)):
            self.undo_act()

    def valid_moves(self) -> np.ndarray:
        """Returns a list of legal moves."""

        moves = np.where(~self.board_all[0, :])[0]
        return moves

    def terminal_value(self) -> typing.Optional[int]:
        """Returns integer value for terminal state, None if non-terminal."""

        if len(self.history) > 0:  # ensure at least one move has been recorded
            last_player = (
                self.init_player if len(self.history) % 2 == 1 else -self.init_player
            )
            last_action = (self.empty_loc[self.history[-1]] + 1, self.history[-1])

            board = self.board_p1 if last_player > 0 else self.board_p2
            rng = np.arange(self.connect_number)  # add for offsets throughout

            # Horizontal
            for offset in range(self.connect_number):
                r = last_action[0]
                c = (rng + last_action[1] - offset) % self.cols
                if np.sum(board[r, c]) >= self.connect_number:
                    return last_player

            # Vertical
            r0 = last_action[0]
            r1 = last_action[0] + self.connect_number
            c = last_action[1]
            if np.sum(board[r0:r1, c]) >= self.connect_number:
                return last_player

            # Diagonals
            for offset in range(self.connect_number):
                r = rng + last_action[0] - offset
                c = (rng + last_action[1] - offset) % self.cols
                if r[0] >= 0 and r[-1] < self.rows:
                    if np.sum(board[r, c]) >= self.connect_number:
                        return last_player

            # Diagonals (Flipped)
            for offset in range(self.connect_number):
                r = rng + last_action[0] - offset
                c = (last_action[1] + offset - rng) % self.cols
                if r[0] >= 0 and r[-1] < self.rows:
                    if np.sum(board[r, c]) >= self.connect_number:
                        return last_player

            # Draws
            if np.sum(self.empty_loc) <= -1 * self.cols:
                return 0
        else:
            # TODO: check if this branch should ever be executed
            # We should (probably?) never initialize into a terminal state,
            # so this is a non-terminal state
            return None

        return None


class Player(PlayerParent):
    def __init__(
        self,
        rows,
        cols,
        connect_number,
        timeout_setup,
        timeout_move,
        max_invalid_moves,
        cylinder,
    ):
        super().__init__(
            rows,
            cols,
            connect_number,
            timeout_setup,
            timeout_move,
            max_invalid_moves,
            cylinder,
        )

    def setup(self, piece_color):
        """
        This method will be called once at the beginning of the game so the player
        can conduct any setup before the move timer begins. The setup method is
        also timed.
        """
        self.piece_color = piece_color

    def play(self, board: np.ndarray):
        """
        Given a 2D array representing the game board, return an integer value (0,1,2,...,number of columns-1) corresponding to
        the column of the board where you want to drop your disc.
        The coordinates of the board increase along the right and down directions. 

        Parameters
        ----------
        board : np.ndarray
            A 2D array where 0s represent empty slots, +1s represent your pieces,
            and -1s represent the opposing player's pieces.

                `index   0   1   2   . column` \\
                `--------------------------` \\
                `0   |   0.  0.  0.  top` \\
                `1   |   -1  0.  0.  .` \\
                `2   |   +1  -1  -1  .` \\
                `.   |   -1  +1  +1  .` \\
                `row |   left        bottom/right`

        Returns
        -------
        integer corresponding to the column of the board where you want to drop your disc.
        """
        return self._monte_carlo_tree_search(board)

    def _monte_carlo_tree_search(self, board: np.ndarray) -> typing.Optional[int]:
        """Monte Carlo Tree Search algorithm."""

        game = CylinderBoard(board, 1, self.connect_number)

        tree = MCTNode(player=1, valid_moves=set(game.valid_moves()))

        end_time = time.monotonic() + MAX_TIME
        while time.monotonic() < end_time:
            leaf = self._select_node(tree, game)
            child = self._expand_node(leaf, game)
            result = self._simulate_node(child, game)
            self._backpropagate(result, child)
            game.undo_all()

        return self._get_most_played(tree).move

    def _select_node(self, tree: MCTNode, game: CylinderBoard):
        """Walk down the tree using UCB1 until an unexpanded node is found."""
        cur = tree

        # While fully expanded and not a terminal leaf
        while len(cur.remaining_moves) == 0 and len(cur.children) > 0:
            # Select the child with the highest UCB1 score
            cur = max(cur.children, key=lambda c: c.ucb1)
            game.act(cur.move)  # Put the game state in sync with the traversal

        return cur

    def _expand_node(self, leaf: MCTNode, game: CylinderBoard):
        if game.terminal_value() is not None or len(leaf.remaining_moves) == 0:
            return leaf
        move = random.choice(list(leaf.remaining_moves))
        game.act(move)
        return leaf.create_child(move, set(game.valid_moves()))

    def _simulate_node(self, child: MCTNode, game: CylinderBoard):
        term = game.terminal_value()
        valid = list(game.valid_moves())

        while term is None:
            best_move = None
            cur_player = game.get_cur_player()

            # 1-Step Lookahead
            for move in valid:
                game.act(move)
                if game.terminal_value() == cur_player:
                    best_move = move
                    game.undo_act()
                    break
                game.undo_act()

            # If no instant win, play randomly
            if best_move is None:
                best_move = random.choice(valid)

            game.act(best_move)
            term = game.terminal_value()

            if game.empty_loc[best_move] < 0:
                valid.remove(best_move)

        return term

    def _backpropagate(self, result, child: MCTNode):
        cur = child
        while cur is not None:
            cur.playouts += 1
            if -1 * cur.player == result:
                cur.wins += 1
            elif -1 * cur.player == -1 * result:
                cur.losses += 1

            # Update UCB metric
            if cur.parent is not None:
                utility = cur.wins - cur.losses
                cur.ucb1 = (utility / cur.playouts) + UCB_CONST * math.sqrt(
                    math.log(cur.parent.playouts + 1) / cur.playouts
                )

            cur = cur.parent

    def _get_most_played(self, tree: MCTNode) -> MCTNode:
        """Returns the immediate child with the most playouts."""
        if not tree.children:
            raise ValueError("No moves explored!")

        return max(tree.children, key=lambda c: c.playouts)


__all__ = ["Player"]
