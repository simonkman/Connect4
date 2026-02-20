import copy
import typing
import numpy as np
from player import Player as PlayerParent

MAX_DEPTH = 6 # TODO: optimize this
WIN_VAL = 10000

class CylinderBoard:
    def __init__(self, board: np.ndarray, connect_number) -> None:
        self.rows, self.cols = board.shape
        self.connect_number = connect_number
        
        self.last_player = None
        # stored as row, col
        # for quick calculation of winner
        self.last_action = None

        self.board_all = board != 0
        self.board_p1 = board > 0
        self.board_p2 = board < 0

        # stores row index of next empty space in each col
        self.empty_loc = self.rows - np.sum(self.board_all, axis=0) - 1

    def act(self, player, action) -> None:
        row = self.empty_loc[action]
        self.empty_loc[action] -= 1

        self.last_player = player
        self.last_action = (row, action)

        self.board_all[row, action] = True

        if player > 0:
            self.board_p1[row, action] = True
        else:
            self.board_p2[row, action] = True

    def valid_moves(self) -> typing.Iterable[int]:
        """Returns a list of legal moves."""

        moves = np.where(~self.board_all[0, :])[0]
        return moves

    def terminal_value(self) -> typing.Optional[int]:
        """Returns integer value for terminal state, None if non-terminal."""

        if self.last_player is not None and self.last_action is not None:
            board = self.board_p1 if self.last_player > 0 else self.board_p2
            rng = np.arange(self.connect_number) # add for offsets throughout

            # Horizontal
            for offset in range(self.connect_number):
                r = self.last_action[0]
                c = (rng + self.last_action[1] - offset) % self.cols
                if np.sum(board[r, c]) >= self.connect_number:
                    return self.last_player * WIN_VAL

            # Vertical
            r0 = self.last_action[0]
            r1 = self.last_action[0] + self.connect_number
            c = self.last_action[1]
            if np.sum(board[r0:r1, c]) >= self.connect_number:
                return self.last_player * WIN_VAL

            # Diagonals
            for offset in range(self.connect_number):
                r = (rng + self.last_action[0] - offset)
                c = (rng + self.last_action[1] - offset) % self.cols
                if r[0] >= 0 and r[-1] < self.rows:
                    if np.sum(board[r, c]) >= self.connect_number:
                        return self.last_player * WIN_VAL

            # Diagonals (Flipped)
            for offset in range(self.connect_number):
                r = (rng + self.last_action[0] - offset)
                c = (self.last_action[1] + offset - rng) % self.cols
                if r[0] >= 0 and r[-1] < self.rows:
                    if np.sum(board[r, c]) >= self.connect_number:
                        return self.last_player * WIN_VAL

            # Draws
            if np.sum(self.empty_loc) <= -1 * self.cols:
                return 0
        else:
            # TODO: check if this branch should ever be executed
            # We should (probably?) never initialize into a terminal state,
            # so this is a non-terminal state
            return None

        return None

    def eval_state(self, player) -> float:
        """Either returns value of terminal state or the heuristic value of a non-terminal state."""
        # TODO: this heuristic sucks

        # just count number of pieces on top for each player
        top_piece_rows = self.empty_loc + 1
        valid_cols = np.where(top_piece_rows < self.rows)[0]
        
        p1_cnt = int(np.sum(self.board_p1[top_piece_rows[valid_cols], valid_cols]))
        p2_cnt = int(np.sum(self.board_p2[top_piece_rows[valid_cols], valid_cols]))
        return (p1_cnt - p2_cnt) / self.cols

class Player(PlayerParent):
    def __init__(self, rows, cols, connect_number, 
                 timeout_setup, timeout_move, max_invalid_moves, 
                 cylinder):
        super().__init__(rows, cols, connect_number, 
                 timeout_setup, timeout_move, max_invalid_moves, 
                 cylinder)

    def setup(self,piece_color):
        """
        This method will be called once at the beginning of the game so the player
        can conduct any setup before the move timer begins. The setup method is
        also timed.
        """
        self.piece_color = piece_color
        self.moves = np.arange(self.cols)

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
        return self._alpha_beta_search(board)

    def _alpha_beta_search(self, board: np.ndarray) -> typing.Optional[int]:
        """Minimax algorithm with alpha-beta pruning."""

        game = CylinderBoard(board, self.connect_number)
        player = 1 # our player is always represented by +1 values
        _, move = self._max_value(game, player, -np.inf, np.inf, 0)
        return move

    def _max_value(self, board: CylinderBoard, player, alpha, beta, depth) -> tuple[float, typing.Optional[int]]:
        """Calculates the max_value according to the minimax algorithm."""

        terminal = board.terminal_value()
        if terminal is not None:
            return terminal, None
        if self._is_cutoff(board, depth):
            return board.eval_state(player), None
        
        v = -np.inf
        other_player = -1 * player
        move = None
        for a in board.valid_moves():
            v2, _ = self._min_value(self._result(board, player, a), other_player, alpha, beta, depth + 1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)

            if v >= beta:
                return v, move
        return v, move

    def _min_value(self, board: CylinderBoard, player, alpha, beta, depth) -> tuple[float, typing.Optional[int]]:
        """Calculates the min_value according to the minimax algorithm."""

        terminal = board.terminal_value()
        if terminal is not None:
            return terminal, None
        if self._is_cutoff(board, depth):
            return board.eval_state(player), None

        v = np.inf
        other_player = -1 * player
        move = None
        for a in board.valid_moves():
            v2, _ = self._max_value(self._result(board, player, a), other_player, alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)

            if v <= alpha:
                return v, move
        return v, move

    def _is_cutoff(self, _: CylinderBoard, depth: int) -> bool:
        """Checks if search depth is too far."""

        return depth >= MAX_DEPTH

    def _result(self, board: CylinderBoard, player, action) -> CylinderBoard:
        """Returns the result of player taking the given action."""
        # TODO: This implementation COPIES A NEW BOARD for every move considered.
        # Extremely inefficient in space and time, needs to be optimized later.
        cpy = copy.deepcopy(board)
        cpy.act(player, action)
        return cpy
