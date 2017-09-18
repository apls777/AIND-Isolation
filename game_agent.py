"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # check if we already know the result of game
    score = utility(player == game.active_player, len(own_moves), len(opp_moves))
    if score:
        return score

    center_loc = (game.width - 1) / 2., (game.height - 1) / 2.
    max_move_score = center_loc[0] ** 2 + center_loc[1] ** 2

    own_loc = game.get_player_location(player)
    own_score = weighted_moves_avg_score(own_moves, center_loc, own_loc, max_move_score, 1.8)
    opp_score = len(opp_moves)

    return (own_score - opp_score) / (own_score + opp_score)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    # check if we already know the result of game
    score = utility(player == game.active_player, len(own_moves), len(opp_moves))
    if score:
        return score

    center_loc = (game.width - 1) / 2., (game.height - 1) / 2.
    max_move_score = center_loc[0] ** 2 + center_loc[1] ** 2

    own_loc = game.get_player_location(player)
    own_score = weighted_moves_avg_score(own_moves, center_loc, own_loc, max_move_score, 1.8)
    opp_score = len(opp_moves)

    return own_score - opp_score


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # check if we already know the result of game
    score = utility(player == game.active_player, own_moves, opp_moves)
    if score:
        return score

    return (own_moves - opp_moves) / (own_moves + opp_moves)


def weighted_moves_avg_score(moves, center_loc, player_loc, max_move_score, a):
    return len(moves) * (a - ((center_loc[0] - player_loc[0]) ** 2 + (center_loc[1] - player_loc[1]) ** 2) / max_move_score)


def weighted_moves_score(moves, center_loc, max_move_score, a):
    score = 0
    for (row, col) in moves:
        score += a - ((center_loc[0] - row) ** 2 + (center_loc[1] - col) ** 2) / max_move_score

    return score


def get_depth_moves_score(game, moves, depth, current_loc=(-1, -1)):
    score = len(moves) * 65 ** depth
    if depth == 0:
        return score

    for m in moves:
        next_moves = get_moves(game, m)
        # remove a move where we came from
        if current_loc != (-1, -1):
            next_moves.remove(current_loc)
        # add depth score for each move
        score += get_depth_moves_score(game, next_moves, depth - 1, m)

    return score


def get_moves(board, loc):
    """Generate the list of possible moves for an L-shaped motion (like a
    knight in chess).
    """
    r, c = loc
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                   if board.move_is_legal((r + dr, c + dc))]

    return valid_moves


def utility(is_active_player, own_moves, opp_moves):
    if is_active_player:
        if own_moves == 0:
            # my turn and I have no moves
            return float('-inf')
        elif opp_moves == 0:
            # my turn and opponent has no moves
            return float('inf')
    else:
        # opponent's turn and he has no moves
        if opp_moves == 0:
            return float('inf')
        # opponent's turn and I have no moves
        elif own_moves == 0:
            return float('-inf')

    return 0



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return -1, -1

        _, move = max([(self.get_minimax_value(False, game.forecast_move(m), depth - 1), m) for m in legal_moves])

        return move

    def get_minimax_value(self, is_max_fn, board, depth):
        # check that we have a time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # get all legal moves
        moves = board.get_legal_moves()
        if not moves:
            # if there are no moves, return game result
            return board.utility(self)

        # if we reached the last ply, return score for that board
        if depth == 0:
            return self.score(board, self)

        # min and max functions are combined to not duplicate terminate logic
        if is_max_fn:
            v = -float('inf')
            for m in moves:
                v = max(v, self.get_minimax_value(False, board.forecast_move(m), depth - 1))
        else:
            v = float('inf')
            for m in moves:
                v = min(v, self.get_minimax_value(True, board.forecast_move(m), depth - 1))

        return v

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        move = (-1, -1)

        depth = 1
        try:
            while True:
                move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            pass

        return move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moves = game.get_legal_moves()
        if not moves:
            return -1, -1

        v = -float('inf')
        move = moves[0]
        for m in moves:
            v = max(v, self.get_minimax_value(False, game.forecast_move(m), depth - 1, alpha, beta))
            if v >= beta:
                move = m
                break
            if v > alpha:
                alpha = v
                move = m

        return move

    def get_minimax_value(self, is_max_fn, board, depth, a, b):
        # check that we have a time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # get all legal moves
        moves = board.get_legal_moves()
        if not moves:
            # if there are no moves, return game result
            return board.utility(self)

        # if we reached the last ply, return score for that board
        if depth == 0:
            return self.score(board, self)

        # min and max functions are combined to not duplicate terminate logic
        if is_max_fn:
            v = -float('inf')
            for m in moves:
                v = max(v, self.get_minimax_value(False, board.forecast_move(m), depth - 1, a, b))
                if v >= b:
                    return v
                a = max(a, v)
        else:
            v = float('inf')
            for m in moves:
                v = min(v, self.get_minimax_value(True, board.forecast_move(m), depth - 1, a, b))
                if v <= a:
                    return v
                b = min(b, v)

        return v
