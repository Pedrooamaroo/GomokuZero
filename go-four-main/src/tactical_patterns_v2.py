import numpy as np
from numba import njit

SCORE_WIN_IMMEDIATE   = 100000.0
SCORE_BLOCK_WIN        = 50000.0
SCORE_OPEN_FOUR        = 10000.0
SCORE_BLOCK_OPEN_4     = 9000.0
SCORE_PENTE_WIN_CAP    = 100000.0

SCORE_MAKE_CAPTURE     = 5000.0  
SCORE_AVOID_CAPTURE    = 2000.0

SCORE_DOUBLE_THREAT    = 20000.0
SCORE_CHAIN_CAPTURE    = 10000.0
SCORE_CLUSTER_BONUS    = 10.0
SCORE_EXTENSION_BONUS  = 1.2
SCORE_CROSS_THREAT     = 1500.0
SCORE_AVOID_CAPTURE_STRONG = 4000.0


@njit(cache=True)
def check_window_15x15(board, row, col, dr, dc, player):
    """Verifica padrões táticos em janelas 5x1 centradas em (row, col)."""
    consecutive = 1
    r, c = row + dr, col + dc
    while 0 <= r < 15 and 0 <= c < 15 and board[r, c] == player:
        consecutive += 1
        r += dr
        c += dc

    r, c = row - dr, col - dc
    while 0 <= r < 15 and 0 <= c < 15 and board[r, c] == player:
        consecutive += 1
        r -= dr
        c -= dc

    if consecutive >= 5:
        return SCORE_WIN_IMMEDIATE

    max_score = 0.0

    for i in range(5):
        stones_count = 0
        valid_window = True
        start_r = row - i * dr
        start_c = col - i * dc
        end_r = start_r + 4 * dr
        end_c = start_c + 4 * dc

        if not (0 <= start_r < 15 and 0 <= start_c < 15 and
                0 <= end_r < 15 and 0 <= end_c < 15):
            continue

        for k in range(5):
            curr_r = start_r + k * dr
            curr_c = start_c + k * dc
            cell = board[curr_r, curr_c]
            if curr_r == row and curr_c == col:
                stones_count += 1
            elif cell == player:
                stones_count += 1
            elif cell != 0:
                valid_window = False
                break

        if valid_window:
            if stones_count == 5:
                return SCORE_WIN_IMMEDIATE
            if stones_count == 4:
                prev_r, prev_c = start_r - dr, start_c - dc
                next_r, next_c = end_r + dr, end_c + dc
                open_sides = 0
                if 0 <= prev_r < 15 and 0 <= prev_c < 15 and board[prev_r, prev_c] == 0:
                    open_sides += 1
                if 0 <= next_r < 15 and 0 <= next_c < 15 and board[next_r, next_c] == 0:
                    open_sides += 1
                
                if open_sides >= 1:
                    score = SCORE_OPEN_FOUR if open_sides == 2 else 2000.0
                    if score > max_score: max_score = score
            if stones_count == 3:
                prev_r, prev_c = start_r - dr, start_c - dc
                next_r, next_c = end_r + dr, end_c + dc
                open_start = (0 <= prev_r < 15 and 0 <= prev_c < 15 and board[prev_r, prev_c] == 0)
                open_end = (0 <= next_r < 15 and 0 <= next_c < 15 and board[next_r, next_c] == 0)
                if open_start and open_end:
                    score = 1500.0
                    if score > max_score: max_score = score
    return max_score


@njit(cache=True)
def score_move_tactical_v3(board, player, move, is_pente, captures_p1, captures_p2):
    row, col = move
    if board[row, col] != 0:
        return -1000000.0

    opponent = 2 if player == 1 else 1
    score = 0.0
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    my_threats = 0
    cross_score = 0.0
    for dr, dc in directions:
        s = check_window_15x15(board, row, col, dr, dc, player)
        if s > 1000.0:
            my_threats += 1
            score += s * 0.7
        if abs(dr) == abs(dc) and s > 1000.0:
            cross_score += SCORE_CROSS_THREAT

    if my_threats >= 2:
        score += SCORE_DOUBLE_THREAT
    score += cross_score
    

    opp_max = 0.0
    for dr, dc in directions:
        s = check_window_15x15(board, row, col, dr, dc, opponent)
        if s > opp_max:
            opp_max = s
            
    if opp_max >= SCORE_WIN_IMMEDIATE:
        score += SCORE_BLOCK_WIN
    elif opp_max >= SCORE_OPEN_FOUR:
        score += SCORE_BLOCK_OPEN_4
    elif opp_max > 1000.0:
        score += opp_max * 0.9

    if is_pente:
        all_dirs = ((0, 1), (1, 0), (1, 1), (1, -1),
                    (0, -1), (-1, 0), (-1, -1), (-1, 1))
        
        my_caps = captures_p1 if player == 1 else captures_p2
        caps_made = 0

        for dr, dc in all_dirs:
            r1, c1 = row - dr, col - dc
            r2, c2 = row - 2*dr, col - 2*dc
            r3, c3 = row - 3*dr, col - 3*dc
            if 0 <= r3 < 15 and 0 <= c3 < 15:
                if (board[r1, c1] == opponent and
                    board[r2, c2] == opponent and
                    board[r3, c3] == player):
                    caps_made += 2

        if caps_made >= 4:
            score += SCORE_CHAIN_CAPTURE
        elif caps_made > 0:
            if my_caps + caps_made >= 10:
                return SCORE_PENTE_WIN_CAP
            score += caps_made * SCORE_MAKE_CAPTURE

        for dr, dc in all_dirs:
            r_friend, c_friend = row + dr, col + dc
            r_tip1, c_tip1 = row - dr, col - dc
            r_tip2, c_tip2 = r_friend + dr, c_friend + dc
            
            if (0 <= r_friend < 15 and 0 <= c_friend < 15 and board[r_friend, c_friend] == player):
                has_enemy_1 = (0 <= r_tip1 < 15 and 0 <= c_tip1 < 15 and board[r_tip1, c_tip1] == opponent)
                has_enemy_2 = (0 <= r_tip2 < 15 and 0 <= c_tip2 < 15 and board[r_tip2, c_tip2] == opponent)
                
                if (has_enemy_1 and 0 <= r_tip2 < 15 and board[r_tip2, c_tip2] == 0) or \
                   (has_enemy_2 and 0 <= r_tip1 < 15 and board[r_tip1, c_tip1] == 0):
                    score -= SCORE_AVOID_CAPTURE_STRONG

    dist = abs(row - 7) + abs(col - 7)
    score += (15 - dist) * 0.5
    
    return score


@njit(cache=True)
def get_tactical_scores_for_moves(board, player, legal_moves, is_pente, captures_p1, captures_p2):
    n = len(legal_moves)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m = legal_moves[i]
        scores[i] = score_move_tactical_v3(board, player, (m[0], m[1]), is_pente, captures_p1, captures_p2)
    return scores


def get_priors(board, player, rules='gomoku'):
    """
    Gera priors táticos.
    Deteta automaticamente se é Pente pelo argumento 'rules' ou por defeito.
    """
    board_np = np.asarray(board, dtype=np.int8)
    is_pente = (rules == 'pente')
    
    priors = np.zeros((15, 15), dtype=np.float64)
    
    for r in range(15):
        for c in range(15):
            if board_np[r, c] == 0:
                priors[r, c] = score_move_tactical_v3(board_np, player, (r, c), is_pente, 0, 0)
    
    max_score = priors.max()
    if max_score > 0:
        priors = np.exp(priors / (max_score / 5.0))
        priors = priors / priors.sum()
    else:
        mask = (board_np == 0).astype(np.float64)
        priors = mask / mask.sum() if mask.sum() > 0 else mask
    
    return priors