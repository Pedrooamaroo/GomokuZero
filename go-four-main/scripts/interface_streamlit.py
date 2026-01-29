import streamlit as st
import numpy as np
import time
import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
src_path = os.path.join(project_root, 'src')

if project_root not in sys.path:
    sys.path.append(project_root)

if src_path not in sys.path:
    sys.path.append(src_path)

try:
    import game_engine
    from player import Player
except ImportError:
    st.error("Erro ao importar módulos. Verifica se estás a correr o script a partir da raiz do projeto.")
    st.stop()

st.set_page_config(page_title="AlphaZero Arena", page_icon="⚔️", layout="wide")


@st.cache_resource
def load_player_agent(game_type, player_ind):
    """
    Carrega uma instância do Player.
    player_ind: 1 (Pretas) ou 2 (Brancas)
    """
    try:
        player = Player(rules=game_type, board_size=15)
        player.set_player_ind(player_ind)
        return player, "Sucesso"
    except Exception as e:
        return None, str(e)


if 'board' not in st.session_state:
    st.session_state.board = np.zeros((15, 15), dtype=np.int8)
if 'turn' not in st.session_state:
    st.session_state.turn = 1
if 'game_over' not in st.session_state:
    st.session_state.game_over = False
if 'winner' not in st.session_state:
    st.session_state.winner = None
if 'captures' not in st.session_state:
    st.session_state.captures = {1: 0, 2: 0}
if 'last_move' not in st.session_state:
    st.session_state.last_move = None

def reset_game():
    st.session_state.board = np.zeros((15, 15), dtype=np.int8)
    st.session_state.turn = 1
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.captures = {1: 0, 2: 0}
    st.session_state.last_move = None
    
    st.cache_resource.clear()


def apply_move(row, col, player_ind, game_type):
    """Aplica uma jogada no estado e verifica vitória"""
    st.session_state.board[row, col] = player_ind
    st.session_state.last_move = (row, col)
    
    if game_type == 'pente':
        caps = game_engine.apply_capture_pente(st.session_state.board, row, col)
        st.session_state.captures[player_ind] += caps

    check_win(player_ind, game_type)
    
    if not st.session_state.game_over:
        st.session_state.turn = 3 - player_ind

def ai_move_logic(game_type, ai_player, player_ind):
    if st.session_state.game_over:
        return

    board = st.session_state.board
    last_move = st.session_state.last_move
    
    with st.spinner(f'IA (P{player_ind}) a pensar...'):
        row, col = ai_player.play(board.tolist(), 0, last_move)

    if row == -1 or st.session_state.board[row, col] != 0:
        st.error(f"IA (P{player_ind}) desistiu ou fez jogada inválida!")
        st.session_state.game_over = True
        return

    apply_move(row, col, player_ind, game_type)

def check_win(player, game_type):
    board = st.session_state.board
    winner = 0
    
    if game_type == 'pente':
        c1 = st.session_state.captures[1]
        c2 = st.session_state.captures[2]
        winner = game_engine.check_win_by_capture(c1, c2)
        if winner == 0:
            winner = game_engine.check_win_pente(board)
    else:
        winner = game_engine.check_win_gomoku(board)
        
    if winner != 0:
        st.session_state.game_over = True
        st.session_state.winner = winner
        if winner == 1:
            st.balloons() 
        else:
            st.snow()    


st.title("Arena AlphaZero")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configurações")
    
    # 1. Escolha do Jogo
    game_mode = st.selectbox("Regras do Jogo", ["gomoku", "pente"], key="gmode")
    
    # 2. Escolha do Modo
    match_mode = st.radio("Modo de Partida", ["Humano vs IA", "IA vs IA"], key="mmode")
    
    st.divider()

    # 3. Carregamento de Agentes
    agent2, msg2 = load_player_agent(game_mode, 2)
    agent1 = None
    
    status_text = ""
    if match_mode == "IA vs IA":
        agent1, msg1 = load_player_agent(game_mode, 1)
        if agent1 and agent2:
            status_text = "Ambas as IAs prontas!"
        else:
            status_text = f"Erro IA: {msg1 or msg2}"
    else:
        if agent2:
            status_text = "IA (P2) pronta!"
        else:
            status_text = f"Erro IA: {msg2}"
    
    if "Erro" in status_text:
        st.error(status_text)
    else:
        st.success(status_text)
        st.caption(f"Modelo usado: {game_mode}_model_best.pth")

    st.divider()
  
    if st.button("🔄 Novo Jogo", type="primary"):
        reset_game()
        st.rerun()

    auto_play = False
    sleep_time = 0.5
    if match_mode == "IA vs IA" and not st.session_state.game_over:
        st.subheader("Auto-Play")
        auto_play = st.checkbox("Executar Automaticamente", value=False)
        if auto_play:
            sleep_time = st.slider("Velocidade (segundos)", 0.0, 2.0, 0.5)

    if game_mode == 'pente':
        st.divider()
        st.subheader("Capturas")
        c1 = st.session_state.captures[1]
        c2 = st.session_state.captures[2]
        
        p1_label = "IA 1 (⚫)" if match_mode == "IA vs IA" else "Humano (⚫)"
        p2_label = "IA 2 (⚪)"
        
        st.metric(p1_label, c1, delta=f"Faltam {5 - c1//2} pares")
        st.metric(p2_label, c2, delta=f"Faltam {5 - c2//2} pares")

    if st.session_state.winner:
        st.divider()
        w = st.session_state.winner
        if w == 1:
            lbl = "IA 1 (Pretas)" if match_mode == "IA vs IA" else "HUMANO"
            st.success(f"VENCEDOR: {lbl}!")
        else:
            st.info(f"VENCEDOR: IA 2 (Brancas)!")

with col2:
    turn_label = ""
    if st.session_state.game_over:
        turn_label = "Jogo Terminado"
    else:
        t = st.session_state.turn
        if match_mode == "Humano vs IA":
            turn_label = "👤 O teu turno (Pretas)" if t == 1 else "🤖 IA a pensar..."
        else:
            turn_label = f"🤖 IA 1 (Pretas) a pensar..." if t == 1 else f"🤖 IA 2 (Brancas) a pensar..."
            
    st.subheader(turn_label)
    
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        height: 36px; width: 36px;
        padding: 0px; border-radius: 50%;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    disable_interaction = st.session_state.game_over or (match_mode == "IA vs IA") or (match_mode == "Humano vs IA" and st.session_state.turn == 2)

    for r in range(15):
        cols = st.columns(15, gap="small")
        for c in range(15):
            val = st.session_state.board[r, c]

            label = "⚫" if val == 1 else "⚪" if val == 2 else " "
            
            is_last = st.session_state.last_move == (r, c)
            btn_type = "primary" if is_last else "secondary"
            
            btn = cols[c].button(
                label, 
                key=f"{r}-{c}",
                disabled=disable_interaction or val != 0,
                type=btn_type
            )
            
            if btn and not disable_interaction:
                apply_move(r, c, 1, game_mode)
                st.rerun()

    if not st.session_state.game_over:
        if match_mode == "Humano vs IA" and st.session_state.turn == 2:
            time.sleep(0.1)
            ai_move_logic(game_mode, agent2, 2)
            st.rerun()
            
        elif match_mode == "IA vs IA":
            if auto_play:
                time.sleep(sleep_time)
                current_agent = agent1 if st.session_state.turn == 1 else agent2
                ai_move_logic(game_mode, current_agent, st.session_state.turn)
                st.rerun()
            else:
                if st.button("▶️ Fazer Jogada IA"):
                    current_agent = agent1 if st.session_state.turn == 1 else agent2
                    ai_move_logic(game_mode, current_agent, st.session_state.turn)
                    st.rerun()