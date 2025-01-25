import random
from nim import Nim, NimAI, train, play
from flask import Flask, request, jsonify, send_from_directory
import os

app = Flask(__name__)

# If modelfile exists, loads the AI from the file
modelfile = 'nim4.pkl'
if os.path.exists(modelfile):
    ai = NimAI.load(modelfile)
else:
    ai = train(10000)
    ai.save(modelfile)

# def display_piles(game: Nim) -> str:
#     """Return the current state of the piles as a string."""
#     piles_str = "Piles:\n"
#     for i, pile in enumerate(game.piles):
#         piles_str += f"Pile {i}: {pile}\n"
#     return piles_str

@app.route('/start', methods=['POST'])
def start_game():
    """Start a new game."""
    global game, human_player, move_history
    human_player = request.json.get('human_player', random.randint(0, 1))
    game = Nim()
    move_history = []
    return jsonify({
        'message': 'Game started',
        'human_player': human_player,
        'piles': game.piles,
        'current_player': 'Human' if game.player == human_player else 'AI',
        'move_history': move_history,
        'ai_info': {
            'exploration_rate': ai.epsilon,
            'learning_rate': ai.alpha,
            'discount_factor': ai.gamma,
            'train_moves': len(ai.train_moves),
            'name': modelfile
        }
    })

@app.route('/restart', methods=['POST'])
def restart_game():
    """Restart the game."""
    global game, human_player, move_history
    human_player = request.json.get('human_player', random.randint(0, 1))
    game = Nim()
    move_history = []
    return jsonify({
        'message': 'Game restarted',
        'human_player': human_player,
        'piles': game.piles,
        'current_player': 'Human' if game.player == human_player else 'AI',
        'move_history': move_history
    })

@app.route('/human_move', methods=['POST'])
def human_move():
    """Make a move in the game by the human player."""
    global game, move_history
    if game.player != human_player:
        return jsonify({'message': 'Not your turn'}), 400

    pile = request.json['pile']
    count = request.json['count']
    if (pile, count) not in Nim.available_actions(game.piles):
        return jsonify({'message': 'Invalid move'}), 400

    game.move((pile, count))
    move_history.append({'player': 'Human', 'pile': pile, 'count': count})
    if game.winner is not None:
        winner = "Human" if game.winner == human_player else "AI"
        return jsonify({
            'message': 'GAME OVER',
            'winner': winner,
            'piles': game.piles,
            'move_history': move_history
        })

    # Trigger AI move if it's AI's turn
    if game.player != human_player:
        pile, count = ai.choose_action(game.piles)
        game.move((pile, count))
        move_history.append({'player': 'AI', 'pile': pile, 'count': count})
        if game.winner is not None:
            winner = "Human" if game.winner == human_player else "AI"
            return jsonify({
                'message': 'GAME OVER',
                'winner': winner,
                'piles': game.piles,
                'move_history': move_history
            })

    return jsonify({
        'message': 'Move made',
        'piles': game.piles,
        'next_player': 'AI' if game.player != human_player else 'Human',
        'current_player': 'Human' if game.player == human_player else 'AI',
        'move_history': move_history
    })

@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Make a move in the game by the AI."""
    global game, move_history
    if game.player == human_player:
        return jsonify({'message': 'Not AI turn'}), 400

    pile, count = ai.choose_action(game.piles)
    game.move((pile, count))  # Fix the order of pile and count
    move_history.append({'player': 'AI', 'pile': pile, 'count': count})
    if game.winner is not None:
        winner = "Human" if game.winner == human_player else "AI"
        return jsonify({
            'message': 'GAME OVER',
            'winner': winner,
            'piles': game.piles,
            'move_history': move_history
        })

    return jsonify({
        'message': 'Move made',
        'piles': game.piles,
        'next_player': 'Human',
        'current_player': 'Human' if game.player == human_player else 'AI',
        'move_history': move_history
    })

@app.route('/piles', methods=['GET'])
def get_piles():
    """Return the current state of the piles."""
    global game
    return jsonify({
        'piles': game.piles
    })

@app.route('/')
def index():
    """Serve the game HTML."""
    return send_from_directory('.', 'game.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
