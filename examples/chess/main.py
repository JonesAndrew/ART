import art
import asyncio
from dotenv import load_dotenv
import chess
import chess.pgn
import csv
import random
import re
from typing import TypedDict, List, Tuple, Dict, Optional, Union
from art.utils.get_trajectory_messages import get_trajectory_messages

load_dotenv()

class ChessPuzzle(TypedDict):
    id: str
    fen: str
    moves: List[str]
    rating: int
    themes: List[str]

def load_puzzles(file_path: str) -> List[ChessPuzzle]:
    """
    Load puzzles from a CSV file and convert them to ChessPuzzle format.
    Only include mate puzzles.
    
    Args:
        file_path: Path to the CSV file containing puzzle data
        
    Returns:
        List of ChessPuzzle objects
    """
    puzzles = []
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            # Extract relevant fields
            # Format: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
            puzzle_id = row[0]
            fen = row[1]
            moves_str = row[2]
            rating = int(row[3])
            themes = row[7].split()
            
            # Only include mate puzzles
            if "mate" not in themes:
                continue
            
            # Convert moves to list
            moves = moves_str.split()
            
            # Create the ChessPuzzle
            puzzle = ChessPuzzle(
                id=puzzle_id,
                fen=fen,
                moves=moves,
                rating=rating,
                themes=themes
            )
            
            puzzles.append(puzzle)
    
    return puzzles

def is_valid_move(board: chess.Board, move_text: str) -> bool:
    """Check if a move is valid in the current board position."""
    try:
        move = chess.Move.from_uci(move_text)
        return move in board.legal_moves
    except ValueError:
        return False

def calculate_move_reward(board: chess.Board, proposed_move: str, correct_move: str) -> float:
    """
    Calculate reward for the proposed move compared to the correct move.
    
    Args:
        board: Current chess board position
        proposed_move: Move proposed by the model (UCI format)
        correct_move: Correct move (UCI format)
    
    Returns:
        Reward value between 0.0 and 1.0
    """
    # If it's not a valid move, return 0
    if not is_valid_move(board, proposed_move):
        return 0.0
    
    # If it's the exact correct move, return 1.0
    if proposed_move == correct_move:
        return 1.0
    
    # Parse moves
    try:
        proposed = chess.Move.from_uci(proposed_move)
        correct = chess.Move.from_uci(correct_move)
        
        # Partial rewards for getting aspects of the move right
        reward = 0.2  # Base reward for making any valid move
        
        # Right piece being moved
        if proposed.from_square == correct.from_square:
            reward += 0.4
        
        # Right destination square
        if proposed.to_square == correct.to_square:
            reward += 0.4
            
        return reward
    except ValueError:
        return 0.0

def board_to_unicode(board: chess.Board) -> str:
    """Convert a chess board to a unicode representation."""
    unicode_str = str(board)
    
    # Replace pieces with Unicode chess symbols
    replacements = {
        'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
        'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙',
        '.': '·'
    }
    
    for char, replacement in replacements.items():
        unicode_str = unicode_str.replace(char, replacement)
    
    # Add row and column coordinates
    rows = unicode_str.split('\n')
    unicode_rows = []
    for i, row in enumerate(rows):
        unicode_rows.append(f"{8-i} {row}")
    
    unicode_rows.append("  a b c d e f g h")
    
    return '\n'.join(unicode_rows)

model = art.TrainableModel(
    name="mate-solver",
    project="chess-mate",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
)

async def rollout(
    client: asyncio.AsyncOpenAI, puzzle: ChessPuzzle
) -> art.Trajectory:
    
    # Set up the chess board with the FEN position
    board = chess.Board(puzzle["fen"])
    correct_moves = puzzle["moves"]
    total_reward = 0.0
    total_moves = len(correct_moves) // 2  # Number of player moves (excluding opponent responses)
    
    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content":
                """
                You are an expert chess mate puzzle solver. Given a chess position, your task is to find the best move that leads to checkmate.

                When analyzing a position:
                1. Understand whose turn it is to move
                2. Analyze the position for mate threats
                3. Find the move that leads to the fastest checkmate
                
                Always return your answer as a single move in UCI notation (e.g., "e2e4", "g1f3").
                UCI notation specifies the start square and end square of a piece. For example, e2e4 means moving from e2 to e4.
                For promotions, add the promotion piece (q for queen, r for rook, b for bishop, n for knight) at the end, e.g. "a7a8q".
                
                First, work through your analysis in a step-by-step manner within <analysis></analysis> tags.
                
                <analysis>
                [Provide your detailed analysis of the position here]
                </analysis>
                
                Then give your final move in UCI notation within <move></move> tags:
                
                <move>e2e4</move>
                """,
            }
        ],
        reward=0,
        metrics={"valid_moves": 0, "correct_moves": 0, "completed_puzzle": 0}
    )
    
    valid_moves = 0
    correct_moves_count = 0
    completed_moves = 0
    
    # Process the puzzle one move at a time
    move_idx = 0
    while move_idx < len(correct_moves):
        player_move = correct_moves[move_idx]
        opponent_move = correct_moves[move_idx + 1] if move_idx + 1 < len(correct_moves) else None
        
        # Visual representation of the board
        board_display = board_to_unicode(board)
        
        # Add message for this move
        trajectory.messages_and_choices.append(
            {
                "role": "user",
                "content": f"Here is the current chess position:\n\n{board_display}\n\nFEN: {board.fen()}\n\nPlease find the move that leads to checkmate."
            }
        )

        # Get model's response
        messages = get_trajectory_messages(trajectory)
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=model.name,
            max_tokens=2048
        )
        choice = chat_completion.choices[0]
        trajectory.messages_and_choices.append(choice)

        content = choice.message.content
        
        # Extract move from the response
        move_match = re.search(r'<move>(.*?)</move>', content, re.DOTALL)
        if not move_match:
            print(f"Cannot parse move for puzzle {puzzle['id']}, move {move_idx//2 + 1}")
            break
        
        proposed_move = move_match.group(1).strip()
        
        # Calculate reward for this move
        move_reward = calculate_move_reward(board, proposed_move, player_move)
        total_reward += move_reward / total_moves  # Normalize by the number of moves
        
        # Update metrics
        if is_valid_move(board, proposed_move):
            valid_moves += 1
        
        if proposed_move == player_move:
            correct_moves_count += 1
            
            # Make the player's move and opponent's response
            board.push_uci(player_move)
            if opponent_move:
                board.push_uci(opponent_move)
            
            completed_moves += 1
            move_idx += 2  # Move to the next pair
        else:
            # Incorrect move, puzzle attempt ends here
            break
    
    # Set final reward and metrics
    trajectory.reward = total_reward
    trajectory.metrics["valid_moves"] = valid_moves / total_moves
    trajectory.metrics["correct_moves"] = correct_moves_count / total_moves
    trajectory.metrics["completed_puzzle"] = 1.0 if completed_moves == total_moves else 0.0
    
    if random.random() < 0.05:
        print(f"\n----- MODEL OUTPUT FOR PUZZLE {puzzle['id']} -----")
        # Print the visual board from the last position
        print(board_to_unicode(board))
        
        # Print all the model's responses to see its analysis
        for i, msg_choice in enumerate(trajectory.messages_and_choices):
            if msg_choice.get("role") == "assistant":
                print(f"\n--- Move {i//2} ---")
                full_content = msg_choice.get("message", {}).get("content", "")
                print(full_content)
                
                # Extract just the move from the content
                move_match = re.search(r'<move>(.*?)</move>', full_content, re.DOTALL)
                if move_match:
                    print(f"\nFinal move: {move_match.group(1).strip()}")
        
        print(f"\n----- METRICS -----")
        print(f"Total moves: {total_moves}")
        print(f"Completed moves: {completed_moves}")
        print(f"Valid moves: {valid_moves}")
        print(f"Correct moves: {correct_moves_count}")
        print(f"Reward: {total_reward:.2f}")
        print(f"Completed puzzle: {trajectory.metrics['completed_puzzle']}")
        print(f"----------------------------------------\n")
    
    return trajectory

async def main():
    # Load all puzzles
    all_puzzles: list[ChessPuzzle] = load_puzzles("examples/chess/mate_puzzles.csv")
    print(f"Loaded {len(all_puzzles)} mate puzzles")
    
    # Ensure reproducibility
    random.seed(42)
    random.shuffle(all_puzzles)

    # Now split into train/val/test sets
    val_size = min(50, len(all_puzzles) // 10)
    test_size = min(50, len(all_puzzles) // 10)

    val_puzzles = all_puzzles[:val_size]
    test_puzzles = all_puzzles[val_size:val_size+test_size]
    train_puzzles = all_puzzles[val_size+test_size:]

    await model.register(art.LocalAPI())

    batch_size = 4  # Process this many puzzles per batch
    num_epochs = 3  # Number of complete passes through the training data
    openai_client = model.openai_client()
    
    start_step = await model.get_step()
    max_steps = 1000
    
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_puzzles)
        
        # Calculate how many batches we can process in this epoch
        num_batches = min(len(train_puzzles) // batch_size, (max_steps - start_step) // num_epochs)
        
        for batch in range(num_batches):
            current_step = start_step + epoch * num_batches + batch
            if current_step >= max_steps:
                break
                
            print(f"Epoch {epoch+1}, Batch {batch+1}/{num_batches}, Step {current_step}")
            
            batch_start_idx = batch * batch_size
            batch_end_idx = (batch + 1) * batch_size
            
            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for _ in range(2))
                        for puzzle in val_puzzles
                    ),
                    pbar_desc=f"val (epoch {epoch+1})",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(rollout(openai_client, puzzle) for _ in range(24))
                        for puzzle in train_puzzles[batch_start_idx:batch_end_idx]
                    ),
                    pbar_desc=f"train (epoch {epoch+1}, batch {batch+1})",
                ),
            )

            await model.log(val_groups)
            await model.delete_checkpoints()
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )

if __name__ == "__main__":
    asyncio.run(main())