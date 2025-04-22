import art
import asyncio
from dotenv import load_dotenv
import chess
import chess.pgn
import csv
import random
import re
import json
from typing import TypedDict, List, Tuple, Dict, Optional, Union, Any
from art.utils.get_trajectory_messages import get_trajectory_messages
from openai import AsyncOpenAI

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

def calculate_move_reward(board: chess.Board, proposed_move: str, correct_move: str, tool_count: int = 0) -> float:
    """
    Calculate reward for the proposed move compared to the correct move.
    
    Args:
        board: Current chess board position
        proposed_move: Move proposed by the model (UCI format)
        correct_move: Correct move (UCI format)
        tool_count: Number of tools used before making a decision
    
    Returns:
        Reward value between 0.0 and 1.0
    """
    # Base reward for using tools, even if the move is invalid
    tool_usage_reward = min(0.1, 0.025 * tool_count)
    
    # If it's not a valid move, return just the tool usage reward
    if not is_valid_move(board, proposed_move):
        return tool_usage_reward
    
    # If it's the exact correct move, return 1.0 plus bonuses
    if proposed_move == correct_move:
        return 1.0
    
    # Parse moves
    try:
        proposed = chess.Move.from_uci(proposed_move)
        correct = chess.Move.from_uci(correct_move)
        
        reward = 0.5
            
        return reward
    except ValueError:
        return tool_usage_reward

model = art.TrainableModel(
    name="003",
    project="chess-mate-xml",
    base_model="Qwen/Qwen2.5-14B-Instruct",
    _internal_config={"init_args": {"gpu_memory_utilization": 0.775}},
)

def get_legal_moves_for_piece(board: chess.Board, square: str, after_moves: Optional[str] = None) -> Dict[str, Any]:
    """Get all legal moves for a piece at a specific square."""
    try:
        # Create a copy of the board to avoid modifying the original
        board_copy = board.copy()
        
        # Apply the after_moves if provided
        if after_moves:
            moves = after_moves.split()
            for move in moves:
                try:
                    board_copy.push_uci(move)
                except ValueError:
                    return {"error": f"Invalid move in after_moves: {move}"}
        
        # Convert algebraic notation (e.g., "e4") to square index
        square_idx = chess.parse_square(square)
        
        # Check if there's a piece at the specified square
        piece = board_copy.piece_at(square_idx)
        if not piece:
            return {"error": f"No piece found at square {square}"}
            
        # Get all legal moves for the piece
        legal_moves = []
        for move in board_copy.legal_moves:
            if move.from_square == square_idx:
                legal_moves.append(move.uci())
                
        piece_type = piece.symbol()
        piece_color = "white" if piece.color == chess.WHITE else "black"
        
        # Include the current board state for context
        return {
            "square": square,
            "piece_type": piece_type,
            "piece_color": piece_color,
            "moves": legal_moves,
            "board_state": str(board_copy),
            "turn": "white" if board_copy.turn == chess.WHITE else "black"
        }
    except ValueError:
        return {"error": f"Invalid square: {square}"}

def try_move(board: chess.Board, move: str, after_moves: Optional[str] = None) -> Dict[str, Any]:
    """Try a move and return the resulting board state without modifying the original board."""
    try:
        # Create a copy of the board to avoid modifying the original
        board_copy = board.copy()
        
        # Apply the after_moves if provided
        if after_moves:
            moves = after_moves.split()
            for prior_move in moves:
                try:
                    board_copy.push_uci(prior_move)
                except ValueError:
                    return {"error": f"Invalid move in after_moves: {prior_move}"}
        
        # Try to make the move
        chess_move = chess.Move.from_uci(move)
        
        # Check if the move is legal
        if chess_move not in board_copy.legal_moves:
            return {"error": f"Illegal move: {move} in the position after moves: {after_moves}" if after_moves else f"Illegal move: {move}"}
            
        # Make the move
        board_copy.push(chess_move)
        
        # Check for special states
        is_check = board_copy.is_check()
        is_checkmate = board_copy.is_checkmate()
        is_stalemate = board_copy.is_stalemate()
        
        return {
            "move": move,
            "after_moves": after_moves,
            "board_state": str(board_copy),
            "is_check": is_check,
            "is_checkmate": is_checkmate,
            "is_stalemate": is_stalemate,
            "turn": "white" if board_copy.turn == chess.WHITE else "black"
        }
    except ValueError:
        return {"error": f"Invalid move format: {move}"}

async def rollout(
    client: AsyncOpenAI, puzzle: ChessPuzzle
) -> art.Trajectory:
    
    # Set up the chess board with the FEN position
    board = chess.Board(puzzle["fen"])
    correct_moves = puzzle["moves"]
    total_reward = 0.0
    total_moves = len(correct_moves) // 2  # Number of player moves (excluding opponent responses)

    # Determine whose turn it is
    side_to_move = "White" if board.turn == chess.WHITE else "Black"

    # Initialize trajectory
    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content":
                f"""
You are an expert chess mate puzzle solver. Given a chess position, your task is to find the best move that leads to checkmate.

It's currently {side_to_move}'s turn to move.

When analyzing a position:

Understand whose turn it is to move
Analyze the position for mate threats
Find the move that leads to the fastest checkmate
You can use the following tools to help you:

To get all legal moves for a piece on a specific square, use:
<get_legal_moves_for_piece square="e4" />
You can also check legal moves after a sequence of moves:
<get_legal_moves_for_piece square="e4" after_moves="e2e4 e7e5" />

To try a move and see the resulting board state without making the move permanent, use:
<try_move move="e2e4" />
You can also try a move after a sequence of moves:
<try_move move="d1h5" after_moves="e2e4 e7e5" />

To submit your final move decision, use:
<make_move move="e2e4" />

Important rules for responses:

Respond using only one XML tag per response
Do not use more than one XML tag at a time
Do not include any explanatory text—respond with the XML tag only
Your final move must be submitted using <make_move move="..." /> with UCI notation (e.g., "e2e4")

Strategy tip: It's a good idea to try a move with <try_move> before submitting it with <make_move>.
"""
            }
        ],
        reward=0,
        metrics={
            "valid_moves": 0, 
            "correct_moves": 0, 
            "completed_puzzle": 0,
            "used_tools": 0,
            "tool_count_avg": 0,
            "get_legal_moves_count": 0,
            "try_move_count": 0,
            "hit_tool_limit": 0,
            "tried_before_submission": 0  # New metric for moves tried before submission
        }
    )

    valid_moves = 0
    correct_moves_count = 0
    completed_moves = 0
    attempted_moves = 0
    used_tools_count = 0
    get_legal_moves_count = 0
    try_move_count = 0
    total_tool_count = 0
    tried_before_submission_count = 0

    move_idx = 0
    while move_idx < len(correct_moves):
        player_move = correct_moves[move_idx]
        opponent_move = correct_moves[move_idx + 1] if move_idx + 1 < len(correct_moves) else None

        # Reset tool usage counter for this move
        current_move_tool_uses = 0
        current_turn_tool_count = 0
        
        trajectory.messages_and_choices.append(
            {
                "role": "user",
                "content": f"Here is the current chess position:\n\n{board}\n\nFEN: {board.fen()}\n\nPlease find the move that leads to checkmate."
            }
        )

        # Process this turn
        turn_complete = False
        proposed_move = None
        # Track moves that have been tried using try_move
        tried_moves = []

        while not turn_complete and current_move_tool_uses < 50:
            # Get model response
            messages = get_trajectory_messages(trajectory)
            chat_completion = await client.chat.completions.create(
                messages=messages,
                model=model.name,
                max_tokens=2048
            )
            
            choice = chat_completion.choices[0]
            trajectory.messages_and_choices.append(choice)
            
            content = choice.message.content or ""
            
            # Parse for XML tool calls
            get_legal_moves_match = re.search(r'<get_legal_moves_for_piece\s+square="([^"]+)"(?:\s+after_moves="([^"]+)")?\s*/>', content)
            try_move_match = re.search(r'<try_move\s+move="([^"]+)"(?:\s+after_moves="([^"]+)")?\s*/>', content)
            make_move_match = re.search(r'<make_move\s+move="([^"]+)"\s*/>', content)

            if not content.startswith("<") or not content.endswith(">") or content.count("<") > 1:
                attempted_moves += 1
                total_reward = 0
                turn_complete = True
                break
            
            # If no tool calls were used in the response
            if not (get_legal_moves_match or try_move_match or make_move_match):
                attempted_moves += 1
                total_reward = 0
                turn_complete = True
                break
            else:
                # Process tool calls
                if get_legal_moves_match:
                    square = get_legal_moves_match.group(1)
                    after_moves = get_legal_moves_match.group(2) if get_legal_moves_match.group(2) else None
                    get_legal_moves_count += 1
                    current_turn_tool_count += 1
                    total_tool_count += 1
                    current_move_tool_uses += 1
                    used_tools_count += 1
                    
                    result = get_legal_moves_for_piece(board, square, after_moves)
                    
                    # Add tool response to conversation
                    after_moves_text = f" after moves {after_moves}" if after_moves else ""
                    trajectory.messages_and_choices.append({
                        "role": "user",
                        "content": f"Legal moves for piece at {square}{after_moves_text}:\n```json\n{json.dumps(result, indent=2)}\n```\nContinue with your analysis."
                    })
                    
                elif try_move_match:
                    move = try_move_match.group(1)
                    after_moves = try_move_match.group(2) if try_move_match.group(2) else None
                    try_move_count += 1
                    current_turn_tool_count += 1
                    total_tool_count += 1
                    current_move_tool_uses += 1
                    used_tools_count += 1
                    
                    # Add this move to the list of tried moves
                    if not after_moves and move not in tried_moves:
                        tried_moves.append(move)
                    
                    result = try_move(board, move, after_moves)
                    
                    # Add tool response to conversation
                    after_moves_text = f" after moves {after_moves}" if after_moves else ""
                    trajectory.messages_and_choices.append({
                        "role": "user",
                        "content": f"Result of trying move {move}{after_moves_text}:\n```json\n{json.dumps(result, indent=2)}\n```\nContinue with your analysis."
                    })
                    
                elif make_move_match:
                    proposed_move = make_move_match.group(1)
                    current_turn_tool_count += 1
                    total_tool_count += 1
                    current_move_tool_uses += 1
                    used_tools_count += 1
                    attempted_moves += 1
                    
                    # Add response and complete the turn
                    trajectory.messages_and_choices.append({
                        "role": "user",
                        "content": f"You've submitted the move: {proposed_move}"
                    })
                    
                    turn_complete = True
            
            # If we got a move proposal (either from make_move or move tag), evaluate it
            if proposed_move:
                # Calculate the reward for this move
                move_reward = calculate_move_reward(
                    board, 
                    proposed_move, 
                    player_move, 
                    tool_count=current_turn_tool_count
                )
                if tried_moves and proposed_move in tried_moves:
                    move_reward = move_reward * 0.8 + 0.2
                    tried_before_submission_count += 1
                total_reward += move_reward / total_moves
                
                if is_valid_move(board, proposed_move):
                    valid_moves += 1
                
                if proposed_move == player_move:
                    correct_moves_count += 1
                    board.push_uci(player_move)
                    if opponent_move:
                        board.push_uci(opponent_move)
                    completed_moves += 1
                    move_idx += 2
                else:
                    # Incorrect move, break out of the puzzle
                    break
                    
                turn_complete = True
        
        # If we reached the tool limit without making a move
        if current_move_tool_uses >= 50 and not turn_complete:
            trajectory.metrics["hit_tool_limit"] += 1
            attempted_moves += 1
            # Add message about reaching tool limit
            trajectory.messages_and_choices.append({
                "role": "user",
                "content": "You've reached the maximum of 50 tool calls without submitting a move. Turn ends."
            })
            break
            
        # If the turn is complete but we didn't have a valid move or correct move,
        # break out of the puzzle
        if turn_complete and (not proposed_move or proposed_move != player_move):
            break

    trajectory.reward = total_reward
    trajectory.metrics["valid_moves"] = valid_moves / max(attempted_moves, 1)
    trajectory.metrics["correct_moves"] = correct_moves_count / max(attempted_moves, 1)
    trajectory.metrics["completed_puzzle"] = 1.0 if completed_moves == total_moves else 0.0
    trajectory.metrics["used_tools"] = used_tools_count / max(attempted_moves, 1)
    trajectory.metrics["tool_count_avg"] = total_tool_count / max(attempted_moves, 1)
    trajectory.metrics["get_legal_moves_count"] = get_legal_moves_count
    trajectory.metrics["try_move_count"] = try_move_count
    trajectory.metrics["tried_before_submission"] = tried_before_submission_count / max(attempted_moves, 1)

    if random.random() < 0.05:
        print(f"\n----- MODEL OUTPUT FOR PUZZLE {puzzle['id']} -----")
        print(board)
        print(trajectory.messages())
        print(f"\n----- METRICS -----")
        print(f"Total moves: {total_moves}")
        print(f"Completed moves: {completed_moves}")
        print(f"Valid moves: {valid_moves}")
        print(f"Correct moves: {correct_moves_count}")
        print(f"Used tools ratio: {used_tools_count}/{attempted_moves}")
        print(f"Get legal moves count: {get_legal_moves_count}")
        print(f"Try move count: {try_move_count}")
        print(f"Tried before submission: {tried_before_submission_count}/{attempted_moves}")
        print(f"Average tools per attempt: {total_tool_count/max(attempted_moves, 1):.2f}")
        print(f"Hit tool limit: {trajectory.metrics['hit_tool_limit']} times")
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