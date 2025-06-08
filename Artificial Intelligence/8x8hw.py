import heapq
import math
from typing import List, Tuple, Optional, Dict


# Utility functions
def manhattanheu(state: List[List[int]], size: int) -> int:
    heuristic = 0
    for i in range(size):
        for j in range(size):
            if state[i][j] != 0:
                x, y = divmod(state[i][j] - 1, size)
                heuristic += abs(x - i) + abs(y - j)
    return heuristic


def euclidean_distanceheu(state: List[List[int]], size: int) -> int:
    heuristic = 0
    for i in range(size):
        for j in range(size):
            if state[i][j] != 0:
                x, y = divmod(state[i][j] - 1, size)
                heuristic += math.sqrt((x - i)**2 + (y - j)**2)
    return heuristic


def missing_tileheu(state: List[List[int]], size: int) -> int:
    heuristic = 0
    goal_state = [[(i * size + j + 1) % (size * size) for j in range(size)]
                  for i in range(size)]
    for i in range(size):
        for j in range(size):
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                heuristic += 1
    return heuristic


def generate_moves(state: List[List[int]], size: int) -> List[List[List[int]]]:
    moves = []
    x, y = next(
        (i, j) for i in range(size) for j in range(size) if state[i][j] == 0)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[
                x][y]
            moves.append(new_state)
    return moves


def count_inversions(state: List[List[int]]) -> int:
    size = len(state)
    flat_state = [num for sublist in state for num in sublist if num != 0]
    inversions = sum(1 for i in range(len(flat_state))
                     for j in range(i + 1, len(flat_state))
                     if flat_state[i] > flat_state[j])
    return inversions


def is_solvable(state: List[List[int]]) -> bool:
    size = len(state)
    inversions = count_inversions(state)
    if size % 2 == 1:
        return inversions % 2 == 0
    else:
        x, _ = next((i, j) for i in range(size) for j in range(size)
                    if state[i][j] == 0)
        return (size - x) % 2 != inversions % 2


# A* algorithm implementation
def astar(
    initial_state: List[List[int]],
    goal_state: List[List[int]],
    heuristic_fn,
    size: int,
    max_steps: int = 4000
) -> Tuple[Optional[List[List[List[int]]]], Optional[int]]:
    queue = []
    heapq.heappush(queue,
                   (heuristic_fn(initial_state, size), 0, initial_state))
    prev: Dict[Tuple[Tuple[int, ...], ...], Tuple[List[List[int]],
                                                  List[List[int]]]] = {}
    visited = set()
    current_steps = 0

    while queue:
        _, moves, state = heapq.heappop(queue)
        current_steps += 1
        visited.add(tuple(map(tuple, state)))

        if current_steps >= max_steps:
            print(
                'The algorithm reached the maximum number of steps without finding a solution'
            )
            return None, None

        if state == goal_state:
            path = []
            while state != initial_state:
                path.append(state)
                state, move = prev[tuple(map(tuple, state))]
            path.append(initial_state)
            path.reverse()
            return path, moves

        for move in generate_moves(state, size):
            if tuple(map(tuple, move)) not in visited:
                heapq.heappush(
                    queue, (moves + heuristic_fn(move, size), moves + 1, move))
                prev[tuple(map(tuple, move))] = (state, move)

    return None, None


# GBFS algorithm implementation
def gbfs(
    initial_state: List[List[int]],
    goal_state: List[List[int]],
    heuristic_fn,
    size: int,
    max_steps: int = 4000
) -> Tuple[Optional[List[List[List[int]]]], Optional[int]]:
    queue = []
    heapq.heappush(queue,
                   (heuristic_fn(initial_state, size), 0, initial_state))
    prev: Dict[Tuple[Tuple[int, ...], ...], Tuple[List[List[int]],
                                                  List[List[int]]]] = {}
    visited = set()
    current_steps = 0

    while queue:
        _, moves, state = heapq.heappop(queue)
        current_steps += 1
        visited.add(tuple(map(tuple, state)))

        if current_steps >= max_steps:
            print(
                'The algorithm reached the maximum number of steps without finding a solution'
            )
            return None, None

        if state == goal_state:
            path = []
            while state != initial_state:
                path.append(state)
                state, move = prev[tuple(map(tuple, state))]
            path.append(initial_state)
            path.reverse()
            return path, moves

        for move in generate_moves(state, size):
            if tuple(map(tuple, move)) not in visited:
                heapq.heappush(queue,
                               (heuristic_fn(move, size), moves + 1, move))
                prev[tuple(map(tuple, move))] = (state, move)

    return None, None


# Utility to print the path
def print_path(path: List[List[List[int]]], moves: int, heuristic_name: str,
               algorithm_name: str):
    print(
        f"Number of steps by {algorithm_name} {heuristic_name} heuristic is {moves}:"
    )
    for state in path:
        for row in state:
            print(row)
        print()


# Main function to solve the puzzle
def solve_puzzle():
    size = int(
        input("Enter the puzzle size (2 for 2x2, 3 for 3x3, 4 for 4x4): "))
    initial_state = []
    print(
        "Enter the initial state row by row like in a matrix, with 0 representing the empty space:"
    )
    for _ in range(size):
        row = list(map(int, input().split()))
        initial_state.append(row)

    if not is_solvable(initial_state):
        print("The puzzle is not solvable.")
        return

    goal_state = []
    counter = 1
    for i in range(size):
        row = []
        for j in range(size):
            if counter == size * size:
                row.append(0)
            else:
                row.append(counter)
            counter += 1
        goal_state.append(row)

    print("Solving using A* Manhattan Heuristic...")
    path, moves = astar(initial_state, goal_state, manhattanheu, size)
    if path:
        print_path(path, moves, "Manhattan", "A*")
    else:
        print("No solution found using A* Manhattan heuristic.")

    print("Solving using A* Euclidean Distance Heuristic...")
    path, moves = astar(initial_state, goal_state, euclidean_distanceheu, size)
    if path:
        print_path(path, moves, "Euclidean Distance", "A*")
    else:
        print("No solution found using A* Euclidean Distance heuristic.")

    print("Solving using A* Missing Tile Heuristic...")
    path, moves = astar(initial_state, goal_state, missing_tileheu, size)
    if path:
        print_path(path, moves, "Missing Tile", "A*")
    else:
        print("No solution found using A* Missing Tile heuristic.")

    print("Solving using GBFS Manhattan Heuristic...")
    path, moves = gbfs(initial_state, goal_state, manhattanheu, size)
    if path:
        print_path(path, moves, "Manhattan", "GBFS")
    else:
        print("No solution found using GBFS Manhattan heuristic.")

    print("Solving using GBFS Euclidean Distance Heuristic...")
    path, moves = gbfs(initial_state, goal_state, euclidean_distanceheu, size)
    if path:
        print_path(path, moves, "Euclidean Distance", "GBFS")
    else:
        print("No solution found using GBFS Euclidean Distance heuristic.")

    print("Solving using GBFS Missing Tile Heuristic...")
    path, moves = gbfs(initial_state, goal_state, missing_tileheu, size)
    if path:
        print_path(path, moves, "Missing Tile", "GBFS")
    else:
        print("No solution found using GBFS Missing Tile heuristic.")


# Example usage
if __name__ == "__main__":
    solve_puzzle()
