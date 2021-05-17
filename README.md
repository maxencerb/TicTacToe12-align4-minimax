# 12 x 12 TicTacToe align 4

&copy; Maxence Raballand, Julio Pique, Valentin Pierrat 2021

## Description

This is a minimax algorithm for TicTacToe on a 12 by 12 grid where you have to align 4 symbols.

Alpha beta pruning is applied.

Because the tree is very large, we created a heuristic that counts the number of symbols aligned and increase or decrease the score.

## Usage

```python
# Start a game with 2 humans
start_game(human_play, human_play)

# Start a game with 1 human and 1 AI
# the AI has a max depth of 4 for the minimax tree
start_game(human_play, ai_play(4))
```

## Improvements needed

The heuristic function is poor. You can check which player turn it is and increase or decrease the score.

You could also have a better sorting function in `legal_moves` for optimal alpha-beta pruning. The current sort takes the distance to the last played move as the key.

The weights choosen (variable `VALUES`) are purely random and based on few tests so they could be changed.

The usage of a bitboard is recommanded for time and space optimization. With a bitboard, you could add conditions for the heurisitic without adding too much time to the execution.

## Disclaimer

The names of the functions and the way the code works is not self-explainatory in general. It was optimized for execution on a small period of time.

The usage of numba package here also constraint forced us to use basic functions. So the code may seem redundant sometimes.