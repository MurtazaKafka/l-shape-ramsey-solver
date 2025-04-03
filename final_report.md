# L-shape Ramsey Problem: Findings and FunSearch Implementation

## Problem Definition

The L-shape Ramsey problem asks for a coloring of a grid such that no L-shape is monochromatic (has all the same color). An L-shape consists of three points where two points are equidistant from the third point, forming a right angle.

Examples of L-shapes:
- Points at (0,0), (2,0), and (2,2) form an L-shape
- Points at (1,1), (1,3), and (3,3) form an L-shape
- Points at (4,2), (2,2), and (2,0) form an L-shape

## Key Findings

### Pattern Effectiveness by Grid Size

Our extensive testing revealed that pattern effectiveness is highly dependent on grid size:

1. **3×3 Grid**:
   - **Latin Square Pattern**: Valid solution with a score of 7.0
   - **Modular Arithmetic Pattern (i + 2*j) % 3**: Valid solution

2. **4×4 Grid**:
   - Most patterns consistently failed with an invalid L-shape (Right+Up) at (0,0) with d=3
   - **Corner-Focused Pattern**: We discovered a valid solution by using a corner-focused approach:
     ```
     [[0 1 1 2]
      [1 0 2 1]
      [1 2 0 1]
      [2 1 1 0]]
     ```
   - This solution has some interesting properties, including symmetry across the diagonal

3. **5×5 Grid and Above**:
   - None of our tested patterns (modular arithmetic, alternating, block, etc.) produced valid solutions

### Verification Challenges

We identified discrepancies in verification methods across different implementations, which led to contradictory results. Our final verification method using the `LShapeGrid` class from the `l_shape_ramsey` module provided consistent and reliable results.

### Pattern Complexity

As grid size increases, the complexity of required patterns increases non-linearly. Simple patterns that work for smaller grids fail for larger ones, suggesting that:

1. Larger grids may require more colors (beyond 3)
2. The patterns likely become more complex and less structured
3. The boundary conditions become increasingly difficult to satisfy

## FunSearch Implementation

We successfully implemented a simplified version of FunSearch using the Llama 3.2 model to generate potential solutions for the L-shape Ramsey problem. Our implementation:

1. Starts with a known valid baseline (Latin square pattern for 3×3)
2. Uses the Llama model to generate candidate Python functions
3. Evaluates and scores generated solutions
4. Saves and visualizes valid solutions

The implementation demonstrated the ability to find multiple valid 3-colorings for the 3×3 grid with varying scores. However, it was less successful for larger grid sizes due to the inherent complexity of the problem.

### Successful Results for 3×3 Grid

Our FunSearch implementation found several valid 3-colorings for the 3×3 grid, including:

```
[[0 1 2]
 [2 0 1]
 [1 2 0]]
```

```
[[0 1 2]
 [1 2 0]
 [2 0 1]]
```

```
[[2 0 1]
 [0 1 2]
 [1 2 0]]
```

All valid solutions maintained a Latin square pattern (each row and column contains each color exactly once), which appears to be optimal for this problem size.

### Specialized 4×4 Solution

For the 4×4 grid, we used a specialized solver that tried various patterns, eventually discovering a valid solution:

```
[[0 1 1 2]
 [1 0 2 1]
 [1 2 0 1]
 [2 1 1 0]]
```

This solution has these key characteristics:
- Symmetry along the main diagonal
- Strategic placement of duplicate colors to avoid forming L-shapes
- Each corner uses a different color (0, 2, 1, 0 clockwise from top-left)
- The pattern doesn't follow any simple formula, suggesting that larger grids may require complex, non-systematic patterns

## Challenges and Limitations

1. **Scaling Difficulty**: Finding valid patterns for larger grids (5×5 and above) proved extremely difficult using our current approaches.

2. **Verification Complexity**: Ensuring consistent verification across different implementations was challenging due to the multiple ways L-shapes can be defined and checked.

3. **LLM Generation Limitations**: While the Llama model could generate valid solutions for the 3×3 grid, its ability to reason about more complex patterns for larger grids was limited.

## Future Directions

1. **Specialized Patterns**: Develop specialized patterns for each grid size rather than seeking a universal formula.

2. **More Colors**: Investigate whether using more than 3 colors allows for valid solutions in larger grids.

3. **Mathematical Analysis**: Perform a deeper mathematical analysis of the L-shape Ramsey numbers to establish theoretical bounds.

4. **Enhanced FunSearch**: Improve the FunSearch implementation with more sophisticated prompting and evaluation techniques to guide the model toward more complex patterns.

5. **Distributed Computation**: Scale up the search by using distributed computation to explore a wider range of potential solutions.

6. **Pattern Recognition**: Analyze the successful 4×4 pattern to see if its principles can be extended to larger grid sizes.

## Conclusion

The L-shape Ramsey problem presents a fascinating challenge at the intersection of combinatorial mathematics and computer science. Our findings demonstrate that while simple solutions exist for small grid sizes, the problem quickly becomes more complex as grid size increases.

Our work revealed valid solutions for both 3×3 and 4×4 grids, with the 4×4 solution requiring a more specialized, non-systematic approach. This suggests that the problem may require increasingly sophisticated patterns as grid size increases, potentially necessitating a combination of human insight and computational search techniques.

Our simplified FunSearch implementation provides a foundation for future work, demonstrating the potential of using language models to explore combinatorial problems. However, the limited success with larger grid sizes highlights the need for more sophisticated approaches and possibly theoretical breakthroughs to make meaningful progress on this problem.