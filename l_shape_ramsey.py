import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

class LShapeGrid:
    def __init__(self, size: int):
        self.size = size
        self.grid = np.full((size, size), None, dtype=object)
    
    def set_color(self, x: int, y: int, color: Color) -> None:
        """Set the color at position (x,y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[y][x] = color
    
    def get_color(self, x: int, y: int) -> Optional[Color]:
        """Get the color at position (x,y)"""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.grid[y][x]
        return None
    
    def has_l_shape(self, x: int, y: int) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Check if there's a monochromatic L-shape starting at position (x,y)
        Returns (has_l_shape, l_shape_points)
        """
        color = self.get_color(x, y)
        if color is None:
            return False, []
            
        # Check all possible L-shapes starting from this point
        for d in range(1, self.size):
            # Check all four orientations of L-shapes
            
            # Right and Up
            if (x + d) < self.size and (y + d) < self.size:
                if (self.get_color(x + d, y) == color and 
                    self.get_color(x + d, y + d) == color):
                    return True, [(x, y), (x + d, y), (x + d, y + d)]
            
            # Right and Down
            if (x + d) < self.size and (y - d) >= 0:
                if (self.get_color(x + d, y) == color and 
                    self.get_color(x + d, y - d) == color):
                    return True, [(x, y), (x + d, y), (x + d, y - d)]
            
            # Left and Up
            if (x - d) >= 0 and (y + d) < self.size:
                if (self.get_color(x - d, y) == color and 
                    self.get_color(x - d, y + d) == color):
                    return True, [(x, y), (x - d, y), (x - d, y + d)]
            
            # Left and Down
            if (x - d) >= 0 and (y - d) >= 0:
                if (self.get_color(x - d, y) == color and 
                    self.get_color(x - d, y - d) == color):
                    return True, [(x, y), (x - d, y), (x - d, y - d)]
                    
        return False, []
    
    def has_any_l_shape(self) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check if the grid contains any monochromatic L-shape"""
        for x in range(self.size):
            for y in range(self.size):
                has_l, points = self.has_l_shape(x, y)
                if has_l:
                    return True, points
        return False, []
    
    def __str__(self) -> str:
        """String representation of the grid"""
        result = ""
        for y in range(self.size):
            for x in range(self.size):
                color = self.get_color(x, y)
                if color is None:
                    result += "_ "
                else:
                    result += color.name[0] + " "
            result += "\n"
        return result

    def visualize(self, highlight_l_shape: bool = True, filename: str = None):
        """
        Visualize the grid and save to file if filename is provided, otherwise display
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for y in range(self.size):
            for x in range(self.size):
                color = self.get_color(x, y)
                if color is not None:
                    rect = patches.Rectangle(
                        (x, self.size - 1 - y), 1, 1,
                        facecolor=color.name.lower(),
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=1
                    )
                    ax.add_patch(rect)
        
        # Draw grid lines
        for i in range(self.size + 1):
            ax.axhline(y=i, color='black', linewidth=1)
            ax.axvline(x=i, color='black', linewidth=1)
        
        # Add coordinates with better visibility
        for y in range(self.size):
            for x in range(self.size):
                # White background for coordinates to ensure visibility
                bbox = dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
                ax.text(x + 0.5, self.size - 0.5 - y, f'({x},{y})',
                       ha='center', va='center', fontsize=9,
                       bbox=bbox)
        
        if highlight_l_shape:
            # Find and highlight L-shapes
            has_l, points = self.has_any_l_shape()
            if has_l:
                # Convert points to plot coordinates
                plot_points = [(x + 0.5, self.size - 0.5 - y) for x, y in points]
                # Draw L-shape
                l_shape = plt.Polygon(plot_points, fill=False, color='black', linewidth=2)
                ax.add_patch(l_shape)
        
        # Remove axes ticks since we don't need them
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        plt.title("L-shape Ramsey Grid")
        ax.set_aspect('equal')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    # Example usage
    grid = LShapeGrid(5)
    
    # Create the L-shape from your example
    grid.set_color(1, 2, Color.RED)
    grid.set_color(2, 2, Color.RED)
    grid.set_color(2, 3, Color.RED)
    
    print("Grid with L-shape:")
    print(grid)
    has_l, points = grid.has_any_l_shape()
    print(f"Has L-shape: {has_l}")
    if has_l:
        print(f"L-shape points: {points}")
    
    # Visualize the grid
    grid.visualize()

if __name__ == "__main__":
    main() 