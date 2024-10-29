import random
import json
import numpy as np
from PIL import Image
import os
import pyperclip  # Clipboard functionality

# Configuration Variables
GRID_SIZE = (100, 100)  # Change grid size here (rows, cols)
TILES_JSON_PATH = "rules2.json"  # Path to the JSON file defining tiles
TILES_FOLDER = "tiles"  # Folder where PNG tiles are stored
OUTPUT_IMAGE = "environment.png"  # Name of the generated image
COPY_GRID_TO_CLIPBOARD = True  # Toggle whether to copy grid to clipboard
CLUMPING_ENABLED = True  # Toggle clumping feature

class Tile:
    def __init__(self, name, can_touch, ideal_percent=None, **kwargs):
        self.name = name
        self.can_touch = can_touch + [name]  # Self-compatibility by default
        self.ideal_percent = ideal_percent  # Optional ideal percentage

class WFC:
    def __init__(self, tiles, grid_size, clumping_enabled):
        self.tiles = tiles
        self.grid_size = grid_size
        self.grid = np.empty(grid_size, dtype=object)
        self.tile_counts = {tile.name: 0 for tile in tiles}
        self.total_tiles = grid_size[0] * grid_size[1]
        self.clumping_enabled = clumping_enabled

        # Ensure all tiles have ideal_percent values
        self.assign_ideal_percentages()

    def assign_ideal_percentages(self):
        """Assign ideal percentages for tiles without defined percentages."""
        total_defined_percent = sum(tile.ideal_percent or 0 for tile in self.tiles)
        undefined_tiles = [tile for tile in self.tiles if tile.ideal_percent is None]

        if undefined_tiles:
            remaining_percent = 100 - total_defined_percent
            share = remaining_percent / len(undefined_tiles)

            for tile in undefined_tiles:
                tile.ideal_percent = share

        total_percent = sum(tile.ideal_percent for tile in self.tiles)
        if total_percent != 100:
            correction = 100 - total_percent
            self.tiles[0].ideal_percent += correction  # Adjust the first tile to match exactly

    def is_compatible(self, tile, neighbor):
        """Check if two tiles are compatible."""
        return neighbor.name in tile.can_touch

    def get_valid_tiles(self, row, col):
        """Return valid tiles based on neighbors and clumping preference."""
        valid_tiles = []
        for tile in self.tiles:
            compatible = True

            if col > 0:
                left_tile = self.grid[row, col - 1]
                if left_tile and not self.is_compatible(tile, left_tile):
                    compatible = False

            if row > 0:
                top_tile = self.grid[row - 1, col]
                if top_tile and not self.is_compatible(tile, top_tile):
                    compatible = False

            if compatible:
                valid_tiles.append(tile)

        return valid_tiles

    def weighted_choice(self, valid_tiles, row, col):
        """Choose a tile based on ideal percentage and clumping."""
        weights = []
        for tile in valid_tiles:
            current_percent = (self.tile_counts[tile.name] / self.total_tiles) * 100
            weight = max(1, tile.ideal_percent - current_percent)

            if self.clumping_enabled:
                neighbors = self.get_neighbors(row, col)
                neighbor_match_count = sum(1 for n in neighbors if n and n.name == tile.name)
                weight += neighbor_match_count

            weights.append(weight)

        return random.choices(valid_tiles, weights=weights, k=1)[0]

    def get_neighbors(self, row, col):
        """Get neighboring tiles for clumping."""
        neighbors = []
        if col > 0:
            neighbors.append(self.grid[row, col - 1])
        if row > 0:
            neighbors.append(self.grid[row - 1, col])
        if col < self.grid_size[1] - 1:
            neighbors.append(self.grid[row, col + 1])
        if row < self.grid_size[0] - 1:
            neighbors.append(self.grid[row + 1, col])
        return neighbors

    def collapse(self):
        """Generate the grid."""
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                valid_tiles = self.get_valid_tiles(row, col)

                if not valid_tiles:
                    print(f"Warning: No valid tiles for ({row}, {col}).")
                    self.grid[row, col] = Tile("empty", [])
                else:
                    selected_tile = self.weighted_choice(valid_tiles, row, col)
                    self.grid[row, col] = selected_tile
                    self.tile_counts[selected_tile.name] += 1

    def get_grid_as_text(self):
        """Return the grid as text."""
        return "\n".join(
            " | ".join(
                (self.grid[row, col].name if self.grid[row, col] else "empty")
                for col in range(self.grid_size[1])
            )
            for row in range(self.grid_size[0])
        )

def load_and_generate_complete_rules(json_file):
    """Load tiles, ensure bi-directional rules, and save the updated file with ideal percentages."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    tile_dict = {tile["name"]: tile for tile in data["tiles"]}

    # Ensure bi-directional compatibility
    for tile in data["tiles"]:
        for compatible_tile in tile["can_touch"]:
            if tile["name"] not in tile_dict[compatible_tile]["can_touch"]:
                tile_dict[compatible_tile]["can_touch"].append(tile["name"])

    # Assign ideal percentages
    total_defined_percent = sum(tile.get("ideal_percent", 0) for tile in data["tiles"])
    undefined_tiles = [tile for tile in data["tiles"] if "ideal_percent" not in tile]

    if undefined_tiles:
        remaining_percent = 100 - total_defined_percent
        share = remaining_percent / len(undefined_tiles)

        for tile in undefined_tiles:
            tile["ideal_percent"] = share

    # Save the updated rules back to the JSON file
    complete_rules = {"tiles": list(tile_dict.values())}

    with open(json_file, 'w') as f:
        json.dump(complete_rules, f, indent=2)

    print(f"Updated rules saved to {json_file}")

    return [Tile(**tile_data) for tile_data in complete_rules["tiles"]]

def load_images_from_folder(folder):
    """Load PNG images from a folder."""
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            tile_name = os.path.splitext(filename)[0]
            images[tile_name] = Image.open(os.path.join(folder, filename))
    return images

def generate_image(grid, tile_images, output_file):
    """Generate an image from the grid."""
    rows, cols = grid.shape
    tile_width, tile_height = list(tile_images.values())[0].size

    output_image = Image.new('RGB', (cols * tile_width, rows * tile_height))

    for row in range(rows):
        for col in range(cols):
            tile_name = grid[row, col].name
            tile_image = tile_images.get(tile_name, Image.new('RGB', (tile_width, tile_height)))
            output_image.paste(tile_image, (col * tile_width, row * tile_height))

    output_image.save(output_file)
    print(f"Generated image saved as {output_file}")

if __name__ == "__main__":
    tiles = load_and_generate_complete_rules(TILES_JSON_PATH)
    wfc = WFC(tiles=tiles, grid_size=GRID_SIZE, clumping_enabled=CLUMPING_ENABLED)

    wfc.collapse()

    grid_text = wfc.get_grid_as_text()
    print(grid_text)

    if COPY_GRID_TO_CLIPBOARD:
        pyperclip.copy(grid_text)
        print("Grid text copied to clipboard.")

    tile_images = load_images_from_folder(TILES_FOLDER)
    generate_image(wfc.grid, tile_images, OUTPUT_IMAGE)
