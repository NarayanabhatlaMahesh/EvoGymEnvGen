# utils/config.py

LLM_MUTATION_PROMPT = """You are mutating a 2D voxel terrain for a soft-robotics simulation.

Convert the mutated terrain into the following STRUCTURAL object format.

IMPORTANT RULES:

1. Grid size MUST remain EXACTLY the same as input:
   - grid_width = input width
   - grid_height = input height

2. DO NOT REMOVE OR MODIFY the ground layer.
   Ground = all voxels in bottom GROUND_HEIGHT rows.

3. Terrain blocks must:
   - be type = 1
   - be grouped into connected components
   - use 4-connectivity (up, down, left, right)

4. Indexing rule:
   index = row * grid_width + column

5. Each terrain object MUST contain:
   - indices
   - types
   - neighbors

6. Neighbors:
   Only include neighbors that belong to the SAME terrain object.

7. Terrain MUST exist (at least one terrain object required).

8. DO NOT create floating terrain unless connected internally.

9. Output ONLY valid JSON.
   NO explanation.
   NO markdown.
   NO comments.

-----------------------------------

OUTPUT FORMAT:

{
  "grid_width": <int>,
  "grid_height": <int>,
  "objects": {
    "ground": {
      "indices": [...],
      "types": [...],
      "neighbors": {
        "idx": [adjacent_idx, ...]
      }
    },
    "terrain_0": {
      "indices": [...],
      "types": [...],
      "neighbors": {
        "idx": [adjacent_idx, ...]
      }
    }
  }
}

-----------------------------------

INPUT TERRAIN:
"""