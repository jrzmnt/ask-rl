"""Generate FrozenLake evaluation and test maps.

Run this script once before any experiment:
    python scripts/generate_maps.py
"""

from awu.envs.frozen_lake import FrozenLake

SIZES = [4, 5, 6, 7, 8]
N_MAPS = 100

for size in SIZES:
    env = FrozenLake(id="FrozenLake-v1", size=size)
    env.create_structures(N_MAPS, eval=True)
    env.create_structures(N_MAPS, eval=False)
    print(f"Generated {N_MAPS} eval + {N_MAPS} test maps for size {size}x{size}")

print("Done. Maps saved under tmp/")
