# PS1 Reaper for Roblox

A low-poly, PS1-style hooded skeleton playing a lute on a stone slab.

- `reaper.glb`: the model. One mesh, about 1,550 triangles, one 1024×1024 pixel-art texture. About 1.9 m (roughly 7 studs) tall.
- `NightLighting.lua`: night fog, a big moon and a cold color grade.
- `build_reaper.py`: regenerates `reaper.glb` (`python3 build_reaper.py`, needs numpy and Pillow).

## Put it in your game

1. In Roblox Studio, open your place and go to **Home → Import 3D**. Pick `reaper.glb` and click **Import**.
2. Select the reaper in the Explorer, tick **Anchored** in Properties, and move it where you want it.
   If it comes in tiny, import it again with **Scale Unit** set to **Meter**, or resize it with the Scale tool.
3. In **ServerScriptService**, add a **Script** and paste in `NightLighting.lua`.
   It replaces any Atmosphere in Lighting, because Roblox ignores classic fog while one exists.
