# AGENTS.md — PoOlympic agent rules

Rules for any AI agent working in this repository (Unity 6000.6 + MuJoCo RL athletes).

## Project and workflow
- Check the `docs/` folder in the repo root first for an overall summary of the project (`docs/DESIGN.md`), then `tasks.md` and `rl_optimization_log.md` for status.
- Only use the main branch (`main`, this repo's master branch) for all work. Use other branches only when specifically asked.
- When doing a git sync, always commit all changes beforehand.
- At the end of any answer longer than 100 words, add a TLDR: a summary of about 20 words.

## Training
- All training is done with MuJoCo / Newton (MuJoCo Warp).
- Ask the user for a skinned mesh before attempting to train a new creature. The rig structure comes from that model and is imported into MuJoCo / Newton for training.
- Creatures move realistically: Earth gravity, realistic joint ranges, and mass that matches their size.
- Joints move at speeds and forces that resemble real humans (when the trained agent is a human).
- All body parts of all creatures collide accurately with each other; creatures cannot pass through each other or through anything in the environment.
- When training starts, always start TensorBoard so the user can view progress.
- When training starts, check TensorBoard for obsolete runs taking up room and remove them.
- When training in MuJoCo or Isaac Lab, show the app's viewer so the user can watch how the creature moves during and after training. Use Newton's viewer if that is the better option.
- When 30+ minutes of RL training is needed, close the Unity / Unreal editor if that dramatically speeds up training, and tell the user when it can be opened again (training is over).

## Unity
- Interact with Unity through whichever tool gives the best result: Unity CLI pipeline (`unity command …`), https://github.com/CoplayDev/unity-mcp, or https://github.com/IvanMurzak/Unity-MCP.
- Create as many prefabs / objects in the scene as possible (via MCP / the editor) rather than generating them from code at runtime, so the user can adjust the positions of static objects directly in the scene.
- To avoid stalling: enable "No Throttling" (Preferences › General › Interaction Mode) in the editor and "Run In Background" in Player settings; keep the editor updating in the background.

## Platforms
- Use https://github.com/joanllobera/mujoco-bin/ to compile MuJoCo for Android phones.
