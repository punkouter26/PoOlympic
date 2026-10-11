# AGENTS.md — PoOlympic agent rules

Rules for any AI agent working in this repository (Unity 6000.6 + MuJoCo RL athletes).

## 1. Project setup and git workflow
- **Project context:** check the `docs/` folder in the repo root first for an overall summary of the project (`docs/DESIGN.md`), then `tasks.md` and `rl_optimization_log.md` for status.
- **Branching and sync:** only use the main branch (`main`, this repo's master branch) for all work. Use other branches only when specifically asked. When doing a git sync, always commit all changes beforehand.
- **Unity tooling:** interact with Unity through whichever tool gives the best result: Unity CLI pipeline (`unity command …`), https://github.com/CoplayDev/unity-mcp, or https://github.com/IvanMurzak/Unity-MCP.
- **Scene construction:** create as many prefabs / objects in the scene as possible (via MCP / the Unity CLI) rather than generating them from code at runtime, so the user can adjust the positions of static objects directly in the scene.
- **Performance safeguards:** to avoid stalling, enable "No Throttling" (Preferences › General › Interaction Mode) in the editor and "Run In Background" in Player settings (via MCP / the Unity CLI); turn on auto-tick so the editor keeps updating in the background.
- **UI documentation:** when a change to the UI is made, take an annotated screenshot showing the old and the new UI, with the changes marked. Put it in an HTML report inside `SCREENSHOTS/`.

## 2. Training environment and execution
- **Engine routing:** all training is done with MuJoCo / Newton (MuJoCo Warp). Use https://github.com/joanllobera/mujoco-bin/ to compile MuJoCo for Android phones. Exception: if an app uses Unity ML-Agents, ignore the MuJoCo / Isaac Lab instructions for that training.
- **Model pipeline:** ask the user for a verified skinned mesh before attempting to train a new creature. Get the rig structure from that model and import it into MuJoCo / Newton for training.
- **Simulation UI:** when training in MuJoCo or Isaac Lab, show the app's viewer so the user can watch how the creature moves during and after training. Use Newton's viewer if that is the better option.
- **Resource management:** when 30+ minutes of RL training is needed, close the Unity / Unreal editor if that dramatically speeds up training, and tell the user when it can be opened again (training is over). When only doing RL training (no Unity changes), close the Unity editor, and open it again when training is complete.
- **TensorBoard:** when training starts, always start TensorBoard so the user can view progress, and check it for obsolete runs taking up room and remove them.
- **Analytics:** when a run has more than 30 minutes of training, take screenshots of the 3 most consequential TensorBoard charts, review all 3 and explain in simple terms what they show. Compare the running training with previous runs of the same task (same charts, both runs visible) and say in simple terms whether it is doing better or worse, and why. Put the screenshots and review in an HTML file, describing each chart at 3 levels: (1) toddler, (2) child, (3) adult. Screenshots: `training/tools/tb_shots.py`; example page: `parity/tb/rs_v2/review.html`.
- **Data logging:** log all agent actions to file logs or Azure Tables, to analyze movement profiles, win / loss patterns and success vectors. Log training data so it can be aggregated into reports over time that show how behaviours and metrics affect the game results.

## 3. Physics, biomechanics and self-collision
- **Physical realism:** creatures move realistically: Earth gravity, realistic joint ranges, and mass that matches their size. Joints move at speeds and forces that resemble real humans (when the trained agent is a human).
- **World collision:** all body parts of all creatures collide accurately with each other; creatures cannot pass through each other or through anything in the environment.
- **Primitive colliders:** every creature collides with itself using simple shapes (capsules, boxes, spheres) fitted inside its skinned mesh. Never use the visual mesh or the bones as colliders.
- **Collision rules:** all body-part pairs collide except parent–child pairs and pairs that overlap in the default standing pose; joint limits handle those. Before training, verify that no pair touches in the T-pose, the default stance and a normal arm and leg swing.
- **Reward shaping:** self-contact never ends an episode. If the policy leans on it, add a small self-contact force penalty.
- **Warm starts:** prefer a warm start for a new skill (a brain trained without self-collision, or the previous rung). Train from scratch only for a body's first brain, or where a warm start has been tried and failed.

## 4. Response formatting
- **TL;DR rule:** at the end of any answer longer than 100 words, add a TL;DR: a summary of about 20 words.
