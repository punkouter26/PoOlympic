# org.mujoco — embedded copy

Source: https://github.com/google-deepmind/mujoco.git?path=unity#3.11.0 (must match native `Assets/Plugins/MuJoCo/mujoco.dll` 3.11.0).
Embedded because Unity 6000.6 promotes some obsolete APIs to compile errors. Patches:

1. `Runtime/Components/Shapes/MjMeshFilter.cs:61` — `mesh.GetInstanceID()` → `mesh.GetEntityId()` (debug mesh name only).
