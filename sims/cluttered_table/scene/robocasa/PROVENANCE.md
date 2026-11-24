# RoboCasa workbench objects

These twelve objects are from the **RoboCasa team's official asset release**,
which credits Objaverse as their source collection:

- [RoboCasa repository](https://github.com/robocasa/robocasa/tree/3443cdf3dd20f9ae8ed41adba442059b506891d0)
- [Pinned RoboCasa asset release](https://huggingface.co/datasets/robocasa/robocasa-assets/tree/1b92c3d02ca4354984fec961357db0bff7b32166)
- Archive: `objaverse.zip`, revision `1b92c3d02ca4354984fec961357db0bff7b32166`
- License: **Creative Commons Attribution 4.0 International**; see the bundled
  [license text](LICENSE-CC-BY-4.0.txt) and [upstream license declaration](https://github.com/robocasa/robocasa#license).
- Attribution: RoboCasa contributors and the original Objaverse asset creators.
  The distributed archive identifies these assets by category/instance paths;
  it does not include per-instance creator or original-host metadata.

| Local object | Path within upstream archive | Uniform scale relative to source XML | Simulation mass |
| --- | --- | --- | --- |
| `mug` | `objaverse/mug/mug_1` | 1.0 | 0.18 kg |
| `water_bottle` | `objaverse/water_bottle/water_bottle_8` | 1.3 | 0.12 kg |
| `scissors` | `objaverse/scissors/scissors_1` | 1.0 | 0.06 kg |
| `tray` | `objaverse/tray/tray_0` | 1.0 | Fixed fixture |
| `apple` | `objaverse/apple/apple_10` | 1.0 | 0.12 kg |
| `banana` | `objaverse/banana/banana_1` | 1.0 | 0.1 kg |
| `lemon` | `objaverse/lemon/lemon_10` | 1.0 | 0.1 kg |
| `sponge` | `objaverse/sponge/sponge_1` | 1.0 | 0.025 kg |
| `soap` | `objaverse/bar_soap/bar_soap_3` | 1.0 | 0.09 kg |
| `drink_can` | `objaverse/can/can_0` | 1.0 | 0.1 kg |
| `spoon` | `objaverse/spoon/spoon_0` | 1.0 | 0.025 kg |
| `bowl` | `objaverse/bowl/bowl_0` | 1.0 | 0.16 kg |

The original instance directory is the identifier, not the sometimes different
`<mujoco model>` label inside its XML. `manifest.json` records these paths, source
revisions, placement, scaling and SHA-256 hashes for every vendored source file.
Only files referenced by the selected models were extracted; the complete
multi-gigabyte library is not required. This subset is about 13 MB and ships in
the cloud image. Loading it never accesses the network.

## Adaptations

Each `source.xml` is the unchanged upstream XML. OBJ meshes and PNG textures
are unchanged. `model.xml` is the adapted copy used here:

- Flatten the static wrapper so each movable object has a root free joint.
- Remove invisible task regions and the mug's liquid marker; they are not physics.
- Keep the textured visual mesh with collision disabled, and the original
  collision pieces (counts recorded in the manifest). Concave objects keep
  their decomposition rather than becoming one solid convex hull. The sponge
  is a single convex contact mesh.
- Put visual geometry in group 2 and collision geometry in hidden group 3,
  matching the robot renderer. Normalize contact friction/softness to the bench.
- Scale the bottle uniformly by 1.3. Scale its collision and visual geometry
  together. Place each asset above the bare tabletop using its source bounding region.
- Set illustrative empty-prop masses; preserve geometry-derived centre of mass
  and inertia proportions, with inertia scaled to the selected mass. These are
  not measured physical parameters.

The mug, bowl and tray retain open cavities. The scissors are a single rigid closed
object; no cutting or hinge actuation is modeled. Bottle contents/cap removal,
liquids and material deformation are not simulated. Fruit, soap and sponge
are rigid props; squashing, squeezing, dispensing and consumption are not modeled. The textured tray replaces
the primitive tray. The USB drive, screwdriver and notebook are authored practice props, not
RoboCasa models. All colored sorting toys and the green mat have been removed.

See [tutorial 14](../../tutorial/14-mujoco-sim.md) for example instructions and
validation scope. This is an asset integration, not a RoboCasa benchmark or
installation of RoboCasa/robosuite.
