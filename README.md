# robotcourse

A dusk obstacle course in the browser. Each generation breeds a humanoid — head, torso, two arms, two legs — whose leg length, torso size, arm length, joint range, jump, posture, and arm drive are genes. Generation 0 is not one body with noise: it seeds clearly different strategies (tall bipeds, stocky walkers, long-legged striders, crawlers, jumpers, stiff vs flexible, arm-assisted, gallopers, lean-heavy). Breeding keeps elites, blends parents, mutates hard enough that lineages can specialize, and injects fresh archetypes so the pool does not collapse into one flop. Flexibility and Jump sliders scale those genes for every robot without rewriting what breeding stores. Jump frequency, from almost never up to every grounded chance, decides how often they may take that hop. The bodies are rigid, the joints are motors, and gravity is Rapier's.

Fitness is distance along the course. Reaching the finish arch adds a large bonus. A faster finish scores higher only among robots that actually get there, so time never outranks a robot that went further.

Drag to orbit. Play runs generation after generation. Space pauses. S steps one generation.

[Live demo](https://davidhanson90.github.io/robotcourse/)
