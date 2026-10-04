# robotcourse

A dusk obstacle course in the browser. Each generation breeds a humanoid — head, torso, two arms, two legs — whose leg length, torso size, arm length, joint range, and jump are genes. Stiff robots stay upright. Flexible ones crouch and crawl. A few hop. Flexibility and Jump sliders scale those genes for every robot without rewriting what breeding stores. Jump frequency, from almost never up to every grounded chance, decides how often they may take that hop. The bodies are rigid, the joints are motors, and gravity is Rapier's.

Fitness is distance along the course. Reaching the finish arch adds a large bonus. A faster finish scores higher only among robots that actually get there, so time never outranks a robot that went further.

Drag to orbit. Play runs generation after generation. Space pauses. S steps one generation.

[Live demo](https://davidhanson90.github.io/robotcourse/)
