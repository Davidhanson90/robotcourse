# robotcourse

A dusk obstacle course in the browser. Each generation breeds a humanoid — head, torso, two arms, two legs — whose leg length, torso size, arm length, joint range, jump, and posture are genes. The first lineup is a wide random spread, not copies of one body. About a third start on all fours, hands planted as front legs. Legs cycle about twice as fast as before. Some barely hop and some jump hard. Flexibility and Jump sliders scale those genes for every robot without rewriting what breeding stores. Jump frequency, from almost never up to every grounded chance, decides how often they may take that hop. The bodies are rigid, the joints are motors, and gravity is Rapier's.

Fitness is distance along the course. Reaching the finish arch adds a large bonus. A faster finish scores higher only among robots that actually get there, so time never outranks a robot that went further.

Drag to orbit. Play runs generation after generation. Space pauses. S steps one generation.

[Live demo](https://davidhanson90.github.io/robotcourse/)
