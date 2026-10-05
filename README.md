# robotcourse

A dusk obstacle course in the browser. Each generation breeds robots of **any shape** — bipeds, quadrupeds, spiders, snakes, blobs, wheeled articulated chassis, centipedes — whose topology, limb length, torso size, joint range, jump, posture, and drive are genes. Mesh and Rapier colliders are rebuilt from each robot’s genome every generation.

Generation 0 seeds clearly different body plans and strategies. Breeding keeps elites whole, blends parents (body and form genes usually copy from one parent so morphs stay crisp), mutates next generations harder, and injects at most one fresh archetype so successful bodies drive the next lineup. When winners outrun or out-jump losers, speed and hop genes (freq, stride/hipAmp, kneeAmp, jump, arm drive, toe) get a directed upward push in offspring — not only random drift. Flexibility and Jump sliders scale those genes for every robot without rewriting what breeding stores. Jump frequency, from almost never up to every grounded chance, decides how often they may hop — any contact part counts (feet, wheels, lobes, snake segments), not only biped soles.

Fitness is distance along the course. Reaching the finish arch adds a large bonus. A faster finish scores higher only among robots that actually get there, so time never outranks a robot that went further.

Drag to orbit. Play runs generation after generation. Space pauses. S steps one generation.

[Live demo](https://davidhanson90.github.io/robotcourse/)
