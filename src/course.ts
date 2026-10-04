/** Obstacle course along +X. Walkable tops sit near y = 0 unless noted. */

export const LANE_HALF = 2.75;
export const FINISH_X = 19.4;
export const GEN_TIME = 8;
export const VALLEY_TOP = -1.55;

export const SPAWN_X0 = 0.4;
export const SPAWN_ROW = 0.82;

export interface Solid {
  shape: "box" | "cylinder";
  x: number;
  y: number;
  z: number;
  hx: number;
  hy: number;
  hz: number;
  qx: number;
  qy: number;
  qz: number;
  qw: number;
  color: number;
  roughness: number;
  metalness: number;
  friction: number;
  tag: string;
}

export interface Walk {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
  yAt: (x: number) => number;
}

const ID = { qx: 0, qy: 0, qz: 0, qw: 1 };

function quatX(rad: number): Pick<Solid, "qx" | "qy" | "qz" | "qw"> {
  return { qx: Math.sin(rad / 2), qy: 0, qz: 0, qw: Math.cos(rad / 2) };
}

function quatZ(rad: number): Pick<Solid, "qx" | "qy" | "qz" | "qw"> {
  return { qx: 0, qy: 0, qz: Math.sin(rad / 2), qw: Math.cos(rad / 2) };
}

function slab(
  x0: number,
  x1: number,
  top: number,
  thickness: number,
  hz: number,
  z: number,
  color: number,
  friction: number,
  tag: string
): { solid: Solid; walk?: Walk } {
  const hx = (x1 - x0) / 2;
  const hy = thickness / 2;
  return {
    solid: {
      shape: "box",
      x: (x0 + x1) / 2,
      y: top - hy,
      z,
      hx,
      hy,
      hz,
      ...ID,
      color,
      roughness: 0.86,
      metalness: 0.02,
      friction,
      tag
    },
    walk: { x0, x1, z0: z - hz, z1: z + hz, yAt: () => top }
  };
}

function ramp(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  hz: number,
  thickness: number,
  color: number,
  friction: number,
  tag: string
): { solid: Solid; walk?: Walk } {
  const run = x1 - x0;
  const rise = y1 - y0;
  const len = Math.hypot(run, rise);
  const angle = Math.atan2(rise, run);
  const nx = -Math.sin(angle);
  const ny = Math.cos(angle);
  const mx = (x0 + x1) / 2;
  const my = (y0 + y1) / 2;
  return {
    solid: {
      shape: "box",
      x: mx - nx * (thickness / 2),
      y: my - ny * (thickness / 2),
      z: 0,
      hx: len / 2,
      hy: thickness / 2,
      hz,
      ...quatZ(angle),
      color,
      roughness: 0.8,
      metalness: 0.04,
      friction,
      tag
    },
    walk: {
      x0,
      x1,
      z0: -hz,
      z1: hz,
      yAt: (x) => y0 + ((x - x0) / (x1 - x0)) * (y1 - y0)
    }
  };
}

const GRASS = 0x3d5340;
const DIRT = 0x6b5344;
const STONE = 0x8d8578;
const WALL = 0xd7a441;
const ARCH = 0xe7d7c4;
const LOG = 0x6a4632;

function build(): { solids: Solid[]; walks: Walk[] } {
  const solids: Solid[] = [];
  const walks: Walk[] = [];
  const add = (piece: { solid: Solid; walk?: Walk }): void => {
    solids.push(piece.solid);
    if (piece.walk) walks.push(piece.walk);
  };
  const lane = LANE_HALF;

  add(slab(-1.3, 7.15, 0, 0.45, lane, 0, GRASS, 1.15, "ground"));

  solids.push({
    shape: "box",
    x: 5.7,
    y: 0.1,
    z: 0,
    hx: 0.14,
    hy: 0.1,
    hz: lane - 0.12,
    ...ID,
    color: WALL,
    roughness: 0.55,
    metalness: 0.08,
    friction: 0.85,
    tag: "wall"
  });

  solids.push({
    shape: "cylinder",
    x: 6.85,
    y: 0.1,
    z: 0,
    hx: 0.1,
    hy: lane - 0.28,
    hz: 0,
    ...quatX(Math.PI / 2),
    color: LOG,
    roughness: 0.78,
    metalness: 0.05,
    friction: 0.9,
    tag: "log"
  });

  add(ramp(7.45, 0, 9.9, 0.62, lane - 0.04, 0.3, 0x4e6248, 1.2, "ramp"));
  add(slab(9.9, 11.05, 0.62, 0.72, lane - 0.04, 0, 0x567055, 1.15, "plateau"));
  add(ramp(11.05, 0.62, 13.5, 0, lane - 0.04, 0.3, 0x4e6248, 1.2, "ramp"));
  add(slab(13.35, 14.15, 0, 0.45, lane, 0, GRASS, 1.15, "ground"));

  const stoneLen = 0.9;
  const stoneHz = 1.7;
  const stones = [
    { x: 14.85, z: 0.45, top: 0.02 },
    { x: 16.15, z: -0.4, top: 0.05 },
    { x: 17.45, z: 0.28, top: 0.02 }
  ];
  for (const s of stones) {
    const hy = 0.16;
    solids.push({
      shape: "box",
      x: s.x,
      y: s.top - hy,
      z: s.z,
      hx: stoneLen / 2,
      hy,
      hz: stoneHz,
      ...ID,
      color: STONE,
      roughness: 0.92,
      metalness: 0.04,
      friction: 1.25,
      tag: "stone"
    });
    walks.push({
      x0: s.x - stoneLen / 2,
      x1: s.x + stoneLen / 2,
      z0: s.z - stoneHz,
      z1: s.z + stoneHz,
      yAt: () => s.top
    });
  }

  add(slab(18.05, 21.6, 0, 0.45, lane, 0, DIRT, 1.15, "ground"));

  const finish = FINISH_X;
  const pillarZ = lane - 0.45;
  for (const z of [-pillarZ, pillarZ]) {
    solids.push({
      shape: "box",
      x: finish,
      y: 1.15,
      z,
      hx: 0.16,
      hy: 1.15,
      hz: 0.18,
      ...ID,
      color: ARCH,
      roughness: 0.42,
      metalness: 0.18,
      friction: 0.6,
      tag: "arch"
    });
  }
  solids.push({
    shape: "box",
    x: finish,
    y: 2.42,
    z: 0,
    hx: 0.2,
    hy: 0.14,
    hz: pillarZ + 0.18,
    ...ID,
    color: ARCH,
    roughness: 0.38,
    metalness: 0.22,
    friction: 0.5,
    tag: "arch"
  });

  solids.push({
    shape: "box",
    x: 10,
    y: VALLEY_TOP - 0.2,
    z: 0,
    hx: 40,
    hy: 0.2,
    hz: 24,
    ...ID,
    color: 0x171c19,
    roughness: 1,
    metalness: 0,
    friction: 0.8,
    tag: "valley"
  });

  return { solids, walks };
}

const BUILT = build();
export const SOLIDS: readonly Solid[] = BUILT.solids;
export const WALKS: readonly Walk[] = BUILT.walks;

export function surfaceY(x: number, z: number): number | null {
  let best: number | null = null;
  for (const w of WALKS) {
    if (x < w.x0 || x > w.x1 || z < w.z0 || z > w.z1) continue;
    const y = w.yAt(x);
    if (best === null || y > best) best = y;
  }
  return best;
}

/** Two rows, spread across the lane, so every robot faces the same obstacles. */
const SPAWN_COLS = [-1.26, -0.42, 0.42, 1.26];

export function spawnSlots(count: number): Array<{ x: number; z: number }> {
  const slots: Array<{ x: number; z: number }> = [];
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / SPAWN_COLS.length);
    const col = i % SPAWN_COLS.length;
    slots.push({ x: SPAWN_X0 + row * 0.76, z: SPAWN_COLS[col] ?? 0 });
  }
  return slots;
}
