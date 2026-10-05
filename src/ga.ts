import { buildMorph, crawlBlend, LIMIT, type Morph, type Pose } from "./body";
import { GEN_TIME } from "./course";

/**
 * Fitness, higher is better.
 *
 * progress = distance traveled from this robot's own spawn, divided by the
 * distance from that spawn to the finish arch, clamped to 0..1.
 * Crossing the arch upright sets progress to 1.
 *
 * score = PROGRESS_WEIGHT * progress
 *       + (finished ? FINISH_BONUS + TIME_BONUS_MAX * (1 - finishTime / generationTime) : 0)
 *
 * Distance is the whole score until someone finishes. The finish bonus is large
 * enough that any finisher outranks every robot still on the course. The time
 * term is smaller than that bonus and is added only after a finish, so it only
 * reorders robots that reached the arch (faster is better).
 */
export const PROGRESS_WEIGHT = 100;
export const FINISH_BONUS = 80;
export const TIME_BONUS_MAX = 20;

export function fitness(
  progress: number,
  finished: boolean,
  finishTime: number
): number {
  if (!finished) {
    const clamped = Math.max(0, Math.min(progress, 0.999));
    return clamped * PROGRESS_WEIGHT;
  }
  const t = Math.max(0, Math.min(1, finishTime / GEN_TIME));
  return PROGRESS_WEIGHT + FINISH_BONUS + TIME_BONUS_MAX * (1 - t);
}

export interface GeneSpec {
  name: string;
  min: number;
  max: number;
}

/**
 * Broader ranges so lineages can specialize: tall vs stocky, slow striders vs
 * fast steppers, stiff vs floppy, biped vs crawl, hoppers vs walkers.
 */
export const GENES: readonly GeneSpec[] = [
  { name: "freq", min: 1.6, max: 6.8 },
  { name: "hipAmp", min: 0.08, max: 1.95 },
  { name: "hipBias", min: -0.18, max: 0.58 },
  { name: "kneeAmp", min: 0.05, max: 2.35 },
  { name: "kneePhase", min: -1.35, max: 1.4 },
  { name: "kneeStance", min: 0, max: 0.55 },
  { name: "armAmp", min: 0.05, max: 1.45 },
  { name: "elbow", min: 0.12, max: 1.75 },
  { name: "lean", min: -0.28, max: 0.62 },
  { name: "turn", min: -0.28, max: 0.28 },
  { name: "toe", min: -0.35, max: 0.42 },
  { name: "split", min: 0, max: 1 },
  { name: "leg", min: 0.48, max: 1.58 },
  { name: "torsoH", min: 0.52, max: 1.48 },
  { name: "torsoW", min: 0.52, max: 1.55 },
  { name: "arm", min: 0.48, max: 1.6 },
  { name: "hipFlex", min: 0, max: 1 },
  { name: "kneeFlex", min: 0, max: 1 },
  // Negative is no jump. Only a positive amplitude can leave the ground on purpose.
  { name: "jump", min: -1, max: 1 },
  // 0 stands. 1 is a low all-fours crawl. Bred and mutated with the rest.
  { name: "quad", min: 0, max: 1 },
  // How hard arms counter-swing (biped) or drive the crawl (quad).
  { name: "armDrive", min: 0, max: 1 }
];

export type Decoded = Record<string, number>;

/** Positive couples the ankle so the sole stays level as the hip and knee move. */
export let ankleCouple = 1;

export function setAnkleCouple(value: number): void {
  ankleCouple = value;
}

/**
 * How much of each robot's hipFlex and kneeFlex genes is allowed when limits
 * are built. Breeding still stores the gene. 0 is stiff, 1 is the full gene.
 */
export let flexScale = 1;

/** How much of each robot's jump gene is allowed. 0 means nobody can jump. */
export let jumpScale = 1;

export function setFlexScale(value: number): void {
  flexScale = clamp01(value);
}

export function setJumpScale(value: number): void {
  jumpScale = clamp01(value);
}

/**
 * Share of grounded jump chances a robot may take. Genes and jump strength
 * are unchanged. 0 is almost never, 1 is every chance the grounded rule allows.
 */
export let jumpFrequency = 0.4;

export function setJumpFrequency(value: number): void {
  jumpFrequency = clamp01(value);
}

export const PROTO: Decoded = {
  freq: 3.8,
  hipAmp: 0.95,
  hipBias: 0.14,
  kneeAmp: 0.85,
  kneePhase: 0,
  kneeStance: 0.08,
  armAmp: 0.45,
  elbow: 0.55,
  lean: 0.08,
  turn: 0,
  toe: 0.12,
  split: 1,
  leg: 1,
  torsoH: 1,
  torsoW: 1,
  arm: 1,
  hipFlex: 0.55,
  kneeFlex: 0.42,
  jump: 0,
  quad: 0,
  armDrive: 0.45
};

export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gauss(rng: Rng): number {
  const u = Math.max(1e-8, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function clamp01(v: number): number {
  return clamp(v, 0, 1);
}

export function decode(genes: readonly number[]): Decoded {
  const proto = encode(PROTO);
  const out: Decoded = {};
  for (let i = 0; i < GENES.length; i++) {
    const spec = GENES[i];
    if (!spec) continue;
    const raw = genes[i];
    const u = clamp01(raw === undefined ? (proto[i] ?? 0.5) : raw);
    out[spec.name] = spec.min + u * (spec.max - spec.min);
  }
  return out;
}

/**
 * Sizes and joint limits implied by a decoded genome. Missing shape genes use PROTO.
 * hipFlex, kneeFlex, and jump are multiplied by the HUD scales here, not written back
 * into the genome.
 */
export function morphFromDecoded(decoded: Decoded): Morph {
  return buildMorph({
    leg: decoded.leg ?? PROTO.leg ?? 1,
    torsoH: decoded.torsoH ?? PROTO.torsoH ?? 1,
    torsoW: decoded.torsoW ?? PROTO.torsoW ?? 1,
    arm: decoded.arm ?? PROTO.arm ?? 1,
    hipFlex: (decoded.hipFlex ?? PROTO.hipFlex ?? 0.55) * flexScale,
    kneeFlex: (decoded.kneeFlex ?? PROTO.kneeFlex ?? 0.42) * flexScale,
    jump: (decoded.jump ?? 0) * jumpScale,
    quad: decoded.quad ?? PROTO.quad ?? 0
  });
}

export function encode(decoded: Decoded): number[] {
  return GENES.map((spec) => {
    // Missing quad stays bipedal. Other gaps use the middle of the range.
    const fallback =
      spec.name === "quad" ? 0 : spec.name === "armDrive" ? 0.45 : (spec.min + spec.max) / 2;
    const value = decoded[spec.name] ?? fallback;
    return clamp01((value - spec.min) / (spec.max - spec.min));
  });
}

export function randomGenome(rng: Rng): number[] {
  return GENES.map(() => rng());
}

function span(name: string, unit: number): number {
  const spec = GENES.find((gene) => gene.name === name);
  const lo = spec?.min ?? 0;
  const hi = spec?.max ?? 1;
  return lo + clamp01(unit) * (hi - lo);
}

function jitter(rng: Rng, amount = 0.12): number {
  return (rng() - 0.5) * 2 * amount;
}

function unitJitter(rng: Rng, center: number, spread = 0.1): number {
  return clamp01(center + jitter(rng, spread));
}

/** Named starting strategies. Noise is applied so no two clones look identical. */
type Archetype =
  | "tallBiped"
  | "stocky"
  | "strider"
  | "crawler"
  | "jumper"
  | "flexy"
  | "stiff"
  | "armAssist"
  | "galloper"
  | "leaner";

const ARCHETYPES: readonly Archetype[] = [
  "tallBiped",
  "stocky",
  "strider",
  "crawler",
  "jumper",
  "flexy",
  "stiff",
  "armAssist",
  "galloper",
  "leaner"
];

function archetypeGenome(kind: Archetype, rng: Rng): number[] {
  const decoded: Decoded = {};
  // Full-range baseline, then overwrite the strategy-defining genes.
  for (const spec of GENES) decoded[spec.name] = spec.min + rng() * (spec.max - spec.min);

  switch (kind) {
    case "tallBiped":
      decoded.leg = span("leg", 0.78 + rng() * 0.22);
      decoded.torsoH = span("torsoH", 0.7 + rng() * 0.28);
      decoded.torsoW = span("torsoW", 0.25 + rng() * 0.35);
      decoded.arm = span("arm", 0.35 + rng() * 0.4);
      decoded.quad = rng() * 0.12;
      decoded.jump = -0.55 + rng() * 0.5;
      decoded.freq = span("freq", 0.28 + rng() * 0.35);
      decoded.hipAmp = span("hipAmp", 0.45 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.4 + rng() * 0.4);
      decoded.armAmp = span("armAmp", 0.25 + rng() * 0.4);
      decoded.lean = span("lean", 0.35 + rng() * 0.35);
      decoded.hipFlex = unitJitter(rng, 0.5, 0.18);
      decoded.kneeFlex = unitJitter(rng, 0.4, 0.18);
      decoded.armDrive = unitJitter(rng, 0.4, 0.2);
      decoded.split = span("split", 0.7 + rng() * 0.3);
      break;
    case "stocky":
      decoded.leg = span("leg", 0.05 + rng() * 0.28);
      decoded.torsoH = span("torsoH", 0.15 + rng() * 0.35);
      decoded.torsoW = span("torsoW", 0.7 + rng() * 0.3);
      decoded.arm = span("arm", 0.2 + rng() * 0.35);
      decoded.quad = rng() * 0.15;
      decoded.jump = -0.7 + rng() * 0.45;
      decoded.freq = span("freq", 0.45 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.3 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.25 + rng() * 0.4);
      decoded.armAmp = span("armAmp", 0.15 + rng() * 0.35);
      decoded.lean = span("lean", 0.2 + rng() * 0.35);
      decoded.hipFlex = unitJitter(rng, 0.28, 0.15);
      decoded.kneeFlex = unitJitter(rng, 0.22, 0.15);
      decoded.armDrive = unitJitter(rng, 0.3, 0.2);
      break;
    case "strider":
      decoded.leg = span("leg", 0.72 + rng() * 0.28);
      decoded.torsoH = span("torsoH", 0.45 + rng() * 0.35);
      decoded.torsoW = span("torsoW", 0.3 + rng() * 0.35);
      decoded.arm = span("arm", 0.4 + rng() * 0.35);
      decoded.quad = rng() * 0.1;
      decoded.jump = -0.4 + rng() * 0.45;
      decoded.freq = span("freq", 0.12 + rng() * 0.28);
      decoded.hipAmp = span("hipAmp", 0.72 + rng() * 0.28);
      decoded.hipBias = span("hipBias", 0.45 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.65 + rng() * 0.35);
      decoded.armAmp = span("armAmp", 0.45 + rng() * 0.4);
      decoded.lean = span("lean", 0.45 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.62, 0.18);
      decoded.kneeFlex = unitJitter(rng, 0.55, 0.18);
      decoded.armDrive = unitJitter(rng, 0.55, 0.2);
      decoded.split = span("split", 0.85 + rng() * 0.15);
      break;
    case "crawler":
      decoded.leg = span("leg", 0.2 + rng() * 0.45);
      decoded.torsoH = span("torsoH", 0.1 + rng() * 0.35);
      decoded.torsoW = span("torsoW", 0.45 + rng() * 0.4);
      decoded.arm = span("arm", 0.65 + rng() * 0.35);
      decoded.quad = 0.78 + rng() * 0.22;
      decoded.jump = -0.85 + rng() * 0.55;
      decoded.freq = span("freq", 0.35 + rng() * 0.45);
      decoded.hipAmp = span("hipAmp", 0.4 + rng() * 0.45);
      decoded.kneeAmp = span("kneeAmp", 0.35 + rng() * 0.45);
      decoded.armAmp = span("armAmp", 0.55 + rng() * 0.45);
      decoded.elbow = span("elbow", 0.45 + rng() * 0.45);
      decoded.lean = span("lean", 0.15 + rng() * 0.35);
      decoded.hipFlex = Math.max(0.66, unitJitter(rng, 0.85, 0.12));
      decoded.kneeFlex = Math.max(0.6, unitJitter(rng, 0.8, 0.15));
      decoded.armDrive = unitJitter(rng, 0.85, 0.12);
      decoded.split = span("split", 0.55 + rng() * 0.45);
      break;
    case "jumper":
      decoded.leg = span("leg", 0.35 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.35 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.4 + rng() * 0.4);
      decoded.arm = span("arm", 0.3 + rng() * 0.4);
      decoded.quad = rng() * 0.18;
      decoded.jump = 0.55 + rng() * 0.45;
      decoded.freq = span("freq", 0.3 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.35 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.45 + rng() * 0.4);
      decoded.kneeStance = span("kneeStance", 0.35 + rng() * 0.5);
      decoded.armAmp = span("armAmp", 0.2 + rng() * 0.4);
      decoded.lean = span("lean", 0.25 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.7, 0.18);
      decoded.kneeFlex = unitJitter(rng, 0.72, 0.18);
      decoded.armDrive = unitJitter(rng, 0.4, 0.2);
      break;
    case "flexy":
      decoded.leg = span("leg", 0.35 + rng() * 0.45);
      decoded.torsoH = span("torsoH", 0.35 + rng() * 0.45);
      decoded.torsoW = span("torsoW", 0.3 + rng() * 0.45);
      decoded.arm = span("arm", 0.4 + rng() * 0.45);
      decoded.quad = rng() * 0.35;
      decoded.jump = -0.2 + rng() * 0.7;
      decoded.freq = span("freq", 0.4 + rng() * 0.45);
      decoded.hipAmp = span("hipAmp", 0.6 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.55 + rng() * 0.45);
      decoded.armAmp = span("armAmp", 0.5 + rng() * 0.5);
      decoded.lean = span("lean", 0.3 + rng() * 0.5);
      decoded.hipFlex = unitJitter(rng, 0.9, 0.1);
      decoded.kneeFlex = unitJitter(rng, 0.88, 0.12);
      decoded.armDrive = unitJitter(rng, 0.65, 0.2);
      break;
    case "stiff":
      decoded.leg = span("leg", 0.4 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.45 + rng() * 0.35);
      decoded.torsoW = span("torsoW", 0.4 + rng() * 0.4);
      decoded.arm = span("arm", 0.35 + rng() * 0.35);
      decoded.quad = rng() * 0.1;
      decoded.jump = -0.8 + rng() * 0.4;
      decoded.freq = span("freq", 0.25 + rng() * 0.35);
      decoded.hipAmp = span("hipAmp", 0.2 + rng() * 0.35);
      decoded.kneeAmp = span("kneeAmp", 0.15 + rng() * 0.3);
      decoded.armAmp = span("armAmp", 0.1 + rng() * 0.3);
      decoded.lean = span("lean", 0.35 + rng() * 0.3);
      decoded.hipFlex = unitJitter(rng, 0.12, 0.1);
      decoded.kneeFlex = unitJitter(rng, 0.1, 0.1);
      decoded.armDrive = unitJitter(rng, 0.2, 0.15);
      break;
    case "armAssist":
      decoded.leg = span("leg", 0.4 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.4 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.35 + rng() * 0.4);
      decoded.arm = span("arm", 0.7 + rng() * 0.3);
      decoded.quad = 0.15 + rng() * 0.35;
      decoded.jump = -0.35 + rng() * 0.55;
      decoded.freq = span("freq", 0.35 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.4 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.35 + rng() * 0.4);
      decoded.armAmp = span("armAmp", 0.7 + rng() * 0.3);
      decoded.elbow = span("elbow", 0.35 + rng() * 0.5);
      decoded.lean = span("lean", 0.4 + rng() * 0.45);
      decoded.hipFlex = unitJitter(rng, 0.55, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.5, 0.2);
      decoded.armDrive = unitJitter(rng, 0.9, 0.1);
      break;
    case "galloper":
      decoded.leg = span("leg", 0.45 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.4 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.35 + rng() * 0.4);
      decoded.arm = span("arm", 0.4 + rng() * 0.4);
      decoded.quad = rng() < 0.35 ? 0.75 + rng() * 0.25 : rng() * 0.2;
      decoded.jump = -0.15 + rng() * 0.7;
      decoded.freq = span("freq", 0.72 + rng() * 0.28);
      decoded.hipAmp = span("hipAmp", 0.55 + rng() * 0.45);
      decoded.kneeAmp = span("kneeAmp", 0.5 + rng() * 0.45);
      decoded.armAmp = span("armAmp", 0.4 + rng() * 0.45);
      decoded.lean = span("lean", 0.4 + rng() * 0.45);
      decoded.hipFlex = unitJitter(rng, 0.65, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.6, 0.2);
      decoded.armDrive = unitJitter(rng, 0.55, 0.25);
      decoded.split = span("split", 0.5 + rng() * 0.5);
      break;
    case "leaner":
      decoded.leg = span("leg", 0.45 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.55 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.25 + rng() * 0.4);
      decoded.arm = span("arm", 0.35 + rng() * 0.4);
      decoded.quad = rng() * 0.15;
      decoded.jump = -0.4 + rng() * 0.55;
      decoded.freq = span("freq", 0.35 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.5 + rng() * 0.4);
      decoded.hipBias = span("hipBias", 0.55 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.4 + rng() * 0.4);
      decoded.armAmp = span("armAmp", 0.35 + rng() * 0.4);
      decoded.lean = span("lean", 0.75 + rng() * 0.25);
      decoded.hipFlex = unitJitter(rng, 0.55, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.45, 0.2);
      decoded.armDrive = unitJitter(rng, 0.5, 0.2);
      break;
  }

  // Soft noise so archetypes stay recognizable but never identical.
  decoded.toe = span("toe", clamp01(0.4 + jitter(rng, 0.35)));
  decoded.turn = span("turn", clamp01(0.5 + jitter(rng, 0.25)));
  decoded.kneePhase = span("kneePhase", clamp01(0.5 + jitter(rng, 0.35)));
  if (decoded.kneeStance === undefined) decoded.kneeStance = span("kneeStance", 0.15 + rng() * 0.5);
  if (decoded.hipBias === undefined) decoded.hipBias = span("hipBias", 0.35 + rng() * 0.4);
  if (decoded.elbow === undefined) decoded.elbow = span("elbow", 0.3 + rng() * 0.5);
  if (decoded.split === undefined) decoded.split = span("split", 0.55 + rng() * 0.45);

  return encode(decoded);
}

/**
 * Generation 0 is a round-robin of clearly different strategies — tall bipeds,
 * stocky walkers, long-legged striders, crawlers, jumpers, stiff vs flexible,
 * arm-driven, gallopers, lean-heavy — with noise so each slot looks unique.
 */
export function initialPopulation(pop: number, rng: Rng): number[][] {
  const order = ARCHETYPES.slice();
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const swap = order[i] ?? order[0]!;
    order[i] = order[j] ?? order[0]!;
    order[j] = swap;
  }
  const out: number[][] = [];
  for (let i = 0; i < pop; i++) {
    const kind = order[i % order.length] ?? "tallBiped";
    out.push(archetypeGenome(kind, rng));
  }
  return out;
}

function tournament(ranked: readonly number[][], rng: Rng): number[] {
  let best = Math.floor(rng() * ranked.length);
  for (let k = 1; k < 3; k++) {
    const i = Math.floor(rng() * ranked.length);
    if (i < best) best = i;
  }
  return ranked[best] ?? ranked[0] ?? randomGenome(rng);
}

/** Body/posture genes. Milder mutation so winners' shapes breed true. */
const SHAPE_GENES = new Set([
  "leg",
  "torsoH",
  "torsoW",
  "arm",
  "hipFlex",
  "kneeFlex",
  "jump",
  "quad",
  "armDrive"
]);

function crossover(a: readonly number[], b: readonly number[], rng: Rng): number[] {
  const len = Math.max(a.length, b.length, GENES.length);
  const child: number[] = [];
  for (let i = 0; i < len; i++) {
    const gene = a[i] ?? b[i] ?? 0.5;
    const other = b[i] ?? gene;
    const name = GENES[i]?.name;
    const shape = name !== undefined && SHAPE_GENES.has(name);
    const roll = rng();
    // Shape: usually inherit one parent's proportions whole so morphs stay crisp.
    // Gait: blend more often.
    if (shape) {
      if (roll < 0.12) child.push((gene + other) * 0.5);
      else if (roll < 0.62) child.push(gene);
      else child.push(other);
    } else if (roll < 0.45) {
      child.push((gene + other) * 0.5);
    } else if (roll < 0.72) {
      child.push(gene);
    } else {
      child.push(other);
    }
  }
  return child;
}

function mutate(genes: readonly number[], rng: Rng): number[] {
  // Gait stays lively. Shape genes move less so fitness clearly reshapes the lineup.
  return genes.map((gene, i) => {
    const name = GENES[i]?.name;
    const shape = name !== undefined && SHAPE_GENES.has(name);
    const rate = shape ? 0.2 : 0.34;
    if (rng() > rate) return gene;
    const sigma = shape
      ? rng() < 0.12
        ? 0.14
        : rng() < 0.4
          ? 0.07
          : 0.03
      : rng() < 0.18
        ? 0.28
        : rng() < 0.45
          ? 0.12
          : 0.055;
    return clamp01(gene + gauss(rng) * sigma);
  });
}

/**
 * `parents` must already be sorted best-first. Returns a new population of genomes.
 * Elites are copied whole (every body and gait gene). Children blend two parents
 * then mutate — shape genes softer than gait — so successful morphs visibly take
 * over. A thin immigrant trickle (at most one) keeps a little diversity without
 * washing out winners.
 */
export function breed(parents: readonly number[][], pop: number, rng: Rng): number[][] {
  if (parents.length === 0) return initialPopulation(pop, rng);
  const eliteN = Math.max(2, Math.round(pop * 0.2));
  // One immigrant at most. Success, not random archetypes, drives shape change.
  const immigrants = pop >= 14 ? 1 : 0;
  const out: number[][] = [];
  for (let i = 0; i < eliteN && i < parents.length; i++) {
    out.push((parents[i] ?? []).slice());
  }
  while (out.length < pop - immigrants) {
    const a = tournament(parents, rng);
    const b = tournament(parents, rng);
    out.push(mutate(crossover(a, b, rng), rng));
  }
  while (out.length < pop) {
    const kind = ARCHETYPES[Math.floor(rng() * ARCHETYPES.length)] ?? "strider";
    out.push(archetypeGenome(kind, rng));
  }
  return out;
}

function legStep(
  cycle: number,
  hipAmp: number,
  hipBias: number,
  kneeAmp: number,
  stance: number
): { hip: number; knee: number } {
  const wrapped = ((cycle % 1) + 1) % 1;
  const swinging = wrapped < 0.5;
  const u = swinging ? wrapped / 0.5 : (wrapped - 0.5) / 0.5;
  if (swinging) {
    return {
      hip: hipBias - hipAmp * Math.cos(Math.PI * u),
      knee: stance + kneeAmp * Math.sin(Math.PI * u)
    };
  }
  return {
    hip: hipBias + hipAmp * Math.cos(Math.PI * u),
    knee: stance
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function writePose(
  decoded: Decoded,
  time: number,
  out: Pose,
  limits: Morph["limits"] = LIMIT,
  crouch = 0
): void {
  const freq = Math.max(0.25, decoded.freq ?? 3.8);
  const split = decoded.split ?? 1;
  const cycle = time * freq;
  const hipAmp = decoded.hipAmp ?? 0.95;
  const hipBias = decoded.hipBias ?? 0.12;
  const turn = decoded.turn ?? 0;
  const kneeAmp = decoded.kneeAmp ?? 0.85;
  const stance = decoded.kneeStance ?? 0.08;
  const armAmp = decoded.armAmp ?? 0.45;
  const elbow = decoded.elbow ?? 0.55;
  const toe = decoded.toe ?? 0;
  const armDrive = clamp01(decoded.armDrive ?? 0.45);
  const crawl = crawlBlend(decoded.quad ?? 0);
  const fold = Math.max(0, Math.min(1, crouch)) * (1 - crawl);
  const left = legStep(
    cycle,
    hipAmp * (1 - 0.78 * fold),
    hipBias * (1 - 0.7 * fold),
    kneeAmp * (1 - fold),
    stance
  );
  const right = legStep(
    cycle + 0.5 * split,
    hipAmp * (1 - 0.78 * fold),
    hipBias * (1 - 0.7 * fold),
    kneeAmp * (1 - fold),
    stance
  );
  const armPhase = cycle * Math.PI * 2;
  const ramp = Math.min(1, time / 1.05);
  const kneeCrouch = fold * 1.05 * ramp;
  const hipCrouch = fold * 0.32 * ramp;

  const phaseL = ((cycle % 1) + 1) % 1;
  const phaseR = ((cycle + 0.5 * split) % 1 + 1) % 1;
  const wave = (phase: number): number => Math.sin(phase * Math.PI * 2);
  // Stronger floors so legs and arms visibly animate even on modest genes.
  const hipSwing = Math.max(0.32, Math.min(0.72, hipAmp * 0.55));
  const kneeSwing = Math.max(0.18, Math.min(0.55, kneeAmp * 0.32));
  const armSwing = Math.max(0.28, Math.min(0.75, armAmp * (0.55 + 0.7 * armDrive)));
  const hipBase = clamp(1.05, limits.hip[0] + 0.04, limits.hip[1] - 0.05);
  const kneeBase = clamp(1.48, limits.knee[0], limits.knee[1] - 0.05);
  const shBase = clamp(1.05, limits.shoulder[0] + 0.04, limits.shoulder[1] - 0.08);
  const elBase = clamp(1.2, limits.elbow[0], limits.elbow[1] - 0.05);
  const mix = (stand: number, low: number, lo: number, hi: number): number =>
    clamp(lerp(stand, low, crawl), lo, hi);

  const bipedArm = armAmp * (0.55 + 0.9 * armDrive);
  const elbowPump = elbow * (0.7 + 0.55 * armDrive);

  out.spine = mix(
    (decoded.lean ?? 0) * (1 - 0.45 * fold) + fold * 0.05 * ramp,
    -0.32,
    limits.spine[0],
    limits.spine[1]
  );
  out.hipL = mix(
    left.hip + turn + hipCrouch,
    hipBase + hipSwing * wave(phaseL) + turn * 0.3,
    limits.hip[0],
    limits.hip[1]
  );
  out.hipR = mix(
    right.hip - turn + hipCrouch,
    hipBase + hipSwing * wave(phaseR) - turn * 0.3,
    limits.hip[0],
    limits.hip[1]
  );
  out.kneeL = mix(
    left.knee + kneeCrouch,
    kneeBase + kneeSwing * Math.max(0, wave(phaseL)),
    limits.knee[0],
    limits.knee[1]
  );
  out.kneeR = mix(
    right.knee + kneeCrouch,
    kneeBase + kneeSwing * Math.max(0, wave(phaseR)),
    limits.knee[0],
    limits.knee[1]
  );
  out.ankleL = mix(toe * Math.sin(armPhase), toe * 0.4 * wave(phaseL), limits.ankle[0], limits.ankle[1]);
  out.ankleR = mix(
    toe * Math.sin(armPhase + Math.PI * split),
    toe * 0.4 * wave(phaseR),
    limits.ankle[0],
    limits.ankle[1]
  );
  // Biped: counter-swing arms. Crawl: arms drive like front legs, scaled by armDrive.
  out.shoulderL = mix(
    -bipedArm * Math.sin(armPhase),
    shBase + armSwing * wave(phaseR),
    limits.shoulder[0],
    limits.shoulder[1]
  );
  out.shoulderR = mix(
    -bipedArm * Math.sin(armPhase + Math.PI * split),
    shBase + armSwing * wave(phaseL),
    limits.shoulder[0],
    limits.shoulder[1]
  );
  out.elbowL = mix(
    elbowPump + 0.18 * armDrive * Math.max(0, Math.sin(armPhase)),
    elBase + 0.22 * armDrive * Math.max(0, wave(phaseR)),
    limits.elbow[0],
    limits.elbow[1]
  );
  out.elbowR = mix(
    elbowPump + 0.18 * armDrive * Math.max(0, Math.sin(armPhase + Math.PI * split)),
    elBase + 0.22 * armDrive * Math.max(0, wave(phaseL)),
    limits.elbow[0],
    limits.elbow[1]
  );
}
