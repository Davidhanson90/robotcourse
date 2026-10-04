import { buildMorph, LIMIT, type Morph, type Pose } from "./body";
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

export const GENES: readonly GeneSpec[] = [
  { name: "freq", min: 0.7, max: 2.7 },
  { name: "hipAmp", min: 0.05, max: 0.95 },
  { name: "hipBias", min: -0.05, max: 0.48 },
  { name: "kneeAmp", min: 0, max: 1.2 },
  { name: "kneePhase", min: -1.1, max: 1.25 },
  { name: "kneeStance", min: 0.02, max: 0.38 },
  { name: "armAmp", min: 0.05, max: 0.85 },
  { name: "elbow", min: 0.25, max: 1.45 },
  { name: "lean", min: -0.12, max: 0.4 },
  { name: "turn", min: -0.2, max: 0.2 },
  { name: "toe", min: -0.22, max: 0.32 },
  { name: "split", min: 0, max: 1 },
  { name: "leg", min: 0.62, max: 1.38 },
  { name: "torsoH", min: 0.68, max: 1.32 },
  { name: "torsoW", min: 0.7, max: 1.3 },
  { name: "arm", min: 0.6, max: 1.4 },
  { name: "hipFlex", min: 0, max: 1 },
  { name: "kneeFlex", min: 0, max: 1 },
  // Negative is no jump. Only a positive amplitude can leave the ground on purpose.
  { name: "jump", min: -1, max: 1 }
];

const SHAPE_GENES = new Set(["leg", "torsoH", "torsoW", "arm", "hipFlex", "kneeFlex", "jump"]);

export type Decoded = Record<string, number>;

/** Positive couples the ankle so the sole stays level as the hip and knee move. */
export let ankleCouple = 1;

export function setAnkleCouple(value: number): void {
  ankleCouple = value;
}

export const PROTO: Decoded = {
  freq: 2.2,
  hipAmp: 0.48,
  hipBias: 0.16,
  kneeAmp: 0.02,
  kneePhase: 0,
  kneeStance: 0.06,
  armAmp: 0.28,
  elbow: 0.5,
  lean: 0.06,
  turn: 0,
  toe: 0.12,
  split: 1,
  leg: 1,
  torsoH: 1,
  torsoW: 1,
  arm: 1,
  hipFlex: 0.55,
  kneeFlex: 0.42,
  jump: 0
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

/** Sizes and joint limits implied by a decoded genome. Missing shape genes use PROTO. */
export function morphFromDecoded(decoded: Decoded): Morph {
  return buildMorph({
    leg: decoded.leg ?? PROTO.leg ?? 1,
    torsoH: decoded.torsoH ?? PROTO.torsoH ?? 1,
    torsoW: decoded.torsoW ?? PROTO.torsoW ?? 1,
    arm: decoded.arm ?? PROTO.arm ?? 1,
    hipFlex: decoded.hipFlex ?? PROTO.hipFlex ?? 0.55,
    kneeFlex: decoded.kneeFlex ?? PROTO.kneeFlex ?? 0.42,
    jump: decoded.jump ?? 0
  });
}

export function encode(decoded: Decoded): number[] {
  return GENES.map((spec) => {
    const value = decoded[spec.name] ?? (spec.min + spec.max) / 2;
    return clamp01((value - spec.min) / (spec.max - spec.min));
  });
}

export function randomGenome(rng: Rng): number[] {
  return GENES.map(() => rng());
}

function jitter(genes: readonly number[], rng: Rng, sigma: number): number[] {
  return genes.map((g) => clamp01(g + gauss(rng) * sigma));
}

/** Spread shape, flexibility, and jump so a generation is not one shared body. */
function diversifyShape(genes: readonly number[], rng: Rng): number[] {
  return genes.map((gene, i) => {
    const name = GENES[i]?.name;
    if (!name || !SHAPE_GENES.has(name)) return gene;
    if (name === "jump") {
      // Most stay grounded. The rest get a real upward amplitude.
      return rng() < 0.62 ? clamp01(0.5 + gauss(rng) * 0.04) : clamp01(0.5 + 0.5 * rng());
    }
    return rng();
  });
}

export function initialPopulation(pop: number, rng: Rng): number[][] {
  const proto = encode(PROTO);
  const out: number[][] = [];
  for (let i = 0; i < pop; i++) {
    if (i < 3) out.push(jitter(proto, rng, 0.02));
    else if (i < Math.floor(pop * 0.75)) out.push(diversifyShape(jitter(proto, rng, 0.07), rng));
    else out.push(randomGenome(rng));
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

function crossover(a: readonly number[], b: readonly number[], rng: Rng): number[] {
  return a.map((gene, i) => {
    const roll = rng();
    const other = b[i] ?? gene;
    if (roll < 0.55) return (gene + other) * 0.5;
    return roll < 0.78 ? gene : other;
  });
}

function mutate(genes: readonly number[], rng: Rng): number[] {
  return genes.map((gene) => {
    if (rng() > 0.28) return gene;
    const sigma = rng() < 0.14 ? 0.2 : 0.065;
    return clamp01(gene + gauss(rng) * sigma);
  });
}

/**
 * `parents` must already be sorted best-first. Returns a new population of genomes.
 * Elites are copied in full, including leg, torso, arm, joint-range, and jump genes.
 * Children mix those genes by crossover and mutation. Nothing is reset to a default body.
 */
export function breed(parents: readonly number[][], pop: number, rng: Rng): number[][] {
  if (parents.length === 0) return initialPopulation(pop, rng);
  const eliteN = Math.max(2, Math.round(pop * 0.125));
  const immigrants = Math.max(1, Math.round(pop * 0.08));
  const out: number[][] = [];
  for (let i = 0; i < eliteN && i < parents.length; i++) {
    out.push((parents[i] ?? []).slice());
  }
  while (out.length < pop - immigrants) {
    const a = tournament(parents, rng);
    const b = tournament(parents, rng);
    out.push(mutate(crossover(a, b, rng), rng));
  }
  while (out.length < pop) out.push(randomGenome(rng));
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

export function writePose(
  decoded: Decoded,
  time: number,
  out: Pose,
  limits: Morph["limits"] = LIMIT,
  crouch = 0
): void {
  const freq = Math.max(0.2, decoded.freq ?? 1.4);
  const split = decoded.split ?? 1;
  const cycle = time * freq;
  const hipAmp = decoded.hipAmp ?? 0.5;
  const hipBias = decoded.hipBias ?? 0.1;
  const turn = decoded.turn ?? 0;
  const kneeAmp = decoded.kneeAmp ?? 0.6;
  const stance = decoded.kneeStance ?? 0.08;
  const armAmp = decoded.armAmp ?? 0.3;
  const elbow = decoded.elbow ?? 0.7;
  const toe = decoded.toe ?? 0;
  const fold = Math.max(0, Math.min(1, crouch));
  const left = legStep(
    cycle,
    hipAmp * (1 - 0.82 * fold),
    hipBias * (1 - 0.75 * fold),
    kneeAmp * (1 - fold),
    stance
  );
  const right = legStep(
    cycle + 0.5 * split,
    hipAmp * (1 - 0.82 * fold),
    hipBias * (1 - 0.75 * fold),
    kneeAmp * (1 - fold),
    stance
  );
  const armPhase = cycle * Math.PI * 2;
  const ramp = Math.min(1, time / 1.4);
  const kneeCrouch = fold * 1.05 * ramp;
  const hipCrouch = fold * 0.32 * ramp;
  const creep = 0;

  out.spine = clamp((decoded.lean ?? 0) * (1 - 0.5 * fold) + fold * 0.05 * ramp, limits.spine[0], limits.spine[1]);
  out.hipL = clamp(left.hip + turn + hipCrouch, limits.hip[0], limits.hip[1]);
  out.hipR = clamp(right.hip - turn + hipCrouch, limits.hip[0], limits.hip[1]);
  out.kneeL = clamp(left.knee + kneeCrouch + creep, limits.knee[0], limits.knee[1]);
  out.kneeR = clamp(right.knee + kneeCrouch - creep, limits.knee[0], limits.knee[1]);
  out.ankleL = clamp(toe * Math.sin(armPhase), limits.ankle[0], limits.ankle[1]);
  out.ankleR = clamp(toe * Math.sin(armPhase + Math.PI * split), limits.ankle[0], limits.ankle[1]);
  out.shoulderL = clamp(-armAmp * Math.sin(armPhase), limits.shoulder[0], limits.shoulder[1]);
  out.shoulderR = clamp(-armAmp * Math.sin(armPhase + Math.PI * split), limits.shoulder[0], limits.shoulder[1]);
  out.elbowL = clamp(elbow, limits.elbow[0], limits.elbow[1]);
  out.elbowR = clamp(elbow, limits.elbow[0], limits.elbow[1]);
}
