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

export const GENES: readonly GeneSpec[] = [
  { name: "freq", min: 3.2, max: 5.4 },
  { name: "hipAmp", min: 0.05, max: 1.55 },
  { name: "hipBias", min: -0.05, max: 0.48 },
  { name: "kneeAmp", min: 0, max: 1.9 },
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
  { name: "jump", min: -1, max: 1 },
  // 0 stands. 1 is a low all-fours crawl. Bred and mutated with the rest.
  { name: "quad", min: 0, max: 1 }
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
  freq: 4.4,
  hipAmp: 0.82,
  hipBias: 0.16,
  kneeAmp: 0.70,
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
  jump: 0,
  quad: 0
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
    const fallback = spec.name === "quad" ? 0 : (spec.min + spec.max) / 2;
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

/**
 * Generation 0 is not a cluster around the prototype. Body proportions are
 * spread across the lane, about a third spawn as all-fours, and jump amplitude
 * runs from barely off the ground to a full hop.
 */
function starterGenome(rng: Rng, quadruped: boolean, index: number, pop: number, jumpBand: number): number[] {
  const decoded: Decoded = {};
  for (const spec of GENES) decoded[spec.name] = spec.min + rng() * (spec.max - spec.min);
  const bins = Math.max(pop, 1);
  const spread = (salt: number): number => {
    const bin = (index * 5 + salt + Math.floor(rng() * bins)) % bins;
    return (bin + 0.15 + rng() * 0.7) / bins;
  };
  decoded.leg = span("leg", spread(0));
  decoded.torsoH = span("torsoH", spread(1));
  decoded.torsoW = span("torsoW", spread(2));
  decoded.arm = span("arm", spread(3));
  decoded.hipFlex = spread(4);
  decoded.kneeFlex = spread(5);
  decoded.hipAmp = span("hipAmp", 0.28 + 0.72 * rng());
  decoded.kneeAmp = span("kneeAmp", 0.25 + 0.75 * rng());
  decoded.armAmp = span("armAmp", 0.15 + 0.85 * rng());
  decoded.elbow = span("elbow", rng());
  decoded.lean = span("lean", rng());
  decoded.split = span("split", 0.35 + 0.65 * rng());
  // Whole freq range is already about twice the old gait. Keep gen 0 in the quicker part.
  decoded.freq = span("freq", 0.35 + 0.65 * rng());
  decoded.quad = quadruped ? 0.78 + rng() * 0.22 : rng() * 0.18;
  if (quadruped) {
    decoded.hipFlex = Math.max(decoded.hipFlex ?? 0, 0.66 + rng() * 0.34);
    decoded.kneeFlex = Math.max(decoded.kneeFlex ?? 0, 0.6 + rng() * 0.4);
    decoded.arm = Math.max(decoded.arm ?? 1, span("arm", 0.55 + 0.45 * rng()));
  }
  if (jumpBand === 0) decoded.jump = -0.45 + rng() * 0.55;
  else if (jumpBand === 1) decoded.jump = 0.12 + rng() * 0.38;
  else decoded.jump = 0.62 + rng() * 0.38;
  return encode(decoded);
}

export function initialPopulation(pop: number, rng: Rng): number[][] {
  const quadCount = Math.max(1, Math.round(pop / 3));
  const order = Array.from({ length: pop }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const swap = order[i] ?? i;
    order[i] = order[j] ?? j;
    order[j] = swap;
  }
  const quads = new Set(order.slice(0, quadCount));
  const bands = Array.from({ length: pop }, (_, i) => i % 3);
  for (let i = bands.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const swap = bands[i] ?? 0;
    bands[i] = bands[j] ?? 0;
    bands[j] = swap;
  }
  const out: number[][] = [];
  for (let i = 0; i < pop; i++) out.push(starterGenome(rng, quads.has(i), i, pop, bands[i] ?? 0));
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
  const freq = Math.max(0.2, decoded.freq ?? 4.4);
  const split = decoded.split ?? 1;
  const cycle = time * freq;
  const hipAmp = decoded.hipAmp ?? 0.82;
  const hipBias = decoded.hipBias ?? 0.1;
  const turn = decoded.turn ?? 0;
  const kneeAmp = decoded.kneeAmp ?? 0.70;
  const stance = decoded.kneeStance ?? 0.08;
  const armAmp = decoded.armAmp ?? 0.3;
  const elbow = decoded.elbow ?? 0.7;
  const toe = decoded.toe ?? 0;
  const crawl = crawlBlend(decoded.quad ?? 0);
  const fold = Math.max(0, Math.min(1, crouch)) * (1 - crawl);
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

  const phaseL = ((cycle % 1) + 1) % 1;
  const phaseR = ((cycle + 0.5 * split) % 1 + 1) % 1;
  const wave = (phase: number): number => Math.sin(phase * Math.PI * 2);
  const hipSwing = Math.max(0.22, Math.min(0.5, hipAmp * 0.42));
  const kneeSwing = Math.max(0.1, Math.min(0.38, kneeAmp * 0.2));
  const armSwing = Math.max(0.16, Math.min(0.4, armAmp * 0.65));
  const hipBase = clamp(1.05, limits.hip[0] + 0.04, limits.hip[1] - 0.05);
  const kneeBase = clamp(1.48, limits.knee[0], limits.knee[1] - 0.05);
  const shBase = clamp(1.0, limits.shoulder[0] + 0.04, limits.shoulder[1] - 0.08);
  const elBase = clamp(1.15, limits.elbow[0], limits.elbow[1] - 0.05);
  const mix = (stand: number, low: number, lo: number, hi: number): number =>
    clamp(lerp(stand, low, crawl), lo, hi);

  out.spine = mix(
    (decoded.lean ?? 0) * (1 - 0.5 * fold) + fold * 0.05 * ramp,
    -0.28,
    limits.spine[0],
    limits.spine[1]
  );
  out.hipL = mix(left.hip + turn + hipCrouch, hipBase + hipSwing * wave(phaseL) + turn * 0.3, limits.hip[0], limits.hip[1]);
  out.hipR = mix(right.hip - turn + hipCrouch, hipBase + hipSwing * wave(phaseR) - turn * 0.3, limits.hip[0], limits.hip[1]);
  out.kneeL = mix(left.knee + kneeCrouch, kneeBase + kneeSwing * Math.max(0, wave(phaseL)), limits.knee[0], limits.knee[1]);
  out.kneeR = mix(right.knee + kneeCrouch, kneeBase + kneeSwing * Math.max(0, wave(phaseR)), limits.knee[0], limits.knee[1]);
  out.ankleL = mix(toe * Math.sin(armPhase), toe * 0.35 * wave(phaseL), limits.ankle[0], limits.ankle[1]);
  out.ankleR = mix(toe * Math.sin(armPhase + Math.PI * split), toe * 0.35 * wave(phaseR), limits.ankle[0], limits.ankle[1]);
  out.shoulderL = mix(-armAmp * Math.sin(armPhase), shBase + armSwing * wave(phaseR), limits.shoulder[0], limits.shoulder[1]);
  out.shoulderR = mix(-armAmp * Math.sin(armPhase + Math.PI * split), shBase + armSwing * wave(phaseL), limits.shoulder[0], limits.shoulder[1]);
  out.elbowL = mix(elbow, elBase, limits.elbow[0], limits.elbow[1]);
  out.elbowR = mix(elbow, elBase, limits.elbow[0], limits.elbow[1]);
}
