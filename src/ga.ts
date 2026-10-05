import {
  applyExpression,
  buildMorph,
  crawlBlend,
  formCenter,
  formFromGene,
  type FormKind,
  type Morph,
  type Pose
} from "./body";
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
 * Broader ranges so lineages can specialize across topologies: bipeds, quads,
 * spiders, snakes, blobs, wheeled chassis, centipedes — plus gait and size.
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
  // 0 stands. 1 is a low all-fours crawl on bipeds.
  { name: "quad", min: 0, max: 1 },
  // How hard arms / front limbs drive.
  { name: "armDrive", min: 0, max: 1 },
  // Topology. Inherited whole so shapes stay crisp.
  { name: "form", min: 0, max: 1 },
  // Segment / lobe / wheel / leg-count nuance within a form.
  { name: "segments", min: 0, max: 1 }
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
  armDrive: 0.45,
  form: formCenter("biped"),
  segments: 0.45
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
 * Sizes, topology, and joint limits implied by a decoded genome.
 * hipFlex, kneeFlex, and jump are multiplied by the HUD scales here.
 */
export function morphFromDecoded(decoded: Decoded, spawnX = 0, spawnZ = 0): Morph {
  return buildMorph({
    form: decoded.form ?? PROTO.form ?? formCenter("biped"),
    leg: decoded.leg ?? PROTO.leg ?? 1,
    torsoH: decoded.torsoH ?? PROTO.torsoH ?? 1,
    torsoW: decoded.torsoW ?? PROTO.torsoW ?? 1,
    arm: decoded.arm ?? PROTO.arm ?? 1,
    hipFlex: (decoded.hipFlex ?? PROTO.hipFlex ?? 0.55) * flexScale,
    kneeFlex: (decoded.kneeFlex ?? PROTO.kneeFlex ?? 0.42) * flexScale,
    jump: (decoded.jump ?? 0) * jumpScale,
    quad: decoded.quad ?? PROTO.quad ?? 0,
    segments: decoded.segments ?? PROTO.segments ?? 0.45,
    spawnX,
    spawnZ
  });
}

/** Refresh joint limits / jump on a live morph after slider changes. */
export function refreshMorphExpression(morph: Morph, decoded: Decoded): void {
  applyExpression(
    morph,
    (decoded.hipFlex ?? 0.55) * flexScale,
    (decoded.kneeFlex ?? 0.42) * flexScale,
    (decoded.jump ?? 0) * jumpScale
  );
}

export function encode(decoded: Decoded): number[] {
  return GENES.map((spec) => {
    const fallback =
      spec.name === "quad"
        ? 0
        : spec.name === "armDrive"
          ? 0.45
          : spec.name === "form"
            ? formCenter("biped")
            : spec.name === "segments"
              ? 0.45
              : (spec.min + spec.max) / 2;
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

/** Named starting strategies across topologies. */
type Archetype =
  | "tallBiped"
  | "stocky"
  | "strider"
  | "crawler"
  | "jumper"
  | "quadWalker"
  | "spider"
  | "snake"
  | "blob"
  | "wheeler"
  | "centipede"
  | "flexy"
  | "galloper"
  | "leaner";

const ARCHETYPES: readonly Archetype[] = [
  "tallBiped",
  "stocky",
  "strider",
  "crawler",
  "jumper",
  "quadWalker",
  "spider",
  "snake",
  "blob",
  "wheeler",
  "centipede",
  "flexy",
  "galloper",
  "leaner"
];

function formSpan(kind: FormKind, rng: Rng): number {
  const i = Math.max(0, ["biped", "quad", "spider", "snake", "blob", "wheeler", "centipede"].indexOf(kind));
  const lo = i / 7;
  const hi = (i + 1) / 7;
  return lo + rng() * (hi - lo) * 0.92 + 0.04 * (hi - lo);
}

function archetypeGenome(kind: Archetype, rng: Rng): number[] {
  const decoded: Decoded = {};
  for (const spec of GENES) decoded[spec.name] = spec.min + rng() * (spec.max - spec.min);

  switch (kind) {
    case "tallBiped":
      decoded.form = formSpan("biped", rng);
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
      decoded.segments = unitJitter(rng, 0.4, 0.2);
      break;
    case "stocky":
      decoded.form = formSpan("biped", rng);
      decoded.leg = span("leg", 0.05 + rng() * 0.28);
      decoded.torsoH = span("torsoH", 0.15 + rng() * 0.35);
      decoded.torsoW = span("torsoW", 0.7 + rng() * 0.3);
      decoded.arm = span("arm", 0.2 + rng() * 0.35);
      decoded.quad = rng() * 0.15;
      decoded.jump = -0.7 + rng() * 0.45;
      decoded.freq = span("freq", 0.45 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.3 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.25 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.28, 0.15);
      decoded.kneeFlex = unitJitter(rng, 0.22, 0.15);
      decoded.armDrive = unitJitter(rng, 0.3, 0.2);
      decoded.segments = unitJitter(rng, 0.35, 0.2);
      break;
    case "strider":
      decoded.form = formSpan("biped", rng);
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
      decoded.lean = span("lean", 0.45 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.62, 0.18);
      decoded.kneeFlex = unitJitter(rng, 0.55, 0.18);
      decoded.armDrive = unitJitter(rng, 0.55, 0.2);
      decoded.split = span("split", 0.85 + rng() * 0.15);
      decoded.segments = unitJitter(rng, 0.4, 0.2);
      break;
    case "crawler":
      decoded.form = formSpan("biped", rng);
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
      decoded.hipFlex = Math.max(0.66, unitJitter(rng, 0.85, 0.12));
      decoded.kneeFlex = Math.max(0.6, unitJitter(rng, 0.8, 0.15));
      decoded.armDrive = unitJitter(rng, 0.85, 0.12);
      decoded.segments = unitJitter(rng, 0.45, 0.2);
      break;
    case "jumper":
      decoded.form = formSpan(rng() < 0.55 ? "biped" : "quad", rng);
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
      decoded.hipFlex = unitJitter(rng, 0.7, 0.18);
      decoded.kneeFlex = unitJitter(rng, 0.72, 0.18);
      decoded.armDrive = unitJitter(rng, 0.4, 0.2);
      decoded.segments = unitJitter(rng, 0.4, 0.25);
      break;
    case "quadWalker":
      decoded.form = formSpan("quad", rng);
      decoded.leg = span("leg", 0.35 + rng() * 0.45);
      decoded.torsoH = span("torsoH", 0.35 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.4 + rng() * 0.45);
      decoded.arm = span("arm", 0.3 + rng() * 0.35);
      decoded.quad = 0.2 + rng() * 0.4;
      decoded.jump = -0.35 + rng() * 0.6;
      decoded.freq = span("freq", 0.4 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.5 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.45 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.6, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.55, 0.2);
      decoded.armDrive = unitJitter(rng, 0.5, 0.2);
      decoded.split = span("split", 0.6 + rng() * 0.4);
      decoded.segments = unitJitter(rng, 0.4, 0.2);
      break;
    case "spider":
      decoded.form = formSpan("spider", rng);
      decoded.leg = span("leg", 0.4 + rng() * 0.45);
      decoded.torsoH = span("torsoH", 0.25 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.45 + rng() * 0.45);
      decoded.arm = span("arm", 0.35 + rng() * 0.4);
      decoded.quad = rng() * 0.3;
      decoded.jump = -0.2 + rng() * 0.7;
      decoded.freq = span("freq", 0.45 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.45 + rng() * 0.45);
      decoded.kneeAmp = span("kneeAmp", 0.4 + rng() * 0.45);
      decoded.hipFlex = unitJitter(rng, 0.7, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.65, 0.2);
      decoded.armDrive = unitJitter(rng, 0.55, 0.25);
      decoded.segments = rng() < 0.5 ? unitJitter(rng, 0.25, 0.15) : unitJitter(rng, 0.75, 0.15);
      break;
    case "snake":
      decoded.form = formSpan("snake", rng);
      decoded.leg = span("leg", 0.2 + rng() * 0.35);
      decoded.torsoH = span("torsoH", 0.35 + rng() * 0.45);
      decoded.torsoW = span("torsoW", 0.25 + rng() * 0.4);
      decoded.arm = span("arm", 0.3 + rng() * 0.35);
      decoded.quad = rng() * 0.2;
      decoded.jump = -0.7 + rng() * 0.5;
      decoded.freq = span("freq", 0.5 + rng() * 0.45);
      decoded.hipAmp = span("hipAmp", 0.55 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.3 + rng() * 0.4);
      decoded.lean = span("lean", 0.3 + rng() * 0.45);
      decoded.hipFlex = unitJitter(rng, 0.75, 0.15);
      decoded.kneeFlex = unitJitter(rng, 0.5, 0.2);
      decoded.armDrive = unitJitter(rng, 0.4, 0.2);
      decoded.segments = unitJitter(rng, 0.55, 0.3);
      break;
    case "blob":
      decoded.form = formSpan("blob", rng);
      decoded.leg = span("leg", 0.25 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.4 + rng() * 0.45);
      decoded.torsoW = span("torsoW", 0.55 + rng() * 0.4);
      decoded.arm = span("arm", 0.45 + rng() * 0.45);
      decoded.quad = rng() * 0.25;
      decoded.jump = -0.15 + rng() * 0.8;
      decoded.freq = span("freq", 0.35 + rng() * 0.45);
      decoded.hipAmp = span("hipAmp", 0.4 + rng() * 0.45);
      decoded.kneeAmp = span("kneeAmp", 0.35 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.65, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.55, 0.2);
      decoded.armDrive = unitJitter(rng, 0.5, 0.25);
      decoded.segments = unitJitter(rng, 0.5, 0.35);
      break;
    case "wheeler":
      decoded.form = formSpan("wheeler", rng);
      decoded.leg = span("leg", 0.3 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.35 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.4 + rng() * 0.45);
      decoded.arm = span("arm", 0.3 + rng() * 0.35);
      decoded.quad = rng() * 0.15;
      decoded.jump = -0.55 + rng() * 0.55;
      decoded.freq = span("freq", 0.55 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.35 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.25 + rng() * 0.35);
      decoded.toe = span("toe", 0.55 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.45, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.4, 0.2);
      decoded.armDrive = unitJitter(rng, 0.35, 0.2);
      decoded.segments = rng() < 0.33 ? unitJitter(rng, 0.2, 0.12) : rng() < 0.66 ? unitJitter(rng, 0.55, 0.12) : unitJitter(rng, 0.85, 0.1);
      break;
    case "centipede":
      decoded.form = formSpan("centipede", rng);
      decoded.leg = span("leg", 0.25 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.3 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.3 + rng() * 0.4);
      decoded.arm = span("arm", 0.3 + rng() * 0.35);
      decoded.quad = rng() * 0.25;
      decoded.jump = -0.45 + rng() * 0.55;
      decoded.freq = span("freq", 0.55 + rng() * 0.4);
      decoded.hipAmp = span("hipAmp", 0.45 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.4 + rng() * 0.4);
      decoded.hipFlex = unitJitter(rng, 0.65, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.6, 0.2);
      decoded.armDrive = unitJitter(rng, 0.45, 0.2);
      decoded.segments = unitJitter(rng, 0.55, 0.3);
      break;
    case "flexy":
      decoded.form = formSpan(rng() < 0.4 ? "biped" : rng() < 0.7 ? "snake" : "blob", rng);
      decoded.leg = span("leg", 0.35 + rng() * 0.45);
      decoded.torsoH = span("torsoH", 0.35 + rng() * 0.45);
      decoded.torsoW = span("torsoW", 0.3 + rng() * 0.45);
      decoded.arm = span("arm", 0.4 + rng() * 0.45);
      decoded.quad = rng() * 0.35;
      decoded.jump = -0.2 + rng() * 0.7;
      decoded.freq = span("freq", 0.4 + rng() * 0.45);
      decoded.hipAmp = span("hipAmp", 0.6 + rng() * 0.4);
      decoded.kneeAmp = span("kneeAmp", 0.55 + rng() * 0.45);
      decoded.hipFlex = unitJitter(rng, 0.9, 0.1);
      decoded.kneeFlex = unitJitter(rng, 0.88, 0.12);
      decoded.armDrive = unitJitter(rng, 0.65, 0.2);
      decoded.segments = unitJitter(rng, 0.5, 0.3);
      break;
    case "galloper":
      decoded.form = formSpan(rng() < 0.55 ? "quad" : "biped", rng);
      decoded.leg = span("leg", 0.45 + rng() * 0.4);
      decoded.torsoH = span("torsoH", 0.4 + rng() * 0.4);
      decoded.torsoW = span("torsoW", 0.35 + rng() * 0.4);
      decoded.arm = span("arm", 0.4 + rng() * 0.4);
      decoded.quad = rng() < 0.35 ? 0.75 + rng() * 0.25 : rng() * 0.2;
      decoded.jump = -0.15 + rng() * 0.7;
      decoded.freq = span("freq", 0.72 + rng() * 0.28);
      decoded.hipAmp = span("hipAmp", 0.55 + rng() * 0.45);
      decoded.kneeAmp = span("kneeAmp", 0.5 + rng() * 0.45);
      decoded.hipFlex = unitJitter(rng, 0.65, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.6, 0.2);
      decoded.armDrive = unitJitter(rng, 0.55, 0.25);
      decoded.split = span("split", 0.5 + rng() * 0.5);
      decoded.segments = unitJitter(rng, 0.4, 0.25);
      break;
    case "leaner":
      decoded.form = formSpan("biped", rng);
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
      decoded.lean = span("lean", 0.75 + rng() * 0.25);
      decoded.hipFlex = unitJitter(rng, 0.55, 0.2);
      decoded.kneeFlex = unitJitter(rng, 0.45, 0.2);
      decoded.armDrive = unitJitter(rng, 0.5, 0.2);
      decoded.segments = unitJitter(rng, 0.4, 0.2);
      break;
  }

  decoded.toe = span("toe", clamp01(0.4 + jitter(rng, 0.35)));
  decoded.turn = span("turn", clamp01(0.5 + jitter(rng, 0.25)));
  decoded.kneePhase = span("kneePhase", clamp01(0.5 + jitter(rng, 0.35)));
  if (decoded.kneeStance === undefined) decoded.kneeStance = span("kneeStance", 0.15 + rng() * 0.5);
  if (decoded.hipBias === undefined) decoded.hipBias = span("hipBias", 0.35 + rng() * 0.4);
  if (decoded.elbow === undefined) decoded.elbow = span("elbow", 0.3 + rng() * 0.5);
  if (decoded.split === undefined) decoded.split = span("split", 0.55 + rng() * 0.45);
  if (decoded.armAmp === undefined) decoded.armAmp = span("armAmp", 0.25 + rng() * 0.5);

  return encode(decoded);
}

/**
 * Generation 0 is a round-robin of clearly different body plans and strategies.
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

/** Body/topology genes. Milder mutation so winners' shapes breed true. */
const SHAPE_GENES = new Set([
  "leg",
  "torsoH",
  "torsoW",
  "arm",
  "hipFlex",
  "kneeFlex",
  "jump",
  "quad",
  "armDrive",
  "form",
  "segments"
]);

/**
 * Speed and jump traits. When winners carry more of these than losers, offspring
 * get a directed upward nudge on top of louder random mutation.
 */
const ACCELERABLE_GENES = new Set([
  "freq",
  "hipAmp",
  "kneeAmp",
  "jump",
  "armDrive",
  "toe",
  "armAmp"
]);

/**
 * Ranked parents (best first). Positive entry = winners hold more of that gene
 * than the bottom quartile, so breeding should accelerate it upward.
 */
function accelerationBias(parents: readonly number[][]): Float64Array {
  const n = GENES.length;
  const bias = new Float64Array(n);
  if (parents.length < 4) return bias;
  const band = Math.max(2, Math.ceil(parents.length * 0.25));
  for (let i = 0; i < n; i++) {
    const name = GENES[i]?.name;
    if (!name || !ACCELERABLE_GENES.has(name)) continue;
    let top = 0;
    let bot = 0;
    for (let p = 0; p < band; p++) {
      top += parents[p]?.[i] ?? 0.5;
      bot += parents[parents.length - 1 - p]?.[i] ?? 0.5;
    }
    const diff = top / band - bot / band;
    // Only steer when the fitness link is clear; cap so form diversity survives.
    if (diff > 0.035) bias[i] = Math.min(0.24, diff * 0.65);
  }
  return bias;
}

function crossover(a: readonly number[], b: readonly number[], rng: Rng): number[] {
  const len = Math.max(a.length, b.length, GENES.length);
  const child: number[] = [];
  for (let i = 0; i < len; i++) {
    const gene = a[i] ?? b[i] ?? 0.5;
    const other = b[i] ?? gene;
    const name = GENES[i]?.name;
    const shape = name !== undefined && SHAPE_GENES.has(name);
    const roll = rng();
    // Form almost always copies whole from one parent so topologies stay crisp.
    if (name === "form") {
      child.push(roll < 0.5 ? gene : other);
      continue;
    }
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

function mutate(genes: readonly number[], rng: Rng, bias?: Float64Array): number[] {
  return genes.map((gene, i) => {
    const name = GENES[i]?.name;
    const shape = name !== undefined && SHAPE_GENES.has(name);
    const accel = name !== undefined && ACCELERABLE_GENES.has(name);
    const directed = bias?.[i] ?? 0;
    // Form mutates rarely: flip to a neighbor topology or stay put.
    if (name === "form") {
      if (rng() > 0.08) return gene;
      if (rng() < 0.55) {
        const kind = formFromGene(gene);
        const idx = Math.max(0, ["biped", "quad", "spider", "snake", "blob", "wheeler", "centipede"].indexOf(kind));
        const next = clamp(idx + (rng() < 0.5 ? -1 : 1), 0, 6);
        return (next + 0.5) / 7;
      }
      return rng();
    }
    // Louder next-gen exploration; jump/speed genes mutate even more often.
    const rate = accel ? 0.55 : shape ? 0.28 : 0.48;
    if (rng() > rate && directed <= 0) return gene;
    if (rng() > rate && directed > 0 && rng() > 0.55) {
      // Still apply a small upward steer even when the random roll skips.
      return clamp01(gene + directed * (0.25 + rng() * 0.55));
    }
    const sigma = accel
      ? rng() < 0.28
        ? 0.4
        : rng() < 0.55
          ? 0.2
          : 0.09
      : shape
        ? rng() < 0.15
          ? 0.18
          : rng() < 0.45
            ? 0.09
            : 0.04
        : rng() < 0.22
          ? 0.36
          : rng() < 0.5
            ? 0.16
            : 0.07;
    let next = gene + gauss(rng) * sigma;
    if (directed > 0) {
      // Winners ran/jumped harder — push offspring up, not just random drift.
      next += directed * (0.4 + rng() * 1.1);
      if (rng() < 0.4) next += directed * (0.6 + rng() * 0.9);
    }
    return clamp01(next);
  });
}

/**
 * `parents` must already be sorted best-first. Returns a new population of genomes.
 * Elites are copied whole. Children blend two parents then mutate harder than
 * before; when jump/speed genes correlate with fitness, those traits get an
 * upward bias so successful hop and stride amplify in the next lineup.
 */
export function breed(parents: readonly number[][], pop: number, rng: Rng): number[][] {
  if (parents.length === 0) return initialPopulation(pop, rng);
  const eliteN = Math.max(2, Math.round(pop * 0.2));
  const immigrants = pop >= 14 ? 1 : 0;
  const bias = accelerationBias(parents);
  const out: number[][] = [];
  for (let i = 0; i < eliteN && i < parents.length; i++) {
    out.push((parents[i] ?? []).slice());
  }
  while (out.length < pop - immigrants) {
    const a = tournament(parents, rng);
    const b = tournament(parents, rng);
    out.push(mutate(crossover(a, b, rng), rng, bias));
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

function wave(phase: number): number {
  return Math.sin(phase * Math.PI * 2);
}

/**
 * Write motor targets for every joint on this morph from gait genes.
 * Topology-aware: biped crawl blend, multi-leg phases, snake undulation,
 * blob wobble, wheel spin, centipede ripple.
 */
export function writePose(decoded: Decoded, time: number, out: Pose, morph: Morph): void {
  for (const key of Object.keys(out)) out[key] = 0;
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
  const lean = decoded.lean ?? 0;
  const crouch = morph.crouch;
  const fold = Math.max(0, Math.min(1, crouch));
  const ramp = Math.min(1, time / 1.05);
  const limits = morph.limits;

  const set = (key: string, value: number, limName: string): void => {
    const lim = limits[limName] ?? limits.hip ?? [-1, 1];
    out[key] = clamp(value, lim[0], lim[1]);
  };

  if (morph.form === "biped") {
    const crawl = crawlBlend(decoded.quad ?? 0);
    const left = legStep(cycle, hipAmp * (1 - 0.78 * fold * (1 - crawl)), hipBias * (1 - 0.7 * fold), kneeAmp * (1 - fold * (1 - crawl)), stance);
    const right = legStep(cycle + 0.5 * split, hipAmp * (1 - 0.78 * fold * (1 - crawl)), hipBias * (1 - 0.7 * fold), kneeAmp * (1 - fold * (1 - crawl)), stance);
    const armPhase = cycle * Math.PI * 2;
    const phaseL = ((cycle % 1) + 1) % 1;
    const phaseR = ((cycle + 0.5 * split) % 1 + 1) % 1;
    const hipSwing = Math.max(0.32, Math.min(0.72, hipAmp * 0.55));
    const kneeSwing = Math.max(0.18, Math.min(0.55, kneeAmp * 0.32));
    const armSwing = Math.max(0.28, Math.min(0.75, armAmp * (0.55 + 0.7 * armDrive)));
    const hipBase = clamp(1.05, (limits.hip?.[0] ?? -1) + 0.04, (limits.hip?.[1] ?? 1) - 0.05);
    const kneeBase = clamp(1.48, limits.knee?.[0] ?? 0, (limits.knee?.[1] ?? 2) - 0.05);
    const shBase = clamp(1.05, (limits.shoulder?.[0] ?? -1) + 0.04, (limits.shoulder?.[1] ?? 1) - 0.08);
    const elBase = clamp(1.2, limits.elbow?.[0] ?? 0, (limits.elbow?.[1] ?? 2) - 0.05);
    const mix = (stand: number, low: number): number => lerp(stand, low, crawl);
    const bipedArm = armAmp * (0.55 + 0.9 * armDrive);
    const elbowPump = elbow * (0.7 + 0.55 * armDrive);
    const kneeCrouch = fold * 1.05 * ramp * (1 - crawl);
    const hipCrouch = fold * 0.32 * ramp * (1 - crawl);
    set("spine", mix(lean * (1 - 0.45 * fold) + fold * 0.05 * ramp, -0.32), "spine");
    set("neck", lean * 0.15, "spine");
    set("hipL", mix(left.hip + turn + hipCrouch, hipBase + hipSwing * wave(phaseL) + turn * 0.3), "hip");
    set("hipR", mix(right.hip - turn + hipCrouch, hipBase + hipSwing * wave(phaseR) - turn * 0.3), "hip");
    set("kneeL", mix(left.knee + kneeCrouch, kneeBase + kneeSwing * Math.max(0, wave(phaseL))), "knee");
    set("kneeR", mix(right.knee + kneeCrouch, kneeBase + kneeSwing * Math.max(0, wave(phaseR))), "knee");
    set("ankleL", mix(toe * Math.sin(armPhase), toe * 0.4 * wave(phaseL)), "ankle");
    set("ankleR", mix(toe * Math.sin(armPhase + Math.PI * split), toe * 0.4 * wave(phaseR)), "ankle");
    set("shoulderL", mix(-bipedArm * Math.sin(armPhase), shBase + armSwing * wave(phaseR)), "shoulder");
    set("shoulderR", mix(-bipedArm * Math.sin(armPhase + Math.PI * split), shBase + armSwing * wave(phaseL)), "shoulder");
    set("elbowL", mix(elbowPump + 0.18 * armDrive * Math.max(0, Math.sin(armPhase)), elBase + 0.22 * armDrive * Math.max(0, wave(phaseR))), "elbow");
    set("elbowR", mix(elbowPump + 0.18 * armDrive * Math.max(0, Math.sin(armPhase + Math.PI * split)), elBase + 0.22 * armDrive * Math.max(0, wave(phaseL))), "elbow");
    return;
  }

  if (morph.form === "snake") {
    const amp = Math.max(0.25, Math.min(1.05, hipAmp * 0.55 + lean * 0.3));
    let i = 0;
    for (const joint of morph.joints) {
      if (!joint.poseKey.startsWith("seg")) continue;
      const phase = cycle + i * (0.35 + 0.25 * split);
      set(joint.poseKey, amp * Math.sin(phase * Math.PI * 2) + turn * 0.4, "segment");
      i += 1;
    }
    return;
  }

  if (morph.form === "blob") {
    const amp = Math.max(0.2, Math.min(0.95, hipAmp * 0.4 + armAmp * 0.25));
    let i = 0;
    for (const joint of morph.joints) {
      if (!joint.poseKey.startsWith("lobe")) continue;
      set(joint.poseKey, amp * Math.sin((cycle + i * 0.37) * Math.PI * 2) + lean * 0.2, "segment");
      i += 1;
    }
    return;
  }

  if (morph.form === "wheeler") {
    set("spine", lean * 0.35, "spine");
    const spin = (toe * 2.8 + hipAmp * 1.6 + 1.2) * (0.6 + 0.8 * armDrive);
    let i = 0;
    for (const joint of morph.joints) {
      if (joint.poseKey.startsWith("susp")) {
        set(joint.poseKey, hipBias * 0.4 + 0.12 * Math.sin((cycle + i * 0.5) * Math.PI * 2), "hip");
      } else if (joint.poseKey.startsWith("wheel")) {
        // Continuous roll target wraps inside ±π via motor position.
        const angle = ((time * spin + i * 0.7) % (Math.PI * 2)) - Math.PI;
        out[joint.poseKey] = angle;
      }
      i += 1;
    }
    return;
  }

  // Quad, spider, centipede: phased multi-leg gait.
  const hipSwing = Math.max(0.28, Math.min(0.85, hipAmp * 0.5));
  const kneeSwing = Math.max(0.15, Math.min(0.7, kneeAmp * 0.35));
  const hipBase = clamp(0.55 + hipBias, (limits.hip?.[0] ?? -1) + 0.05, (limits.hip?.[1] ?? 1) - 0.05);
  const kneeBase = clamp(0.85 + stance, limits.knee?.[0] ?? 0, (limits.knee?.[1] ?? 2) - 0.05);
  set("spine", lean * (morph.form === "quad" ? 0.45 : 0.25), "spine");

  const hipKeys = morph.joints.filter((j) => j.poseKey.startsWith("hip")).map((j) => j.poseKey);
  const n = Math.max(1, hipKeys.length);
  for (let i = 0; i < hipKeys.length; i++) {
    const key = hipKeys[i]!;
    const phaseOff =
      morph.form === "quad"
        ? (i % 2 === 0 ? 0 : 0.5 * split)
        : morph.form === "centipede"
          ? (Math.floor(i / 2) * 0.18 + (i % 2) * 0.5 * split)
          : (i / n) * split;
    const phase = ((cycle + phaseOff) % 1 + 1) % 1;
    const step = legStep(cycle + phaseOff, hipSwing, hipBias * 0.5, kneeSwing, stance);
    set(key, hipBase + step.hip * 0.85 + turn * (i % 2 === 0 ? 0.25 : -0.25), "hip");
    const kneeKey = key.replace("hip", "knee");
    const ankleKey = key.replace("hip", "ankle");
    if (out[kneeKey] !== undefined || morph.joints.some((j) => j.poseKey === kneeKey)) {
      set(kneeKey, kneeBase + step.knee, "knee");
    }
    if (morph.joints.some((j) => j.poseKey === ankleKey)) {
      set(ankleKey, toe * 0.45 * wave(phase), "ankle");
    }
  }

  // Centipede body links undulate mildly.
  if (morph.form === "centipede") {
    let i = 0;
    for (const joint of morph.joints) {
      if (!joint.poseKey.startsWith("link")) continue;
      set(joint.poseKey, 0.35 * hipAmp * Math.sin((cycle + i * 0.28) * Math.PI * 2) + turn * 0.2, "segment");
      i += 1;
    }
  }
}
