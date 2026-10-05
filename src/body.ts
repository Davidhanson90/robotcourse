/** Shared humanoid dimensions (meters). Visuals and colliders use the same numbers. */

export const THIGH = 0.42;
export const SHIN = 0.4;
export const LIMB_R = 0.068;
export const SHIN_R = 0.058;
export const UPPER_ARM = 0.3;
export const FOREARM = 0.28;
export const ARM_R = 0.046;
export const FORE_R = 0.04;

export const PELVIS = { hx: 0.16, hy: 0.11, hz: 0.2 };
export const CHEST = { hx: 0.19, hy: 0.22, hz: 0.17 };
export const HEAD_R = 0.15;

export const HIP_Z = 0.16;
export const HIP_DROP = 0.045;
export const SHOULDER_Y = 0.14;
export const SHOULDER_Z = 0.27;

export const FOOT = { hx: 0.22, hy: 0.045, hz: 0.11, heel: -0.1 };

export const LIFT = 0.025;

export const LIMIT = {
  spine: [-0.48, 0.62] as const,
  hip: [-1.55, 1.95] as const,
  knee: [0, 2.55] as const,
  ankle: [-0.85, 0.95] as const,
  shoulder: [-1.35, 1.35] as const,
  elbow: [0.1, 2.15] as const
};

/** Pelvis center height when the sole is `lift` meters above y = 0. */
export function pelvisHeight(lift = LIFT): number {
  return lift + FOOT.hy * 2 + SHIN + THIGH + HIP_DROP;
}

export function capsuleHalf(length: number, radius: number): number {
  return Math.max(0.02, length / 2 - radius);
}

export function capsuleVolume(length: number, radius: number): number {
  const half = capsuleHalf(length, radius);
  return Math.PI * radius * radius * (2 * half + (4 / 3) * radius);
}

export function cuboidVolume(hx: number, hy: number, hz: number): number {
  return 8 * hx * hy * hz;
}

export function densityFor(volume: number, mass: number): number {
  return mass / Math.max(volume, 1e-6);
}

export const BODY_KEYS = [
  "pelvis",
  "chest",
  "thighL",
  "shinL",
  "footL",
  "thighR",
  "shinR",
  "footR",
  "armL",
  "foreL",
  "armR",
  "foreR"
] as const;

export type BodyKey = (typeof BODY_KEYS)[number];

export interface Pose {
  spine: number;
  hipL: number;
  hipR: number;
  kneeL: number;
  kneeR: number;
  ankleL: number;
  ankleR: number;
  shoulderL: number;
  shoulderR: number;
  elbowL: number;
  elbowR: number;
}

export interface Morph {
  thigh: number;
  shin: number;
  upperArm: number;
  forearm: number;
  pelvis: { hx: number; hy: number; hz: number };
  chest: { hx: number; hy: number; hz: number };
  headR: number;
  hipZ: number;
  hipDrop: number;
  shoulderY: number;
  shoulderZ: number;
  limits: {
    spine: readonly [number, number];
    hip: readonly [number, number];
    knee: readonly [number, number];
    ankle: readonly [number, number];
    shoulder: readonly [number, number];
    elbow: readonly [number, number];
  };
  /** Upward jump strength. Zero means this robot cannot jump on purpose. */
  jump: number;
  /** Heritable posture. 0 stands. 1 is a low four-point crawl. */
  quad: number;
  /** 0 stands tall. 1 folds low, including an all-fours stance. */
  crouch: number;
  /** Pelvis center height with straight legs and the sole just off the ground. */
  pelvisStand: number;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Piecewise lerp that passes through `mid` at `tMid`, so the prototype limits stay put. */
function through(stiff: number, mid: number, loose: number, t: number, tMid: number): number {
  const u = Math.max(0, Math.min(1, t));
  if (u <= tMid) return lerp(stiff, mid, u / Math.max(tMid, 1e-6));
  return lerp(mid, loose, (u - tMid) / (1 - tMid));
}

/**
 * 0 is a standing biped. 1 plants the hands and holds a low four-point stance.
 * Values between the cutoffs are a partial crouch so breeding can slide into it.
 */
export function crawlBlend(quad: number): number {
  const q = Math.max(0, Math.min(1, quad));
  if (q <= 0.34) return 0;
  if (q >= 0.72) return 1;
  return (q - 0.34) / 0.38;
}

/** Rigid-body sizes, joint anchors, and joint limits for one genome. */
export function buildMorph(params: {
  leg: number;
  torsoH: number;
  torsoW: number;
  arm: number;
  hipFlex: number;
  kneeFlex: number;
  jump: number;
  quad?: number;
}): Morph {
  const hipFlex = Math.max(0, Math.min(1, params.hipFlex));
  const kneeFlex = Math.max(0, Math.min(1, params.kneeFlex));
  const thigh = THIGH * params.leg;
  const shin = SHIN * params.leg;
  const pelvis = {
    hx: PELVIS.hx * params.torsoW,
    hy: PELVIS.hy * params.torsoH,
    hz: PELVIS.hz * params.torsoW
  };
  const chest = {
    hx: CHEST.hx * params.torsoW,
    hy: CHEST.hy * params.torsoH,
    hz: CHEST.hz * params.torsoW
  };
  const hipDrop = HIP_DROP * params.torsoH;
  const kneeCrouch = Math.max(0, Math.min(1, (kneeFlex - 0.62) / 0.38));
  const hipFold = Math.max(0, Math.min(1, (hipFlex - 0.4) / 0.6));
  const quad = Math.max(0, Math.min(1, params.quad ?? 0));
  const crawl = crawlBlend(quad);
  return {
    thigh,
    shin,
    upperArm: UPPER_ARM * params.arm,
    forearm: FOREARM * params.arm,
    pelvis,
    chest,
    headR: HEAD_R * (0.82 + 0.18 * params.torsoH),
    hipZ: HIP_Z * params.torsoW,
    hipDrop,
    shoulderY: SHOULDER_Y * params.torsoH,
    shoulderZ: SHOULDER_Z * params.torsoW,
    limits: {
      spine: LIMIT.spine,
      hip: [through(-0.32, LIMIT.hip[0], -2.35, hipFlex, 0.55), through(0.45, LIMIT.hip[1], 2.95, hipFlex, 0.55)],
      knee: [0, through(0.32, LIMIT.knee[1], 3.35, kneeFlex, 0.42)],
      ankle: [
        through(-0.4, LIMIT.ankle[0], -1.15, kneeFlex, 0.42),
        through(0.45, LIMIT.ankle[1], 1.25, kneeFlex, 0.42)
      ],
      shoulder: [
        LIMIT.shoulder[0] - 0.45 * crawl,
        LIMIT.shoulder[1] + 0.85 * crawl
      ] as const,
      elbow: LIMIT.elbow
    },
    jump: Math.max(0, params.jump),
    quad,
    crouch: Math.max(kneeCrouch * (0.35 + 0.65 * hipFold), crawl),
    pelvisStand: LIFT + FOOT.hy * 2 + shin + thigh + hipDrop
  };
}
