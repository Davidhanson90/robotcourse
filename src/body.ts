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
  spine: [-0.4, 0.55] as const,
  hip: [-0.95, 1.2] as const,
  knee: [0, 1.9] as const,
  ankle: [-0.75, 0.85] as const,
  shoulder: [-1.15, 1.15] as const,
  elbow: [0.12, 2.05] as const
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
