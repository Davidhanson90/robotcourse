/** Genome-driven body blueprints. Robots can be any topology under Rapier. */

export const LIFT = 0.025;

export type FormKind =
  | "biped"
  | "quad"
  | "spider"
  | "snake"
  | "blob"
  | "wheeler"
  | "centipede";

export const FORM_KINDS: readonly FormKind[] = [
  "biped",
  "quad",
  "spider",
  "snake",
  "blob",
  "wheeler",
  "centipede"
];

export type PartShape = "box" | "capsule" | "ball" | "cylinder";

export interface V3 {
  x: number;
  y: number;
  z: number;
}

export interface PartSpec {
  id: string;
  shape: PartShape;
  /** Box half-extents. */
  hx?: number;
  hy?: number;
  hz?: number;
  /** Ball / capsule / cylinder radius. */
  radius?: number;
  /** Capsule or cylinder length along local Y (full). */
  length?: number;
  mass: number;
  friction: number;
  /** World spawn of the part center. */
  x: number;
  y: number;
  z: number;
  /** Local rotation as quaternion. Identity if omitted. */
  qx?: number;
  qy?: number;
  qz?: number;
  qw?: number;
  ccd?: boolean;
  /** Counts as a contact for grounded jump. */
  contact?: boolean;
  /** Visual accent: foot, wheel, head, core. */
  role?: "core" | "limb" | "foot" | "wheel" | "head" | "blob" | "segment";
}

export interface JointSpec {
  id: string;
  parent: string;
  child: string;
  anchorParent: V3;
  anchorChild: V3;
  axis: V3;
  min: number;
  max: number;
  /** Motor stiffness family. */
  gain: "spine" | "hip" | "knee" | "ankle" | "shoulder" | "elbow" | "segment";
  /** Pose key written by the gait controller. */
  poseKey: string;
}

export interface Morph {
  form: FormKind;
  parts: PartSpec[];
  joints: JointSpec[];
  rootId: string;
  contactIds: string[];
  /** Upward jump strength. Zero means no intentional jump. */
  jump: number;
  /** 0 stands tall. 1 folds low. */
  crouch: number;
  /** Root center height when resting above the ground. */
  pelvisStand: number;
  /** Shared nominal limb sizes for gait amplitude scaling. */
  thigh: number;
  shin: number;
  upperArm: number;
  forearm: number;
  /** Heritable posture leftover for biped crawl blend. */
  quad: number;
  limits: Record<string, readonly [number, number]>;
}

/** Pose targets keyed by joint poseKey. */
export type Pose = Record<string, number>;

export function emptyPose(morph: Morph): Pose {
  const out: Pose = {};
  for (const joint of morph.joints) out[joint.poseKey] = 0;
  return out;
}

export function formFromGene(form: number): FormKind {
  const u = Math.max(0, Math.min(0.999999, form));
  return FORM_KINDS[Math.floor(u * FORM_KINDS.length)] ?? "biped";
}

export function formCenter(kind: FormKind): number {
  const i = FORM_KINDS.indexOf(kind);
  return (i + 0.5) / FORM_KINDS.length;
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

export function ballVolume(radius: number): number {
  return (4 / 3) * Math.PI * radius * radius * radius;
}

export function cylinderVolume(halfHeight: number, radius: number): number {
  return Math.PI * radius * radius * 2 * halfHeight;
}

export function densityFor(volume: number, mass: number): number {
  return mass / Math.max(volume, 1e-6);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function through(stiff: number, mid: number, loose: number, t: number, tMid: number): number {
  const u = Math.max(0, Math.min(1, t));
  if (u <= tMid) return lerp(stiff, mid, u / Math.max(tMid, 1e-6));
  return lerp(mid, loose, (u - tMid) / (1 - tMid));
}

/** 0 biped stand. 1 low four-point crawl (biped only). */
export function crawlBlend(quad: number): number {
  const q = Math.max(0, Math.min(1, quad));
  if (q <= 0.34) return 0;
  if (q >= 0.72) return 1;
  return (q - 0.34) / 0.38;
}

const HIP_BASE = [-1.55, 1.95] as const;
const KNEE_BASE = [0, 2.55] as const;
const ANKLE_BASE = [-0.85, 0.95] as const;
const SHOULDER_BASE = [-1.35, 1.35] as const;
const ELBOW_BASE = [0.1, 2.15] as const;
const SPINE_BASE = [-0.48, 0.62] as const;
const SEG_BASE = [-1.1, 1.1] as const;

function flexLimits(hipFlex: number, kneeFlex: number): Morph["limits"] {
  return {
    spine: SPINE_BASE,
    hip: [
      through(-0.32, HIP_BASE[0], -2.35, hipFlex, 0.55),
      through(0.45, HIP_BASE[1], 2.95, hipFlex, 0.55)
    ],
    knee: [0, through(0.32, KNEE_BASE[1], 3.35, kneeFlex, 0.42)],
    ankle: [
      through(-0.4, ANKLE_BASE[0], -1.15, kneeFlex, 0.42),
      through(0.45, ANKLE_BASE[1], 1.25, kneeFlex, 0.42)
    ],
    shoulder: SHOULDER_BASE,
    elbow: ELBOW_BASE,
    segment: SEG_BASE
  };
}

function id(prefix: string, i: number, side = ""): string {
  return side ? `${prefix}${side}${i}` : `${prefix}${i}`;
}

/**
 * Build a full part/joint blueprint from shape genes.
 * `spawnX`/`spawnZ` place the creature; Y is standing height.
 */
export function buildMorph(params: {
  form: number;
  leg: number;
  torsoH: number;
  torsoW: number;
  arm: number;
  hipFlex: number;
  kneeFlex: number;
  jump: number;
  quad?: number;
  segments?: number;
  spawnX?: number;
  spawnZ?: number;
}): Morph {
  const form = formFromGene(params.form);
  const hipFlex = Math.max(0, Math.min(1, params.hipFlex));
  const kneeFlex = Math.max(0, Math.min(1, params.kneeFlex));
  const leg = Math.max(0.4, params.leg);
  const torsoH = Math.max(0.4, params.torsoH);
  const torsoW = Math.max(0.4, params.torsoW);
  const arm = Math.max(0.4, params.arm);
  const quad = Math.max(0, Math.min(1, params.quad ?? 0));
  const segGene = Math.max(0, Math.min(1, params.segments ?? 0.5));
  const sx = params.spawnX ?? 0;
  const sz = params.spawnZ ?? 0;
  const limits = flexLimits(hipFlex, kneeFlex);
  const kneeCrouch = Math.max(0, Math.min(1, (kneeFlex - 0.62) / 0.38));
  const hipFold = Math.max(0, Math.min(1, (hipFlex - 0.4) / 0.6));
  const crouch = Math.max(kneeCrouch * (0.35 + 0.65 * hipFold), crawlBlend(quad) * (form === "biped" ? 1 : 0.35));
  const jump = Math.max(0, params.jump);

  const thigh = 0.42 * leg;
  const shin = 0.4 * leg;
  const upperArm = 0.3 * arm;
  const forearm = 0.28 * arm;

  let morph: Morph;
  switch (form) {
    case "biped":
      morph = buildBiped(sx, sz, thigh, shin, upperArm, forearm, torsoH, torsoW, limits, jump, crouch, quad);
      break;
    case "quad":
      morph = buildQuad(sx, sz, thigh, shin, torsoH, torsoW, limits, jump, crouch, quad);
      break;
    case "spider":
      morph = buildSpider(sx, sz, thigh, shin, torsoH, torsoW, segGene, limits, jump, crouch, quad);
      break;
    case "snake":
      morph = buildSnake(sx, sz, torsoH, torsoW, segGene, limits, jump, crouch, quad);
      break;
    case "blob":
      morph = buildBlob(sx, sz, torsoH, torsoW, arm, segGene, limits, jump, crouch, quad);
      break;
    case "wheeler":
      morph = buildWheeler(sx, sz, thigh, torsoH, torsoW, segGene, limits, jump, crouch, quad);
      break;
    case "centipede":
      morph = buildCentipede(sx, sz, thigh, shin, torsoH, torsoW, segGene, limits, jump, crouch, quad);
      break;
  }
  morph.thigh = thigh;
  morph.shin = shin;
  morph.upperArm = upperArm;
  morph.forearm = forearm;
  return morph;
}

function finish(
  form: FormKind,
  parts: PartSpec[],
  joints: JointSpec[],
  rootId: string,
  jump: number,
  crouch: number,
  pelvisStand: number,
  quad: number,
  limits: Morph["limits"]
): Morph {
  const contactIds = parts.filter((p) => p.contact).map((p) => p.id);
  return {
    form,
    parts,
    joints,
    rootId,
    contactIds,
    jump,
    crouch,
    pelvisStand,
    thigh: 0.42,
    shin: 0.4,
    upperArm: 0.3,
    forearm: 0.28,
    quad,
    limits
  };
}

function buildBiped(
  sx: number,
  sz: number,
  thigh: number,
  shin: number,
  upperArm: number,
  forearm: number,
  torsoH: number,
  torsoW: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const foot = { hx: 0.22, hy: 0.045, hz: 0.11, heel: -0.1 };
  const limbR = 0.068;
  const shinR = 0.058;
  const armR = 0.046;
  const foreR = 0.04;
  const pelvis = { hx: 0.16 * torsoW, hy: 0.11 * torsoH, hz: 0.2 * torsoW };
  const chest = { hx: 0.19 * torsoW, hy: 0.22 * torsoH, hz: 0.17 * torsoW };
  const hipZ = 0.16 * torsoW;
  const hipDrop = 0.045 * torsoH;
  const headR = 0.15 * (0.82 + 0.18 * torsoH);
  const shoulderY = 0.14 * torsoH;
  const shoulderZ = 0.27 * torsoW;
  const pelvisStand = LIFT + foot.hy * 2 + shin + thigh + hipDrop;
  const pelvisY = pelvisStand;
  const chestY = pelvisY + pelvis.hy + chest.hy;
  const hipY = pelvisY - hipDrop;
  const thighY = hipY - thigh / 2;
  const kneeY = hipY - thigh;
  const shinY = kneeY - shin / 2;
  const ankleY = kneeY - shin;
  const footY = ankleY - foot.hy;
  const shoulderTop = chestY + shoulderY;
  const armY = shoulderTop - upperArm / 2;
  const elbowY = shoulderTop - upperArm;
  const foreY = elbowY - forearm / 2;

  const parts: PartSpec[] = [
    { id: "root", shape: "box", hx: pelvis.hx, hy: pelvis.hy, hz: pelvis.hz, mass: 7, friction: 0.45, x: sx, y: pelvisY, z: sz, ccd: true, role: "core" },
    { id: "chest", shape: "box", hx: chest.hx, hy: chest.hy, hz: chest.hz, mass: 9, friction: 0.4, x: sx, y: chestY, z: sz, role: "core" },
    { id: "head", shape: "ball", radius: headR, mass: 3.2, friction: 0.35, x: sx, y: chestY + chest.hy + headR * 0.55, z: sz, role: "head" },
    { id: "thighL", shape: "capsule", length: thigh, radius: limbR, mass: 2.8, friction: 0.4, x: sx, y: thighY, z: sz - hipZ, role: "limb" },
    { id: "thighR", shape: "capsule", length: thigh, radius: limbR, mass: 2.8, friction: 0.4, x: sx, y: thighY, z: sz + hipZ, role: "limb" },
    { id: "shinL", shape: "capsule", length: shin, radius: shinR, mass: 1.8, friction: 0.4, x: sx, y: shinY, z: sz - hipZ, ccd: true, role: "limb" },
    { id: "shinR", shape: "capsule", length: shin, radius: shinR, mass: 1.8, friction: 0.4, x: sx, y: shinY, z: sz + hipZ, ccd: true, role: "limb" },
    { id: "footL", shape: "box", hx: foot.hx, hy: foot.hy, hz: foot.hz, mass: 1.6, friction: 1.55, x: sx + foot.heel, y: footY, z: sz - hipZ, ccd: true, contact: true, role: "foot" },
    { id: "footR", shape: "box", hx: foot.hx, hy: foot.hy, hz: foot.hz, mass: 1.6, friction: 1.55, x: sx + foot.heel, y: footY, z: sz + hipZ, ccd: true, contact: true, role: "foot" },
    { id: "armL", shape: "capsule", length: upperArm, radius: armR, mass: 1, friction: 0.3, x: sx, y: armY, z: sz - shoulderZ, role: "limb" },
    { id: "armR", shape: "capsule", length: upperArm, radius: armR, mass: 1, friction: 0.3, x: sx, y: armY, z: sz + shoulderZ, role: "limb" },
    { id: "foreL", shape: "capsule", length: forearm, radius: foreR, mass: 0.7, friction: quad >= 0.5 ? 1.2 : 0.3, x: sx, y: foreY, z: sz - shoulderZ, role: "limb" },
    { id: "foreR", shape: "capsule", length: forearm, radius: foreR, mass: 0.7, friction: quad >= 0.5 ? 1.2 : 0.3, x: sx, y: foreY, z: sz + shoulderZ, role: "limb" }
  ];

  const z = { x: 0, y: 0, z: 1 };
  const kz = { x: 0, y: 0, z: -1 };
  const joints: JointSpec[] = [
    { id: "spine", parent: "root", child: "chest", anchorParent: { x: 0, y: pelvis.hy, z: 0 }, anchorChild: { x: 0, y: -chest.hy, z: 0 }, axis: z, min: limits.spine![0], max: limits.spine![1], gain: "spine", poseKey: "spine" },
    { id: "neck", parent: "chest", child: "head", anchorParent: { x: 0, y: chest.hy, z: 0 }, anchorChild: { x: 0, y: -headR * 0.85, z: 0 }, axis: z, min: -0.35, max: 0.45, gain: "spine", poseKey: "neck" },
    { id: "hipL", parent: "root", child: "thighL", anchorParent: { x: 0, y: -hipDrop, z: -hipZ }, anchorChild: { x: 0, y: thigh / 2, z: 0 }, axis: z, min: limits.hip![0], max: limits.hip![1], gain: "hip", poseKey: "hipL" },
    { id: "hipR", parent: "root", child: "thighR", anchorParent: { x: 0, y: -hipDrop, z: hipZ }, anchorChild: { x: 0, y: thigh / 2, z: 0 }, axis: z, min: limits.hip![0], max: limits.hip![1], gain: "hip", poseKey: "hipR" },
    { id: "kneeL", parent: "thighL", child: "shinL", anchorParent: { x: 0, y: -thigh / 2, z: 0 }, anchorChild: { x: 0, y: shin / 2, z: 0 }, axis: kz, min: limits.knee![0], max: limits.knee![1], gain: "knee", poseKey: "kneeL" },
    { id: "kneeR", parent: "thighR", child: "shinR", anchorParent: { x: 0, y: -thigh / 2, z: 0 }, anchorChild: { x: 0, y: shin / 2, z: 0 }, axis: kz, min: limits.knee![0], max: limits.knee![1], gain: "knee", poseKey: "kneeR" },
    { id: "ankleL", parent: "shinL", child: "footL", anchorParent: { x: 0, y: -shin / 2, z: 0 }, anchorChild: { x: -foot.heel, y: foot.hy, z: 0 }, axis: z, min: limits.ankle![0], max: limits.ankle![1], gain: "ankle", poseKey: "ankleL" },
    { id: "ankleR", parent: "shinR", child: "footR", anchorParent: { x: 0, y: -shin / 2, z: 0 }, anchorChild: { x: -foot.heel, y: foot.hy, z: 0 }, axis: z, min: limits.ankle![0], max: limits.ankle![1], gain: "ankle", poseKey: "ankleR" },
    { id: "shoulderL", parent: "chest", child: "armL", anchorParent: { x: 0, y: shoulderY, z: -shoulderZ }, anchorChild: { x: 0, y: upperArm / 2, z: 0 }, axis: z, min: limits.shoulder![0], max: limits.shoulder![1], gain: "shoulder", poseKey: "shoulderL" },
    { id: "shoulderR", parent: "chest", child: "armR", anchorParent: { x: 0, y: shoulderY, z: shoulderZ }, anchorChild: { x: 0, y: upperArm / 2, z: 0 }, axis: z, min: limits.shoulder![0], max: limits.shoulder![1], gain: "shoulder", poseKey: "shoulderR" },
    { id: "elbowL", parent: "armL", child: "foreL", anchorParent: { x: 0, y: -upperArm / 2, z: 0 }, anchorChild: { x: 0, y: forearm / 2, z: 0 }, axis: z, min: limits.elbow![0], max: limits.elbow![1], gain: "elbow", poseKey: "elbowL" },
    { id: "elbowR", parent: "armR", child: "foreR", anchorParent: { x: 0, y: -upperArm / 2, z: 0 }, anchorChild: { x: 0, y: forearm / 2, z: 0 }, axis: z, min: limits.elbow![0], max: limits.elbow![1], gain: "elbow", poseKey: "elbowR" }
  ];

  return finish("biped", parts, joints, "root", jump, crouch, pelvisStand, quad, limits);
}

function buildQuad(
  sx: number,
  sz: number,
  thigh: number,
  shin: number,
  torsoH: number,
  torsoW: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const bodyLen = 0.55 * torsoH + 0.35;
  const bodyH = 0.12 * torsoH + 0.08;
  const bodyW = 0.18 * torsoW + 0.1;
  const limbR = 0.055;
  const footR = 0.07;
  const stand = LIFT + footR + shin + thigh * 0.85 + bodyH;
  const rootY = stand;
  const hipDrop = bodyH * 0.55;
  const spanX = bodyLen * 0.42;
  const spanZ = bodyW * 0.95;
  const parts: PartSpec[] = [
    { id: "root", shape: "box", hx: bodyLen / 2, hy: bodyH, hz: bodyW, mass: 14, friction: 0.5, x: sx, y: rootY, z: sz, ccd: true, role: "core" },
    { id: "head", shape: "ball", radius: 0.12 * torsoW + 0.08, mass: 2.5, friction: 0.35, x: sx + bodyLen * 0.55, y: rootY + bodyH * 0.2, z: sz, role: "head" }
  ];
  const joints: JointSpec[] = [
    {
      id: "neck",
      parent: "root",
      child: "head",
      anchorParent: { x: bodyLen / 2, y: bodyH * 0.3, z: 0 },
      anchorChild: { x: -0.1, y: 0, z: 0 },
      axis: { x: 0, y: 0, z: 1 },
      min: -0.6,
      max: 0.6,
      gain: "spine",
      poseKey: "spine"
    }
  ];
  const legs: Array<{ name: string; x: number; z: number; phase: number }> = [
    { name: "FL", x: spanX, z: -spanZ, phase: 0 },
    { name: "FR", x: spanX, z: spanZ, phase: 1 },
    { name: "BL", x: -spanX, z: -spanZ, phase: 1 },
    { name: "BR", x: -spanX, z: spanZ, phase: 0 }
  ];
  for (const leg of legs) {
    const thighId = `thigh${leg.name}`;
    const shinId = `shin${leg.name}`;
    const footId = `foot${leg.name}`;
    const hipY = rootY - hipDrop;
    parts.push(
      { id: thighId, shape: "capsule", length: thigh, radius: limbR, mass: 2.2, friction: 0.4, x: sx + leg.x, y: hipY - thigh / 2, z: sz + leg.z, role: "limb" },
      { id: shinId, shape: "capsule", length: shin, radius: limbR * 0.9, mass: 1.4, friction: 0.4, x: sx + leg.x, y: hipY - thigh - shin / 2, z: sz + leg.z, ccd: true, role: "limb" },
      { id: footId, shape: "ball", radius: footR, mass: 0.8, friction: 1.6, x: sx + leg.x, y: LIFT + footR, z: sz + leg.z, ccd: true, contact: true, role: "foot" }
    );
    joints.push(
      {
        id: `hip${leg.name}`,
        parent: "root",
        child: thighId,
        anchorParent: { x: leg.x, y: -hipDrop, z: leg.z },
        anchorChild: { x: 0, y: thigh / 2, z: 0 },
        axis: { x: 0, y: 0, z: 1 },
        min: limits.hip![0],
        max: limits.hip![1],
        gain: "hip",
        poseKey: `hip${leg.name}`
      },
      {
        id: `knee${leg.name}`,
        parent: thighId,
        child: shinId,
        anchorParent: { x: 0, y: -thigh / 2, z: 0 },
        anchorChild: { x: 0, y: shin / 2, z: 0 },
        axis: { x: 0, y: 0, z: -1 },
        min: limits.knee![0],
        max: limits.knee![1],
        gain: "knee",
        poseKey: `knee${leg.name}`
      },
      {
        id: `ankle${leg.name}`,
        parent: shinId,
        child: footId,
        anchorParent: { x: 0, y: -shin / 2, z: 0 },
        anchorChild: { x: 0, y: footR * 0.4, z: 0 },
        axis: { x: 0, y: 0, z: 1 },
        min: limits.ankle![0],
        max: limits.ankle![1],
        gain: "ankle",
        poseKey: `ankle${leg.name}`
      }
    );
  }
  return finish("quad", parts, joints, "root", jump, crouch, stand, quad, limits);
}

function buildSpider(
  sx: number,
  sz: number,
  thigh: number,
  shin: number,
  torsoH: number,
  torsoW: number,
  segGene: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const legCount = segGene < 0.45 ? 6 : 8;
  const bodyR = 0.18 * torsoW + 0.14;
  const bodyH = 0.1 * torsoH + 0.07;
  const limbR = 0.045;
  const footR = 0.055;
  const stand = LIFT + footR + shin * 0.9 + thigh * 0.75 + bodyH;
  const rootY = stand;
  const parts: PartSpec[] = [
    { id: "root", shape: "cylinder", radius: bodyR, length: bodyH * 2, mass: 12, friction: 0.55, x: sx, y: rootY, z: sz, ccd: true, role: "core" },
    { id: "head", shape: "ball", radius: bodyR * 0.55, mass: 2, friction: 0.3, x: sx + bodyR * 0.85, y: rootY + bodyH * 0.4, z: sz, role: "head" }
  ];
  const joints: JointSpec[] = [
    {
      id: "neck",
      parent: "root",
      child: "head",
      anchorParent: { x: bodyR * 0.7, y: bodyH * 0.2, z: 0 },
      anchorChild: { x: -bodyR * 0.35, y: 0, z: 0 },
      axis: { x: 0, y: 0, z: 1 },
      min: -0.5,
      max: 0.5,
      gain: "spine",
      poseKey: "spine"
    }
  ];
  for (let i = 0; i < legCount; i++) {
    const ang = (i / legCount) * Math.PI * 2 + Math.PI / legCount;
    const lx = Math.cos(ang) * bodyR * 0.95;
    const lz = Math.sin(ang) * bodyR * 0.95;
    const thighId = id("thigh", i);
    const shinId = id("shin", i);
    const footId = id("foot", i);
    const out = 0.12;
    parts.push(
      { id: thighId, shape: "capsule", length: thigh, radius: limbR, mass: 1.6, friction: 0.35, x: sx + lx * 1.4, y: rootY - thigh * 0.35, z: sz + lz * 1.4, role: "limb" },
      { id: shinId, shape: "capsule", length: shin, radius: limbR * 0.85, mass: 1.1, friction: 0.35, x: sx + lx * 2.1, y: rootY - thigh * 0.7 - shin * 0.35, z: sz + lz * 2.1, ccd: true, role: "limb" },
      { id: footId, shape: "ball", radius: footR, mass: 0.5, friction: 1.7, x: sx + lx * 2.55, y: LIFT + footR, z: sz + lz * 2.55, ccd: true, contact: true, role: "foot" }
    );
    joints.push(
      {
        id: `hip${i}`,
        parent: "root",
        child: thighId,
        anchorParent: { x: lx, y: -bodyH * 0.2, z: lz },
        anchorChild: { x: 0, y: thigh / 2, z: 0 },
        axis: { x: -Math.sin(ang), y: 0, z: Math.cos(ang) },
        min: limits.hip![0],
        max: limits.hip![1],
        gain: "hip",
        poseKey: `hip${i}`
      },
      {
        id: `knee${i}`,
        parent: thighId,
        child: shinId,
        anchorParent: { x: 0, y: -thigh / 2, z: 0 },
        anchorChild: { x: 0, y: shin / 2, z: 0 },
        axis: { x: -Math.sin(ang), y: 0, z: Math.cos(ang) },
        min: limits.knee![0],
        max: limits.knee![1],
        gain: "knee",
        poseKey: `knee${i}`
      },
      {
        id: `ankle${i}`,
        parent: shinId,
        child: footId,
        anchorParent: { x: 0, y: -shin / 2, z: 0 },
        anchorChild: { x: 0, y: footR * 0.3, z: 0 },
        axis: { x: -Math.sin(ang), y: 0, z: Math.cos(ang) },
        min: limits.ankle![0],
        max: limits.ankle![1],
        gain: "ankle",
        poseKey: `ankle${i}`
      }
    );
    void out;
  }
  return finish("spider", parts, joints, "root", jump, crouch, stand, quad, limits);
}

function buildSnake(
  sx: number,
  sz: number,
  torsoH: number,
  torsoW: number,
  segGene: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const n = 5 + Math.round(segGene * 5);
  const r = 0.08 * torsoW + 0.07;
  const segLen = 0.22 * torsoH + 0.16;
  const stand = LIFT + r * 1.15;
  const parts: PartSpec[] = [];
  const joints: JointSpec[] = [];
  for (let i = 0; i < n; i++) {
    const x = sx - i * segLen * 0.92;
    parts.push({
      id: i === 0 ? "root" : `seg${i}`,
      shape: "capsule",
      length: segLen,
      radius: r * (i === 0 ? 1.15 : 1 - i * 0.04),
      mass: i === 0 ? 4 : 2.2,
      friction: 1.1,
      x,
      y: stand,
      z: sz,
      ccd: i === 0 || i === n - 1,
      contact: true,
      role: i === 0 ? "head" : "segment",
      qx: 0,
      qy: 0,
      qz: 0.7071,
      qw: 0.7071
    });
    if (i > 0) {
      const parent = i === 1 ? "root" : `seg${i - 1}`;
      const child = `seg${i}`;
      joints.push({
        id: `segj${i}`,
        parent,
        child,
        anchorParent: { x: -segLen / 2, y: 0, z: 0 },
        anchorChild: { x: segLen / 2, y: 0, z: 0 },
        axis: { x: 0, y: 1, z: 0 },
        min: limits.segment![0],
        max: limits.segment![1],
        gain: "segment",
        poseKey: `seg${i}`
      });
    }
  }
  return finish("snake", parts, joints, "root", jump, crouch * 0.2, stand, quad, limits);
}

function buildBlob(
  sx: number,
  sz: number,
  torsoH: number,
  torsoW: number,
  arm: number,
  segGene: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const coreR = 0.28 * torsoW + 0.18 * torsoH + 0.16;
  const lobeN = 3 + Math.round(segGene * 3);
  const lobeR = coreR * (0.35 + 0.15 * arm);
  const stand = LIFT + coreR * 0.85;
  const parts: PartSpec[] = [
    { id: "root", shape: "ball", radius: coreR, mass: 16, friction: 1.2, x: sx, y: stand, z: sz, ccd: true, contact: true, role: "blob" }
  ];
  const joints: JointSpec[] = [];
  for (let i = 0; i < lobeN; i++) {
    const ang = (i / lobeN) * Math.PI * 2;
    const elev = -0.25 + (i % 2) * 0.35;
    const dist = coreR + lobeR * 0.85;
    const lx = Math.cos(ang) * Math.cos(elev) * dist;
    const ly = Math.sin(elev) * dist * 0.7;
    const lz = Math.sin(ang) * Math.cos(elev) * dist;
    const idL = `lobe${i}`;
    parts.push({
      id: idL,
      shape: "ball",
      radius: lobeR * (0.85 + (i % 3) * 0.08),
      mass: 2.5,
      friction: 1.35,
      x: sx + lx,
      y: Math.max(LIFT + lobeR, stand + ly),
      z: sz + lz,
      contact: true,
      role: "blob"
    });
    joints.push({
      id: `blob${i}`,
      parent: "root",
      child: idL,
      anchorParent: { x: lx * 0.55, y: ly * 0.55, z: lz * 0.55 },
      anchorChild: { x: -lx * 0.25, y: -ly * 0.25, z: -lz * 0.25 },
      axis: { x: -Math.sin(ang), y: 0, z: Math.cos(ang) },
      min: limits.segment![0],
      max: limits.segment![1],
      gain: "segment",
      poseKey: `lobe${i}`
    });
  }
  return finish("blob", parts, joints, "root", jump, crouch * 0.15, stand, quad, limits);
}

function buildWheeler(
  sx: number,
  sz: number,
  thigh: number,
  torsoH: number,
  torsoW: number,
  segGene: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const wheelN = segGene < 0.4 ? 2 : segGene < 0.75 ? 3 : 4;
  const chassisL = 0.45 * torsoH + 0.35;
  const chassisH = 0.1 * torsoH + 0.07;
  const chassisW = 0.22 * torsoW + 0.14;
  const wheelR = 0.12 * torsoW + 0.1;
  const stub = Math.max(0.12, thigh * 0.45);
  const stand = LIFT + wheelR + stub * 0.35 + chassisH;
  const rootY = stand;
  const parts: PartSpec[] = [
    { id: "root", shape: "box", hx: chassisL / 2, hy: chassisH, hz: chassisW, mass: 14, friction: 0.45, x: sx, y: rootY, z: sz, ccd: true, role: "core" },
    { id: "cab", shape: "box", hx: chassisL * 0.22, hy: chassisH * 1.1, hz: chassisW * 0.7, mass: 4, friction: 0.35, x: sx + chassisL * 0.22, y: rootY + chassisH * 1.4, z: sz, role: "head" }
  ];
  const joints: JointSpec[] = [
    {
      id: "cabj",
      parent: "root",
      child: "cab",
      anchorParent: { x: chassisL * 0.15, y: chassisH, z: 0 },
      anchorChild: { x: 0, y: -chassisH * 1.1, z: 0 },
      axis: { x: 0, y: 0, z: 1 },
      min: -0.25,
      max: 0.35,
      gain: "spine",
      poseKey: "spine"
    }
  ];
  for (let i = 0; i < wheelN; i++) {
    const t = wheelN <= 1 ? 0.5 : i / (wheelN - 1);
    const wx = lerp(chassisL * 0.35, -chassisL * 0.35, t);
    const side = i % 2 === 0 ? -1 : 1;
    const wz = side * (chassisW + stub * 0.4);
    const stubId = `stub${i}`;
    const wheelId = `wheel${i}`;
    parts.push(
      { id: stubId, shape: "capsule", length: stub, radius: 0.04, mass: 1.2, friction: 0.3, x: sx + wx, y: rootY - stub * 0.15, z: sz + wz * 0.55, role: "limb" },
      {
        id: wheelId,
        shape: "cylinder",
        radius: wheelR,
        length: wheelR * 0.55,
        mass: 2.2,
        friction: 1.8,
        x: sx + wx,
        y: LIFT + wheelR,
        z: sz + wz,
        ccd: true,
        contact: true,
        role: "wheel",
        qx: 0.7071,
        qy: 0,
        qz: 0,
        qw: 0.7071
      }
    );
    joints.push(
      {
        id: `susp${i}`,
        parent: "root",
        child: stubId,
        anchorParent: { x: wx, y: -chassisH * 0.6, z: wz * 0.35 },
        anchorChild: { x: 0, y: stub / 2, z: 0 },
        axis: { x: 0, y: 0, z: 1 },
        min: limits.hip![0] * 0.45,
        max: limits.hip![1] * 0.45,
        gain: "hip",
        poseKey: `susp${i}`
      },
      {
        id: `axle${i}`,
        parent: stubId,
        child: wheelId,
        anchorParent: { x: 0, y: -stub / 2, z: 0 },
        anchorChild: { x: 0, y: 0, z: 0 },
        axis: { x: 0, y: 0, z: 1 },
        min: -Math.PI,
        max: Math.PI,
        gain: "ankle",
        poseKey: `wheel${i}`
      }
    );
  }
  return finish("wheeler", parts, joints, "root", jump, crouch * 0.2, stand, quad, limits);
}

function buildCentipede(
  sx: number,
  sz: number,
  thigh: number,
  shin: number,
  torsoH: number,
  torsoW: number,
  segGene: number,
  limits: Morph["limits"],
  jump: number,
  crouch: number,
  quad: number
): Morph {
  const n = 4 + Math.round(segGene * 4);
  const segLen = 0.2 * torsoH + 0.14;
  const segH = 0.07 * torsoH + 0.05;
  const segW = 0.12 * torsoW + 0.08;
  const limbR = 0.035;
  const footR = 0.045;
  const legLen = Math.max(0.14, thigh * 0.55);
  const shinLen = Math.max(0.12, shin * 0.5);
  const stand = LIFT + footR + shinLen + legLen * 0.7 + segH;
  const parts: PartSpec[] = [];
  const joints: JointSpec[] = [];
  for (let i = 0; i < n; i++) {
    const x = sx - i * segLen * 0.95;
    const sid = i === 0 ? "root" : `seg${i}`;
    parts.push({
      id: sid,
      shape: "box",
      hx: segLen / 2,
      hy: segH,
      hz: segW,
      mass: i === 0 ? 5 : 3.2,
      friction: 0.5,
      x,
      y: stand,
      z: sz,
      ccd: i === 0,
      role: i === 0 ? "head" : "segment"
    });
    if (i > 0) {
      const parent = i === 1 ? "root" : `seg${i - 1}`;
      joints.push({
        id: `link${i}`,
        parent,
        child: sid,
        anchorParent: { x: -segLen / 2, y: 0, z: 0 },
        anchorChild: { x: segLen / 2, y: 0, z: 0 },
        axis: { x: 0, y: 1, z: 0 },
        min: limits.segment![0] * 0.7,
        max: limits.segment![1] * 0.7,
        gain: "segment",
        poseKey: `link${i}`
      });
    }
    for (const side of [-1, 1] as const) {
      const tag = side < 0 ? "L" : "R";
      const thighId = `thigh${i}${tag}`;
      const shinId = `shin${i}${tag}`;
      const footId = `foot${i}${tag}`;
      const zOff = side * (segW + 0.02);
      parts.push(
        { id: thighId, shape: "capsule", length: legLen, radius: limbR, mass: 0.9, friction: 0.35, x, y: stand - legLen * 0.4, z: sz + zOff, role: "limb" },
        { id: shinId, shape: "capsule", length: shinLen, radius: limbR * 0.85, mass: 0.6, friction: 0.35, x, y: stand - legLen * 0.75 - shinLen * 0.35, z: sz + zOff * 1.15, role: "limb" },
        { id: footId, shape: "ball", radius: footR, mass: 0.35, friction: 1.55, x, y: LIFT + footR, z: sz + zOff * 1.25, ccd: true, contact: true, role: "foot" }
      );
      joints.push(
        {
          id: `hip${i}${tag}`,
          parent: sid,
          child: thighId,
          anchorParent: { x: 0, y: -segH * 0.4, z: zOff * 0.7 },
          anchorChild: { x: 0, y: legLen / 2, z: 0 },
          axis: { x: 0, y: 0, z: 1 },
          min: limits.hip![0],
          max: limits.hip![1],
          gain: "hip",
          poseKey: `hip${i}${tag}`
        },
        {
          id: `knee${i}${tag}`,
          parent: thighId,
          child: shinId,
          anchorParent: { x: 0, y: -legLen / 2, z: 0 },
          anchorChild: { x: 0, y: shinLen / 2, z: 0 },
          axis: { x: 0, y: 0, z: -1 },
          min: limits.knee![0],
          max: limits.knee![1],
          gain: "knee",
          poseKey: `knee${i}${tag}`
        },
        {
          id: `ankle${i}${tag}`,
          parent: shinId,
          child: footId,
          anchorParent: { x: 0, y: -shinLen / 2, z: 0 },
          anchorChild: { x: 0, y: footR * 0.3, z: 0 },
          axis: { x: 0, y: 0, z: 1 },
          min: limits.ankle![0],
          max: limits.ankle![1],
          gain: "ankle",
          poseKey: `ankle${i}${tag}`
        }
      );
    }
  }
  return finish("centipede", parts, joints, "root", jump, crouch, stand, quad, limits);
}

/** Re-apply flex/jump scales onto an existing morph's joint limits and jump. */
export function applyExpression(morph: Morph, hipFlex: number, kneeFlex: number, jump: number): void {
  const lim = flexLimits(hipFlex, kneeFlex);
  morph.limits = lim;
  morph.jump = Math.max(0, jump);
  for (const joint of morph.joints) {
    if (joint.gain === "hip") {
      joint.min = lim.hip![0];
      joint.max = lim.hip![1];
    } else if (joint.gain === "knee") {
      joint.min = lim.knee![0];
      joint.max = lim.knee![1];
    } else if (joint.gain === "ankle") {
      joint.min = lim.ankle![0];
      joint.max = lim.ankle![1];
    } else if (joint.gain === "segment") {
      joint.min = lim.segment![0];
      joint.max = lim.segment![1];
    } else if (joint.gain === "spine" && joint.poseKey === "spine") {
      joint.min = lim.spine![0];
      joint.max = lim.spine![1];
    } else if (joint.gain === "shoulder") {
      joint.min = lim.shoulder![0];
      joint.max = lim.shoulder![1];
    } else if (joint.gain === "elbow") {
      joint.min = lim.elbow![0];
      joint.max = lim.elbow![1];
    }
  }
}
