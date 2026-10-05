import RAPIER from "@dimforge/rapier3d-compat";
import {
  ARM_R,
  BODY_KEYS,
  FORE_R,
  FOOT,
  LIFT,
  LIMB_R,
  SHIN_R,
  crawlBlend,
  capsuleHalf,
  capsuleVolume,
  cuboidVolume,
  densityFor,
  type BodyKey,
  type Morph,
  type Pose
} from "./body";
import { FINISH_X, GEN_TIME, LANE_HALF, SOLIDS, spawnSlots } from "./course";
import {
  ankleCouple,
  breed,
  decode,
  fitness,
  initialPopulation,
  morphFromDecoded,
  mulberry32,
  jumpFrequency,
  setFlexScale,
  setJumpScale,
  writePose,
  type Decoded,
  type Rng
} from "./ga";

export const SIM_DT = 1 / 60;

const GROUND_GROUP = (0x0001 << 16) | 0x0002;
const ROBOT_GROUP = (0x0002 << 16) | 0x0001;

const GAIN = {
  spine: { k: 580, d: 62, max: 3200 },
  hip: { k: 1180, d: 105, max: 7800 },
  knee: { k: 1080, d: 98, max: 7200 },
  ankle: { k: 920, d: 82, max: 3600 },
  shoulder: { k: 520, d: 42, max: 3000 },
  elbow: { k: 440, d: 38, max: 2400 }
};

export interface BodyPose {
  x: number;
  y: number;
  z: number;
  qx: number;
  qy: number;
  qz: number;
  qw: number;
}

export interface RobotState {
  index: number;
  hue: number;
  genes: number[];
  spawnX: number;
  spawnZ: number;
  maxTravel: number;
  finished: boolean;
  finishTime: number;
  fallen: boolean;
  hidden: boolean;
  score: number;
  morph: Morph;
  bodies: Record<BodyKey, BodyPose>;
}

interface Hinge {
  joint: RAPIER.RevoluteImpulseJoint;
  min: number;
  max: number;
  k: number;
  d: number;
  pick: (pose: Pose) => number;
  limit: keyof Morph["limits"];
}

interface Robot {
  state: RobotState;
  decoded: Decoded;
  pose: Pose;
  hinges: Hinge[];
  rigid: RAPIER.RigidBody[];
  pelvis: RAPIER.RigidBody;
  fallTimer: number;
  motorsCut: boolean;
  lastX: number;
  upY: number;
  jumpCycle: number;
  jumpWindow: number;
  footL: RAPIER.Collider;
  footR: RAPIER.Collider;
  /** At least one foot collider has a contact with a solid. */
  grounded: boolean;
  /** True after a jump until the robot has left the ground and a foot lands again. */
  jumpSpent: boolean;
  /** Seen airborne since the last landing, so a spent jump can reset. */
  leftGround: boolean;
}

let ready: Promise<void> | null = null;
let nextSpawnSerial = 1;

function initPhysics(): Promise<void> {
  if (!ready) ready = RAPIER.init();
  return ready;
}

function emptyPose(): Record<BodyKey, BodyPose> {
  const bodies = {} as Record<BodyKey, BodyPose>;
  for (const key of BODY_KEYS) {
    bodies[key] = { x: 0, y: 0, z: 0, qx: 0, qy: 0, qz: 0, qw: 1 };
  }
  return bodies;
}

function hueOf(index: number): number {
  return (index * 0.618033988749895) % 1;
}

function upYOf(body: RAPIER.RigidBody): number {
  const q = body.rotation();
  return 1 - 2 * (q.x * q.x + q.z * q.z);
}

function zPitch(body: RAPIER.RigidBody): number {
  const q = body.rotation();
  return Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
}

function clampRange(angle: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, angle));
}

/** Stable 0..1 value. Not the breed rng, so a skipped hop does not change the next generation. */
function unitHash(index: number, cycle: number): number {
  let x = Math.imul(index + 1, 0x9e3779b1) ^ Math.imul(cycle + 1, 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 16), 0xc2b2ae35);
  x = Math.imul(x ^ (x >>> 13), 0x27d4eb2f);
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/**
 * Slider 0 still allows a rare hop. Slider 1 takes every grounded chance.
 * In between, the slider is the share of those chances.
 */
function jumpChanceRate(): number {
  if (jumpFrequency >= 1) return 1;
  if (jumpFrequency <= 0) return 0.02;
  return jumpFrequency;
}


interface Q {
  x: number;
  y: number;
  z: number;
  w: number;
}
interface V3 {
  x: number;
  y: number;
  z: number;
}
interface PartPose {
  x: number;
  y: number;
  z: number;
  q: Q;
}

function qZ(angle: number): Q {
  const h = angle * 0.5;
  return { x: 0, y: 0, z: Math.sin(h), w: Math.cos(h) };
}

function qMul(a: Q, b: Q): Q {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  };
}

function qRot(q: Q, p: V3): V3 {
  const ix = q.w * p.x + q.y * p.z - q.z * p.y;
  const iy = q.w * p.y + q.z * p.x - q.x * p.z;
  const iz = q.w * p.z + q.x * p.y - q.y * p.x;
  const iw = -q.x * p.x - q.y * p.y - q.z * p.z;
  return {
    x: ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
    y: iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
    z: iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x
  };
}

function vadd(a: V3, b: V3): V3 {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

/** Limb layout for a pitched crawl. Joint angles match the motor targets. */
function crawlLayout(morph: Morph, pose: Pose, pitch: number): {
  parts: Record<BodyKey, V3>;
  rots: Record<BodyKey, Q>;
  soleL: number;
  soleR: number;
  handL: number;
  handR: number;
  footXL: number;
  footXR: number;
  handXL: number;
  handXR: number;
  headY: number;
} {
  const qPelvis = qZ(pitch);
  const qChest = qMul(qPelvis, qZ(pose.spine));
  const qThighL = qMul(qPelvis, qZ(pose.hipL));
  const qThighR = qMul(qPelvis, qZ(pose.hipR));
  const qShinL = qMul(qThighL, qZ(-pose.kneeL));
  const qShinR = qMul(qThighR, qZ(-pose.kneeR));
  const qFootL = qMul(qShinL, qZ(pose.ankleL));
  const qFootR = qMul(qShinR, qZ(pose.ankleR));
  const qArmL = qMul(qChest, qZ(pose.shoulderL));
  const qArmR = qMul(qChest, qZ(pose.shoulderR));
  const qForeL = qMul(qArmL, qZ(pose.elbowL));
  const qForeR = qMul(qArmR, qZ(pose.elbowR));
  const origin: V3 = { x: 0, y: 0, z: 0 };
  const hipL = vadd(origin, qRot(qPelvis, { x: 0, y: -morph.hipDrop, z: -morph.hipZ }));
  const hipR = vadd(origin, qRot(qPelvis, { x: 0, y: -morph.hipDrop, z: morph.hipZ }));
  const kneeL = vadd(hipL, qRot(qThighL, { x: 0, y: -morph.thigh, z: 0 }));
  const kneeR = vadd(hipR, qRot(qThighR, { x: 0, y: -morph.thigh, z: 0 }));
  const ankleL = vadd(kneeL, qRot(qShinL, { x: 0, y: -morph.shin, z: 0 }));
  const ankleR = vadd(kneeR, qRot(qShinR, { x: 0, y: -morph.shin, z: 0 }));
  const footL = vadd(ankleL, qRot(qFootL, { x: FOOT.heel, y: -FOOT.hy, z: 0 }));
  const footR = vadd(ankleR, qRot(qFootR, { x: FOOT.heel, y: -FOOT.hy, z: 0 }));
  const soleL = vadd(footL, qRot(qFootL, { x: 0, y: -FOOT.hy, z: 0 }));
  const soleR = vadd(footR, qRot(qFootR, { x: 0, y: -FOOT.hy, z: 0 }));
  const chest = vadd(
    vadd(origin, qRot(qPelvis, { x: 0, y: morph.pelvis.hy, z: 0 })),
    qRot(qChest, { x: 0, y: morph.chest.hy, z: 0 })
  );
  const shL = vadd(chest, qRot(qChest, { x: 0, y: morph.shoulderY, z: -morph.shoulderZ }));
  const shR = vadd(chest, qRot(qChest, { x: 0, y: morph.shoulderY, z: morph.shoulderZ }));
  const elbL = vadd(shL, qRot(qArmL, { x: 0, y: -morph.upperArm, z: 0 }));
  const elbR = vadd(shR, qRot(qArmR, { x: 0, y: -morph.upperArm, z: 0 }));
  const foreL = vadd(elbL, qRot(qForeL, { x: 0, y: -morph.forearm / 2, z: 0 }));
  const foreR = vadd(elbR, qRot(qForeR, { x: 0, y: -morph.forearm / 2, z: 0 }));
  const handL = vadd(elbL, qRot(qForeL, { x: 0, y: -morph.forearm - FORE_R * 0.35, z: 0 }));
  const handR = vadd(elbR, qRot(qForeR, { x: 0, y: -morph.forearm - FORE_R * 0.35, z: 0 }));
  const head = vadd(chest, qRot(qChest, { x: 0, y: morph.chest.hy + morph.headR * 1.3, z: 0 }));
  const thighL = vadd(hipL, qRot(qThighL, { x: 0, y: -morph.thigh / 2, z: 0 }));
  const thighR = vadd(hipR, qRot(qThighR, { x: 0, y: -morph.thigh / 2, z: 0 }));
  const shinL = vadd(kneeL, qRot(qShinL, { x: 0, y: -morph.shin / 2, z: 0 }));
  const shinR = vadd(kneeR, qRot(qShinR, { x: 0, y: -morph.shin / 2, z: 0 }));
  const parts: Record<BodyKey, V3> = {
    pelvis: origin,
    chest,
    thighL,
    shinL,
    footL,
    thighR,
    shinR,
    footR,
    armL: vadd(shL, qRot(qArmL, { x: 0, y: -morph.upperArm / 2, z: 0 })),
    foreL,
    armR: vadd(shR, qRot(qArmR, { x: 0, y: -morph.upperArm / 2, z: 0 })),
    foreR
  };
  // thigh centers computed for clarity; arm centers already stored.
  parts.thighL = thighL;
  parts.thighR = thighR;
  parts.shinL = shinL;
  parts.shinR = shinR;
  const rots: Record<BodyKey, Q> = {
    pelvis: qPelvis,
    chest: qChest,
    thighL: qThighL,
    shinL: qShinL,
    footL: qFootL,
    thighR: qThighR,
    shinR: qShinR,
    footR: qFootR,
    armL: qArmL,
    foreL: qForeL,
    armR: qArmR,
    foreR: qForeR
  };
  return {
    parts,
    rots,
    soleL: soleL.y,
    soleR: soleR.y,
    handL: handL.y,
    handR: handR.y,
    footXL: soleL.x,
    footXR: soleR.x,
    handXL: handL.x,
    handXR: handR.x,
    headY: head.y
  };
}

function fitCrawlPitch(morph: Morph, pose: Pose, crawl: number): number {
  const prefer = -1.12 * crawl;
  let best = prefer;
  let bestScore = Infinity;
  for (let i = 0; i <= 46; i++) {
    const alpha = -0.4 - (1.1 * i) / 46;
    const frame = crawlLayout(morph, pose, alpha);
    const gap =
      Math.abs(frame.handL - frame.soleL) + Math.abs(frame.handR - frame.soleR);
    const front = (frame.handXL < 0.04 ? 0.55 : 0) + (frame.handXR < 0.04 ? 0.55 : 0);
    const back = (frame.footXL > 0.06 ? 0.4 : 0) + (frame.footXR > 0.06 ? 0.4 : 0);
    const span = (frame.handXL + frame.handXR) * 0.5 - (frame.footXL + frame.footXR) * 0.5;
    const spanPen = span < 0.32 ? 0.65 : span > 1.6 ? 0.3 : 0;
    const head = frame.headY < 0.18 ? 0.9 : 0;
    const score = gap + front + back + spanPen + head + 0.2 * Math.abs(alpha - prefer);
    if (score < bestScore) {
      bestScore = score;
      best = alpha;
    }
  }
  return best;
}

function placeCrawl(
  morph: Morph,
  pose: Pose,
  crawl: number,
  spawnX: number,
  spawnZ: number
): Record<BodyKey, PartPose> {
  const pitch = fitCrawlPitch(morph, pose, crawl);
  const frame = crawlLayout(morph, pose, pitch);
  const low = Math.min(frame.soleL, frame.soleR, frame.handL, frame.handR);
  const shift = LIFT + 0.02 - low;
  const out = {} as Record<BodyKey, PartPose>;
  for (const key of BODY_KEYS) {
    const part = frame.parts[key];
    out[key] = {
      x: part.x + spawnX,
      y: part.y + shift,
      z: part.z + spawnZ,
      q: frame.rots[key]
    };
  }
  return out;
}

export class CourseSim {
  readonly world: RAPIER.World;
  readonly robots: RobotState[] = [];
  generation = 1;
  stepCount = 0;
  readonly maxSteps: number;
  time = 0;
  pop: number;

  private rng: Rng;
  private internals: Robot[] = [];
  private disposed = false;
  spawnSerial = 0;

  private constructor(pop: number, seed: number, genomes: number[][] | undefined, maxSteps: number) {
    this.pop = pop;
    this.maxSteps = maxSteps;
    this.rng = mulberry32(seed);
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    this.world.timestep = SIM_DT;
    this.world.integrationParameters.numSolverIterations = 10;
    this.world.integrationParameters.numInternalPgsIterations = 2;
    this.addCourse();
    const first = genomes ?? initialPopulation(pop, this.rng);
    this.spawnAll(first);
  }

  static async create(pop: number, seed: number, genomes?: number[][]): Promise<CourseSim> {
    await initPhysics();
    return new CourseSim(pop, seed, genomes, Math.round(GEN_TIME / SIM_DT));
  }

  get complete(): boolean {
    return this.stepCount >= this.maxSteps;
  }

  free(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.world.free();
  }

  best(): RobotState {
    let best = this.robots[0];
    for (const robot of this.robots) {
      if (!best || robot.score > best.score) best = robot;
    }
    if (!best) {
      throw new Error("Population is empty.");
    }
    return best;
  }

  bestFinishTime(): number | null {
    let best: number | null = null;
    for (const robot of this.robots) {
      if (!robot.finished) continue;
      if (best === null || robot.finishTime < best) best = robot.finishTime;
    }
    return best;
  }

  aliveCount(): number {
    let n = 0;
    for (const robot of this.robots) if (!robot.fallen) n++;
    return n;
  }

  step(): void {
    if (this.complete || this.disposed) return;
    this.applyMotors();
    this.world.step();
    this.stepCount += 1;
    this.time = this.stepCount * SIM_DT;
    this.collect();
  }

  /**
   * Slider scales. Genes stay stored. Limits and jump strength are rebuilt
   * for the robots already on the course, so the current generation picks it up.
   */
  setExpression(flexibility: number, jump: number): void {
    setFlexScale(flexibility);
    setJumpScale(jump);
    for (const robot of this.internals) {
      robot.state.morph = morphFromDecoded(robot.decoded);
      for (const hinge of robot.hinges) {
        const pair = robot.state.morph.limits[hinge.limit];
        hinge.min = pair[0];
        hinge.max = pair[1];
        hinge.joint.setLimits(pair[0], pair[1]);
      }
    }
  }

  /**
   * Rank by fitness, breed (elites keep full genomes; children inherit body genes),
   * then rebuild every robot from its own genome — not a shared morph.
   */
  nextGeneration(pop = this.pop): void {
    const ranked = [...this.internals].sort(
      (a, b) => b.state.score - a.state.score || b.state.maxTravel - a.state.maxTravel
    );
    const genomes = breed(
      ranked.map((robot) => robot.state.genes),
      pop,
      this.rng
    );
    this.clearRobots();
    this.pop = pop;
    this.generation += 1;
    this.stepCount = 0;
    this.time = 0;
    this.spawnAll(genomes);
  }

  private addCourse(): void {
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    for (const solid of SOLIDS) {
      const desc =
        solid.shape === "box"
          ? RAPIER.ColliderDesc.cuboid(solid.hx, solid.hy, solid.hz)
          : RAPIER.ColliderDesc.cylinder(solid.hy, solid.hx);
      desc
        .setTranslation(solid.x, solid.y, solid.z)
        .setRotation({ x: solid.qx, y: solid.qy, z: solid.qz, w: solid.qw })
        .setFriction(solid.friction)
        .setRestitution(0.02)
        .setCollisionGroups(GROUND_GROUP);
      this.world.createCollider(desc, body);
    }
  }

  private clearRobots(): void {
    for (const robot of this.internals) {
      for (const hinge of robot.hinges) this.world.removeImpulseJoint(hinge.joint, false);
      for (const body of robot.rigid) this.world.removeRigidBody(body);
    }
    this.internals = [];
    this.robots.length = 0;
  }

  private spawnAll(genomes: number[][]): void {
    this.spawnSerial = nextSpawnSerial++;
    const slots = spawnSlots(genomes.length);
    genomes.forEach((genes, index) => {
      const slot = slots[index] ?? { x: 0.4, z: 0 };
      this.internals.push(this.spawnRobot(index, genes, slot.x, slot.z));
    });
    for (const robot of this.internals) this.robots.push(robot.state);
    this.capturePoses();
    this.scoreAll();
  }

  /** Physics and colliders for one genome. Morph sizes come only from these genes. */
  private spawnRobot(index: number, genes: number[], spawnX: number, spawnZ: number): Robot {
    const decoded = decode(genes);
    const morph = morphFromDecoded(decoded);
    const pelvisY = morph.pelvisStand;
    const chestY = pelvisY + morph.pelvis.hy + morph.chest.hy;
    const pose0: Pose = {
      spine: 0,
      hipL: 0,
      hipR: 0,
      kneeL: 0,
      kneeR: 0,
      ankleL: 0,
      ankleR: 0,
      shoulderL: 0,
      shoulderR: 0,
      elbowL: 0,
      elbowR: 0
    };
    writePose(decoded, 0, pose0, morph.limits, morph.crouch);
    const crawl = crawlBlend(decoded.quad ?? 0);
    const posed = crawl > 0.02 ? placeCrawl(morph, pose0, crawl, spawnX, spawnZ) : null;
    const make = (
      key: BodyKey,
      x: number,
      y: number,
      z: number,
      ccd: boolean
    ): RAPIER.RigidBody => {
      const part = posed?.[key];
      const desc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(part?.x ?? x, part?.y ?? y, part?.z ?? z)
        .setCanSleep(false)
        .setLinearDamping(0.05)
        .setAngularDamping(1.25)
        .setCcdEnabled(ccd);
      if (part) desc.setRotation(part.q);
      return this.world.createRigidBody(desc);
    };

    const pelvis = make("pelvis", spawnX, pelvisY, spawnZ, true);
    const chest = make("chest", spawnX, chestY, spawnZ, false);

    const hipY = pelvisY - morph.hipDrop;
    const thighY = hipY - morph.thigh / 2;
    const kneeY = hipY - morph.thigh;
    const shinY = kneeY - morph.shin / 2;
    const ankleY = kneeY - morph.shin;
    const footY = ankleY - FOOT.hy;

    const thighL = make("thighL", spawnX, thighY, spawnZ - morph.hipZ, false);
    const thighR = make("thighR", spawnX, thighY, spawnZ + morph.hipZ, false);
    const shinL = make("shinL", spawnX, shinY, spawnZ - morph.hipZ, true);
    const shinR = make("shinR", spawnX, shinY, spawnZ + morph.hipZ, true);
    const footL = make("footL", spawnX + FOOT.heel, footY, spawnZ - morph.hipZ, true);
    const footR = make("footR", spawnX + FOOT.heel, footY, spawnZ + morph.hipZ, true);

    const shoulderY = chestY + morph.shoulderY;
    const armY = shoulderY - morph.upperArm / 2;
    const elbowY = shoulderY - morph.upperArm;
    const foreY = elbowY - morph.forearm / 2;
    const armL = make("armL", spawnX, armY, spawnZ - morph.shoulderZ, false);
    const armR = make("armR", spawnX, armY, spawnZ + morph.shoulderZ, false);
    const foreL = make("foreL", spawnX, foreY, spawnZ - morph.shoulderZ, false);
    const foreR = make("foreR", spawnX, foreY, spawnZ + morph.shoulderZ, false);

    const collide = (
      body: RAPIER.RigidBody,
      desc: RAPIER.ColliderDesc,
      friction: number
    ): RAPIER.Collider => {
      desc.setFriction(friction).setRestitution(0).setCollisionGroups(ROBOT_GROUP);
      return this.world.createCollider(desc, body);
    };

    collide(
      pelvis,
      RAPIER.ColliderDesc.cuboid(morph.pelvis.hx, morph.pelvis.hy, morph.pelvis.hz).setDensity(
        densityFor(cuboidVolume(morph.pelvis.hx, morph.pelvis.hy, morph.pelvis.hz), 7)
      ),
      0.45
    );
    collide(
      chest,
      RAPIER.ColliderDesc.cuboid(morph.chest.hx, morph.chest.hy, morph.chest.hz).setDensity(
        densityFor(cuboidVolume(morph.chest.hx, morph.chest.hy, morph.chest.hz), 9)
      ),
      0.4
    );
    collide(
      chest,
      RAPIER.ColliderDesc.ball(morph.headR)
        .setTranslation(0, morph.chest.hy + morph.headR * 0.55, 0)
        .setDensity(densityFor((4 / 3) * Math.PI * morph.headR ** 3, 3.2)),
      0.35
    );

    const limb = (length: number, radius: number, mass: number, friction: number): RAPIER.ColliderDesc =>
      RAPIER.ColliderDesc.capsule(capsuleHalf(length, radius), radius).setDensity(
        densityFor(capsuleVolume(length, radius), mass)
      ).setFriction(friction);

    collide(thighL, limb(morph.thigh, LIMB_R, 2.8, 0.4), 0.4);
    collide(thighR, limb(morph.thigh, LIMB_R, 2.8, 0.4), 0.4);
    collide(shinL, limb(morph.shin, SHIN_R, 1.8, 0.4), 0.4);
    collide(shinR, limb(morph.shin, SHIN_R, 1.8, 0.4), 0.4);
    collide(armL, limb(morph.upperArm, ARM_R, 1.0, 0.3), 0.3);
    collide(armR, limb(morph.upperArm, ARM_R, 1.0, 0.3), 0.3);
    const handFriction = morph.quad >= 0.5 ? 1.2 : 0.3;
    collide(foreL, limb(morph.forearm, FORE_R, 0.7, handFriction), handFriction);
    collide(foreR, limb(morph.forearm, FORE_R, 0.7, handFriction), handFriction);

    const footDesc = (): RAPIER.ColliderDesc =>
      RAPIER.ColliderDesc.cuboid(FOOT.hx, FOOT.hy, FOOT.hz)
        .setDensity(densityFor(cuboidVolume(FOOT.hx, FOOT.hy, FOOT.hz), 1.6))
        .setFriction(1.5)
        .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max);
    const footColL = collide(footL, footDesc(), 1.55);
    const footColR = collide(footR, footDesc(), 1.55);

    const hinges: Hinge[] = [];
    const hinge = (
      a: RAPIER.RigidBody,
      b: RAPIER.RigidBody,
      anchorA: { x: number; y: number; z: number },
      anchorB: { x: number; y: number; z: number },
      axis: { x: number; y: number; z: number },
      min: number,
      max: number,
      gain: { k: number; d: number; max: number },
      pick: (pose: Pose) => number,
      limit: keyof Morph["limits"]
    ): void => {
      const data = RAPIER.JointData.revolute(anchorA, anchorB, axis);
      const joint = this.world.createImpulseJoint(data, a, b, true) as RAPIER.RevoluteImpulseJoint;
      joint.setLimits(min, max);
      joint.configureMotorModel(RAPIER.MotorModel.ForceBased);
      joint.setMotorMaxForce(gain.max);
      joint.setContactsEnabled(false);
      hinges.push({ joint, min, max, k: gain.k, d: gain.d, pick, limit });
    };

    const zAxis = { x: 0, y: 0, z: 1 };
    const kneeAxis = { x: 0, y: 0, z: -1 };

    hinge(
      pelvis,
      chest,
      { x: 0, y: morph.pelvis.hy, z: 0 },
      { x: 0, y: -morph.chest.hy, z: 0 },
      zAxis,
      morph.limits.spine[0],
      morph.limits.spine[1],
      GAIN.spine,
      (pose) => pose.spine,
      "spine"
    );

    const hip = (side: -1 | 1, thigh: RAPIER.RigidBody, pick: (pose: Pose) => number): void => {
      hinge(
        pelvis,
        thigh,
        { x: 0, y: -morph.hipDrop, z: side * morph.hipZ },
        { x: 0, y: morph.thigh / 2, z: 0 },
        zAxis,
        morph.limits.hip[0],
        morph.limits.hip[1],
        GAIN.hip,
        pick,
        "hip"
      );
    };
    hip(-1, thighL, (pose) => pose.hipL);
    hip(1, thighR, (pose) => pose.hipR);

    const knee = (thigh: RAPIER.RigidBody, shin: RAPIER.RigidBody, pick: (pose: Pose) => number): void => {
      hinge(
        thigh,
        shin,
        { x: 0, y: -morph.thigh / 2, z: 0 },
        { x: 0, y: morph.shin / 2, z: 0 },
        kneeAxis,
        morph.limits.knee[0],
        morph.limits.knee[1],
        GAIN.knee,
        pick,
        "knee"
      );
    };
    knee(thighL, shinL, (pose) => pose.kneeL);
    knee(thighR, shinR, (pose) => pose.kneeR);

    const ankle = (shin: RAPIER.RigidBody, foot: RAPIER.RigidBody, pick: (pose: Pose) => number): void => {
      hinge(
        shin,
        foot,
        { x: 0, y: -morph.shin / 2, z: 0 },
        { x: -FOOT.heel, y: FOOT.hy, z: 0 },
        zAxis,
        morph.limits.ankle[0],
        morph.limits.ankle[1],
        GAIN.ankle,
        pick,
        "ankle"
      );
    };
    ankle(shinL, footL, (pose) => pose.ankleL);
    ankle(shinR, footR, (pose) => pose.ankleR);

    const shoulder = (side: -1 | 1, arm: RAPIER.RigidBody, pick: (pose: Pose) => number): void => {
      hinge(
        chest,
        arm,
        { x: 0, y: morph.shoulderY, z: side * morph.shoulderZ },
        { x: 0, y: morph.upperArm / 2, z: 0 },
        zAxis,
        morph.limits.shoulder[0],
        morph.limits.shoulder[1],
        GAIN.shoulder,
        pick,
        "shoulder"
      );
    };
    shoulder(-1, armL, (pose) => pose.shoulderL);
    shoulder(1, armR, (pose) => pose.shoulderR);

    const elbow = (arm: RAPIER.RigidBody, fore: RAPIER.RigidBody, pick: (pose: Pose) => number): void => {
      hinge(
        arm,
        fore,
        { x: 0, y: -morph.upperArm / 2, z: 0 },
        { x: 0, y: morph.forearm / 2, z: 0 },
        zAxis,
        morph.limits.elbow[0],
        morph.limits.elbow[1],
        GAIN.elbow,
        pick,
        "elbow"
      );
    };
    elbow(armL, foreL, (pose) => pose.elbowL);
    elbow(armR, foreR, (pose) => pose.elbowR);

    const rigid = [pelvis, chest, thighL, shinL, footL, thighR, shinR, footR, armL, foreL, armR, foreR];
    const state: RobotState = {
      index,
      hue: hueOf(index),
      genes: genes.slice(),
      spawnX,
      spawnZ,
      maxTravel: 0,
      finished: false,
      finishTime: 0,
      fallen: false,
      hidden: false,
      score: 0,
      morph,
      bodies: emptyPose()
    };
    return {
      state,
      decoded,
      pose: {
        spine: 0,
        hipL: 0,
        hipR: 0,
        kneeL: 0,
        kneeR: 0,
        ankleL: 0,
        ankleR: 0,
        shoulderL: 0,
        shoulderR: 0,
        elbowL: 0,
        elbowR: 0
      },
      hinges,
      rigid,
      pelvis,
      fallTimer: 0,
      motorsCut: false,
      lastX: spawnX,
      upY: 1,
      jumpCycle: 0,
      jumpWindow: 0,
      footL: footColL,
      footR: footColR,
      grounded: false,
      jumpSpent: false,
      leftGround: false
    };
  }

  private applyMotors(): void {
    for (const robot of this.internals) {
      if (robot.state.hidden) continue;
      if (robot.state.fallen) {
        if (!robot.motorsCut) {
          for (const hinge of robot.hinges) hinge.joint.configureMotorVelocity(0, 2.5);
          robot.motorsCut = true;
        }
        continue;
      }
      const limits = robot.state.morph.limits;
      writePose(robot.decoded, this.time, robot.pose, limits, robot.state.morph.crouch);
      this.applyJump(robot);
      const shinL = robot.rigid[3];
      const shinR = robot.rigid[6];
      if (shinL && shinR) {
        robot.pose.ankleL = clampRange(zPitch(shinL) * ankleCouple + robot.pose.ankleL, limits.ankle[0], limits.ankle[1]);
        robot.pose.ankleR = clampRange(zPitch(shinR) * ankleCouple + robot.pose.ankleR, limits.ankle[0], limits.ankle[1]);
      }
      for (const hinge of robot.hinges) {
        const target = Math.max(hinge.min, Math.min(hinge.max, hinge.pick(robot.pose)));
        hinge.joint.configureMotorPosition(target, hinge.k, hinge.d);
      }
    }
  }

  /** True when this foot collider has at least one contact point with a solid. */
  private footOnSolid(foot: RAPIER.Collider): boolean {
    let touching = false;
    this.world.contactPairsWith(foot, (other) => {
      if (touching) return;
      this.world.contactPair(foot, other, (manifold) => {
        if (manifold.numContacts() > 0) touching = true;
      });
    });
    return touching;
  }

  /** Grounded flag from the contact pairs of the last physics step. */
  private updateGrounded(robot: Robot): void {
    const grounded = this.footOnSolid(robot.footL) || this.footOnSolid(robot.footR);
    if (!grounded) robot.leftGround = true;
    else if (robot.leftGround) {
      robot.jumpSpent = false;
      robot.leftGround = false;
    }
    robot.grounded = grounded;
  }

  private applyJump(robot: Robot): void {
    const amp = robot.state.morph.jump;
    if (amp <= 0.02 || robot.state.fallen || robot.state.hidden) {
      robot.jumpWindow = 0;
      return;
    }
    const gait = Math.max(0.2, robot.decoded.freq ?? 4.4);
    // Gait is about twice as fast as before. Keep hop chances near the old ceiling
    // so the Jump frequency slider still means the same share of grounded chances.
    const cycle = Math.floor(this.time * Math.min(gait, 2.6));
    // Impulse only with a foot on a solid, and only once until they leave and land.
    // Jump frequency keeps some of those chances. 1 takes all of them.
    if (this.time > 0.2 && cycle !== robot.jumpCycle && robot.grounded && !robot.jumpSpent) {
      robot.jumpCycle = cycle;
      const rate = jumpChanceRate();
      const take = rate >= 1 || unitHash(robot.state.index, cycle) < rate;
      if (take) {
        robot.jumpSpent = true;
        robot.jumpWindow = 0.1;
        const vy = amp * 3.4;
        const vx = amp * 0.45;
        for (const body of robot.rigid) {
          const mass = body.mass();
          body.applyImpulse({ x: vx * mass, y: vy * mass, z: 0 }, true);
        }
      }
    }
    if (robot.jumpWindow > 0) {
      if (!robot.grounded) {
        robot.jumpWindow = 0;
      } else {
        const burst = amp * (robot.jumpWindow / 0.1);
        const knee = robot.state.morph.limits.knee;
        const hip = robot.state.morph.limits.hip;
        robot.pose.kneeL = Math.max(knee[0], robot.pose.kneeL - burst);
        robot.pose.kneeR = Math.max(knee[0], robot.pose.kneeR - burst);
        robot.pose.hipL = Math.max(hip[0], robot.pose.hipL - burst * 0.7);
        robot.pose.hipR = Math.max(hip[0], robot.pose.hipR - burst * 0.7);
        robot.jumpWindow -= SIM_DT;
      }
    }
  }

  private collect(): void {
    for (const robot of this.internals) {
      this.updateGrounded(robot);
      this.copyBodies(robot);
      if (robot.state.hidden) continue;
      const pelvis = robot.state.bodies.pelvis;
      const upY = upYOf(robot.pelvis);
      robot.upY = upY;
      const span = Math.max(0.5, FINISH_X - robot.state.spawnX);
      const stand = robot.state.morph.pelvisStand;
      const crouch = robot.state.morph.crouch;
      const yFall = stand * (0.34 - 0.2 * crouch);
      const yScore = stand * (crouch > 0.4 ? 0.16 : 0.46);
      const upCut = 0.38 - 0.34 * crouch;
      const offCourse =
        Math.abs(pelvis.z) > LANE_HALF + 0.08 && pelvis.y < stand * 0.5;
      const tipped = upY < upCut || pelvis.y < yFall || offCourse;
      if (!robot.state.fallen && !robot.state.finished) {
        const travel = pelvis.x - robot.state.spawnX;
        const dx = pelvis.x - robot.lastX;
        if (dx < 0.85 && pelvis.y < 2.6 && pelvis.y > yScore && Math.abs(pelvis.z) < LANE_HALF + 0.15) {
          if (travel > robot.state.maxTravel) robot.state.maxTravel = Math.min(span, travel);
        }
        if (
          pelvis.x >= FINISH_X &&
          pelvis.y > stand * 0.56 &&
          upY > 0.55 &&
          Math.abs(pelvis.z) < LANE_HALF - 0.15
        ) {
          robot.state.finished = true;
          robot.state.finishTime = this.time;
          robot.state.maxTravel = span;
        }
      }
      robot.lastX = pelvis.x;
      if (tipped) robot.fallTimer += SIM_DT;
      else robot.fallTimer = 0;
      if (!robot.state.fallen && robot.fallTimer > 0.4) robot.state.fallen = true;
      if (robot.state.fallen && (robot.fallTimer > 2.35 || pelvis.y > 5 || pelvis.y < -3)) {
        robot.state.hidden = true;
        for (const body of robot.rigid) body.setEnabled(false);
      }
    }
    this.scoreAll();
  }

  private scoreAll(): void {
    for (const robot of this.internals) {
      const span = Math.max(0.5, FINISH_X - robot.state.spawnX);
      const progress = robot.state.maxTravel / span;
      robot.state.score = fitness(progress, robot.state.finished, robot.state.finishTime);
    }
  }

  private capturePoses(): void {
    for (const robot of this.internals) this.copyBodies(robot);
  }

  private copyBodies(robot: Robot): void {
    for (let i = 0; i < BODY_KEYS.length; i++) {
      const key = BODY_KEYS[i];
      const body = robot.rigid[i];
      if (!key || !body) continue;
      const p = body.translation();
      const q = body.rotation();
      const slot = robot.state.bodies[key];
      slot.x = p.x;
      slot.y = p.y;
      slot.z = p.z;
      slot.qx = q.x;
      slot.qy = q.y;
      slot.qz = q.z;
      slot.qw = q.w;
    }
  }
}

