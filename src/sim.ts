import RAPIER from "@dimforge/rapier3d-compat";
import {
  ARM_R,
  BODY_KEYS,
  FORE_R,
  FOOT,
  LIMB_R,
  SHIN_R,
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
  writePose,
  type Decoded,
  type Rng
} from "./ga";

export const SIM_DT = 1 / 60;

const GROUND_GROUP = (0x0001 << 16) | 0x0002;
const ROBOT_GROUP = (0x0002 << 16) | 0x0001;

const GAIN = {
  spine: { k: 420, d: 55, max: 2500 },
  hip: { k: 520, d: 70, max: 3500 },
  knee: { k: 480, d: 65, max: 3200 },
  ankle: { k: 760, d: 70, max: 2800 },
  shoulder: { k: 160, d: 18, max: 800 },
  elbow: { k: 120, d: 14, max: 600 }
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

  private spawnRobot(index: number, genes: number[], spawnX: number, spawnZ: number): Robot {
    const decoded = decode(genes);
    const morph = morphFromDecoded(decoded);
    const pelvisY = morph.pelvisStand;
    const chestY = pelvisY + morph.pelvis.hy + morph.chest.hy;
    const make = (
      x: number,
      y: number,
      z: number,
      ccd: boolean
    ): RAPIER.RigidBody => {
      const desc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(x, y, z)
        .setCanSleep(false)
        .setLinearDamping(0.05)
        .setAngularDamping(2.2)
        .setCcdEnabled(ccd);
      return this.world.createRigidBody(desc);
    };

    const pelvis = make(spawnX, pelvisY, spawnZ, true);
    const chest = make(spawnX, chestY, spawnZ, false);

    const hipY = pelvisY - morph.hipDrop;
    const thighY = hipY - morph.thigh / 2;
    const kneeY = hipY - morph.thigh;
    const shinY = kneeY - morph.shin / 2;
    const ankleY = kneeY - morph.shin;
    const footY = ankleY - FOOT.hy;

    const thighL = make(spawnX, thighY, spawnZ - morph.hipZ, false);
    const thighR = make(spawnX, thighY, spawnZ + morph.hipZ, false);
    const shinL = make(spawnX, shinY, spawnZ - morph.hipZ, true);
    const shinR = make(spawnX, shinY, spawnZ + morph.hipZ, true);
    const footL = make(spawnX + FOOT.heel, footY, spawnZ - morph.hipZ, true);
    const footR = make(spawnX + FOOT.heel, footY, spawnZ + morph.hipZ, true);

    const shoulderY = chestY + morph.shoulderY;
    const armY = shoulderY - morph.upperArm / 2;
    const elbowY = shoulderY - morph.upperArm;
    const foreY = elbowY - morph.forearm / 2;
    const armL = make(spawnX, armY, spawnZ - morph.shoulderZ, false);
    const armR = make(spawnX, armY, spawnZ + morph.shoulderZ, false);
    const foreL = make(spawnX, foreY, spawnZ - morph.shoulderZ, false);
    const foreR = make(spawnX, foreY, spawnZ + morph.shoulderZ, false);

    const collide = (
      body: RAPIER.RigidBody,
      desc: RAPIER.ColliderDesc,
      friction: number
    ): void => {
      desc.setFriction(friction).setRestitution(0).setCollisionGroups(ROBOT_GROUP);
      this.world.createCollider(desc, body);
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
    collide(foreL, limb(morph.forearm, FORE_R, 0.7, 0.3), 0.3);
    collide(foreR, limb(morph.forearm, FORE_R, 0.7, 0.3), 0.3);

    const footDesc = (): RAPIER.ColliderDesc =>
      RAPIER.ColliderDesc.cuboid(FOOT.hx, FOOT.hy, FOOT.hz)
        .setDensity(densityFor(cuboidVolume(FOOT.hx, FOOT.hy, FOOT.hz), 1.6))
        .setFriction(1.5)
        .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max);
    collide(footL, footDesc(), 1.55);
    collide(footR, footDesc(), 1.55);

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
      pick: (pose: Pose) => number
    ): void => {
      const data = RAPIER.JointData.revolute(anchorA, anchorB, axis);
      const joint = this.world.createImpulseJoint(data, a, b, true) as RAPIER.RevoluteImpulseJoint;
      joint.setLimits(min, max);
      joint.configureMotorModel(RAPIER.MotorModel.ForceBased);
      joint.setMotorMaxForce(gain.max);
      joint.setContactsEnabled(false);
      hinges.push({ joint, min, max, k: gain.k, d: gain.d, pick });
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
      (pose) => pose.spine
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
        pick
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
        pick
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
        pick
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
        pick
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
        pick
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
      jumpWindow: 0
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

  private applyJump(robot: Robot): void {
    const amp = robot.state.morph.jump;
    if (amp <= 0.02 || robot.state.fallen || robot.state.hidden) {
      robot.jumpWindow = 0;
      return;
    }
    const freq = Math.max(0.2, robot.decoded.freq ?? 1.4);
    const cycle = Math.floor(this.time * freq);
    if (this.time > 0.2 && cycle !== robot.jumpCycle) {
      robot.jumpCycle = cycle;
      robot.jumpWindow = 0.1;
      const vy = amp * 3.4;
      const vx = amp * 0.45;
      for (const body of robot.rigid) {
        const mass = body.mass();
        body.applyImpulse({ x: vx * mass, y: vy * mass, z: 0 }, true);
      }
    }
    if (robot.jumpWindow > 0) {
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

  private collect(): void {
    for (const robot of this.internals) {
      this.copyBodies(robot);
      if (robot.state.hidden) continue;
      const pelvis = robot.state.bodies.pelvis;
      const upY = upYOf(robot.pelvis);
      robot.upY = upY;
      const span = Math.max(0.5, FINISH_X - robot.state.spawnX);
      const stand = robot.state.morph.pelvisStand;
      const crouch = robot.state.morph.crouch;
      const yFall = stand * (0.36 - 0.16 * crouch);
      const yScore = stand * (crouch > 0.4 ? 0.26 : 0.46);
      const upCut = 0.38 - 0.32 * crouch;
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

