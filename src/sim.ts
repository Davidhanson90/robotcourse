import RAPIER from "@dimforge/rapier3d-compat";
import {
  ballVolume,
  capsuleHalf,
  capsuleVolume,
  cuboidVolume,
  cylinderVolume,
  densityFor,
  emptyPose,
  type Morph,
  type PartSpec,
  type Pose
} from "./body";
import { FINISH_X, GEN_TIME, LANE_HALF, SOLIDS, spawnSlots } from "./course";
import {
  breed,
  decode,
  fitness,
  initialPopulation,
  morphFromDecoded,
  mulberry32,
  jumpFrequency,
  refreshMorphExpression,
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
  elbow: { k: 440, d: 38, max: 2400 },
  segment: { k: 480, d: 55, max: 2800 }
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
  bodies: Record<string, BodyPose>;
}

interface Hinge {
  joint: RAPIER.RevoluteImpulseJoint;
  min: number;
  max: number;
  k: number;
  d: number;
  poseKey: string;
}

interface Robot {
  state: RobotState;
  decoded: Decoded;
  pose: Pose;
  hinges: Hinge[];
  rigid: RAPIER.RigidBody[];
  partIndex: Map<string, number>;
  root: RAPIER.RigidBody;
  contacts: RAPIER.Collider[];
  fallTimer: number;
  motorsCut: boolean;
  lastX: number;
  upY: number;
  jumpCycle: number;
  jumpWindow: number;
  grounded: boolean;
  jumpSpent: boolean;
  leftGround: boolean;
}

let ready: Promise<void> | null = null;
let nextSpawnSerial = 1;

function initPhysics(): Promise<void> {
  if (!ready) ready = RAPIER.init();
  return ready;
}

function hueOf(index: number): number {
  return (index * 0.61803398875) % 1;
}

function upYOf(body: RAPIER.RigidBody): number {
  const r = body.rotation();
  // Local +Y in world.
  return 1 - 2 * (r.x * r.x + r.z * r.z);
}

function unitHash(index: number, cycle: number): number {
  let x = Math.imul(index + 1, 374761393) ^ Math.imul(cycle + 1, 668265263);
  x = (x ^ (x >>> 13)) >>> 0;
  return x / 4294967296;
}

function jumpChanceRate(): number {
  if (jumpFrequency >= 1) return 1;
  if (jumpFrequency <= 0) return 0.02;
  return jumpFrequency;
}

function partVolume(part: PartSpec): number {
  if (part.shape === "box") return cuboidVolume(part.hx ?? 0.1, part.hy ?? 0.1, part.hz ?? 0.1);
  if (part.shape === "ball") return ballVolume(part.radius ?? 0.1);
  if (part.shape === "cylinder") {
    const r = part.radius ?? 0.1;
    const half = (part.length ?? r * 2) / 2;
    return cylinderVolume(half, r);
  }
  return capsuleVolume(part.length ?? 0.3, part.radius ?? 0.05);
}

function colliderDesc(part: PartSpec): RAPIER.ColliderDesc {
  let desc: RAPIER.ColliderDesc;
  if (part.shape === "box") {
    desc = RAPIER.ColliderDesc.cuboid(part.hx ?? 0.1, part.hy ?? 0.1, part.hz ?? 0.1);
  } else if (part.shape === "ball") {
    desc = RAPIER.ColliderDesc.ball(part.radius ?? 0.1);
  } else if (part.shape === "cylinder") {
    const r = part.radius ?? 0.1;
    const half = Math.max(0.02, (part.length ?? r * 2) / 2);
    desc = RAPIER.ColliderDesc.cylinder(half, r);
  } else {
    const r = part.radius ?? 0.05;
    const len = part.length ?? 0.3;
    desc = RAPIER.ColliderDesc.capsule(capsuleHalf(len, r), r);
  }
  desc.setDensity(densityFor(partVolume(part), part.mass));
  return desc;
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
    if (!best) throw new Error("Population is empty.");
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
   * for the robots already on the course.
   */
  setExpression(flexibility: number, jump: number): void {
    setFlexScale(flexibility);
    setJumpScale(jump);
    for (const robot of this.internals) {
      refreshMorphExpression(robot.state.morph, robot.decoded);
      for (const hinge of robot.hinges) {
        const jointSpec = robot.state.morph.joints.find((j) => j.poseKey === hinge.poseKey);
        if (!jointSpec) continue;
        hinge.min = jointSpec.min;
        hinge.max = jointSpec.max;
        hinge.joint.setLimits(jointSpec.min, jointSpec.max);
      }
    }
  }

  /**
   * Rank by fitness, breed, then rebuild every robot from its own genome.
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

  /** Physics and colliders for one genome. Mesh keys match part ids. */
  private spawnRobot(index: number, genes: number[], spawnX: number, spawnZ: number): Robot {
    const decoded = decode(genes);
    const morph = morphFromDecoded(decoded, spawnX, spawnZ);
    const bodies: Record<string, BodyPose> = {};
    const rigid: RAPIER.RigidBody[] = [];
    const partIndex = new Map<string, number>();
    const contacts: RAPIER.Collider[] = [];

    for (const part of morph.parts) {
      const desc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(part.x, part.y, part.z)
        .setCanSleep(false)
        .setLinearDamping(0.05)
        .setAngularDamping(morph.form === "snake" || morph.form === "blob" ? 0.85 : 1.25)
        .setCcdEnabled(!!part.ccd);
      if (part.qw !== undefined) {
        desc.setRotation({
          x: part.qx ?? 0,
          y: part.qy ?? 0,
          z: part.qz ?? 0,
          w: part.qw
        });
      }
      const body = this.world.createRigidBody(desc);
      const col = colliderDesc(part)
        .setFriction(part.friction)
        .setRestitution(0)
        .setCollisionGroups(ROBOT_GROUP);
      if (part.friction >= 1.4) {
        col.setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max);
      }
      const collider = this.world.createCollider(col, body);
      if (part.contact) contacts.push(collider);
      partIndex.set(part.id, rigid.length);
      rigid.push(body);
      bodies[part.id] = {
        x: part.x,
        y: part.y,
        z: part.z,
        qx: part.qx ?? 0,
        qy: part.qy ?? 0,
        qz: part.qz ?? 0,
        qw: part.qw ?? 1
      };
    }

    // Fixed weld for head/cab attached without a motor when needed — all joints are revolute here.
    const hinges: Hinge[] = [];
    const byId = (id: string): RAPIER.RigidBody => {
      const i = partIndex.get(id);
      if (i === undefined) throw new Error(`Missing part ${id}`);
      return rigid[i]!;
    };

    for (const spec of morph.joints) {
      const data = RAPIER.JointData.revolute(spec.anchorParent, spec.anchorChild, spec.axis);
      const joint = this.world.createImpulseJoint(
        data,
        byId(spec.parent),
        byId(spec.child),
        true
      ) as RAPIER.RevoluteImpulseJoint;
      joint.setLimits(spec.min, spec.max);
      joint.configureMotorModel(RAPIER.MotorModel.ForceBased);
      const gain = GAIN[spec.gain] ?? GAIN.hip;
      joint.setMotorMaxForce(gain.max);
      joint.setContactsEnabled(false);
      hinges.push({
        joint,
        min: spec.min,
        max: spec.max,
        k: gain.k,
        d: gain.d,
        poseKey: spec.poseKey
      });
    }

    const root = byId(morph.rootId);
    const pose = emptyPose(morph);
    writePose(decoded, 0, pose, morph);

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
      bodies
    };

    return {
      state,
      decoded,
      pose,
      hinges,
      rigid,
      partIndex,
      root,
      contacts,
      fallTimer: 0,
      motorsCut: false,
      lastX: spawnX,
      upY: 1,
      jumpCycle: 0,
      jumpWindow: 0,
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
      writePose(robot.decoded, this.time, robot.pose, robot.state.morph);
      this.applyJump(robot);
      for (const hinge of robot.hinges) {
        const raw = robot.pose[hinge.poseKey] ?? 0;
        const target = Math.max(hinge.min, Math.min(hinge.max, raw));
        hinge.joint.configureMotorPosition(target, hinge.k, hinge.d);
      }
    }
  }

  /** True when this collider has at least one contact point with a solid. */
  private colliderOnSolid(col: RAPIER.Collider): boolean {
    let touching = false;
    this.world.contactPairsWith(col, (other) => {
      if (touching) return;
      this.world.contactPair(col, other, (manifold) => {
        if (manifold.numContacts() > 0) touching = true;
      });
    });
    return touching;
  }

  /** Grounded from any contact part (feet, wheels, blob lobes, snake segments). */
  private updateGrounded(robot: Robot): void {
    let grounded = false;
    if (robot.contacts.length === 0) {
      // Fallback: root near ground.
      grounded = robot.root.translation().y < robot.state.morph.pelvisStand * 0.55;
    } else {
      for (const col of robot.contacts) {
        if (this.colliderOnSolid(col)) {
          grounded = true;
          break;
        }
      }
    }
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
    const cycle = Math.floor(this.time * Math.min(gait, 2.6));
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
        for (const key of Object.keys(robot.pose)) {
          if (key.startsWith("knee") || key.startsWith("hip") || key.startsWith("susp")) {
            robot.pose[key] = (robot.pose[key] ?? 0) - burst * (key.startsWith("hip") || key.startsWith("susp") ? 0.7 : 1);
          }
        }
        robot.jumpWindow -= SIM_DT;
      }
    }
  }

  private collect(): void {
    for (const robot of this.internals) {
      this.updateGrounded(robot);
      this.copyBodies(robot);
      if (robot.state.hidden) continue;
      const rootPose = robot.state.bodies[robot.state.morph.rootId];
      if (!rootPose) continue;
      const upY = upYOf(robot.root);
      robot.upY = upY;
      const span = Math.max(0.5, FINISH_X - robot.state.spawnX);
      const stand = robot.state.morph.pelvisStand;
      const crouch = robot.state.morph.crouch;
      const form = robot.state.morph.form;
      // Low forms (snake/blob) use softer fall rules.
      const lowForm = form === "snake" || form === "blob";
      const yFall = stand * (lowForm ? 0.12 : 0.34 - 0.2 * crouch);
      const yScore = stand * (lowForm ? 0.05 : crouch > 0.4 ? 0.16 : 0.46);
      const upCut = lowForm ? -0.2 : 0.38 - 0.34 * crouch;
      const offCourse =
        Math.abs(rootPose.z) > LANE_HALF + 0.08 && rootPose.y < stand * 0.5;
      const tipped = upY < upCut || rootPose.y < yFall || offCourse;
      if (!robot.state.fallen && !robot.state.finished) {
        const travel = rootPose.x - robot.state.spawnX;
        const dx = rootPose.x - robot.lastX;
        if (dx < 0.85 && rootPose.y < 2.6 && rootPose.y > yScore && Math.abs(rootPose.z) < LANE_HALF + 0.15) {
          if (travel > robot.state.maxTravel) robot.state.maxTravel = Math.min(span, travel);
        }
        const finishUp = lowForm ? 0.15 : 0.55;
        const finishY = stand * (lowForm ? 0.35 : 0.56);
        if (
          rootPose.x >= FINISH_X &&
          rootPose.y > finishY &&
          upY > finishUp &&
          Math.abs(rootPose.z) < LANE_HALF - 0.15
        ) {
          robot.state.finished = true;
          robot.state.finishTime = this.time;
          robot.state.maxTravel = span;
        }
      }
      robot.lastX = rootPose.x;
      if (tipped) robot.fallTimer += SIM_DT;
      else robot.fallTimer = 0;
      if (!robot.state.fallen && robot.fallTimer > (lowForm ? 0.7 : 0.4)) robot.state.fallen = true;
      if (robot.state.fallen && (robot.fallTimer > 2.35 || rootPose.y > 5 || rootPose.y < -3)) {
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
    for (const [id, index] of robot.partIndex) {
      const body = robot.rigid[index];
      if (!body) continue;
      const p = body.translation();
      const q = body.rotation();
      const slot = robot.state.bodies[id] ?? (robot.state.bodies[id] = {
        x: 0,
        y: 0,
        z: 0,
        qx: 0,
        qy: 0,
        qz: 0,
        qw: 1
      });
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
