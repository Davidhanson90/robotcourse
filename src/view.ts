import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { capsuleHalf, type Morph, type PartSpec } from "./body";
import { FINISH_X, LANE_HALF, SOLIDS, VALLEY_TOP, surfaceY } from "./course";
import type { CourseSim } from "./sim";
import type { RobotState } from "./sim";

interface RobotVisual {
  parts: Record<string, THREE.Object3D>;
  visor: THREE.MeshStandardMaterial;
  accent: THREE.MeshStandardMaterial;
}

const SKY_VERT = `
  varying vec3 vPos;
  void main() {
    vPos = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const SKY_FRAG = `
  varying vec3 vPos;
  uniform vec3 topColor;
  uniform vec3 horizonColor;
  uniform vec3 groundColor;
  void main() {
    float h = normalize(vPos).y;
    vec3 col = mix(horizonColor, topColor, smoothstep(0.0, 0.62, h));
    col = mix(groundColor, col, smoothstep(-0.35, 0.08, h));
    gl_FragColor = vec4(col, 1.0);
  }
`;

export class CourseView {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly controls: OrbitControls;
  private readonly lookAt = new THREE.Vector3();
  private readonly robots = new Map<number, RobotVisual>();
  private readonly shared = new Map<string, THREE.BufferGeometry>();
  private readonly ring: THREE.Mesh;
  private readonly trail: THREE.Line;
  private readonly ghost: THREE.Line;
  private readonly trailPositions = new Float32Array(180 * 3);
  private readonly ghostPositions = new Float32Array(180 * 3);
  private trailCount = 0;
  private ghostCount = 0;
  private generation = -1;
  private leaderHue = 0.08;
  private builtFor = -1;
  private readonly clock = new THREE.Clock();
  private disposed = false;
  follow = false;
  private readonly onUserOrbit: () => void;
  private bannerMat: THREE.MeshBasicMaterial | null = null;

  constructor(canvas: HTMLCanvasElement, onUserOrbit: () => void) {
    this.onUserOrbit = onUserOrbit;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x1a120e, 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.12;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.1, 220);
    this.camera.position.set(4.5, 8.6, 16.5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(9.2, 1.05, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 2.8;
    this.controls.maxDistance = 42;
    this.controls.addEventListener("start", () => this.onUserOrbit());

    this.buildScene();

    const ringGeo = new THREE.RingGeometry(0.46, 0.62, 40);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffc56b,
      transparent: true,
      opacity: 0.92,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    this.ring = new THREE.Mesh(ringGeo, ringMat);
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.renderOrder = 2;
    this.scene.add(this.ring);

    this.trail = this.makeLine(this.trailPositions, 0.95);
    this.ghost = this.makeLine(this.ghostPositions, 0.28);
    this.scene.add(this.ghost);
    this.scene.add(this.trail);
    this.resize();
  }

  resize(): void {
    const canvas = this.renderer.domElement;
    const parent = canvas.parentElement;
    const width = parent?.clientWidth ?? window.innerWidth;
    const height = parent?.clientHeight ?? window.innerHeight;
    if (width < 2 || height < 2) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  sync(sim: CourseSim): void {
    if (this.builtFor !== sim.spawnSerial) this.rebuildRobots(sim);
    if (this.generation !== sim.generation) {
      this.stashTrail();
      this.generation = sim.generation;
    }
    let leader: RobotState | null = null;
    for (const robot of sim.robots) {
      if (!leader || robot.score > leader.score) leader = robot;
      const visual = this.robots.get(robot.index);
      if (!visual) continue;
      for (const [key, obj] of Object.entries(visual.parts)) {
        const pose = robot.bodies[key];
        if (!obj || !pose) continue;
        obj.visible = !robot.hidden;
        obj.position.set(pose.x, pose.y, pose.z);
        obj.quaternion.set(pose.qx, pose.qy, pose.qz, pose.qw);
      }
      const glow = robot === leader ? 0.95 : 0.28;
      visual.visor.emissiveIntensity = glow;
      visual.accent.emissiveIntensity = robot === leader ? 0.55 : 0.12;
    }
    if (!leader || leader.hidden) {
      this.ring.visible = false;
      return;
    }
    this.leaderHue = leader.hue;
    const root = leader.bodies[leader.morph.rootId] ?? Object.values(leader.bodies)[0];
    if (!root) {
      this.ring.visible = false;
      return;
    }
    const ground = surfaceY(root.x, root.z);
    this.ring.visible = ground !== null && !leader.fallen;
    if (ground !== null) this.ring.position.set(root.x, ground + 0.035, root.z);
    this.lookAt.set(root.x, root.y + 0.35, root.z);
    (this.ring.material as THREE.MeshBasicMaterial).color.setHSL(leader.hue, 0.85, 0.62);
    this.pushTrail(root.x, (ground ?? root.y) + 0.06, root.z);
  }

  render(): void {
    if (this.disposed) return;
    const t = this.clock.getElapsedTime();
    if (this.bannerMat) {
      this.bannerMat.opacity = 0.14 + Math.sin(t * 2.2) * 0.05;
    }
    if (this.follow) this.controls.target.lerp(this.lookAt, 0.16);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.disposed = true;
    this.controls.dispose();
    this.renderer.dispose();
    this.scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const material = mesh.material;
      if (Array.isArray(material)) material.forEach((m) => m.dispose());
      else material?.dispose();
    });
  }

  /** One visual per robot, sized from that robot's morph (its genome). */
  private rebuildRobots(sim: CourseSim): void {
    for (const visual of this.robots.values()) {
      for (const obj of Object.values(visual.parts)) {
        if (obj) this.scene.remove(obj);
      }
    }
    this.robots.clear();
    for (const robot of sim.robots) this.robots.set(robot.index, this.makeRobot(robot.hue, robot.morph));
    this.builtFor = sim.spawnSerial;
    this.generation = -1;
    this.trailCount = 0;
    this.ghostCount = 0;
  }

  private makeRobot(hue: number, morph: Morph): RobotVisual {
    const armor = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.62, 0.48),
      roughness: 0.42,
      metalness: 0.38
    });
    const dark = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.45, 0.16),
      roughness: 0.5,
      metalness: 0.55
    });
    const accent = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.7, 0.58),
      emissive: new THREE.Color().setHSL(hue, 0.85, 0.32),
      emissiveIntensity: 0.12,
      roughness: 0.32,
      metalness: 0.45
    });
    const visor = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.9, 0.55),
      emissive: new THREE.Color().setHSL(hue, 0.95, 0.48),
      emissiveIntensity: 0.28,
      roughness: 0.18,
      metalness: 0.2
    });
    const sole = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.3, 0.12),
      roughness: 0.85,
      metalness: 0.15
    });
    const joint = new THREE.MeshStandardMaterial({
      color: 0x1b1e24,
      roughness: 0.35,
      metalness: 0.7
    });
    const blobMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setHSL(hue, 0.7, 0.42),
      roughness: 0.28,
      metalness: 0.15,
      emissive: new THREE.Color().setHSL(hue, 0.8, 0.2),
      emissiveIntensity: 0.2
    });

    const parts: Record<string, THREE.Object3D> = {};
    for (const part of morph.parts) {
      const obj = this.partMesh(part, { armor, dark, accent, visor, sole, joint, blobMat });
      this.scene.add(obj);
      parts[part.id] = obj;
    }
    return { parts, visor, accent };
  }

  private partMesh(
    part: PartSpec,
    mats: {
      armor: THREE.Material;
      dark: THREE.Material;
      accent: THREE.Material;
      visor: THREE.Material;
      sole: THREE.Material;
      joint: THREE.Material;
      blobMat: THREE.Material;
    }
  ): THREE.Object3D {
    const group = new THREE.Group();
    const role = part.role ?? "limb";
    const cast = role !== "foot";

    if (part.shape === "box") {
      const mat =
        role === "core" || role === "head"
          ? mats.armor
          : role === "foot"
            ? mats.sole
            : mats.dark;
      const mesh = this.mesh(
        this.box((part.hx ?? 0.1) * 2, (part.hy ?? 0.1) * 2, (part.hz ?? 0.1) * 2),
        mat,
        cast
      );
      group.add(mesh);
      if (role === "core") {
        const stripe = this.mesh(
          this.box((part.hx ?? 0.1) * 1.05, 0.02, (part.hz ?? 0.1) * 2.05),
          mats.accent,
          false
        );
        stripe.position.y = (part.hy ?? 0.1) + 0.01;
        group.add(stripe);
      }
      if (role === "foot") {
        const stripe = this.mesh(
          this.box((part.hx ?? 0.1) * 1.1, 0.012, (part.hz ?? 0.1) * 2.05),
          mats.accent,
          false
        );
        stripe.position.y = (part.hy ?? 0.05) + 0.004;
        group.add(stripe);
      }
      if (role === "head") {
        const eye = this.mesh(this.geo("eye", () => new THREE.SphereGeometry(0.04, 10, 8)), mats.visor, false);
        eye.position.set((part.hx ?? 0.1) * 0.9, (part.hy ?? 0.1) * 0.2, 0);
        group.add(eye);
      }
      return group;
    }

    if (part.shape === "ball") {
      const r = part.radius ?? 0.1;
      const mat =
        role === "blob"
          ? mats.blobMat
          : role === "head"
            ? mats.armor
            : role === "foot"
              ? mats.sole
              : mats.armor;
      group.add(this.mesh(this.geo(`ball-${r.toFixed(3)}`, () => new THREE.SphereGeometry(r, 22, 16)), mat, cast));
      if (role === "head") {
        const visorMesh = this.mesh(this.box(0.05, 0.06, r * 1.1), mats.visor, false);
        visorMesh.position.set(r * 0.72, 0.02, 0);
        group.add(visorMesh);
      }
      if (role === "blob") {
        const spot = this.mesh(
          this.geo(`blobspot-${(r * 0.35).toFixed(3)}`, () => new THREE.SphereGeometry(r * 0.28, 10, 8)),
          mats.visor,
          false
        );
        spot.position.set(r * 0.55, r * 0.25, 0);
        group.add(spot);
      }
      if (role === "foot") {
        const ring = this.mesh(
          this.geo(`footring-${r.toFixed(3)}`, () => new THREE.TorusGeometry(r * 0.7, 0.012, 6, 14)),
          mats.accent,
          false
        );
        ring.rotation.x = Math.PI / 2;
        ring.position.y = -r * 0.15;
        group.add(ring);
      }
      return group;
    }

    if (part.shape === "cylinder") {
      const r = part.radius ?? 0.1;
      const h = part.length ?? r * 2;
      const mat = role === "wheel" ? mats.dark : mats.armor;
      const cyl = this.mesh(
        this.geo(`cyl-${r.toFixed(3)}-${h.toFixed(3)}`, () => new THREE.CylinderGeometry(r, r, h, 18)),
        mat,
        cast
      );
      group.add(cyl);
      if (role === "wheel") {
        const hub = this.mesh(
          this.geo(`hub-${(r * 0.35).toFixed(3)}`, () => new THREE.CylinderGeometry(r * 0.28, r * 0.28, h * 1.08, 12)),
          mats.accent,
          false
        );
        group.add(hub);
        const rim = this.mesh(
          this.geo(`rim-${r.toFixed(3)}`, () => new THREE.TorusGeometry(r * 0.82, 0.02, 6, 18)),
          mats.visor,
          false
        );
        rim.rotation.x = Math.PI / 2;
        group.add(rim);
      }
      if (role === "core") {
        const belt = this.mesh(
          this.geo(`belt-${r.toFixed(3)}`, () => new THREE.TorusGeometry(r * 0.9, 0.018, 8, 18)),
          mats.accent,
          false
        );
        belt.rotation.x = Math.PI / 2;
        group.add(belt);
      }
      return group;
    }

    // Capsule limb / segment.
    const r = part.radius ?? 0.05;
    const len = part.length ?? 0.3;
    const mat = role === "segment" ? mats.armor : role === "head" ? mats.armor : mats.dark;
    const cyl = Math.max(0.04, capsuleHalf(len, r) * 2);
    const geo = this.geo(`cap-${len.toFixed(3)}-${r.toFixed(3)}`, () => new THREE.CapsuleGeometry(r, cyl, 6, 12));
    group.add(this.mesh(geo, mat, cast));
    const cuff = this.mesh(
      this.geo(`cuff-${(r * 1.15).toFixed(3)}`, () => new THREE.SphereGeometry(r * 1.15, 12, 10)),
      mats.joint,
      cast
    );
    cuff.position.y = len / 2;
    group.add(cuff);
    if (role === "head") {
      const eye = this.mesh(this.geo("eye", () => new THREE.SphereGeometry(r * 0.45, 10, 8)), mats.visor, false);
      eye.position.set(r * 0.7, 0, 0);
      group.add(eye);
    }
    return group;
  }

  private mesh(geo: THREE.BufferGeometry, material: THREE.Material, cast: boolean): THREE.Mesh {
    const mesh = new THREE.Mesh(geo, material);
    mesh.castShadow = cast;
    mesh.receiveShadow = false;
    return mesh;
  }

  private box(w: number, h: number, d: number): THREE.BufferGeometry {
    return this.geo(`box-${w}-${h}-${d}`, () => new THREE.BoxGeometry(w, h, d));
  }

  private geo(key: string, build: () => THREE.BufferGeometry): THREE.BufferGeometry {
    const found = this.shared.get(key);
    if (found) return found;
    const created = build();
    this.shared.set(key, created);
    return created;
  }

  private makeLine(positions: Float32Array, opacity: number): THREE.Line {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setDrawRange(0, 0);
    const mat = new THREE.LineBasicMaterial({
      color: 0xffd7a1,
      transparent: true,
      opacity,
      depthWrite: false
    });
    const line = new THREE.Line(geo, mat);
    line.frustumCulled = false;
    return line;
  }

  private pushTrail(x: number, y: number, z: number): void {
    const last = this.trailCount - 1;
    if (last >= 0) {
      const ox = this.trailPositions[last * 3] ?? 0;
      const oz = this.trailPositions[last * 3 + 2] ?? 0;
      if ((x - ox) ** 2 + (z - oz) ** 2 < 0.006) return;
    }
    if (this.trailCount >= 180) {
      this.trailPositions.copyWithin(0, 3);
      this.trailCount = 179;
    }
    const i = this.trailCount * 3;
    this.trailPositions[i] = x;
    this.trailPositions[i + 1] = y;
    this.trailPositions[i + 2] = z;
    this.trailCount += 1;
    const attr = this.trail.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailCount);
    (this.trail.material as THREE.LineBasicMaterial).color.setHSL(this.leaderHue, 0.8, 0.66);
  }

  private stashTrail(): void {
    this.ghostPositions.set(this.trailPositions);
    this.ghostCount = this.trailCount;
    const attr = this.ghost.geometry.getAttribute("position") as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.ghost.geometry.setDrawRange(0, this.ghostCount);
    (this.ghost.material as THREE.LineBasicMaterial).color.setHSL(this.leaderHue, 0.4, 0.7);
    this.trailCount = 0;
    this.trail.geometry.setDrawRange(0, 0);
  }

  private buildScene(): void {
    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(140, 32, 20),
      new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        uniforms: {
          topColor: { value: new THREE.Color("#1c3358") },
          horizonColor: { value: new THREE.Color("#e39a6a") },
          groundColor: { value: new THREE.Color("#241810") }
        },
        vertexShader: SKY_VERT,
        fragmentShader: SKY_FRAG
      })
    );
    this.scene.add(sky);
    this.scene.fog = new THREE.Fog(0xc98b68, 28, 78);

    const hemi = new THREE.HemisphereLight(0xc5d7ff, 0x3a2a22, 0.72);
    this.scene.add(hemi);
    this.scene.add(new THREE.AmbientLight(0xfff0e4, 0.18));

    const sun = new THREE.DirectionalLight(0xffd2a6, 2.8);
    sun.position.set(-6, 14, 9);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 48;
    sun.shadow.camera.left = -16;
    sun.shadow.camera.right = 16;
    sun.shadow.camera.top = 16;
    sun.shadow.camera.bottom = -16;
    sun.shadow.bias = -0.00035;
    sun.shadow.normalBias = 0.035;
    this.scene.add(sun);
    sun.target.position.set(9, 0.5, 0);
    this.scene.add(sun.target);

    const fill = new THREE.DirectionalLight(0x8eb4ff, 0.45);
    fill.position.set(10, 6, -8);
    this.scene.add(fill);

    const finishLight = new THREE.PointLight(0xffb15c, 18, 14, 2);
    finishLight.position.set(FINISH_X, 2.1, 0);
    this.scene.add(finishLight);
    const startLight = new THREE.PointLight(0x49ffe2, 8, 10, 2);
    startLight.position.set(2.2, 1.6, 0);
    this.scene.add(startLight);

    const stripe = stripeTexture();
    for (const solid of SOLIDS) {
      let geo: THREE.BufferGeometry;
      if (solid.shape === "box") {
        geo = new THREE.BoxGeometry(solid.hx * 2, solid.hy * 2, solid.hz * 2);
      } else {
        geo = new THREE.CylinderGeometry(solid.hx, solid.hx, solid.hy * 2, 20);
      }
      const mat = new THREE.MeshStandardMaterial({
        color: solid.color,
        roughness: solid.roughness,
        metalness: solid.metalness,
        map: solid.tag === "wall" ? stripe : null
      });
      if (solid.tag === "valley") mat.color.set(0x141816);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(solid.x, solid.y, solid.z);
      mesh.quaternion.set(solid.qx, solid.qy, solid.qz, solid.qw);
      mesh.receiveShadow = true;
      mesh.castShadow = solid.tag !== "valley";
      this.scene.add(mesh);
    }

    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(4.6, 0.03, 3.3),
      new THREE.MeshStandardMaterial({
        color: 0x1c8f86,
        emissive: 0x083e3a,
        emissiveIntensity: 0.7,
        roughness: 0.55,
        metalness: 0.1
      })
    );
    pad.position.set(2.15, 0.02, 0);
    pad.receiveShadow = true;
    this.scene.add(pad);

    for (const side of [-1, 1]) {
      const edge = new THREE.Mesh(
        new THREE.BoxGeometry(22.4, 0.025, 0.045),
        new THREE.MeshStandardMaterial({
          color: 0xd7fff6,
          emissive: 0x1a6e66,
          emissiveIntensity: 0.45,
          roughness: 0.4,
          metalness: 0.2
        })
      );
      edge.position.set(9.6, 0.03, side * (LANE_HALF - 0.05));
      this.scene.add(edge);
    }

    const banner = new THREE.Mesh(
      new THREE.PlaneGeometry(3.5, 2.15),
      new THREE.MeshBasicMaterial({
        color: 0xffc56a,
        transparent: true,
        opacity: 0.16,
        side: THREE.DoubleSide,
        depthWrite: false
      })
    );
    banner.position.set(FINISH_X, 1.15, 0);
    banner.rotation.y = Math.PI / 2;
    banner.renderOrder = 1;
    this.bannerMat = banner.material as THREE.MeshBasicMaterial;
    this.scene.add(banner);

    this.addLabel("START", 1.5, 0.58, 2.15, 1.5, 0.42);
    this.addLabel("FINISH", FINISH_X - 0.32, 2.15, 0, 1.35, 0.38, Math.PI / 2);

    const moundMat = new THREE.MeshStandardMaterial({
      color: 0x243028,
      roughness: 1,
      metalness: 0,
      flatShading: true
    });
    const mounds: Array<[number, number, number]> = [
      [-4, -7, 3.2],
      [3, 8.5, 2.4],
      [12, -9, 3.6],
      [20, 8, 2.8],
      [24, -6, 2.2]
    ];
    for (const [x, z, s] of mounds) {
      const mound = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), moundMat);
      mound.scale.set(s, s * 0.42, s);
      mound.position.set(x, VALLEY_TOP + 0.15, z);
      mound.castShadow = true;
      mound.receiveShadow = true;
      this.scene.add(mound);
    }

    const trunkMat = new THREE.MeshStandardMaterial({ color: 0x3a2a22, roughness: 0.9 });
    const leafMat = new THREE.MeshStandardMaterial({ color: 0x1d3b2c, roughness: 0.85, flatShading: true });
    const trees: Array<[number, number, number]> = [
      [-2.2, 5.4, 1],
      [7.5, -5.8, 1.15],
      [14.2, 6.1, 0.9],
      [19.5, -5.5, 1.25]
    ];
    for (const [x, z, s] of trees) {
      const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.1 * s, 0.14 * s, 1.3 * s, 8), trunkMat);
      trunk.position.set(x, VALLEY_TOP + 0.65 * s, z);
      trunk.castShadow = true;
      this.scene.add(trunk);
      const crown = new THREE.Mesh(new THREE.ConeGeometry(0.85 * s, 1.7 * s, 7), leafMat);
      crown.position.set(x, VALLEY_TOP + 1.7 * s, z);
      crown.castShadow = true;
      this.scene.add(crown);
    }
  }

  private addLabel(
    text: string,
    x: number,
    y: number,
    z: number,
    w: number,
    h: number,
    rotY = 0
  ): void {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({
        map: textTexture(text),
        transparent: true,
        depthWrite: false
      })
    );
    plane.position.set(x, y, z);
    plane.rotation.y = rotY;
    this.scene.add(plane);
  }
}

function stripeTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.CanvasTexture(canvas);
  ctx.fillStyle = "#e2b34a";
  ctx.fillRect(0, 0, 64, 64);
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 8;
  for (let i = -64; i < 128; i += 16) {
    ctx.beginPath();
    ctx.moveTo(i, 64);
    ctx.lineTo(i + 32, 0);
    ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function textTexture(text: string): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 96;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, 256, 96);
    ctx.fillStyle = "#fff6e8";
    ctx.font = "700 54px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 128, 50);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}
