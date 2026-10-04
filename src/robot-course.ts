import { LitElement, css, html } from "lit";
import { customElement, query, state } from "lit/decorators.js";
import { FINISH_BONUS, PROGRESS_WEIGHT, TIME_BONUS_MAX, setFlexScale, setJumpFrequency, setJumpScale } from "./ga";
import { GEN_TIME } from "./course";
import { CourseSim, SIM_DT } from "./sim";
import { CourseView } from "./view";

const MIN_POP = 12;
const MAX_POP = 24;

@customElement("robot-course")
export class RobotCourse extends LitElement {
  static override styles = css`
    :host {
      display: block;
      height: 100%;
      background: #07080c;
      color: #e8eef6;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }
    .stage {
      position: relative;
      height: 100%;
      overflow: hidden;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    .overlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 16px;
      gap: 12px;
    }
    .panel {
      pointer-events: auto;
      background: rgba(8, 10, 14, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 16px;
      backdrop-filter: blur(12px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
    }
    header.panel {
      display: flex;
      flex-wrap: wrap;
      gap: 18px 28px;
      align-items: stretch;
      padding: 14px 16px 12px;
      max-width: min(980px, 100%);
    }
    .brand h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.14em;
      font-weight: 700;
    }
    .brand p {
      margin: 4px 0 0;
      color: #b7c3d1;
      font-size: 12.5px;
      max-width: 280px;
      line-height: 1.4;
    }
    .mark {
      width: 12px;
      height: 12px;
      border-radius: 99px;
      display: inline-block;
      margin-right: 8px;
      background: var(--swatch, #ffc56b);
      box-shadow: 0 0 12px var(--swatch, #ffc56b);
      vertical-align: -1px;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: stretch;
    }
    .stat {
      min-width: 108px;
      padding: 6px 10px 7px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.03);
    }
    .stat span {
      display: block;
      color: #93a0b0;
      font-size: 10px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .stat strong {
      display: block;
      margin-top: 3px;
      font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 20px;
      font-weight: 650;
      font-variant-numeric: tabular-nums;
    }
    .stat em {
      display: block;
      margin-top: 2px;
      color: #8e9bab;
      font-style: normal;
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }
    footer.panel {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      padding: 10px 12px;
      max-width: min(1180px, 100%);
    }
    button, label.follow {
      font: inherit;
      color: inherit;
    }
    button {
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: #161b22;
      color: #f3f6fb;
      border-radius: 999px;
      padding: 8px 14px;
      cursor: pointer;
    }
    button.primary {
      background: #f0b56a;
      color: #1a1208;
      border-color: transparent;
      font-weight: 700;
    }
    button:hover {
      filter: brightness(1.08);
    }
    .pop {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #c5d0dc;
      font-size: 13px;
    }
    input[type="range"] {
      accent-color: #f0b56a;
      width: 110px;
    }
    label.follow {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #c5d0dc;
      cursor: pointer;
    }
    .track {
      flex: 1 1 140px;
      height: 6px;
      border-radius: 99px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
      min-width: 120px;
    }
    .fill {
      height: 100%;
      background: linear-gradient(90deg, #1c8f86, #f0b56a);
      width: 0%;
    }
    .formula {
      margin: 0;
      color: #9aa8b8;
      font-size: 11.5px;
      line-height: 1.35;
      flex-basis: 100%;
    }
    .loading {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      color: #d5deea;
      background: rgba(7, 8, 12, 0.45);
      pointer-events: none;
    }
  `;

  @state() private generation = 1;
  @state() private bestDistance = 0;
  @state() private bestTime: number | null = null;
  @state() private recordDistance = 0;
  @state() private recordTime: number | null = null;
  @state() private alive = 0;
  @state() private playing = true;
  @state() private popSize = 16;
  @state() private simPop = 16;
  @state() private flexibility = 100;
  @state() private jump = 100;
  @state() private jumpFrequency = 40;
  @state() private follow = false;
  @state() private progress = 0;
  @state() private leaderHue = 0.08;
  @state() private ready = false;
  @state() private error = "";
  @state() private seed = 20261004;

  @query("canvas") private canvas!: HTMLCanvasElement;

  private view: CourseView | null = null;
  private sim: CourseSim | null = null;
  private raf = 0;
  private lastNow = 0;
  private acc = 0;
  private pauseAfter = false;
  private lastHud = 0;
  private onKey = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (event.code === "Space") {
      event.preventDefault();
      this.onPlay();
    } else if (event.key === "s" || event.key === "S") {
      this.onStep();
    }
  };

  override connectedCallback(): void {
    super.connectedCallback();
    window.addEventListener("keydown", this.onKey);
  }

  override disconnectedCallback(): void {
    window.removeEventListener("keydown", this.onKey);
    cancelAnimationFrame(this.raf);
    this.view?.dispose();
    this.sim?.free();
    super.disconnectedCallback();
  }

  override firstUpdated(): void {
    this.view = new CourseView(this.canvas, () => {
      this.follow = false;
    });
    const observer = new ResizeObserver(() => this.view?.resize());
    observer.observe(this);
    void this.boot(this.seed);
    this.raf = requestAnimationFrame(this.frame);
  }

  override render() {
    const swatch = `hsl(${Math.round(this.leaderHue * 360)} 72% 58%)`;
    const distance = this.bestDistance.toFixed(2);
    const time = this.bestTime === null ? "—" : `${this.bestTime.toFixed(2)} s`;
    const recordTime = this.recordTime === null ? "no finish yet" : `best finish ${this.recordTime.toFixed(2)} s`;
    const popNote = this.popSize === this.simPop ? `${this.simPop}` : `${this.simPop} → ${this.popSize}`;
    return html`
      <div class="stage">
        <canvas></canvas>
        ${this.error ? html`<div class="loading">${this.error}</div>` : this.ready ? null : html`<div class="loading">Starting physics…</div>`}
        <div class="overlay">
          <header class="panel">
            <div class="brand">
              <h1><span class="mark" style="--swatch:${swatch}"></span>ROBOT COURSE</h1>
              <p>Shape, joint range, and jump are genes. Flexibility and Jump scale how much of that range and hop the field may use. Jump frequency sets how often a grounded robot may take that hop. Distance first, then a finish bonus, then time.</p>
            </div>
            <div class="stats">
              <div class="stat">
                <span>Generation</span>
                <strong>${this.generation}</strong>
                <em>${this.alive} upright</em>
              </div>
              <div class="stat">
                <span>Best distance</span>
                <strong>${distance} m</strong>
                <em>record ${this.recordDistance.toFixed(2)} m</em>
              </div>
              <div class="stat">
                <span>Best time</span>
                <strong>${time}</strong>
                <em>${recordTime}</em>
              </div>
              <div class="stat">
                <span>Population</span>
                <strong>${popNote}</strong>
                <em>seed ${this.seed}</em>
              </div>
            </div>
          </header>
          <footer class="panel">
            <button class="primary" @click=${this.onPlay}>${this.playing ? "Pause" : "Play"}</button>
            <button @click=${this.onStep}>Step</button>
            <button @click=${this.onRestart}>Restart</button>
            <label class="pop">
              Pop ${this.popSize}
              <input
                type="range"
                min=${MIN_POP}
                max=${MAX_POP}
                step="1"
                .value=${String(this.popSize)}
                @input=${this.onPop}
              />
            </label>
            <label class="pop" title="0 is stiff. 100 allows the full hip and knee gene, including crouch and crawl.">
              Flexibility ${this.flexibility}
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                .value=${String(this.flexibility)}
                @input=${this.onFlexibility}
              />
            </label>
            <label class="pop" title="0 means nobody can jump. 100 is the full jump gene. A foot still has to be on the ground.">
              Jump ${this.jump}
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                .value=${String(this.jump)}
                @input=${this.onJump}
              />
            </label>
            <label class="pop" title="0 is almost never. 100 is as often as a foot on the ground allows. No jump in the air, and no second jump until they land.">
              Jump frequency ${this.jumpFrequency}
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                .value=${String(this.jumpFrequency)}
                @input=${this.onJumpFrequency}
              />
            </label>
            <label class="follow">
              <input type="checkbox" .checked=${this.follow} @change=${this.onFollow} />
              Follow leader
            </label>
            <div class="track" title="Generation clock">
              <div class="fill" style="width:${this.progress}%"></div>
            </div>
            <p class="formula">
              score = ${PROGRESS_WEIGHT} × progress
              + (finished ? ${FINISH_BONUS} + ${TIME_BONUS_MAX} × (1 − time / ${GEN_TIME}s) : 0).
              Progress is how far that robot got toward the arch. Time only ranks finishers. Space plays, S steps a generation.
            </p>
          </footer>
        </div>
      </div>
    `;
  }

  private frame = (now: number): void => {
    this.advance(now);
    if (this.view && this.sim) {
      this.view.follow = this.follow;
      this.view.sync(this.sim);
      this.view.render();
      if (now - this.lastHud > 100) {
        this.lastHud = now;
        this.capture(false);
      }
    }
    this.raf = requestAnimationFrame(this.frame);
  };

  private advance(now: number): void {
    const sim = this.sim;
    if (!sim || !this.playing) return;
    if (this.lastNow === 0) this.lastNow = now;
    let dt = (now - this.lastNow) / 1000;
    this.lastNow = now;
    if (dt > 0.05) dt = 0.05;
    this.acc += dt;
    let steps = 0;
    while (this.acc >= SIM_DT && steps < 3 && !sim.complete) {
      sim.step();
      this.acc -= SIM_DT;
      steps += 1;
    }
    if (this.acc > SIM_DT) this.acc = SIM_DT;
    if (!sim.complete) return;
    this.capture(true);
    if (this.pauseAfter) {
      this.playing = false;
      this.pauseAfter = false;
      this.lastNow = 0;
      return;
    }
    sim.nextGeneration(this.popSize);
    this.capture(false);
  }

  private capture(endOfGen: boolean): void {
    const sim = this.sim;
    if (!sim || sim.robots.length === 0) return;
    const best = sim.best();
    this.generation = sim.generation;
    this.bestDistance = best.maxTravel;
    this.bestTime = sim.bestFinishTime();
    this.alive = sim.aliveCount();
    this.simPop = sim.pop;
    this.progress = Math.max(0, Math.min(100, (sim.time / GEN_TIME) * 100));
    this.leaderHue = best.hue;
    if (!endOfGen) return;
    if (best.maxTravel > this.recordDistance) this.recordDistance = best.maxTravel;
    const finish = sim.bestFinishTime();
    if (finish !== null && (this.recordTime === null || finish < this.recordTime)) {
      this.recordTime = finish;
    }
  }

  private onPlay(): void {
    if (!this.sim) return;
    if (this.playing) {
      this.playing = false;
      this.pauseAfter = false;
      this.lastNow = 0;
      return;
    }
    if (this.sim.complete) this.sim.nextGeneration(this.popSize);
    this.pauseAfter = false;
    this.playing = true;
    this.lastNow = 0;
  }

  private onStep(): void {
    if (!this.sim) return;
    if (this.sim.complete) this.sim.nextGeneration(this.popSize);
    this.pauseAfter = true;
    this.playing = true;
    this.lastNow = 0;
  }

  private onPop(event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    if (Number.isFinite(value)) this.popSize = Math.max(MIN_POP, Math.min(MAX_POP, Math.round(value)));
  }

  private onFlexibility(event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    if (!Number.isFinite(value)) return;
    this.flexibility = Math.max(0, Math.min(100, Math.round(value)));
    this.pushExpression();
  }

  private onJump(event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    if (!Number.isFinite(value)) return;
    this.jump = Math.max(0, Math.min(100, Math.round(value)));
    this.pushExpression();
  }

  private onJumpFrequency(event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    if (!Number.isFinite(value)) return;
    this.jumpFrequency = Math.max(0, Math.min(100, Math.round(value)));
    this.pushExpression();
  }

  /** Writes the slider scales. Genes are unchanged. Live robots pick it up immediately. */
  private pushExpression(): void {
    setFlexScale(this.flexibility / 100);
    setJumpScale(this.jump / 100);
    setJumpFrequency(this.jumpFrequency / 100);
    this.sim?.setExpression(this.flexibility / 100, this.jump / 100);
  }

  private onFollow(event: Event): void {
    this.follow = (event.target as HTMLInputElement).checked;
  }

  private async onRestart(): Promise<void> {
    const seed = (Math.random() * 0xffffffff) >>> 0;
    this.seed = seed;
    this.recordDistance = 0;
    this.recordTime = null;
    this.playing = true;
    this.pauseAfter = false;
    this.lastNow = 0;
    this.acc = 0;
    await this.boot(seed);
  }

  private async boot(seed: number): Promise<void> {
    this.sim?.free();
    this.sim = null;
    setFlexScale(this.flexibility / 100);
    setJumpScale(this.jump / 100);
    setJumpFrequency(this.jumpFrequency / 100);
    try {
      this.sim = await CourseSim.create(this.popSize, seed);
      this.sim.setExpression(this.flexibility / 100, this.jump / 100);
      this.ready = true;
      this.error = "";
      this.capture(false);
    } catch (err) {
      this.error = err instanceof Error ? err.message : "Physics failed to start.";
    }
  }
}

declare global {
  interface HTMLElementTagNameMap {
    "robot-course": RobotCourse;
  }
}
