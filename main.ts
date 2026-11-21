// Sokoban solver: decompose a level into rooms by its doors, solve every room
// push-optimal with A* over pushes, then stitch the rooms together with walks
// through the hallway. Walking steps never cost pushes, so per-room push-optimal
// solutions concatenate into a push-optimal whole-level solution.
//
// The generator lays one hallway row across the board with rooms above and below,
// each room opening through a single door cell in the wall row beside the hallway.
// A box pushed through a door into the hallway can never come back, so in an
// optimal solution every box stays in its room: rooms are independent problems.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const START = Date.now();
const TIME_BUDGET_MS = 48_000; // the whole set must finish inside the 60 s wall
const MAX_EXPANSIONS = 1_500_000; // per room search; deterministic cap, not a clock

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet: new Set(goals), boxes, player };
}

const keyOf = (p: number, bs: number[]): string => p + '|' + bs.join(',');

/** BFS distances from `from` over the given floor cells (boxes ignored). */
function bfsDist(from: number, floor: Set<number>, w: number): Map<number, number> {
  const dist = new Map<number, number>([[from, 0]]);
  const q = [from];
  for (let i = 0; i < q.length; i++) {
    const c = q[i]!;
    const cx = c % w;
    const cy = (c - cx) / w;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (!floor.has(n) || dist.has(n)) continue;
      dist.set(n, dist.get(c)! + 1);
      q.push(n);
    }
  }
  return dist;
}

/** Shortest walk for the player from `from` to `to`, avoiding boxes. */
function walkPath(
  from: number,
  to: number,
  floor: Set<number>,
  boxSet: Set<number>,
  w: number,
): string | null {
  if (from === to) return '';
  const prev = new Map<number, [number, string]>();
  const q = [from];
  prev.set(from, [-1, '']);
  for (let i = 0; i < q.length; i++) {
    const c = q[i]!;
    const cx = c % w;
    const cy = (c - cx) / w;
    for (const [dx, dy, ch] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (!floor.has(n) || boxSet.has(n) || prev.has(n)) continue;
      prev.set(n, [c, ch]);
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          const [p, step] = prev.get(cur)!;
          s = step + s;
          cur = p;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

interface RoomProblem {
  floor: Set<number>; // cells the player may stand on
  boxCells: Set<number>; // cells a box may occupy
  goals: number[];
  boxes: number[];
  player: number; // start square of the player
}

/** Push-optimal A* over pushes. Returns the move string and the end player square. */
function solveRoom(prob: RoomProblem, w: number): { moves: string; player: number } | null {
  const { floor, boxCells, goals, boxes } = prob;
  const goalSet = new Set(goals);
  if (boxes.every((b) => goalSet.has(b))) return { moves: '', player: prob.player };
  if (boxes.length > goals.length) return null;

  // All-pairs box distances over the floor (boxes ignored): a lower bound on pushes.
  const distTo = new Map<number, Map<number, number>>();
  for (const c of floor) distTo.set(c, bfsDist(c, floor, w));
  const dist = (a: number, b: number): number => distTo.get(a)?.get(b) ?? Infinity;
  // A box cell is dead when no goal is reachable from it at all.
  const dead = new Set<number>();
  for (const c of boxCells) {
    if (goalSet.has(c)) continue;
    if (!goals.some((g) => dist(c, g) < Infinity)) dead.add(c);
  }
  if (boxes.some((b) => dead.has(b))) return null;

  // Minimum assignment of boxes to goals (brute force; rooms hold at most 4 boxes).
  const heuristic = (bs: number[]): number => {
    const n = bs.length;
    if (n === 0) return 0;
    if (n > 4) {
      let s = 0;
      for (const b of bs) {
        let m = Infinity;
        for (const g of goals) m = Math.min(m, dist(b, g));
        s += m;
      }
      return s;
    }
    let best = Infinity;
    const used = new Array<boolean>(goals.length).fill(false);
    const rec = (i: number, acc: number): void => {
      if (acc >= best) return;
      if (i === n) {
        best = acc;
        return;
      }
      for (let j = 0; j < goals.length; j++) {
        if (used[j]) continue;
        const d = dist(bs[i]!, goals[j]!);
        if (d === Infinity) continue;
        used[j] = true;
        rec(i + 1, acc + d);
        used[j] = false;
      }
    };
    rec(0, 0);
    return best;
  };

  // Binary heap on (f asc, g desc, seq asc) so ties break deterministically.
  interface HeapEntry {
    f: number;
    g: number;
    seq: number;
    state: number;
  }
  const heap: HeapEntry[] = [];
  const less = (a: HeapEntry, b: HeapEntry): boolean =>
    a.f < b.f || (a.f === b.f && (a.g > b.g || (a.g === b.g && a.seq < b.seq)));
  const heapPush = (e: HeapEntry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): HeapEntry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const players: number[] = [prob.player];
  const states: number[][] = [boxes.slice()];
  const gs: number[] = [0];
  const parents: number[] = [-1];
  const segs: string[] = [''];
  const best = new Map<string, number>([[keyOf(prob.player, boxes), 0]]);
  let seq = 0;
  heapPush({ f: heuristic(boxes), g: 0, seq: seq++, state: 0 });
  let expansions = 0;

  while (heap.length > 0 && expansions < MAX_EXPANSIONS) {
    const { g, state } = heapPop();
    const p = players[state]!;
    const bs = states[state]!;
    if (best.get(keyOf(p, bs)) !== g) continue;
    if (bs.every((b) => goalSet.has(b))) {
      let moves = '';
      for (let t = state; t > 0; t = parents[t]!) moves = segs[t] + moves;
      return { moves, player: p };
    }
    expansions++;
    const boxSet = new Set(bs);
    // One BFS for everywhere the player can stand in this state; every push
    // candidate reads its walk from this parent map.
    const reach = new Map<number, [number, string]>();
    {
      const q = [p];
      reach.set(p, [-1, '']);
      for (let i = 0; i < q.length; i++) {
        const c = q[i]!;
        const cx = c % w;
        const cy = (c - cx) / w;
        for (const [dx, dy, ch] of DIRS) {
          const nx = cx + dx;
          const ny = cy + dy;
          if (nx < 0 || ny < 0) continue;
          const n = ny * w + nx;
          if (!floor.has(n) || boxSet.has(n) || reach.has(n)) continue;
          reach.set(n, [c, ch]);
          q.push(n);
        }
      }
    }
    const pathTo = (to: number): string | null => {
      if (!reach.has(to)) return null;
      let s = '';
      let cur = to;
      while (cur !== p) {
        const [par, step] = reach.get(cur)!;
        s = step + s;
        cur = par;
      }
      return s;
    };
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi]!;
      const bx = b % w;
      const by = (b - bx) / w;
      for (const [dx, dy, ch] of DIRS) {
        const nx = bx + dx;
        const ny = by + dy;
        if (nx < 0 || ny < 0) continue;
        const t = ny * w + nx;
        if (!boxCells.has(t) || boxSet.has(t)) continue;
        const fromX = bx - dx;
        const fromY = by - dy;
        if (fromX < 0 || fromY < 0) continue;
        const from = fromY * w + fromX;
        if (!floor.has(from) || boxSet.has(from)) continue;
        const path = pathTo(from);
        if (path === null) continue;
        const nb = bs.slice();
        nb[bi] = t;
        nb.sort((a, b2) => a - b2);
        if (nb.some((cell) => dead.has(cell))) continue;
        const h = heuristic(nb);
        if (h === Infinity) continue;
        const ng = g + 1;
        const key = keyOf(b, nb);
        const prev = best.get(key);
        if (prev !== undefined && prev <= ng) continue;
        best.set(key, ng);
        const idx = states.length;
        players.push(b);
        states.push(nb);
        gs.push(ng);
        parents.push(state);
        segs.push(path + ch);
        heapPush({ f: ng + h, g: ng, seq: seq++, state: idx });
      }
    }
  }
  return null;
}

interface Room {
  door: number;
  entry: number; // hallway square the player stands on to enter
  cells: number[];
  boxes: number[];
  goals: number[];
}

/** Split a level into rooms. Every box must land in exactly one room. */
function decompose(lv: Level): Room[] | null {
  const { w, wall, player } = lv;
  const floor = new Set<number>();
  for (let i = 0; i < wall.length; i++) if (!wall[i]) floor.add(i);
  const hallwayRow = Math.floor(player / w);
  const boxSet = new Set(lv.boxes);
  const goalSet = lv.goalSet;
  const rooms: Room[] = [];
  const assigned = new Set<number>();

  const reachedFrom = (start: number, blocked: number): Set<number> => {
    const seen = new Set<number>([start]);
    const q = [start];
    for (let i = 0; i < q.length; i++) {
      const c = q[i]!;
      const cx = c % w;
      const cy = (c - cx) / w;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0) continue;
        const n = ny * w + nx;
        if (n === blocked || !floor.has(n) || seen.has(n)) continue;
        seen.add(n);
        q.push(n);
      }
    }
    return seen;
  };

  for (const d of [...floor].sort((a, b) => a - b)) {
    const dy = Math.floor(d / w) - hallwayRow;
    if (dy !== 1 && dy !== -1) continue; // doors sit in the wall row beside the hallway
    const reached = reachedFrom(player, d);
    if (reached.has(d)) continue;
    // Cells cut off by removing d, grouped into connected components.
    const cut = [...floor].filter((c) => c !== d && !reached.has(c));
    const cutSet = new Set(cut);
    const seen = new Set<number>();
    for (const c of cut) {
      if (seen.has(c)) continue;
      const comp: number[] = [];
      const q = [c];
      seen.add(c);
      for (let i = 0; i < q.length; i++) {
        const cur = q[i]!;
        comp.push(cur);
        const cx = cur % w;
        const cy = (cur - cx) / w;
        for (const [dx, dy2] of DIRS) {
          const nx = cx + dx;
          const ny = cy + dy2;
          if (nx < 0 || ny < 0) continue;
          const n = ny * w + nx;
          if (!cutSet.has(n) || seen.has(n)) continue;
          seen.add(n);
          q.push(n);
        }
      }
      const compSet = new Set(comp);
      const roomBoxes = lv.boxes.filter((b) => compSet.has(b) || b === d);
      if (roomBoxes.length === 0) continue;
      // The hallway square beside the door: a reached neighbour of d.
      let entry = -1;
      const dx = d % w;
      const dyy = (d - dx) / w;
      for (const [ox, oy] of DIRS) {
        const n = (dyy + oy) * w + (dx + ox);
        if (reached.has(n)) {
          entry = n;
          break;
        }
      }
      if (entry < 0) return null;
      const roomGoals = lv.goals.filter((g) => compSet.has(g) || g === d);
      for (const b of roomBoxes) assigned.add(b);
      rooms.push({ door: d, entry, cells: comp, boxes: roomBoxes, goals: roomGoals });
    }
  }
  if (lv.boxes.some((b) => !assigned.has(b))) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

/** Apply a move string to the global board; returns false on an illegal step. */
function applyMoves(
  lv: Level,
  player: number,
  boxes: Set<number>,
  moves: string,
): { player: number } | null {
  let p = player;
  for (const ch of moves) {
    const dir = DIRS.find(([, , c]) => c === ch);
    if (!dir) return null;
    const d = dir[1] * lv.w + dir[0];
    const next = p + d;
    if (lv.wall[next]) return null;
    if (boxes.has(next)) {
      const beyond = next + d;
      if (lv.wall[beyond] || boxes.has(beyond)) return null;
      boxes.delete(next);
      boxes.add(beyond);
    }
    p = next;
  }
  return { player: p };
}

function solve(text: string): string {
  const lv = parse(text);
  const floor = new Set<number>();
  for (let i = 0; i < lv.wall.length; i++) if (!lv.wall[i]) floor.add(i);
  const boxes = new Set(lv.boxes);
  let player = lv.player;
  let moves = '';

  const rooms = decompose(lv);
  const jobs: { prob: RoomProblem }[] = [];
  if (rooms !== null) {
    for (const r of rooms) {
      const roomFloor = new Set(r.cells);
      roomFloor.add(r.door);
      roomFloor.add(r.entry);
      const boxCells = new Set(r.cells);
      boxCells.add(r.door);
      jobs.push({
        prob: { floor: roomFloor, boxCells, goals: r.goals, boxes: r.boxes, player: r.entry },
      });
    }
  } else {
    // Fallback: no clean decomposition, solve the whole level as one room.
    jobs.push({
      prob: { floor, boxCells: new Set(floor), goals: lv.goals, boxes: lv.boxes, player: lv.player },
    });
  }

  for (const { prob } of jobs) {
    const isRoom = rooms !== null;
    if (isRoom) {
      // Walk through the hallway to the room's entry, never touching a box.
      const path = walkPath(player, prob.player, floor, boxes, lv.w);
      if (path === null) return '';
      const applied = applyMoves(lv, player, boxes, path);
      if (applied === null) return '';
      player = applied.player;
      moves += path;
    }
    const solved = solveRoom(prob, lv.w);
    if (solved === null) return '';
    const applied = applyMoves(lv, player, boxes, solved.moves);
    if (applied === null) return '';
    player = applied.player;
    moves += solved.moves;
  }
  if ([...boxes].some((b) => !lv.goalSet.has(b))) return '';
  return moves;
}

for (const id of readdirSync(I).sort()) {
  if (Date.now() - START > TIME_BUDGET_MS) {
    mkdirSync(join(O, id), { recursive: true });
    writeFileSync(join(O, id, 'solution.txt'), '\n');
    continue;
  }
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
