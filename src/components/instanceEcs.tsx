import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three/webgpu";

// ---------------------------------------------------------------------------
// Shared temporaries (module-level, avoids GC pressure)
// ---------------------------------------------------------------------------

const _mat4 = new THREE.Matrix4();
const _quat = new THREE.Quaternion();
const _color = new THREE.Color();

// ---------------------------------------------------------------------------
// Radix-sort helpers (module-level to avoid GC pressure)
// ---------------------------------------------------------------------------

const _sortVec = new THREE.Vector3();
const _camDir = new THREE.Vector3();

// Shared float↔uint conversion buffer
const _f2uBuf = new ArrayBuffer(4);
const _f2uFloat = new Float32Array(_f2uBuf);
const _f2uUint = new Uint32Array(_f2uBuf);

/** Convert an IEEE-754 float to a uint32 that preserves total ordering. */
function floatToSortableUint(f: number): number {
  _f2uFloat[0] = f;
  let bits = _f2uUint[0];
  // Negative floats: flip all bits. Positive: flip sign bit only.
  if (bits & 0x80000000) {
    bits = ~bits;
  } else {
    bits |= 0x80000000;
  }
  return bits >>> 0;
}

// Scratch arrays – grown once, reused every frame
let _rKeys = new Uint32Array(0);
let _rIndices = new Uint32Array(0);
let _rKeysT = new Uint32Array(0);
let _rIndicesT = new Uint32Array(0);

const RADIX_BITS = 8;
const RADIX_SIZE = 1 << RADIX_BITS; // 256
const RADIX_MASK = RADIX_SIZE - 1;
const _counts = new Uint32Array(RADIX_SIZE);

/**
 * LSB radix sort (4 passes of 8-bit digits over 32-bit keys).
 * Sorts `keys` and `indices` arrays in ascending key order, in-place.
 */
function radixSortPairs(
  keys: Uint32Array,
  indices: Uint32Array,
  n: number,
): void {
  let srcK = keys,
    srcI = indices;
  let dstK = _rKeysT,
    dstI = _rIndicesT;

  for (let shift = 0; shift < 32; shift += RADIX_BITS) {
    // --- histogram ---
    _counts.fill(0);
    for (let i = 0; i < n; i++) {
      _counts[(srcK[i] >>> shift) & RADIX_MASK]++;
    }
    // --- prefix sum ---
    let sum = 0;
    for (let b = 0; b < RADIX_SIZE; b++) {
      const c = _counts[b];
      _counts[b] = sum;
      sum += c;
    }
    // --- scatter ---
    for (let i = 0; i < n; i++) {
      const bucket = (srcK[i] >>> shift) & RADIX_MASK;
      const pos = _counts[bucket]++;
      dstK[pos] = srcK[i];
      dstI[pos] = srcI[i];
    }
    // --- swap src/dst ---
    [srcK, dstK] = [dstK, srcK];
    [srcI, dstI] = [dstI, srcI];
  }

  // After 4 passes result lives in srcK/srcI.
  // If that's *not* the original buffer, copy back.
  if (srcK !== keys) {
    keys.set(srcK.subarray(0, n));
    indices.set(srcI.subarray(0, n));
  }
}

function ensureScratch(n: number): void {
  if (_rKeys.length >= n) return;
  _rKeys = new Uint32Array(n);
  _rIndices = new Uint32Array(n);
  _rKeysT = new Uint32Array(n);
  _rIndicesT = new Uint32Array(n);
}

// ---------------------------------------------------------------------------
// InstancedEntity
// ---------------------------------------------------------------------------

export class InstancedEntity {
  /** @internal */ _index: number;
  /** @internal */ _pool: InstancePool;
  /** @internal */ _removed: boolean;

  readonly position: THREE.Vector3;
  readonly rotation: THREE.Euler;
  readonly quaternion: THREE.Quaternion;
  readonly scale: THREE.Vector3;
  readonly color: THREE.Color;

  // Allow arbitrary user data (velocity, currentTime, etc.)
  [key: string]: unknown;

  constructor(pool: InstancePool, index: number) {
    this._pool = pool;
    this._index = index;
    this._removed = false;
    this.position = new THREE.Vector3();
    this.rotation = new THREE.Euler();
    this.quaternion = new THREE.Quaternion();
    this.scale = new THREE.Vector3(1, 1, 1);
    this.color = new THREE.Color(1, 1, 1);
  }

  updateMatrix(): void {
    _mat4.compose(this.position, this.quaternion, this.scale);
    this._pool._mesh.setMatrixAt(this._index, _mat4);
    if (this._pool._mesh.instanceColor) {
      this._pool._mesh.setColorAt(this._index, this.color);
    }
  }

  /** Mark this entity for removal. Actual removal (swap-and-pop) happens at
   *  the end of the next `updateInstances` call. */
  remove(): void {
    this._removed = true;
  }
}

// ---------------------------------------------------------------------------
// InstancePool
// ---------------------------------------------------------------------------

const DEFAULT_CAPACITY = 128;
const GROW_FACTOR = 2;

export class InstancePool {
  /** @internal */ _mesh: THREE.InstancedMesh;
  /** Current buffer capacity. Grows automatically when exceeded. */
  capacity: number;
  activeCount: number;
  /** @internal */ _entities: (InstancedEntity | null)[];

  constructor(mesh: THREE.InstancedMesh, capacity?: number) {
    this._mesh = mesh;
    this.capacity = capacity ?? mesh.count ?? DEFAULT_CAPACITY;
    this.activeCount = 0;
    this._entities = new Array<InstancedEntity | null>(this.capacity).fill(
      null,
    );

    // Start with nothing visible
    mesh.count = 0;
  }

  // ---- Public API ---------------------------------------------------------

  /**
   * Allocate `count` new instances at the end of the active range.
   * If the buffer is too small, it grows automatically.
   */
  addInstances(
    count: number,
    setup: (entity: InstancedEntity, index: number) => void,
  ): void {
    const needed = this.activeCount + count;
    if (needed > this.capacity) {
      this._grow(needed);
    }

    for (let i = this.activeCount; i < needed; i++) {
      const entity = new InstancedEntity(this, i);
      this._entities[i] = entity;
      setup(entity, i - this.activeCount);
      entity.updateMatrix();
    }

    this.activeCount = needed;
    this._mesh.count = this.activeCount;
    this._mesh.instanceMatrix.needsUpdate = true;
  }

  /**
   * Iterate all active entities with `callback`, then process deferred
   * removals (swap-and-pop) and auto-call `updateMatrix()` on surviving
   * entities.
   *
   * Two-phase design:
   *  1. Forward pass — user callback on every active entity.
   *  2. Backward pass — removals + matrix writes.
   *
   * Backward iteration in phase 2 ensures swapped-in entities are not
   * skipped.
   */
  updateInstances(
    callback: (entity: InstancedEntity, index: number) => void,
  ): void {
    // Phase 1: user logic
    for (let i = 0; i < this.activeCount; i++) {
      callback(this._entities[i]!, i);
    }

    // Phase 2: removals + matrix writes (iterate backward)
    for (let i = this.activeCount - 1; i >= 0; i--) {
      const entity = this._entities[i]!;
      if (entity._removed) {
        this._swapRemove(i);
      } else {
        entity.updateMatrix();
      }
    }

    this._mesh.count = this.activeCount;
    this._mesh.instanceMatrix.needsUpdate = true;
    if (this._mesh.instanceColor) {
      this._mesh.instanceColor.needsUpdate = true;
    }
  }

  /**
   * Radix-sort all active instances **back-to-front** relative to `camera`.
   * Call once per frame (before rendering) so each refractive cube's
   * `viewportSharedTexture` / `backdropNode` captures the cubes behind it.
   */
  sortByDepth(camera: THREE.Camera): void {
    const n = this.activeCount;
    if (n <= 1) return;

    ensureScratch(n);

    // Camera world-space forward & position
    camera.getWorldDirection(_camDir);
    const cp = camera.getWorldPosition(_sortVec);

    // Build keys: negate signed depth so ascending sort → back-to-front
    for (let i = 0; i < n; i++) {
      const e = this._entities[i]!;
      const depth =
        (e.position.x - cp.x) * _camDir.x +
        (e.position.y - cp.y) * _camDir.y +
        (e.position.z - cp.z) * _camDir.z;
      _rKeys[i] = floatToSortableUint(-depth); // negate → far first
      _rIndices[i] = i;
    }

    radixSortPairs(_rKeys, _rIndices, n);

    // Check if order actually changed (skip expensive reorder when static)
    let changed = false;
    for (let i = 0; i < n; i++) {
      if (_rIndices[i] !== i) {
        changed = true;
        break;
      }
    }
    if (!changed) return;

    // Reorder entity array + rewrite matrices according to new order
    const tmp = new Array<InstancedEntity | null>(this.capacity).fill(null);
    for (let i = 0; i < n; i++) {
      const entity = this._entities[_rIndices[i]]!;
      entity._index = i;
      tmp[i] = entity;
      entity.updateMatrix();
    }
    this._entities = tmp;

    this._mesh.instanceMatrix.needsUpdate = true;
    if (this._mesh.instanceColor) {
      this._mesh.instanceColor.needsUpdate = true;
    }
  }

  clear(): void {
    for (let i = 0; i < this.activeCount; i++) {
      this._entities[i] = null;
    }
    this.activeCount = 0;
    this._mesh.count = 0;
    this._mesh.instanceMatrix.needsUpdate = true;
  }

  // ---- Internal -----------------------------------------------------------

  /** @internal Swap entity at `index` with the last active entity, then
   *  shrink activeCount by 1. */
  private _swapRemove(index: number): void {
    const lastIndex = this.activeCount - 1;

    if (index !== lastIndex) {
      const lastEntity = this._entities[lastIndex]!;
      lastEntity._index = index;
      this._entities[index] = lastEntity;
      lastEntity.updateMatrix(); // re-write matrix at the new index
    }

    this._entities[lastIndex] = null;
    this.activeCount--;
  }

  /** @internal Grow the underlying InstancedMesh buffer and entity array. */
  private _grow(minCapacity: number): void {
    let newCapacity = this.capacity || DEFAULT_CAPACITY;
    while (newCapacity < minCapacity) {
      newCapacity *= GROW_FACTOR;
    }

    const mesh = this._mesh;

    // Resize the InstancedMesh — Three.js r149+ supports resize()
    // but for broader compat we rebuild the instanceMatrix buffer.
    const oldArray = mesh.instanceMatrix.array as Float32Array;
    const newArray = new Float32Array(newCapacity * 16);
    newArray.set(oldArray); // copy existing data

    // Fill new slots with identity matrices
    const identity = new THREE.Matrix4();
    const identityElements = identity.elements;
    for (let i = this.capacity; i < newCapacity; i++) {
      newArray.set(identityElements, i * 16);
    }

    mesh.instanceMatrix = new THREE.InstancedBufferAttribute(newArray, 16);
    mesh.instanceMatrix.needsUpdate = true;

    // Grow the entity array
    const newEntities = new Array<InstancedEntity | null>(newCapacity).fill(
      null,
    );
    for (let i = 0; i < this.activeCount; i++) {
      newEntities[i] = this._entities[i];
    }
    this._entities = newEntities;

    this.capacity = newCapacity;
  }
}

// ---------------------------------------------------------------------------
// Hook: attach a pool to any InstancedMesh ref
// ---------------------------------------------------------------------------

/**
 * Attach an InstancePool to an existing `<instancedMesh>` ref.
 *
 * ```tsx
 * const meshRef = useRef<THREE.InstancedMesh>(null!)
 * const pool = useInstancedMesh2(meshRef)
 *
 * useFrame((_, delta) => {
 *   pool.current.addInstances(1, (obj) => { ... })
 *   pool.current.updateInstances((obj) => { ... })
 * })
 *
 * return (
 *   <instancedMesh ref={meshRef} args={[geometry, material, 100]}>
 *     ...
 *   </instancedMesh>
 * )
 * ```
 */
export function useInstancedMesh2(
  meshRef: React.RefObject<THREE.InstancedMesh | null>,
): React.RefObject<InstancePool | null> {
  const poolRef = useRef<InstancePool | null>(null);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    poolRef.current = new InstancePool(mesh);
    return () => {
      poolRef.current = null;
    };
  }, [meshRef]);

  return poolRef;
}

export interface InstancesRef {
  addInstances: (
    count: number,
    setup: (entity: InstancedEntity, index: number) => void,
  ) => void;
  updateInstances: (
    callback: (entity: InstancedEntity, index: number) => void,
  ) => void;
  sortByDepth: (camera: THREE.Camera) => void;
  clear: () => void;
  readonly activeCount: number;
  readonly mesh: THREE.InstancedMesh;
}

export interface InstancesProps {
  /** Pass [geometry, material, count] to set the mesh directly, similar to <instancedMesh args={...}>. */
  args?: [THREE.BufferGeometry, THREE.Material, number];
  maxInstances?: number;
  frustumCulled?: boolean;
  castShadow?: boolean;
  receiveShadow?: boolean;
  renderOrder?: number;
  children?: React.ReactNode;
}

/**
 * Declarative R3F wrapper around InstancedMesh with a pool-based API.
 *
 * Option A — declarative children:
 * ```tsx
 * <Instances ref={ref} maxInstances={1000}>
 *   <sphereGeometry args={[0.1, 16, 16]} />
 *   <meshStandardNodeMaterial color="hotpink" />
 * </Instances>
 * ```
 *
 * Option B — use the hook with your own instancedMesh:
 * ```tsx
 * const meshRef = useRef(null!)
 * const pool = useInstancedMesh2(meshRef)
 * <instancedMesh ref={meshRef} args={[geometry, material, 100]} />
 * ```
 */
export const Instances = forwardRef<InstancesRef, InstancesProps>(
  function Instances(
    {
      args,
      maxInstances,
      frustumCulled = false,
      castShadow = false,
      receiveShadow = false,
      renderOrder,
      children,
    },
    ref,
  ) {
    const meshRef = useRef<THREE.InstancedMesh>(null!);
    const poolRef = useRef<InstancePool | null>(null);

    const initialCapacity = args?.[2] ?? maxInstances ?? DEFAULT_CAPACITY;
    const geometry = args?.[0];
    const material = args?.[1];

    useEffect(() => {
      const mesh = meshRef.current;
      if (!mesh) return;

      const identity = new THREE.Matrix4();
      const white = new THREE.Color(1, 1, 1);
      for (let i = 0; i < initialCapacity; i++) {
        mesh.setMatrixAt(i, identity);
        mesh.setColorAt(i, white);
      }
      mesh.count = 0;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

      poolRef.current = new InstancePool(mesh, initialCapacity);

      return () => {
        poolRef.current = null;
      };
    }, [initialCapacity]);

    useImperativeHandle(
      ref,
      () => ({
        addInstances(count, setup) {
          poolRef.current?.addInstances(count, setup);
        },
        updateInstances(callback) {
          poolRef.current?.updateInstances(callback);
        },
        sortByDepth(camera: THREE.Camera) {
          poolRef.current?.sortByDepth(camera);
        },
        clear() {
          poolRef.current?.clear();
        },
        get activeCount() {
          return poolRef.current?.activeCount ?? 0;
        },
        get mesh() {
          return meshRef.current;
        },
      }),
      [],
    );

    return (
      <instancedMesh
        ref={meshRef}
        args={[geometry ?? undefined!, material ?? undefined!, initialCapacity]}
        frustumCulled={frustumCulled}
        castShadow={castShadow}
        receiveShadow={receiveShadow}
        renderOrder={renderOrder}
      >
        {children}
      </instancedMesh>
    );
  },
);
