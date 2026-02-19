import { useGLTF } from "@react-three/drei/webgpu";
import { useFrame, useTexture } from "@react-three/fiber/webgpu";
import { useCallback, useMemo, useRef } from "react";
import {
  dot,
  float,
  normalView,
  positionViewDirection,
  pow,
  screenUV,
  texture,
  vec3,
  viewportSharedTexture,
} from "three/tsl";
import * as THREE from "three/webgpu";
import { MeshStandardNodeMaterial, RepeatWrapping } from "three/webgpu";

const GRID = 3;
const SPACING = 2.25;
const HALF = ((GRID - 1) * SPACING) / 2;
const INNER_SCALE = 0.98;

// Reusable vectors for depth sorting
const _depthVec = new THREE.Vector3();
const _camDir = new THREE.Vector3();

export const Cube = () => {
  const { nodes } = useGLTF("/cube.glb");
  const nTex = useTexture("./noise.png");
  const wrappedNoiseTexture = useMemo(() => {
    const tex = nTex.clone();
    tex.wrapS = tex.wrapT = RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
  }, [nTex]);

  const createCubeMaterial = useCallback(
    (side: THREE.Side, isInner: boolean) => {
      const m = new MeshStandardNodeMaterial({
        roughness: 0,
        transparent: true,
        side,
        opacity: 0.99,
      });
      m.depthWrite = true;
      const ca = float(0.005);

      if(isInner) {
        const n = texture(wrappedNoiseTexture, screenUV.mul(40));
        const fresnel = pow(
          dot(normalView, positionViewDirection).oneMinus(),
          0.5,
        );
  
        const vUv = screenUV.add(
          normalView.xy.mul(0.1).add(n.rg.sub(0.5).mul(0.03)),
        );
        const r = viewportSharedTexture(vUv.add(ca.mul(fresnel))).r;
        const g = viewportSharedTexture(vUv).g;
        const b = viewportSharedTexture(vUv.sub(ca.mul(fresnel))).b;


      const tint = vec3(0.9412, 0.2902, 0.0); // #f04a00
      m.backdropNode = vec3(r, g, b).mul(tint).add(fresnel.mul(0.1));
      } else {
        const r = viewportSharedTexture(screenUV).r;
        const g = viewportSharedTexture(screenUV).g;
        const b = viewportSharedTexture(screenUV).b;

        const tint = vec3(0.9412, 0.2902, 0.0); // #f04a00
        m.backdropNode = vec3(r, g, b).mul(tint)
        
      }

      return m;
    },
    [wrappedNoiseTexture],
  );

  // One material pair per cube so each gets its own viewportSharedTexture capture
  const { outerMaterials, innerMaterials } = useMemo(() => {
    return {
      outerMaterials: Array.from({ length: GRID * GRID * GRID }, () =>
        createCubeMaterial(THREE.FrontSide, false),
      ),
      innerMaterials: Array.from({ length: GRID * GRID * GRID }, () =>
        createCubeMaterial(THREE.BackSide, true),
      ),
    };
  }, [createCubeMaterial]);

  // Build a 5×5×5 grid of positions centred at origin
  const positions = useMemo<[number, number, number][]>(() => {
    const out: [number, number, number][] = [];
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        for (let k = 0; k < GRID; k++) {
          out.push([
            i * SPACING - HALF,
            j * SPACING - HALF,
            k * SPACING - HALF,
          ]);
        }
      }
    }
    return out;
  }, []);

  // Keep refs to every cube pair for per-frame renderOrder sorting
  const outerRefs = useRef<(THREE.Mesh | null)[]>([]);
  const innerRefs = useRef<(THREE.Mesh | null)[]>([]);
  const setOuterRef = useCallback(
    (index: number) => (el: THREE.Mesh | null) => {
      outerRefs.current[index] = el;
    },
    [],
  );
  const setInnerRef = useCallback(
    (index: number) => (el: THREE.Mesh | null) => {
      innerRefs.current[index] = el;
    },
    [],
  );

  // Sort renderOrder back-to-front each frame so each cube's
  // viewportSharedTexture / backdropNode captures the cubes behind it
  useFrame(({ camera }) => {
    const outerMeshes = outerRefs.current;
    const innerMeshes = innerRefs.current;
    const n = outerMeshes.length;
    if (n <= 1) return;

    camera.getWorldDirection(_camDir);
    const cp = camera.position;

    const entries: { idx: number; depth: number }[] = [];
    for (let i = 0; i < n; i++) {
      const m = outerMeshes[i];
      if (!m) continue;
      m.getWorldPosition(_depthVec);
      const depth =
        (_depthVec.x - cp.x) * _camDir.x +
        (_depthVec.y - cp.y) * _camDir.y +
        (_depthVec.z - cp.z) * _camDir.z;
      entries.push({ idx: i, depth });
    }

    // Farthest first → lowest renderOrder → drawn first
    entries.sort((a, b) => b.depth - a.depth);

    for (let order = 0; order < entries.length; order++) {
      const idx = entries[order].idx;
      const inner = innerMeshes[idx];
      const outer = outerMeshes[idx];
      // Draw inner first, then outer, while preserving cube depth ordering.
      if (inner) inner.renderOrder = order * 2;
      if (outer) outer.renderOrder = order * 2 + 1;
    }
  });

  return (
    <group>
      {positions.map((pos, i) => (
        <group key={i} position={pos}>
          <mesh
            ref={setOuterRef(i)}
            geometry={(nodes.Cube as THREE.Mesh).geometry}
            material={outerMaterials[i]}
          />
          <mesh
            ref={setInnerRef(i)}
            geometry={(nodes.Cube as THREE.Mesh).geometry}
            material={innerMaterials[i]}
            scale={INNER_SCALE}
          />
        </group>
      ))}
    </group>
  );
};
