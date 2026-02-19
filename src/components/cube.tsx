import { useGLTF } from "@react-three/drei/webgpu";
import { useFrame, useTexture } from "@react-three/fiber/webgpu";
import { useCallback, useMemo, useRef } from "react";
import {
  dot,
  float,
  normalView,
  normalWorld,
  positionViewDirection,
  pow,
  refract,
  screenUV,
  texture,
  vec3,
  viewportSharedTexture,
} from "three/tsl";
import * as THREE from "three/webgpu";
import { MeshStandardNodeMaterial, RepeatWrapping } from "three/webgpu";

const GRID = 5;
const SPACING = 4;
const HALF = ((GRID - 1) * SPACING) / 2;

// Reusable vectors for depth sorting
const _depthVec = new THREE.Vector3();
const _camDir = new THREE.Vector3();

export const Cube = () => {
  const { nodes } = useGLTF("/cube.glb");
  const nTex = useTexture("./noise.png");
  nTex.wrapS = nTex.wrapT = RepeatWrapping;

  // One material per cube so each gets its own viewportSharedTexture capture
  const materials = useMemo(() => {
    return Array.from({ length: GRID * GRID }, () => {
      const m = new MeshStandardNodeMaterial({ roughness: 0 });
      m.side = THREE.DoubleSide;
      m.transparent = true;
      const ca = float(0.0035);

      const n = texture(nTex, screenUV.mul(40));
      const fresnel = pow(
        dot(normalView, positionViewDirection).oneMinus(),
        0.5,
      );

      const vUv = screenUV.add(
        refract(normalView.add(n.r.mul(0.2)), normalWorld, 1.0005).mul(0.1),
      );
      const r = viewportSharedTexture(vUv.add(ca.mul(fresnel))).r;
      const g = viewportSharedTexture(vUv).g;
      const b = viewportSharedTexture(vUv.sub(ca.mul(fresnel))).b;

      m.backdropNode = vec3(r, g, b).add(fresnel.mul(0.1));
      m.opacityNode = 0.8;

      return m;
    });
  }, [nTex]);

  // Build a 5×5 grid of positions centred at origin
  const positions = useMemo<[number, number, number][]>(() => {
    const out: [number, number, number][] = [];
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        out.push([i * SPACING - HALF, 0, j * SPACING - HALF]);
      }
    }
    return out;
  }, []);

  // Keep refs to every mesh for per-frame renderOrder sorting
  const meshRefs = useRef<(THREE.Mesh | null)[]>([]);
  const setMeshRef = useCallback(
    (index: number) => (el: THREE.Mesh | null) => {
      meshRefs.current[index] = el;
    },
    [],
  );

  // Sort renderOrder back-to-front each frame so each cube's
  // viewportSharedTexture / backdropNode captures the cubes behind it
  useFrame(({ camera }) => {
    const meshes = meshRefs.current;
    const n = meshes.length;
    if (n <= 1) return;

    camera.getWorldDirection(_camDir);
    const cp = camera.position;

    const entries: { idx: number; depth: number }[] = [];
    for (let i = 0; i < n; i++) {
      const m = meshes[i];
      if (!m) continue;
      m.getWorldPosition(_depthVec);
      const depth =
        (_depthVec.x - cp.x) * _camDir.x +
        (_depthVec.y - cp.y) * _camDir.y +
        (_depthVec.z - cp.z) * _camDir.z;
      entries.push({ idx: i, depth });
    }

    // Farthest first → lowest renderOrder → drawn first
    entries.sort((a, b) => a.depth - b.depth);

    for (let order = 0; order < entries.length; order++) {
      const m = meshes[entries[order].idx];
      if (m) m.renderOrder = order;
    }
  });

  return (
    <group>
      {positions.map((pos, i) => (
        <mesh
          key={i}
          ref={setMeshRef(i)}
          geometry={(nodes.Cube as THREE.Mesh).geometry}
          material={materials[i]}
          position={pos}
        />
      ))}
    </group>
  );
};
