import { useGLTF } from "@react-three/drei/webgpu";
import { useMemo } from "react";
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
import { useTexture } from "@react-three/fiber/webgpu";

const GRID = 3;
const SPACING = 2.25;
const HALF = ((GRID - 1) * SPACING) / 2;

export const Cube = () => {
  const { nodes } = useGLTF("/cube.glb");
  const nTex = useTexture("./noise.png");
  const wrappedNoiseTexture = useMemo(() => {
    const tex = nTex.clone();
    tex.wrapS = tex.wrapT = RepeatWrapping;
    tex.needsUpdate = true;
    return tex;
  }, [nTex]);

  // Create a single shared material, using a custom node on both sides
  const cubeMaterial = useMemo(() => {
    const m = new MeshStandardNodeMaterial({
      roughness: 0,
      transparent: true,
      opacity: 0.99,
      side: THREE.DoubleSide,
    });
    m.depthWrite = true;
    const ca = float(0.005);

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

    return m;
  }, [wrappedNoiseTexture]);

  // Build a 3×3×3 grid of positions centred at origin
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

  return (
    <group>
      {positions.map((pos, i) => (
        <mesh
          key={i}
          geometry={(nodes.Cube as THREE.Mesh).geometry}
          material={cubeMaterial}
          position={pos}
        />
      ))}
    </group>
  );
};
