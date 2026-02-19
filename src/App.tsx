import { Canvas } from "@react-three/fiber";

import { Lights } from "./components/lights";
import { OrbitControls } from "@react-three/drei";
import { PostProcessing } from "./components/postprocessing";

import { Model } from "./components/Checkered_tile_floor";
import { VFXParticles } from "r3f-vfx";
import { Cube } from "./components/cube";

function App() {
  // wobblysphere2 update the material so each changes trigger a re-render, better developer experience but doesn't follow the new R3F v10 API
  return (
    <>
      <Canvas renderer={{ forceWebGL: false}} hmr={true}>
        <Lights />
        <Model />
        <Cube />
        <OrbitControls />
        <PostProcessing />
      </Canvas>
    </>
  );
}

export default App;
