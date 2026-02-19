import { Environment } from "@react-three/drei";

export const Lights = () => {
  return (
    <>
      <directionalLight
      castShadow
      
      position={[1000, 100, 100]}
      intensity={1}
      color={"#ffffffff"}
      shadow-mapSize={[4096, 4096]}
    />

    <Environment preset="sunset" background backgroundBlurriness={1} environmentIntensity={0.2} />
    </>
  );
}