import { Canvas } from "@react-three/fiber";
import { Edges, Line, OrbitControls } from "@react-three/drei";
import { Color } from "three";
import { Ink } from "./MemoryScene";
import { abs2 } from "../models/quantum";
import type { ComplexMatrix } from "../models/compiler";

type Props = {
  matrix: ComplexMatrix;
  selected: [number, number];
  onSelect: (cell: [number, number]) => void;
};

function Coefficients({ matrix, selected, onSelect }: Props) {
  const size = matrix.length;
  return (
    <group position={[-(size - 1) * 0.8, 0, -(size - 1) * 0.8]}>
      <mesh position={[(size - 1) * 0.8, -0.15, (size - 1) * 0.8]}>
        <boxGeometry args={[size * 1.6 + 0.35, 0.18, size * 1.6 + 0.35]} />
        <meshStandardMaterial
          color="#e9e6df"
          roughness={0.8}
          metalness={0.25}
        />
        <Edges color="#c3bfb5" />
      </mesh>
      {matrix.flatMap((row, r) =>
        row.map((value, c) => {
          const magnitude = Math.sqrt(abs2(value)),
            phase = magnitude > 1e-14 ? Math.atan2(value[1], value[0]) : 0;
          const active = selected[0] === r && selected[1] === c;
          const shade = new Color().setHSL(
            (phase / (2 * Math.PI) + 1) % 1,
            0.25,
            active ? 0.69 : 0.42,
          );
          const x = c * 1.6,
            z = r * 1.6,
            height = magnitude * 1.8;
          return (
            <group
              key={`${r}-${c}`}
              position={[x, 0, z]}
              onClick={(event) => {
                event.stopPropagation();
                onSelect([r, c]);
              }}
            >
              <mesh position={[0, -0.015, 0]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[0.44, 0.48, 40]} />
                <meshBasicMaterial
                  color={active ? "#2457b5" : "#c3bfb5"}
                  side={2}
                />
              </mesh>
              <Line
                points={[
                  [-0.45, 0.015, 0],
                  [0.45, 0.015, 0],
                ]}
                color="#c3bfb5"
                lineWidth={0.7}
              />
              <Line
                points={[
                  [0, 0.015, 0.45],
                  [0, 0.015, -0.45],
                ]}
                color="#c3bfb5"
                lineWidth={0.7}
              />
              {active && (
                <>
                  <Ink
                    text="+Re"
                    position={[0.6, -0.04, 0]}
                    color="#57554f"
                    size={0.085}
                    flat
                  />
                  <Ink
                    text="+Im"
                    position={[0, -0.04, -0.6]}
                    color="#57554f"
                    size={0.085}
                    flat
                  />
                </>
              )}
              {magnitude > 0 && (
                <mesh position={[0, height / 2, 0]}>
                  <cylinderGeometry args={[0.17, 0.2, height, 20]} />
                  <meshStandardMaterial
                    color={shade}
                    roughness={0.48}
                    metalness={0.25}
                  />
                  <Edges color={active ? "#1c1b18" : "#8f8b82"} />
                </mesh>
              )}
              {magnitude > 1e-14 && (
                <Line
                  points={[
                    [0, height + 0.03, 0],
                    [
                      0.42 * Math.cos(phase),
                      height + 0.03,
                      -0.42 * Math.sin(phase),
                    ],
                  ]}
                  color={active ? "#b8432c" : "#2457b5"}
                  lineWidth={2}
                />
              )}
              <Ink
                text={`${r},${c}`}
                position={[0, -0.04, 0.66]}
                color={active ? "#1c1b18" : "#6a6862"}
                size={0.13}
                flat
              />
            </group>
          );
        }),
      )}
    </group>
  );
}

export default function OperatorScene(props: Props) {
  const size = props.matrix.length;
  return (
    <div
      className="operator-scene"
      role="img"
      aria-label={`${size} by ${size} complex matrix. Column heights are coefficient magnitudes; spokes encode complex phase. Select coefficients with the matrix table below.`}
    >
      <Canvas
        orthographic
        frameloop="demand"
        dpr={[1, 1.5]}
        camera={{ position: [9, 10, 11], zoom: size === 2 ? 64 : 43 }}
        gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
      >
        <ambientLight intensity={1.25} />
        <directionalLight
          position={[3, 8, 5]}
          intensity={2.4}
          color="#1c1b18"
        />
        <directionalLight
          position={[-4, 3, -4]}
          intensity={1.1}
          color="#57554f"
        />
        <Coefficients {...props} />
        <OrbitControls
          enableDamping={false}
          enablePan={false}
          minZoom={28}
          maxZoom={100}
          minPolarAngle={0.2}
          maxPolarAngle={Math.PI / 2.1}
        />
      </Canvas>
    </div>
  );
}
