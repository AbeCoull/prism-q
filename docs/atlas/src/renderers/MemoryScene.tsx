import { useEffect, useMemo, useState } from "react";
import { Canvas, type ThreeEvent, useThree } from "@react-three/fiber";
import { Edges, Line, OrbitControls } from "@react-three/drei";
import { CanvasTexture, OrthographicCamera, SRGBColorSpace } from "three";
import { memoryRows } from "./AnatomyMemory";

export type MemorySceneProps = {
  n: number;
  target: number;
  pair: [number, number];
  phase: number;
  onSelect: (index: number) => void;
  architecture: "avx2" | "neon" | "scalar";
  width: number;
  skipLow: boolean;
};

const SAGE = "#99b7a5";
const COPPER = "#d6a17c";
const MUTED = "#77877d";
const COLUMN = 1.44;
const ROW = 0.62;

export function Ink({
  text,
  position,
  color = MUTED,
  size = 0.13,
  flat = false,
  align = "center",
}: {
  text: string;
  position: [number, number, number];
  color?: string;
  size?: number;
  flat?: boolean;
  align?: CanvasTextAlign;
}) {
  const textureWidth = Math.min(
    2048,
    Math.max(128, 2 ** Math.ceil(Math.log2(text.length * 26 + 16))),
  );
  const texture = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = textureWidth;
    canvas.height = 64;
    const context = canvas.getContext("2d");
    if (context) {
      context.font = "42px ui-monospace, SFMono-Regular, Consolas, monospace";
      context.fillStyle = color;
      context.textAlign = align;
      context.textBaseline = "middle";
      context.fillText(
        text,
        align === "left"
          ? 4
          : align === "right"
            ? textureWidth - 4
            : textureWidth / 2,
        32,
      );
    }
    const map = new CanvasTexture(canvas);
    map.colorSpace = SRGBColorSpace;
    map.anisotropy = 4;
    return map;
  }, [text, color, align, textureWidth]);
  useEffect(() => () => texture.dispose(), [texture]);
  return (
    <mesh
      position={position}
      rotation={flat ? [-Math.PI / 2, 0, 0] : [0, 0, 0]}
    >
      <planeGeometry args={[(size * 1.5 * textureWidth) / 64, size * 1.5]} />
      <meshBasicMaterial
        map={texture}
        transparent
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  );
}

function Amplitude({
  index,
  column,
  row,
  selected,
  phase,
  bypassed,
  onSelect,
}: {
  index: number;
  column: number;
  row: number;
  selected: "a" | "b" | null;
  phase: number;
  bypassed: boolean;
  onSelect: (index: number) => void;
}) {
  const [hovered, setHovered] = useState(false);
  const color = selected === "a" ? SAGE : selected === "b" ? COPPER : "#65756a";
  const highlighted = selected !== null;
  const activate = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation();
    onSelect(index);
  };
  return (
    <group
      position={[column * COLUMN, 0, row * ROW]}
      onClick={activate}
      onPointerOver={(event) => {
        event.stopPropagation();
        setHovered(true);
      }}
      onPointerOut={() => setHovered(false)}
    >
      <mesh position={[0, -0.025, 0]}>
        <boxGeometry args={[1.35, 0.15, 0.51]} />
        <meshStandardMaterial
          color={highlighted ? "#36483d" : "#1b2420"}
          roughness={0.88}
        />
        <Edges
          color={highlighted || hovered ? color : "#36433b"}
          threshold={15}
        />
      </mesh>
      {[-1, 1].map((side) => (
        <mesh key={side} position={[side * 0.323, 0.068, 0]}>
          <boxGeometry args={[0.61, 0.035, 0.424]} />
          <meshStandardMaterial
            color={highlighted ? color : hovered ? "#4c6152" : "#2d3930"}
            roughness={0.7}
            metalness={0.08}
            emissive={
              highlighted && !bypassed && phase === 3 ? color : "#000000"
            }
            emissiveIntensity={
              highlighted && !bypassed && phase === 3 ? 0.12 : 0
            }
          />
        </mesh>
      ))}
      <Ink
        text={`${index}.r`}
        position={[-0.323, 0.092, 0]}
        color={highlighted ? "#152018" : "#b2c4b1"}
        size={index < 100 ? 0.15 : 0.11}
        flat
      />
      <Ink
        text={`${index}.i`}
        position={[0.323, 0.092, 0]}
        color={highlighted ? "#152018" : "#b2c4b1"}
        size={index < 100 ? 0.15 : 0.11}
        flat
      />
    </group>
  );
}

function Register({
  name,
  index,
  width,
  x,
  z,
  color,
  active,
}: {
  name: string;
  index: number;
  width: number;
  x: number;
  z: number;
  color: string;
  active: boolean;
}) {
  const lanes = width === 256 ? 4 : 2;
  const laneWidth = 0.72;
  const bankWidth = lanes * laneWidth + 0.16;
  return (
    <group position={[x, 1.12, z]}>
      <mesh position={[0, -0.085, 0]}>
        <boxGeometry args={[bankWidth, 0.11, 0.69]} />
        <meshStandardMaterial color="#1b2620" roughness={0.86} />
        <Edges color={active ? color : "#4a5b4d"} />
      </mesh>
      {Array.from({ length: lanes }, (_, lane) => {
        const slotX = (lane - (lanes - 1) / 2) * laneWidth;
        const label = `${lane % 2 === 0 ? "re" : "im"} ${index + Math.floor(lane / 2)}`;
        return (
          <group key={lane} position={[slotX, 0, 0]}>
            <mesh>
              <boxGeometry args={[laneWidth - 0.025, 0.06, 0.58]} />
              <meshStandardMaterial
                color={active ? color : "#36463b"}
                roughness={0.8}
              />
            </mesh>
            <Ink
              text={label}
              position={[0, 0.035, 0.02]}
              color={active ? "#16231b" : "#c9d6c1"}
              size={0.16}
              flat
            />
            <Ink
              text={`${lane}`}
              position={[0, 0.036, -0.19]}
              color={active ? "#314b37" : "#8fa58f"}
              size={0.09}
              flat
            />
          </group>
        );
      })}
      <Ink
        text={name}
        position={[0, 0.04, -0.52]}
        color={color}
        size={0.2}
        flat
      />
    </group>
  );
}

function Storage({
  n,
  target,
  pair,
  phase,
  onSelect,
  architecture,
  width,
  skipLow,
}: MemorySceneProps) {
  const rows = useMemo(() => memoryRows(n, pair), [n, pair[0], pair[1]]);
  const count = 2 ** n;
  const rowFor = (index: number) => rows.indexOf(Math.floor(index / 4));
  const first = rowFor(pair[0]);
  const second = rowFor(pair[1]);
  const a: [number, number, number] = [
    (pair[0] % 4) * COLUMN,
    0.19,
    first * ROW,
  ];
  const b: [number, number, number] = [
    (pair[1] % 4) * COLUMN,
    0.19,
    second * ROW,
  ];
  const bankBase = width === 256 ? Math.floor(pair[0] / 2) * 2 : pair[0];
  const laneTitle =
    width === 0
      ? "SCALAR FIELDS"
      : `${width}-BIT ${architecture === "neon" ? "NEON" : width === 256 ? "AVX2" : "FMA"}`;
  const registerName =
    width === 256
      ? "2 complex / input vector"
      : width === 128
        ? "1 complex / input vector"
        : "two scalar f64 components";
  const depth = Math.max(0.62, rows.length * ROW);

  return (
    <group>
      <ambientLight intensity={1.2} />
      <directionalLight
        position={[-5, 12, 8]}
        intensity={2.1}
        color="#e3e9cf"
      />
      <directionalLight position={[9, 6, -4]} intensity={0.7} color="#93aea2" />

      <mesh position={[2.16, -0.19, (depth - ROW) / 2]}>
        <boxGeometry args={[6.12, 0.12, depth + 0.32]} />
        <meshStandardMaterial color="#121c16" roughness={1} />
        <Edges color="#344336" />
      </mesh>

      {rows.map((memoryRow, visualRow) => (
        <group key={memoryRow}>
          <Line
            points={[
              [-0.73, -0.095, visualRow * ROW - 0.28],
              [5.05, -0.095, visualRow * ROW - 0.28],
            ]}
            color="#485648"
            lineWidth={0.7}
          />
          <Ink
            text={`0x${(memoryRow * 64).toString(16).padStart(4, "0")}`}
            position={[-1.37, -0.06, visualRow * ROW]}
            color="#8fa58f"
            size={0.15}
            flat
          />
          {[0, 1, 2, 3].map((column) => {
            const index = memoryRow * 4 + column;
            return index < count ? (
              <Amplitude
                key={index}
                index={index}
                column={column}
                row={visualRow}
                selected={
                  index === pair[0] ? "a" : index === pair[1] ? "b" : null
                }
                phase={phase}
                bypassed={skipLow && index === pair[0]}
                onSelect={onSelect}
              />
            ) : null;
          })}
          {visualRow > 0 && memoryRow !== rows[visualRow - 1] + 1 && (
            <Ink
              text="address window break"
              position={[2.16, 0.12, visualRow * ROW - ROW / 2]}
              color={COPPER}
              size={0.105}
              flat
            />
          )}
        </group>
      ))}

      <Line
        points={[a, [5.35, 0.19, a[2]], [5.35, 0.19, b[2]], b]}
        color={COPPER}
        lineWidth={1}
        dashed
        dashSize={0.09}
        gapSize={0.065}
      />
      <Ink
        text={`PAIR +${16 * 2 ** target} B`}
        position={[2.16, 0.02, depth + 0.37]}
        color={COPPER}
        size={0.2}
        flat
      />
      <Ink
        text="4 amplitudes / 64 B model row"
        position={[2.16, -0.02, depth + 0.76]}
        color="#a3b5a1"
        size={0.15}
        flat
      />
      <Ink
        text="re, im : f64 + f64"
        position={[2.16, -0.02, depth + 1.05]}
        color="#8fa58f"
        size={0.13}
        flat
      />

      <Register
        name={skipLow ? "a: BYPASSED (d0 = 1)" : "INPUT a"}
        index={bankBase}
        width={width}
        x={0.6}
        z={-1.6}
        color={SAGE}
        active={!skipLow && phase >= 1 && phase < 3}
      />
      <Register
        name="INPUT b"
        index={bankBase + 2 ** target}
        width={width}
        x={3.88}
        z={-1.6}
        color={COPPER}
        active={phase >= 1 && phase < 3}
      />
      <Ink
        text={laneTitle}
        position={[2.16, 1.13, -2.88]}
        color="#c9d6c1"
        size={0.24}
        flat
      />
      <Ink
        text={registerName}
        position={[2.16, 1.13, -2.51]}
        color="#a3b5a1"
        size={0.16}
        flat
      />
      <Ink
        text="Ascending lane number = ascending load address"
        position={[2.16, 1.13, -3.25]}
        color="#8fa58f"
        size={0.13}
        flat
      />

      {[0, 1, 2, 3].map((column) => (
        <Ink
          key={column}
          text={`+${column * 16} B`}
          position={[column * COLUMN, -0.04, -0.59]}
          color="#8fa58f"
          size={0.14}
          flat
        />
      ))}
    </group>
  );
}

function FitCamera({ rows }: { rows: number }) {
  const { camera, size, invalidate } = useThree();
  useEffect(() => {
    if (camera instanceof OrthographicCamera) {
      camera.zoom = Math.min(
        size.width / (rows > 8 ? 13.5 : 11),
        size.height / (rows > 8 ? 10.5 : rows > 4 ? 8.5 : 6.8),
      );
      camera.updateProjectionMatrix();
      invalidate();
    }
  }, [camera, size.width, size.height, rows, invalidate]);
  return null;
}

export default function MemoryScene(props: MemorySceneProps) {
  const rows = memoryRows(props.n, props.pair).length;
  const centerZ = Math.max(-0.4, (rows * ROW - 3.4) / 2);
  return (
    <div
      className="memory-scene"
      style={{ width: "100%", height: "100%", minHeight: 0 }}
      role="img"
      aria-label={`Isometric address diagram: amplitude ${props.pair[0]} pairs with ${props.pair[1]}, separated by ${16 * 2 ** props.target} bytes. ${props.skipLow ? "Low half bypassed: no load or store because d0 is one." : "Both input streams participate."} Each amplitude has adjacent real and imaginary f64 components. Drag to orbit; use the amplitude table for keyboard selection.`}
    >
      <Canvas
        orthographic
        frameloop="demand"
        dpr={[1, 1.5]}
        camera={{
          position: [13, 15, 18],
          zoom: rows > 8 ? 39 : rows > 4 ? 45 : 54,
          near: 0.1,
          far: 100,
        }}
        gl={{ antialias: true, alpha: true, powerPreference: "low-power" }}
        style={{ minHeight: 0 }}
      >
        <Storage {...props} />
        <FitCamera rows={rows} />
        <OrbitControls
          makeDefault
          target={[2.16, 0.3, centerZ]}
          enableDamping={false}
          enablePan
          minZoom={22}
          maxZoom={100}
          minPolarAngle={0.15}
          maxPolarAngle={Math.PI / 2.1}
        />
      </Canvas>
    </div>
  );
}
