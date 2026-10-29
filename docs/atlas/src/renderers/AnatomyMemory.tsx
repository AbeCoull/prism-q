import geometry from "../data/anatomy-memory-geometry.json";

type Props = {
  n: number;
  target: number;
  pair: [number, number];
  phase: number;
  onSelect: (index: number) => void;
  width?: number;
  architecture?: "avx2" | "neon" | "scalar";
  skipLow: boolean;
};

type SolidPaths = typeof geometry.amplitude;

export function memoryRows(n: number, pair: [number, number]) {
  const total = Math.ceil(2 ** n / 4);
  if (total <= 16) return Array.from({ length: total }, (_, index) => index);
  const rows = new Set<number>();
  for (const index of pair) {
    const first = Math.max(0, Math.min(total - 8, Math.floor(index / 4) - 3));
    for (let offset = 0; offset < 8; offset += 1) rows.add(first + offset);
  }
  return [...rows].sort((a, b) => a - b);
}

function Solid({
  paths,
  top,
  edge = "#aeaaa0",
}: {
  paths: SolidPaths;
  top: string;
  edge?: string;
}) {
  return (
    <g>
      <path d={paths.fill} fill="#e9e6df" />
      {paths.shades.map((path, index) => (
        <path
          key={index}
          d={path}
          fill={["#f4f2ed", "#e6e3dc", "#d9d6ce", "#cdc9c0"][index]}
        />
      ))}
      <path d={paths.top} fill={top} />
      <path d={paths.crease} stroke={edge} strokeWidth="0.45" fill="none" />
      <path
        d={paths.bevel}
        stroke={edge}
        strokeWidth="0.3"
        opacity="0.7"
        fill="none"
      />
      <path d={paths.outline} stroke={edge} strokeWidth="0.65" fill="none" />
    </g>
  );
}

export default function AnatomyMemory({
  n,
  target,
  pair,
  phase,
  onSelect,
  width = 0,
  architecture = "scalar",
  skipLow,
}: Props) {
  const rows = memoryRows(n, pair);
  const total = 2 ** n;
  const coordinate = (column: number, row: number) => [
    column * geometry.column[0] + row * geometry.row[0],
    column * geometry.column[1] + row * geometry.row[1],
  ];
  const cells = rows
    .flatMap((memoryRow, row) =>
      [0, 1, 2, 3].map((column) => {
        const index = memoryRow * 4 + column;
        return {
          index,
          column,
          row,
          depth:
            column * 28 * Math.cos((48 * Math.PI) / 180) +
            row * 13.8 * Math.sin((48 * Math.PI) / 180),
        };
      }),
    )
    .filter(({ index }) => index < total)
    .sort((a, b) => a.depth - b.depth);
  const minX = (rows.length - 1) * geometry.row[0] - 150;
  const maxX = 3 * geometry.column[0] + 100;
  const maxY =
    3 * geometry.column[1] + (rows.length - 1) * geometry.row[1] + 90;
  const point = (index: number) =>
    coordinate(index % 4, rows.indexOf(Math.floor(index / 4)));
  const [ax, ay] = point(pair[0]);
  const [bx, by] = point(pair[1]);
  const lanes = width === 256 ? 4 : 2;
  const bankBase = width === 256 ? Math.floor(pair[0] / 2) * 2 : pair[0];
  const labelX = (minX + maxX) / 2;

  return (
    <svg
      viewBox={`${minX} -165 ${maxX - minX} ${maxY + 165}`}
      width="100%"
      height="100%"
      style={{ minHeight: 0, display: "block" }}
      role="group"
      aria-label={`Interactive isometric storage diagram. Pair ${pair[0]}, ${pair[1]}. ${skipLow ? "Low half bypassed: no load or store because d0 is one." : "Both input streams participate."} Select an amplitude to change the pair.`}
      data-renderer="anatomy-svg"
    >
      <title>Anatomy of contiguous complex128 storage</title>
      <desc>
        Each amplitude has adjacent real and imaginary f64 slots. Four
        amplitudes occupy one assumed 64-byte cache row. Geometry generated with
        the Anatomy isometric kit, commit {geometry.provenance.commit}, under
        the MIT license. The full license is retained in the geometry asset.
      </desc>
      <g fontFamily="ui-monospace, SFMono-Regular, Consolas, monospace">
        {rows.map((memoryRow, row) => {
          const [x, y] = coordinate(0, row);
          return (
            <g key={memoryRow} transform={`translate(${x},${y})`}>
              <Solid paths={geometry.rowBase} top="#e9e6df" edge="#c3bfb5" />
              <text
                x="-52"
                y="-18"
                fill="#57554f"
                fontSize="10"
                textAnchor="end"
              >
                0x{(memoryRow * 64).toString(16).padStart(4, "0")}
              </text>
              {row > 0 && memoryRow !== rows[row - 1] + 1 && (
                <text
                  x="-53"
                  y="-30"
                  fill="#b8432c"
                  fontSize="7"
                  textAnchor="end"
                >
                  window break
                </text>
              )}
            </g>
          );
        })}
        {cells.map(({ index, column, row }) => {
          const [x, y] = coordinate(column, row);
          const selected = index === pair[0] || index === pair[1];
          const color =
            index === pair[0]
              ? "#2457b5"
              : index === pair[1]
                ? "#b8432c"
                : "#fdfcf9";
          return (
            <g
              key={index}
              transform={`translate(${x},${y})`}
              role="button"
              tabIndex={0}
              aria-label={`Amplitude ${index}, byte offset ${index * 16}${selected ? ", selected pair member" : ""}`}
              onClick={() => onSelect(index)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(index);
                }
              }}
              style={{ cursor: "pointer" }}
            >
              <title>
                Amplitude {index}: real at +{index * 16} B, imaginary at +
                {index * 16 + 8} B
              </title>
              <Solid
                paths={geometry.amplitude}
                top="#f4f2ed"
                edge={selected ? color : "#aeaaa0"}
              />
              <Solid
                paths={geometry.real}
                top={color}
                edge={selected ? color : "#aeaaa0"}
              />
              <Solid
                paths={geometry.imag}
                top={color}
                edge={selected ? color : "#aeaaa0"}
              />
              <text
                x="-12"
                y="-12"
                fill={selected ? "#ffffff" : "#33322e"}
                fontSize="6.3"
                textAnchor="middle"
              >
                {index}.r
              </text>
              <text
                x="13"
                y="2"
                fill={selected ? "#ffffff" : "#33322e"}
                fontSize="6.3"
                textAnchor="middle"
              >
                {index}.i
              </text>
              {selected && !(skipLow && index === pair[0]) && phase === 3 && (
                <path
                  data-testid="memory-store-marker"
                  d={geometry.amplitude.outline}
                  stroke={color}
                  strokeWidth="1.4"
                  fill="none"
                />
              )}
            </g>
          );
        })}

        <path
          d={`M${ax},${ay - 18} L${ax + 38},${ay - 40} L${bx + 38},${by - 40} L${bx},${by - 18}`}
          fill="none"
          stroke="#b8432c"
          strokeWidth="0.9"
          strokeDasharray="4 4"
          pointerEvents="none"
        />

        {[
          {
            name: skipLow ? "a: skipped (d0 = 1)" : "Input a",
            bypassed: skipLow,
            index: bankBase,
            color: "#2457b5",
            x: labelX - 92,
          },
          {
            name: "Input b",
            bypassed: false,
            index: bankBase + 2 ** target,
            color: "#b8432c",
            x: labelX + 75,
          },
        ].map((bank) => (
          <g
            key={bank.name}
            transform={`translate(${bank.x},-97)`}
            data-execution={
              bank.bypassed
                ? "bypassed"
                : phase === 1 || phase === 2
                  ? "loaded"
                  : "idle"
            }
          >
            <text x="0" y="-27" fill={bank.color} fontSize="11">
              {bank.name}
            </text>
            {Array.from({ length: lanes }, (_, lane) => (
              <g key={lane} transform={`translate(${lane * 25},0)`}>
                <Solid
                  paths={geometry.registerLane}
                  top={
                    !bank.bypassed && (phase === 1 || phase === 2)
                      ? bank.color
                      : "#f4f2ed"
                  }
                />
                <text
                  x="0"
                  y="-2"
                  fill={
                    !bank.bypassed && (phase === 1 || phase === 2)
                      ? "#ffffff"
                      : "#33322e"
                  }
                  fontSize="6"
                  textAnchor="middle"
                >
                  {bank.index + Math.floor(lane / 2)}.{lane % 2 ? "i" : "r"}
                </text>
              </g>
            ))}
          </g>
        ))}
        <text
          x={labelX}
          y="-151"
          fill="#1c1b18"
          fontSize="14"
          textAnchor="middle"
        >
          {width === 0
            ? "Scalar f64 fields"
            : `${width}-bit ${architecture === "neon" ? "NEON" : width === 256 ? "AVX2" : "FMA"} input lanes`}
        </text>
        <text
          x={labelX}
          y={maxY - 34}
          fill="#b8432c"
          fontSize="14"
          textAnchor="middle"
        >
          Pair distance {16 * 2 ** target} B
        </text>
        <text
          x={labelX}
          y={maxY - 13}
          fill="#57554f"
          fontSize="11"
          textAnchor="middle"
        >
          4 amplitudes / 64 B model row · r = real · i = imaginary
        </text>
      </g>
    </svg>
  );
}
