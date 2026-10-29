import { useMemo } from "react";
import katex from "katex";

export function Equation({
  tex,
  block = false,
}: {
  tex: string;
  block?: boolean;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        throwOnError: false,
        displayMode: block,
        output: "htmlAndMathml",
      }),
    [tex, block],
  );
  return (
    <span
      className={block ? "equation block" : "equation"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
