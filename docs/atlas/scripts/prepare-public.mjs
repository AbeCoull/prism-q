import { copyFileSync } from "node:fs";
for (const file of ["native-gates.json", "native-fusion.json"]) {
  copyFileSync(
    new URL(`../fixtures/${file}`, import.meta.url),
    new URL(`../public/${file}`, import.meta.url),
  );
}
