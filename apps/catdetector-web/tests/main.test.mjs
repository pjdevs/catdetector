import { readFileSync } from "node:fs";
import assert from "node:assert/strict";

const mainSource = readFileSync(new URL("../src/main.ts", import.meta.url), "utf8");

assert.match(mainSource, /import\s+\{\s*mount\s*\}\s+from\s+["']svelte["']/);
assert.doesNotMatch(mainSource, /new\s+App\s*\(/);
