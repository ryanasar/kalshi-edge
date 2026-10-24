// Bundles test/quoter.test.ts (stubbing the `cloudflare:workers` runtime module)
// and runs it under Node. esbuild ships with wrangler, so no extra dependency.
import { build } from "esbuild";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";

const dir = fileURLToPath(new URL(".", import.meta.url));
const out = join(tmpdir(), `quoter-test-${process.pid}.mjs`);
await build({
  entryPoints: [join(dir, "quoter.test.ts")], bundle: true, platform: "node",
  format: "esm", outfile: out, logLevel: "warning",
  plugins: [{ name: "cf-workers-stub", setup(b) {
    b.onResolve({ filter: /^cloudflare:workers$/ }, () => ({ path: join(dir, "stub-workers.ts") }));
  } }],
});
await import(pathToFileURL(out).href);
