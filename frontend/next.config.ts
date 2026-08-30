import type { NextConfig } from "next";

// Two build shapes from one config:
//
//   standalone (default) - the Docker image, where Next serves itself behind the API.
//   export               - the desktop application, where there is no Node process at all:
//                          the pages are plain files that FastAPI serves from its own port.
//
// The desktop shape is why the dashboard has no dynamic route segments. It also means the
// browser talks to the API on its own origin, so NEXT_PUBLIC_API_URL is set to the empty
// string for that build and every request becomes a relative one.
const isExport = process.env.NEXT_OUTPUT === "export";

const config: NextConfig = {
  reactStrictMode: true,
  output: isExport ? "export" : "standalone",
  // The dashboard is a pure client of the API; it holds no secrets of its own.
  // NEXT_PUBLIC_API_URL is baked in at build time and is therefore public by definition —
  // never put anything sensitive in a NEXT_PUBLIC_ variable.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
  },
  // A static export has no image optimiser behind it.
  images: { unoptimized: true },
  // `headers()` requires a server. In the export build FastAPI sets the same headers on every
  // response, including the ones serving these files, so nothing is lost.
  ...(isExport
    ? {}
    : {
        async headers() {
          return [
            {
              source: "/:path*",
              headers: [
                { key: "X-Content-Type-Options", value: "nosniff" },
                { key: "X-Frame-Options", value: "DENY" },
                { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
              ],
            },
          ];
        },
      }),
};

export default config;
