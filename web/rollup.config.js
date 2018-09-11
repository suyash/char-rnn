import commonjs from "rollup-plugin-commonjs";
import copy from "rollup-plugin-copy";
import resolve from "rollup-plugin-node-resolve";
import typescript from "rollup-plugin-typescript2";
import { terser } from "rollup-plugin-terser";

const tsconfig = process.env.NODE_ENV === "production" ? "tsconfig.json" : "tsconfig.dev.json";

export default {
    input: "./src/index.ts",
    output: {
        file: "./lib/index.js",
        format: "iife",
        globals: {
            "@tensorflow/tfjs": "tf",
        },
        sourcemap: process.env.NODE_ENV !== "production",
    },
    external: [
        "@tensorflow/tfjs",
    ],
    plugins: [
        typescript({ tsconfig, tsconfigOverride: { compilerOptions: { module: "es6" } } }),
        resolve({ browser: true }),
        commonjs(),
        copy({
            verbose: true,
            "./src/html/index.html": "./lib/index.html",
        }),
        process.env.NODE_ENV === "production" && terser(),
    ],
}
