#!/usr/bin/env node

import { access, chmod, copyFile, mkdir, realpath, stat, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { spawn } from 'node:child_process';

const root = process.cwd();
const sourceDir = resolve(root, 'native/maxflow');
const outDir = resolve(root, 'public/wasm/maxflow');
const emccCommand = process.env.EMXX || 'em++';
const emCache = process.env.EM_CACHE || resolve('/tmp', 'brenizer-emcache');
const synthesizedEmConfigPath = resolve('/tmp', 'brenizer-emscripten-config.py');
const compilerWrapperPath = process.env.EM_COMPILER_WRAPPER || resolve('/tmp', 'brenizer-emcc-wrapper.py');

const sourceFiles = [
  resolve(sourceDir, 'brenizer_maxflow.cpp'),
  resolve(sourceDir, 'vendor/atcoder/maxflow.hpp'),
  resolve(sourceDir, 'vendor/atcoder/internal_queue.hpp'),
  resolve(sourceDir, 'vendor/LICENSE.atcoder-CC0.txt'),
  resolve(sourceDir, 'vendor/LICENSE.emscripten.txt'),
  resolve(root, 'scripts/prepare-maxflow.mjs'),
];

const outputs = [
  resolve(outDir, 'maxflow-simd.module.js'),
  resolve(outDir, 'maxflow-simd.wasm'),
  resolve(outDir, 'maxflow-threads.module.js'),
  resolve(outDir, 'maxflow-threads.wasm'),
  resolve(outDir, 'maxflow-threads.worker.js'),
  resolve(outDir, 'LICENSE.atcoder-CC0.txt'),
  resolve(outDir, 'LICENSE.emscripten.txt'),
  resolve(outDir, 'THIRD_PARTY_NOTICES.txt'),
];

let resolvedToolchain = null;

async function fileExists(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function executableExists(path) {
  try {
    await access(path, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function formatPathDetails(meta) {
  const parts = [];
  if (meta.activeConfigPath) parts.push(`EM_CONFIG=${meta.activeConfigPath}`);
  if (meta.emRoot) parts.push(`EMSCRIPTEN_ROOT=${meta.emRoot}`);
  if (meta.llvmRoot) parts.push(`LLVM_ROOT=${meta.llvmRoot}`);
  if (meta.binaryenRoot) parts.push(`BINARYEN_ROOT=${meta.binaryenRoot}`);
  if (meta.nodePath) parts.push(`NODE_JS=${meta.nodePath}`);
  return parts.join(', ');
}

async function resolvePathlikeEnvFile(raw) {
  if (!raw || typeof raw !== 'string') return null;
  if (raw.includes('\n') || raw.includes('\r')) return null;
  const candidate = resolve(raw);
  return (await fileExists(candidate)) ? candidate : null;
}

async function findBinaryenTool(binaryenRoot) {
  if (!binaryenRoot) return null;
  const candidates = [
    join(binaryenRoot, 'bin', 'wasm-opt'),
    join(binaryenRoot, 'wasm-opt'),
    join(binaryenRoot, 'bin', 'wasm-opt.exe'),
    join(binaryenRoot, 'wasm-opt.exe'),
  ];
  for (const candidate of candidates) {
    if (await executableExists(candidate)) return candidate;
  }
  return null;
}

async function validateSynthesizedToolchain(meta) {
  const clangCandidates = [
    join(meta.llvmRoot, 'clang'),
    join(meta.llvmRoot, 'clang.exe'),
  ];
  let clangPath = null;
  for (const candidate of clangCandidates) {
    if (await executableExists(candidate)) {
      clangPath = candidate;
      break;
    }
  }
  const binaryenTool = await findBinaryenTool(meta.binaryenRoot);
  if (!clangPath || !binaryenTool) {
    const missing = [];
    if (!clangPath) missing.push(`clang executable not found under LLVM_ROOT (${meta.llvmRoot})`);
    if (!binaryenTool) missing.push(`wasm-opt executable not found under BINARYEN_ROOT (${meta.binaryenRoot})`);
    throw new Error(`${missing.join('; ')}. Resolved paths: ${formatPathDetails(meta)}`);
  }
  return { clangPath, binaryenTool };
}

function compilerInvocation(args) {
  if (emccCommand.endsWith('.py')) {
    return {
      cmd: 'python3',
      args: [emccCommand, ...args],
    };
  }
  return {
    cmd: emccCommand,
    args,
  };
}

function log(msg) {
  console.log(`[prepare:maxflow] ${msg}`);
}

async function fileMtime(path) {
  const info = await stat(path);
  return info.mtimeMs;
}

async function outputsAreFresh() {
  let newestInput = 0;
  for (const file of sourceFiles) {
    newestInput = Math.max(newestInput, await fileMtime(file));
  }
  for (const out of outputs) {
    try {
      const outMtime = await fileMtime(out);
      if (outMtime < newestInput) return false;
    } catch {
      return false;
    }
  }
  return true;
}

function run(cmd, args) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(cmd, args, {
      cwd: '/',
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONPATH: process.env.PYTHONPATH || '',
        ...(resolvedToolchain?.env ?? { EM_CACHE: emCache }),
      },
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.on('error', rejectPromise);
    child.on('close', (code) => {
      if (code === 0) {
        resolvePromise({ stdout, stderr });
      } else {
        const err = new Error(`${cmd} exited with code ${code}\n${stderr || stdout}`);
        rejectPromise(err);
      }
    });
  });
}

async function resolveToolchain() {
  if (resolvedToolchain) return resolvedToolchain;

  const activeEmConfigPath = await resolvePathlikeEnvFile(process.env.EM_CONFIG);
  if (activeEmConfigPath) {
    resolvedToolchain = {
      mode: 'active-config',
      env: {
        EM_CACHE: emCache,
      },
      meta: {
        activeConfigPath: activeEmConfigPath,
      },
    };
    return resolvedToolchain;
  }

  const compilerLookup = emccCommand.includes('/')
    ? emccCommand
    : (await run('which', [emccCommand])).stdout.trim();
  const compilerPath = await realpath(compilerLookup);
  const compilerDir = dirname(compilerPath);
  const explicitRootsRequested = [
    process.env.EMSCRIPTEN_ROOT,
    process.env.EM_LLVM_ROOT,
    process.env.EM_BINARYEN_ROOT,
    process.env.EM_NODE_JS,
  ].some(Boolean);
  const emsdkLayout = basename(compilerDir) === 'emscripten' && basename(dirname(compilerDir)) === 'upstream';

  const emRoot = process.env.EMSCRIPTEN_ROOT
    ? resolve(process.env.EMSCRIPTEN_ROOT)
    : basename(compilerDir) === 'bin'
      ? resolve(dirname(compilerDir), 'libexec')
      : compilerDir;
  const llvmRoot = process.env.EM_LLVM_ROOT
    ? resolve(process.env.EM_LLVM_ROOT)
    : emsdkLayout
      ? resolve(dirname(compilerDir), 'bin')
      : resolve(emRoot, 'llvm/bin');
  const binaryenRoot = process.env.EM_BINARYEN_ROOT
    ? resolve(process.env.EM_BINARYEN_ROOT)
    : emsdkLayout
      ? resolve(dirname(compilerDir))
      : resolve(emRoot, 'binaryen');
  const nodeLookup = process.env.EM_NODE_JS
    ? process.env.EM_NODE_JS
    : (await run('which', ['node'])).stdout.trim();
  const nodePath = await realpath(nodeLookup);
  const useCompilerWrapper = process.platform === 'darwin';

  if (useCompilerWrapper) {
    const wrapperSource = `#!/usr/bin/env python3
import os
import re
import sys

PREFIX_RE = re.compile(r'^(\\.\\./)+(private/tmp/|opt/homebrew/)')

def canonicalize(arg: str) -> str:
    if arg.startswith('-I') or arg.startswith('-L'):
        prefix = arg[:2]
        value = arg[2:]
        return prefix + canonicalize(value)
    if not PREFIX_RE.match(arg):
        return arg
    absolute = '/' + PREFIX_RE.sub(lambda match: match.group(2), arg)
    return absolute if os.path.exists(absolute) else arg

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('compiler wrapper expected a compiler argv')
    argv = [canonicalize(arg) for arg in sys.argv[1:]]
    os.execv(argv[0], argv)

if __name__ == '__main__':
    main()
`;
    await writeFile(compilerWrapperPath, wrapperSource, 'utf8');
    await chmod(compilerWrapperPath, 0o755);
  }

  const configBody = [
    `LLVM_ROOT = ${JSON.stringify(llvmRoot)}`,
    `BINARYEN_ROOT = ${JSON.stringify(binaryenRoot)}`,
    `NODE_JS = ${JSON.stringify(nodePath)}`,
    `CACHE = ${JSON.stringify(emCache)}`,
    useCompilerWrapper ? `COMPILER_WRAPPER = ${JSON.stringify(compilerWrapperPath)}` : null,
    '',
  ].filter(Boolean).join('\n');
  const mode = explicitRootsRequested ? 'explicit-roots' : 'autodetect';
  const meta = {
    emRoot,
    llvmRoot,
    binaryenRoot,
    nodePath,
  };
  const { clangPath, binaryenTool } = await validateSynthesizedToolchain(meta);
  await writeFile(synthesizedEmConfigPath, configBody, 'utf8');

  resolvedToolchain = {
    mode,
    env: {
      EM_CONFIG: synthesizedEmConfigPath,
      EM_CACHE: emCache,
      EMSCRIPTEN_ROOT: emRoot,
      EM_LLVM_ROOT: llvmRoot,
      EM_BINARYEN_ROOT: binaryenRoot,
      EM_NODE_JS: nodePath,
      ...(useCompilerWrapper ? { EM_COMPILER_WRAPPER: compilerWrapperPath } : {}),
    },
    meta: {
      ...meta,
      synthesizedEmConfigPath,
      clangPath,
      binaryenTool,
    },
  };
  return resolvedToolchain;
}

async function ensureCompilerAvailable() {
  try {
    const toolchain = await resolveToolchain();
    if (toolchain.mode === 'active-config') {
      log(`Using active Emscripten config at ${toolchain.meta.activeConfigPath}`);
    } else {
      log(`Resolved ${toolchain.mode} toolchain: ${formatPathDetails(toolchain.meta)}`);
    }
    const invocation = compilerInvocation(['--version']);
    const { stdout, stderr } = await run(invocation.cmd, invocation.args);
    const banner = (stdout || stderr).trim().split('\n')[0] || 'unknown em++ version';
    log(`Using ${banner}`);
  } catch (err) {
    throw new Error(
      `Unable to run ${emccCommand}. Install Emscripten or set EMXX to a working em++ binary.\n${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

function commonArgs(exportName) {
  return [
    resolve(sourceDir, 'brenizer_maxflow.cpp'),
    '-std=c++20',
    '-O3',
    '-DNDEBUG',
    '-fno-exceptions',
    '-fno-rtti',
    '-s', 'WASM=1',
    '-s', 'ENVIRONMENT=worker',
    '-s', 'MODULARIZE=1',
    '-s', `EXPORT_NAME=${exportName}`,
    '-s', 'EXPORT_ES6=0',
    '-s', 'FILESYSTEM=0',
    '-s', 'ALLOW_MEMORY_GROWTH=1',
    '-s', 'MALLOC=emmalloc',
    '-s', 'INITIAL_MEMORY=33554432',
    '-s', 'AUTO_NATIVE_LIBRARIES=0',
    '-s', 'GL_ENABLE_GET_PROC_ADDRESS=0',
    '-s', 'MIN_WEBGL_VERSION=0',
    '-s', 'MAX_WEBGL_VERSION=0',
    '-s', 'EXPORT_ALL=1',
    '-s', "EXPORTED_FUNCTIONS=['_solve_grid','_last_error_message','_malloc','_free']",
    '-s', "EXPORTED_RUNTIME_METHODS=['UTF8ToString']",
    '-I', resolve(sourceDir, 'vendor'),
  ];
}

async function emitThirdPartyNotices() {
  await copyFile(
    resolve(sourceDir, 'vendor/LICENSE.atcoder-CC0.txt'),
    resolve(outDir, 'LICENSE.atcoder-CC0.txt'),
  );
  await copyFile(
    resolve(sourceDir, 'vendor/LICENSE.emscripten.txt'),
    resolve(outDir, 'LICENSE.emscripten.txt'),
  );
  const notice = [
    'Third-party notices for the compiled maxflow seam solver artifacts.',
    '',
    'Files:',
    '- maxflow-simd.module.js / maxflow-simd.wasm',
    '- maxflow-threads.module.js / maxflow-threads.wasm',
    '- maxflow-simd.js / maxflow-threads.js',
    '',
    'Included third-party components:',
    '- AtCoder Library mf_graph (vendored C++ source), licensed under CC0 1.0.',
    '- Emscripten-generated runtime glue, available under the MIT license and the',
    '  University of Illinois/NCSA Open Source License.',
    '',
    'See LICENSE.atcoder-CC0.txt and LICENSE.emscripten.txt in this directory for',
    'the full upstream license texts.',
    '',
  ].join('\n');
  await writeFile(resolve(outDir, 'THIRD_PARTY_NOTICES.txt'), notice, 'utf8');
}

async function compileSimd() {
  log('Compiling SIMD maxflow module');
  const invocation = compilerInvocation([
    ...commonArgs('createBrenizerMaxflowSimdModule'),
    '-msimd128',
    '-o', resolve(outDir, 'maxflow-simd.module.js'),
  ]);
  await run(invocation.cmd, invocation.args);
  await copyFile(
    resolve(outDir, 'maxflow-simd.module.wasm'),
    resolve(outDir, 'maxflow-simd.wasm'),
  );
}

async function compileThreads() {
  log('Compiling threaded maxflow module');
  const invocation = compilerInvocation([
    ...commonArgs('createBrenizerMaxflowThreadsModule'),
    '-pthread',
    '-msimd128',
    '-s', 'PTHREAD_POOL_SIZE=4',
    '-o', resolve(outDir, 'maxflow-threads.module.js'),
  ]);
  await run(invocation.cmd, invocation.args);
  await copyFile(
    resolve(outDir, 'maxflow-threads.module.wasm'),
    resolve(outDir, 'maxflow-threads.wasm'),
  );
}

async function main() {
  await mkdir(dirname(resolve(outDir, 'placeholder')), { recursive: true });
  await mkdir(emCache, { recursive: true });
  if (await outputsAreFresh()) {
    log('Maxflow artifacts are up to date');
    return;
  }
  await ensureCompilerAvailable();
  await mkdir(outDir, { recursive: true });
  await compileSimd();
  await compileThreads();
  await emitThirdPartyNotices();
  log('Prepared browser-WASM maxflow artifacts');
}

main().catch((err) => {
  console.error(`[prepare:maxflow] ${err.message}`);
  process.exit(1);
});
