#!/usr/bin/env node

import { mkdir, copyFile, stat } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const source = resolve(process.cwd(), 'node_modules/@techstark/opencv-js/dist/opencv.js');
const dest = resolve(process.cwd(), 'public/opencv/opencv.js');

async function main() {
  try {
    await stat(source);
  } catch {
    throw new Error(
      'OpenCV source not found at node_modules/@techstark/opencv-js/dist/opencv.js. Run `npm install` first.',
    );
  }

  await mkdir(dirname(dest), { recursive: true });
  await copyFile(source, dest);
  console.log('Prepared OpenCV runtime at public/opencv/opencv.js');
}

main().catch((err) => {
  console.error(`[prepare:opencv] ${err.message}`);
  process.exit(1);
});
