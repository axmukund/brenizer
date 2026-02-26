import type { Capabilities } from './capabilities';

/** All tuneable pipeline parameters. */
export interface PipelineSettings {
  maxImages: number;
  alignScale: number;
  orbFeatures: number;
  pairWindowW: number;
  matchAllPairs: boolean;
  ratioTest: number;
  ransacThreshPx: number;
  refineIters: number;
  meshGrid: number;       // 0 = off
  depthEnabled: boolean;
  depthInputSize: number;
  seamMethod: 'graphcut' | 'feather';
  seamBlockSize: number;
  depthSeamBias: number;   // reserved (not currently applied in seam cost)
  featherWidth: number;
  multibandEnabled: boolean;
  multibandLevels: number; // 0 = auto
  exposureComp: boolean;
  exportScale: number;     // fraction of alignment scale; 'max' handled separately
  exportFormat: 'png' | 'jpeg';
  exportJpegQuality: number;
  /** AI saliency-aware seam placement (Itti-Koch-Niebur + Achanta CVPR 2009) */
  saliencyEnabled: boolean;
  /** PTGui-style polynomial radial vignetting correction V(r) = 1 + ar² + br⁴ */
  vignetteCorrection: boolean;
  /** Export at maximum possible resolution (no downscale). */
  maxResExport: boolean;
  /** Blur-aware feature weighting for Brenizer method. */
  blurAwareStitching: boolean;
  /** Reserved for future cylindrical pre-warp integration. */
  cylindricalProjection: boolean;
  /** Reserved for future Brown-Conrady correction pass. */
  lensDistortionCorrection: boolean;
  /** Assume all photos share same camera settings (aperture, focal length,
   *  shutter speed, ISO) and were shot within ~2 min of each other.
   *  Enables: pooled vignetting, shared-intrinsics BA, exposure skip,
   *  color-transfer skip, tighter APAP regularization. */
  sameCameraSettings: boolean;
}

export type ModeName = 'auto' | 'desktopHQ' | 'mobileQuality' | 'mobileSafe';

const DESKTOP_HQ: PipelineSettings = {
  maxImages: 25,
  alignScale: 1536,
  orbFeatures: 5000,
  pairWindowW: 6,
  matchAllPairs: false,
  ratioTest: 0.75,
  ransacThreshPx: 3,
  refineIters: 30,
  meshGrid: 12,
  depthEnabled: false,
  depthInputSize: 256,
  seamMethod: 'graphcut',
  seamBlockSize: 16,
  depthSeamBias: 1.0,
  featherWidth: 60,
  multibandEnabled: true,
  multibandLevels: 0, // auto ≤ 6
  exposureComp: true,
  exportScale: 0.5,
  exportFormat: 'png',
  exportJpegQuality: 0.92,
  saliencyEnabled: true,
  vignetteCorrection: true,
  maxResExport: false,
  blurAwareStitching: true,
  cylindricalProjection: false,
  lensDistortionCorrection: false,
  sameCameraSettings: true,
};

const MOBILE_QUALITY: PipelineSettings = {
  maxImages: 18,
  alignScale: 1024,
  orbFeatures: 3500,
  pairWindowW: 4,
  matchAllPairs: false,
  ratioTest: 0.75,
  ransacThreshPx: 3,
  refineIters: 15,
  meshGrid: 10,
  depthEnabled: false,
  depthInputSize: 192,
  seamMethod: 'graphcut',
  seamBlockSize: 24,
  depthSeamBias: 1.0,
  featherWidth: 40,
  multibandEnabled: true,
  multibandLevels: 4,
  exposureComp: true,
  exportScale: 0.33,
  exportFormat: 'jpeg',
  exportJpegQuality: 0.90,
  saliencyEnabled: true,
  vignetteCorrection: true,
  maxResExport: false,
  blurAwareStitching: true,
  cylindricalProjection: false,
  lensDistortionCorrection: false,
  sameCameraSettings: true,
};

const MOBILE_SAFE: PipelineSettings = {
  maxImages: 12,
  alignScale: 768,
  orbFeatures: 2000,
  pairWindowW: 3,
  matchAllPairs: false,
  ratioTest: 0.75,
  ransacThreshPx: 3,
  refineIters: 8,
  meshGrid: 8,
  depthEnabled: false,
  depthInputSize: 128,
  seamMethod: 'graphcut',
  seamBlockSize: 32,
  depthSeamBias: 1.0,
  featherWidth: 30,
  multibandEnabled: true,
  multibandLevels: 3,
  exposureComp: true,
  exportScale: 0.25,
  exportFormat: 'jpeg',
  exportJpegQuality: 0.85,
  saliencyEnabled: false,
  vignetteCorrection: false,
  maxResExport: false,
  blurAwareStitching: false,
  cylindricalProjection: false,
  lensDistortionCorrection: false,
  sameCameraSettings: true,
};

const MOBILE_LITE: PipelineSettings = {
  maxImages: 10,
  alignScale: 768,
  orbFeatures: 1500,
  pairWindowW: 3,
  matchAllPairs: false,
  ratioTest: 0.75,
  ransacThreshPx: 3,
  refineIters: 4,
  meshGrid: 0,
  depthEnabled: false,
  depthInputSize: 128,
  seamMethod: 'feather',
  seamBlockSize: 32,
  depthSeamBias: 0,
  featherWidth: 30,
  multibandEnabled: false,
  multibandLevels: 0,
  exposureComp: false,
  exportScale: 0.25,
  exportFormat: 'jpeg',
  exportJpegQuality: 0.80,
  saliencyEnabled: false,
  vignetteCorrection: false,
  maxResExport: false,
  blurAwareStitching: false,
  cylindricalProjection: false,
  lensDistortionCorrection: false,
  sameCameraSettings: true,
};

export const PRESETS: Record<string, PipelineSettings> = {
  desktopHQ: DESKTOP_HQ,
  mobileQuality: MOBILE_QUALITY,
  mobileSafe: MOBILE_SAFE,
  mobileLite: MOBILE_LITE,
};

/** Select the effective mode given user choice + capabilities. */
export function resolveMode(userMode: ModeName, mobileSafeFlag: boolean, caps: Capabilities): string {
  if (mobileSafeFlag) return 'mobileSafe';

  if (userMode !== 'auto') return userMode;

  // Auto selection
  if (caps.isMobile) {
    const strong = (caps.deviceMemory !== null && caps.deviceMemory >= 4)
      || (caps.hardwareConcurrency >= 6 && caps.floatFBO);
    return strong ? 'mobileQuality' : 'mobileSafe';
  }

  return 'desktopHQ';
}

/** Get a full settings object for a resolved mode name. Returns a copy. */
export function getPreset(mode: string): PipelineSettings {
  const base = PRESETS[mode] || PRESETS.desktopHQ;
  return { ...base };
}
