//========================================================================
//  This software is free: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License Version 3,
//  as published by the Free Software Foundation.
//
//  This software is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  Version 3 in the file COPYING that came with this distribution.
//  If not, see <http://www.gnu.org/licenses/>.
//========================================================================
/*!
  \file    benchmark_threshold_rgb.cpp
  \brief   Benchmark of RGB color thresholding on real camera frames

  Compares CMVisionThreshold::thresholdImageRGB (exact rgb2yuv per pixel + YUV LUT)
  against the previous approach (lookup in an RGB LUT with 5 bits per channel derived
  from the YUV LUT) and against a plain scalar implementation, which also serves as
  the reference for a correctness check.

  Build and run (on the machine you want numbers for, as the build uses -march=native):
    make -C build benchmark_threshold_rgb
    ./bin/benchmark_threshold_rgb [test-data dir]   (get test data with `make install_test_data`)
*/
//========================================================================

#include "lut3d.h"
#include "conversions.h"
#include "cmvision_threshold.h"
#include "rawimage.h"
#include "image.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

/// Previous approach: index a derived RGB LUT by truncating each channel (as in thresholdImageRGB before the exact conversion).
static void thresholdRgbLut(uint8_t *out, const uint8_t *src, const uint8_t *mask, int n, RGBLUT *lut) {
  const lut_mask_t *LUT = lut->getTable();
  const int XS = lut->X_SHIFT, YS = lut->Y_SHIFT, ZS = lut->Z_SHIFT, ZYB = lut->Z_AND_Y_BITS, ZB = lut->Z_BITS;
  int i = 0;
#ifdef __AVX2__
  const __m128i r0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
  const __m128i r1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
  const __m128i r2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i g0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
  const __m128i g1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
  const __m128i g2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  const __m128i b0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
  const __m128i b1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
  const __m128i b2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  alignas(32) uint16_t idx[16];
  for (; i + 16 <= n; i += 16, src += 48) {
    const __m128i c0 = _mm_loadu_si128((const __m128i *)src);
    const __m128i c1 = _mm_loadu_si128((const __m128i *)(src + 16));
    const __m128i c2 = _mm_loadu_si128((const __m128i *)(src + 32));
    __m256i r = _mm256_cvtepu8_epi16(_mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(c0, r0), _mm_shuffle_epi8(c1, r1)), _mm_shuffle_epi8(c2, r2)));
    __m256i g = _mm256_cvtepu8_epi16(_mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(c0, g0), _mm_shuffle_epi8(c1, g1)), _mm_shuffle_epi8(c2, g2)));
    __m256i b = _mm256_cvtepu8_epi16(_mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(c0, b0), _mm_shuffle_epi8(c1, b1)), _mm_shuffle_epi8(c2, b2)));
    __m256i res = _mm256_or_si256(_mm256_slli_epi16(_mm256_srli_epi16(r, XS), ZYB),
                                  _mm256_or_si256(_mm256_slli_epi16(_mm256_srli_epi16(g, YS), ZB), _mm256_srli_epi16(b, ZS)));
    _mm256_store_si256((__m256i *)idx, res);
    for (int j = 0; j < 16; j++) out[i + j] = mask[i + j] & LUT[idx[j]];
  }
#endif
  for (; i < n; i++, src += 3) {
    out[i] = mask[i] & LUT[((src[0] >> XS) << ZYB) | ((src[1] >> YS) << ZB) | (src[2] >> ZS)];
  }
}

/// Exact conversion without SIMD: what thresholdImageRGB computes, written the obvious way.
static void thresholdExactScalar(uint8_t *out, const uint8_t *src, const uint8_t *mask, int n, YUVLUT *lut) {
  const lut_mask_t *LUT = lut->getTable();
  const int XS = lut->X_SHIFT, YS = lut->Y_SHIFT, ZS = lut->Z_SHIFT, ZYB = lut->Z_AND_Y_BITS, ZB = lut->Z_BITS;
  for (int i = 0; i < n; i++, src += 3) {
    int y, u, v;
    Conversions::rgb2yuv(src[0], src[1], src[2], y, u, v);
    out[i] = mask[i] & LUT[((y >> XS) << ZYB) | ((u >> YS) << ZB) | (v >> ZS)];
  }
}

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "test-data";
  const size_t max_frames = 16;
  const int reps = 7;

  // pick frames spread over all test sets
  std::vector<cv::String> files;
  cv::glob(dir + "/*.png", files, true);
  std::vector<RawImage> frames;
  frames.reserve(max_frames);
  int W = 0, H = 0;
  for (size_t i = 0; i < files.size() && frames.size() < max_frames; i += std::max<size_t>(1, files.size() / max_frames)) {
    cv::Mat bgr = cv::imread(files[i], cv::IMREAD_COLOR), rgb;
    if (bgr.empty() || (W != 0 && (bgr.cols != W || bgr.rows != H))) continue;
    W = bgr.cols;
    H = bgr.rows;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    frames.emplace_back();
    frames.back().allocate(COLOR_RGB8, W, H);
    for (int y = 0; y < H; y++) memcpy(frames.back().getData() + (size_t)y * W * 3, rgb.ptr(y), (size_t)W * 3);
  }
  if (frames.empty()) {
    fprintf(stderr, "No PNG frames found in '%s'. Run `make install_test_data` first.\n", dir.c_str());
    return 1;
  }
  const int N = W * H;
#ifdef __AVX2__
  const char *simd = "AVX2";
#else
  const char *simd = "no AVX2";
#endif
  printf("%zu frames of %dx%d (%.1f MP), single thread, %s, median of %d passes\n\n", frames.size(), W, H, N / 1e6, simd, reps);

  // LUT sizes as in stack_robocup_ssl.cpp; every cell gets its nearest label so no lookup can be skipped
  YUVLUT yuv(4, 6, 6, "");
  auto *rgb_lut = new RGBLUT(5, 5, 5, "");
  yuv.loadRoboCupChannels(LUTChannelMode_Numeric);
  yuv.addDerivedLUT(rgb_lut);
  yuv.computeLUTfromLabels();
  yuv.updateDerivedLUTs();

  Image<raw8> mask;
  mask.allocate(W, H);
  memset(mask.getData(), 0xFF, N);
  const auto *mask_data = (const uint8_t *)mask.getData();
  Image<raw8> target;
  target.allocate(W, H);
  std::vector<uint8_t> reference(N);

  auto run = [&](const char *name, const std::function<void(RawImage &)> &fn) {
    for (auto &f : frames) fn(f);  // warm-up
    std::vector<double> ms;
    for (int r = 0; r < reps; r++) {
      for (auto &f : frames) {
        auto start = Clock::now();
        fn(f);
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
      }
    }
    std::sort(ms.begin(), ms.end());
    double median = ms[ms.size() / 2];
    printf("  %-44s %7.2f ms/frame  %5.2f ns/px  %6.0f MP/s\n", name, median, median * 1e6 / N, N / median / 1e3);
    return median;
  };

  double t_old = run("RGB LUT 5/5/5 (previous)", [&](RawImage &f) {
    thresholdRgbLut((uint8_t *)target.getData(), f.getData(), mask_data, N, rgb_lut);
  });
  double t_exact = run("exact, CMVisionThreshold::thresholdImageRGB", [&](RawImage &f) {
    CMVisionThreshold::thresholdImageRGB(&target, &f, &yuv, &mask);
  });
  double t_scalar = run("exact, scalar", [&](RawImage &f) {
    thresholdExactScalar(reference.data(), f.getData(), mask_data, N, &yuv);
  });
  printf("\nrelative to previous: thresholdImageRGB %.2fx, scalar %.2fx\n", t_exact / t_old, t_scalar / t_old);

  // thresholdImageRGB must match the scalar reference exactly; count how often the previous approach disagrees
  long mismatches = 0, differs_from_old = 0;
  std::vector<uint8_t> old(N);
  for (auto &f : frames) {
    CMVisionThreshold::thresholdImageRGB(&target, &f, &yuv, &mask);
    thresholdExactScalar(reference.data(), f.getData(), mask_data, N, &yuv);
    thresholdRgbLut(old.data(), f.getData(), mask_data, N, rgb_lut);
    const auto *t = (const uint8_t *)target.getData();
    for (int i = 0; i < N; i++) {
      mismatches += t[i] != reference[i];
      differs_from_old += old[i] != reference[i];
    }
  }
  printf("thresholdImageRGB vs scalar reference: %ld mismatching pixels\n", mismatches);
  printf("previous approach classifies %.2f%% of pixels differently (with this synthetic LUT)\n",
         100.0 * differs_from_old / ((double)N * frames.size()));
  return mismatches == 0 ? 0 : 1;
}
