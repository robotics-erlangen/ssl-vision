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
  \file    cmvision_threshold.cpp
  \brief   C++ Implementation: cmvision_threshold
  \author  James Bruce (Original CMVision implementation and algorithms),
           Some Code Restructuring, and data structure changes: Stefan Zickler 2008
*/
//========================================================================
#include "cmvision_threshold.h"
#ifdef __AVX2__
#include <x86intrin.h>
#endif

CMVisionThreshold::CMVisionThreshold()
{
}


CMVisionThreshold::~CMVisionThreshold()
{
}

bool CMVisionThreshold::thresholdImageYUV422_UYVY(Image<raw8> * target, const RawImage * source, YUVLUT * lut, const ImageInterface* mask) {
  if (source->getColorFormat()!=COLOR_YUV422_UYVY) {
    //TODO add YUV444 and maybe even 411 mode
    fprintf(stderr,"CMVision thresholdImageYUV422_UYVY assumes YUV422 as input, but found %s\n", Colors::colorFormatToString(source->getColorFormat()).c_str());
    return false;
  }

  register lut_mask_t * LUT = lut->getTable();

  register unsigned int          target_size    = target->getNumPixels();
  register uyvy *       source_pointer = (uyvy*)(source->getData());
  register raw8 *      target_pointer = target->getPixelData();
  register unsigned char *      mask_pointer = mask->getData();

  if (target->getNumPixels() != source->getNumPixels()) {
    fprintf(stderr, "CMVision YUV422_UYVY thresholding: source (num=%d  w=%d  h=%d) and target (num=%d w=%d h=%d) pixel counts do not match!\n", source->getNumPixels(),source->getWidth(),source->getHeight(), target->getNumPixels(),target->getWidth(),target->getHeight());
    return false;
  }

  int X_SHIFT=lut->X_SHIFT;
  int Y_SHIFT=lut->Y_SHIFT;
  int Z_SHIFT=lut->Z_SHIFT;
  int Z_AND_Y_BITS=lut->Z_AND_Y_BITS;
  int Z_BITS = lut->Z_BITS;
  uyvy p;
  for (unsigned int i=0;i<target_size;i+=2) {
    p=source_pointer[(i >> 0x01)];
    register int B=((p.u >> Y_SHIFT) << Z_BITS);
    register int C=(p.v >> Z_SHIFT);
    target_pointer[i] =  mask_pointer[i] & LUT[(((p.y1 >> X_SHIFT) << Z_AND_Y_BITS) | B | C)];
    target_pointer[i+1] =  mask_pointer[i+1] & LUT[(((p.y2 >> X_SHIFT) << Z_AND_Y_BITS) | B | C)];
  }
  return true;
}

bool CMVisionThreshold::thresholdImageYUV444(Image<raw8> * target, const ImageInterface * source, YUVLUT * lut, const ImageInterface* mask) {
  if (source->getColorFormat()!=COLOR_YUV444) {
    fprintf(stderr,"CMVision thresholdImageYUV444 assumes YUV444 as input, but found %s\n", Colors::colorFormatToString(source->getColorFormat()).c_str());
    return false;
  }

  register lut_mask_t * LUT = lut->getTable();

  register unsigned int          target_size    = target->getNumPixels();
  register yuv  *                source_pointer = (yuv*)(source->getData());
  register raw8 *                target_pointer = target->getPixelData();
  register unsigned char *       mask_pointer = mask->getData();

  if (target->getNumPixels() != source->getNumPixels()) {
     fprintf(stderr, "CMVision YUV444 thresholding: source (num=%d  w=%d  h=%d) and target (num=%d w=%d h=%d) pixel counts do not match!\n", source->getNumPixels(),source->getWidth(),source->getHeight(), target->getNumPixels(),target->getWidth(),target->getHeight());
    return false;
  }

  int X_SHIFT=lut->X_SHIFT;
  int Y_SHIFT=lut->Y_SHIFT;
  int Z_SHIFT=lut->Z_SHIFT;
  int Z_AND_Y_BITS=lut->Z_AND_Y_BITS;
  int Z_BITS = lut->Z_BITS;
  yuv p;
  for (unsigned int i=0;i<target_size;i++) {
    p=source_pointer[i];
    target_pointer[i] =  mask_pointer[i] & LUT[(((p.y >> X_SHIFT) << Z_AND_Y_BITS) | ((p.u >> Y_SHIFT) << Z_BITS) | (p.v >> Z_SHIFT))];
  }

  return true;
}



bool CMVisionThreshold::thresholdImageRGB(Image<raw8> * target, const ImageInterface * source, YUVLUT * lut, const ImageInterface* mask) {
  // Each pixel is converted to YUV exactly as the LUT GUI does (Conversions::rgb2yuv) and looked up in the YUV LUT,
  // so the classification matches what is shown while calibrating.
  if (source->getColorFormat()!=COLOR_RGB8) {
    fprintf(stderr,"CMVision RGB thresholding assumes RGB8 as input, but found %s\n", Colors::colorFormatToString(source->getColorFormat()).c_str());
    return false;
  }

  lut_mask_t * LUT = lut->getTable();
  int source_size    = source->getNumPixels();
  const uint8_t * source_pixel = source->getData();
  auto * target_pointer = (uint8_t*) target->getPixelData();
  auto * mask_pointer = mask->getData();

  if (target->getNumPixels() != source->getNumPixels()) {
    fprintf(stderr, "CMVision RGB thresholding: source (num=%d  w=%d  h=%d) and target (num=%d w=%d h=%d) pixel counts do not match!\n", source->getNumPixels(),source->getWidth(),source->getHeight(), target->getNumPixels(),target->getWidth(),target->getHeight());
    return false;
  }

  int X_SHIFT=lut->X_SHIFT;
  int Y_SHIFT=lut->Y_SHIFT;
  int Z_SHIFT=lut->Z_SHIFT;
  int Z_AND_Y_BITS=lut->Z_AND_Y_BITS;
  int Z_BITS = lut->Z_BITS;

  int i=0;
#ifdef __AVX2__
  // unpacking from: https://docs.google.com/presentation/d/1I0-SiHid1hTsv7tjLST2dYW5YF5AJVfs9l4Rg9rvz48/edit#slide=id.g1eefe20b_0_125
  __m128i ssse3_red_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
  __m128i ssse3_red_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
  __m128i ssse3_red_indeces_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  __m128i ssse3_green_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
  __m128i ssse3_green_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
  __m128i ssse3_green_indeces_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
  __m128i ssse3_blue_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
  __m128i ssse3_blue_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
  __m128i ssse3_blue_indeces_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

  // rgb2yuv coefficients as 16-bit pairs for madd: (r,g) pairs and (b,0) pairs
  const __m256i y_rg = _mm256_set1_epi32((601 << 16) | 306);
  const __m256i y_b  = _mm256_set1_epi32(117);
  const __m256i u_rg = _mm256_set1_epi32((int)(((uint32_t)(uint16_t)-340 << 16) | (uint16_t)-172));
  const __m256i u_b  = _mm256_set1_epi32(512);
  const __m256i v_rg = _mm256_set1_epi32((int)(((uint32_t)(uint16_t)-429 << 16) | 512));
  const __m256i v_b  = _mm256_set1_epi32((uint16_t)-83);
  const __m256i c128 = _mm256_set1_epi32(128);
  const __m256i c255 = _mm256_set1_epi32(255);
  const __m256i zero = _mm256_setzero_si256();

  // 8 pixels as (r,g) and (b,0) pairs -> 8 32-bit LUT indices
  auto lut_index = [&](__m256i rg, __m256i b0) {
    __m256i y = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(rg, y_rg), _mm256_madd_epi16(b0, y_b)), 10);
    __m256i u = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(rg, u_rg), _mm256_madd_epi16(b0, u_b)), 10), c128);
    __m256i v = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(rg, v_rg), _mm256_madd_epi16(b0, v_b)), 10), c128);
    y = _mm256_min_epi32(_mm256_max_epi32(y, zero), c255);
    u = _mm256_min_epi32(_mm256_max_epi32(u, zero), c255);
    v = _mm256_min_epi32(_mm256_max_epi32(v, zero), c255);
    return _mm256_or_si256(_mm256_slli_epi32(_mm256_srli_epi32(y, X_SHIFT), Z_AND_Y_BITS),
                           _mm256_or_si256(_mm256_slli_epi32(_mm256_srli_epi32(u, Y_SHIFT), Z_BITS), _mm256_srli_epi32(v, Z_SHIFT)));
  };

  alignas(32) uint16_t idx[16];

  for (; i+16<=source_size; i+=16) {

    // crazy RGB unpacking
    const __m128i chunk0 = _mm_loadu_si128((const __m128i*)(source_pixel));
    const __m128i chunk1 = _mm_loadu_si128((const __m128i*)(source_pixel + 16));
    const __m128i chunk2 = _mm_loadu_si128((const __m128i*)(source_pixel + 32));
    source_pixel += 48;

    const __m128i red = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, ssse3_red_indeces_0),
                                                  _mm_shuffle_epi8(chunk1, ssse3_red_indeces_1)), _mm_shuffle_epi8(chunk2, ssse3_red_indeces_2));
    const __m128i green = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, ssse3_green_indeces_0),
                                                    _mm_shuffle_epi8(chunk1, ssse3_green_indeces_1)), _mm_shuffle_epi8(chunk2, ssse3_green_indeces_2));
    const __m128i blue = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, ssse3_blue_indeces_0),
                                                   _mm_shuffle_epi8(chunk1, ssse3_blue_indeces_1)), _mm_shuffle_epi8(chunk2, ssse3_blue_indeces_2));

    // widen pixel values to 16bit
    __m256i r = _mm256_cvtepu8_epi16(red);
    __m256i g = _mm256_cvtepu8_epi16(green);
    __m256i b = _mm256_cvtepu8_epi16(blue);

    // unpack works per 128-bit lane: lo holds pixels 0-3 and 8-11, hi holds 4-7 and 12-15.
    // packus (also per lane) restores the original order.
    __m256i lo = lut_index(_mm256_unpacklo_epi16(r, g), _mm256_unpacklo_epi16(b, zero));
    __m256i hi = lut_index(_mm256_unpackhi_epi16(r, g), _mm256_unpackhi_epi16(b, zero));
    _mm256_store_si256((__m256i*)idx, _mm256_packus_epi32(lo, hi));

#pragma GCC unroll 16
    for(int j=0; j<16; j++) {
      target_pointer[i+j] = mask_pointer[i+j] & LUT[idx[j]];
    }
  }
#endif
  // remaining pixels (all pixels without AVX2)
  for (; i<source_size; i++, source_pixel+=3) {
    int y, u, v;
    Conversions::rgb2yuv(source_pixel[0], source_pixel[1], source_pixel[2], y, u, v);
    target_pointer[i] = mask_pointer[i] & LUT[((y >> X_SHIFT) << Z_AND_Y_BITS) | ((u >> Y_SHIFT) << Z_BITS) | (v >> Z_SHIFT)];
  }

  return true;
}
