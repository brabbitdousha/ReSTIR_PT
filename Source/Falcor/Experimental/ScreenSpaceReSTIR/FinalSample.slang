/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Utils.Math.PackedFormats;

// Use compact packing. Changing this requires restarting the renderer.
#define USE_COMPACT_FINAL_SAMPLE 0

#if USE_COMPACT_FINAL_SAMPLE
typedef uint4 PackedFinalSample;
#else
struct PackedFinalSample
{
    float3 dir;
    float distance;
    float3 Li;
    float _pad;
};
#endif

/** Represents a final light sample used for shading pixels.
*/
struct FinalSample
{
    float3 dir;     ///< Direction to the sampled light in world-space.
    float distance; ///< Distance to the sampled light.
    float3 Li;      ///< Weighted incident radiance.

    static FinalSample createFromPacked(const PackedFinalSample packed)
    {
        FinalSample finalSample;
        finalSample.unpack(packed);
        return finalSample;
    }

    PackedFinalSample pack()
    {
        PackedFinalSample packed;
#if USE_COMPACT_FINAL_SAMPLE
        packed.x = encodeNormal2x16(dir);
        packed.y = asuint(distance);
        packed.z = encodeLogLuvHDR(Li);
        packed.w = 0;
#else
        packed.dir = dir;
        packed.distance = distance;
        packed.Li = Li;
        packed._pad = 0.f;
#endif
        return packed;
    }

    [mutating] void unpack(const PackedFinalSample packed)
    {
#if USE_COMPACT_FINAL_SAMPLE
        dir = decodeNormal2x16(packed.x);
        distance = asfloat(packed.y);
        Li = decodeLogLuvHDR(packed.z);
#else
        dir = packed.dir;
        distance = packed.distance;
        Li = packed.Li;
#endif
    }
};
