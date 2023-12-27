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
__exported import LightSample;

// Use compact packing. Changing this requires restarting the renderer.
#define USE_COMPACT_RESERVOIR 0

#if USE_COMPACT_RESERVOIR
typedef uint2 PackedReservoir;
#else
typedef uint4 PackedReservoir;
#endif

/** Represents a reservoir containing a weighted light sample obtained through resampled importance sampling.
*/
struct Reservoir
{
    LightSample lightSample;    ///< Selected light sample.
    uint M;                     ///< Number of samples seen so far.
    float weight;               ///< Probabilistic weight.

    static Reservoir createEmpty()
    {
        return { LightSample::createInvalid(), 0.f, 0.f };
    }

    static Reservoir createFromPacked(const PackedReservoir packed)
    {
        Reservoir reservoir = {};
        reservoir.unpack(packed);
        return reservoir;
    }

    bool isValid()
    {
        return lightSample.isValid();
    }

    PackedReservoir pack()
    {
        PackedReservoir packed;
#if USE_COMPACT_RESERVOIR
        packed.x = uint(lightSample.pack());
        packed.y = f32tof16(weight) | (clamp(M, 0, 0xffff) << 16);
#else
        packed.x = lightSample.pack();
        packed.y = asuint(weight);
        packed.z = M;
        packed.w = 0;
#endif
        return packed;
    }

    [mutating] void unpack(const PackedReservoir packed)
    {
#if USE_COMPACT_RESERVOIR
        lightSample.unpack(packed.x);
        weight = f16tof32(packed.y & 0xffff);
        M = packed.y >> 16;
#else
        lightSample.unpack(packed.x);
        weight = asfloat(packed.y);
        M = packed.z;
#endif
        // Replace with empty reservoir if invalid.
        if (isinf(weight) || isnan(weight))
        {
            lightSample = LightSample::createInvalid();
            weight = 0.f;
            M = 0.f;
        }
    }
};
