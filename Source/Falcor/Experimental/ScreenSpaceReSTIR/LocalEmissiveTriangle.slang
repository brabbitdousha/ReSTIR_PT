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

struct PackedLocalEmissiveTriangle
{
    uint4 posNormal;
    uint4 edgesRadiance;
};

/** Compact representation of an emissive triangle.
    This is used in place of the scene data for more efficient
    sampling and evaluation of emissive triangles.
*/
struct LocalEmissiveTriangle
{
    float3 pos;     ///< Position in world-space (i.e. p0).
    float3 normal;  ///< Normal in world-space.
    float3 edge0;   ///< First edge in world-space (i.e. p1 - p0).
    float3 edge1;   ///< Second edge in world-space (i.e. p2 - p0).
    float radiance; ///< Average radiance luminance.

    static LocalEmissiveTriangle createFromPacked(const PackedLocalEmissiveTriangle packed)
    {
        LocalEmissiveTriangle tri;
        tri.unpack(packed);
        return tri;
    }

    float3 getPosition(const float2 uv)
    {
        return pos + edge0 * uv.x + edge1 * uv.y;
    }

    float getArea()
    {
        return 0.5f * length(cross(edge0, edge1));
    }

    PackedLocalEmissiveTriangle pack()
    {
        PackedLocalEmissiveTriangle packed;

        packed.posNormal.xyz = asuint(pos);
        packed.posNormal.w = encodeNormal2x8(normal);

        packed.edgesRadiance.xyz = f32tof16(edge0) | (f32tof16(edge1) << 16);
        packed.edgesRadiance.w = asuint(radiance);

        return packed;
    }

    [mutating] void unpack(const PackedLocalEmissiveTriangle packed)
    {
        pos = asfloat(packed.posNormal.xyz);
        normal = decodeNormal2x8(packed.posNormal.w);

        edge0 = f16tof32(packed.edgesRadiance.xyz);
        edge1 = f16tof32(packed.edgesRadiance.xyz >> 16);
        radiance = asfloat(packed.edgesRadiance.w);
    }
};
