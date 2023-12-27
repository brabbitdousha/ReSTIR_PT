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
import Scene.Scene;
import Utils.Color.ColorHelpers;
import LocalEmissiveTriangle;

/** Collects emissive triangles from the scene representation and copies
    them to a compact local array for fast access during light sampling and evaluation.
*/
struct UpdateEmissiveTriangles
{
    // Resources.
    RWStructuredBuffer<PackedLocalEmissiveTriangle> emissiveTriangles;

    uint emissiveTriangleCount;

    void process(uint triangleIndex)
    {
        if (triangleIndex >= emissiveTriangleCount) return;

        EmissiveTriangle inTriangle = gScene.lightCollection.getTriangle(triangleIndex);

        LocalEmissiveTriangle outTriangle = {};
        outTriangle.pos = inTriangle.posW[0];
        outTriangle.normal = inTriangle.normal;
        outTriangle.edge0 = inTriangle.posW[1] - inTriangle.posW[0];
        outTriangle.edge1 = inTriangle.posW[2] - inTriangle.posW[0];
        outTriangle.radiance = luminance(gScene.lightCollection.getAverageRadiance(triangleIndex));

        emissiveTriangles[triangleIndex] = outTriangle.pack();
    }
};

cbuffer CB
{
    UpdateEmissiveTriangles gUpdateEmissiveTriangles;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gUpdateEmissiveTriangles.process(dispatchThreadId.x);
}
