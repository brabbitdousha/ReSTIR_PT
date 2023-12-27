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

/** Compute shader implementing a reservoir reset pass.

    The shader resets initial sample buffer. If frameCount is 0, or forceClearReservoirs is set,
    It also clears two GI reservoir buffers.

    The dispatch dimension is over pixels (XY).
*/

import GIReservoir;

struct GIClearReservoirs
{
    uint2 frameDim;
    uint frameCount;
    uint reservoirCount;
    bool forceClearReservoirs;

    RWStructuredBuffer<PackedGIReservoir> initialSamples;
    RWStructuredBuffer<PackedGIReservoir> reservoirBuffer0;
    RWStructuredBuffer<PackedGIReservoir> reservoirBuffer1;

    uint pixelToLinearIndex(uint2 pixel)
    {
        return pixel.y * frameDim.x + pixel.x;
    }

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        uint pixelIndex = pixelToLinearIndex(pixel);

        GIReservoir reservoir = GIReservoir.createEmpty();

        for (int sampleIndex = 0; sampleIndex < reservoirCount && (frameCount == 0 || forceClearReservoirs); sampleIndex++)
        {
            writeReservoir(reservoirBuffer0, pixelIndex, sampleIndex * 2, frameDim.x * frameDim.y, reservoir);
            writeReservoir(reservoirBuffer0, pixelIndex, sampleIndex * 2 + 1, frameDim.x * frameDim.y, reservoir);
            writeReservoir(reservoirBuffer1, pixelIndex, sampleIndex * 2, frameDim.x * frameDim.y, reservoir);
            writeReservoir(reservoirBuffer1, pixelIndex, sampleIndex * 2 + 1, frameDim.x * frameDim.y, reservoir);
        }

        writeReservoir(initialSamples, pixelIndex, 0, frameDim.x* frameDim.y, reservoir);
        //writeReservoir(initialSamples, pixelIndex, 1, frameDim.x * frameDim.y, reservoir);
    }
};

cbuffer CB
{
    GIClearReservoirs gGIClearReservoirs;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gGIClearReservoirs.execute(dispatchThreadId.xy);
}
