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
import Params;
import Shift;
import Scene.HitInfo;
import PathReservoir;
import PathTracer;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Rendering.Utils.PixelStats;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;
import LoadShadingData;
import Rendering.Materials.MaterialShading;

/** TODO.
*/
struct PathRetracePass
{
    static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
    static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

    // Resources
    RestirPathTracerParams params;                        ///< Runtime parameters.

    Texture2D<PackedHitInfo> vbuffer;                     ///< Fullscreen V-buffer for the primary hits.
    Texture2D<PackedHitInfo> temporalVbuffer;
    Texture2D<float2> motionVectors;

    ByteAddressBuffer nRooksPattern;

    RWTexture2D<float4> outputColor;                      ///< Output resolved color.
    RWStructuredBuffer<PathReservoir> outputReservoirs;                  
    RWStructuredBuffer<PathReservoir> temporalReservoirs;
    RWStructuredBuffer<PixelReconnectionData> reconnectionDataBuffer;

    int  gNumSpatialRounds;
    bool gEnableTemporalReprojection;

    float gTemporalHistoryLength;
    bool gNoResamplingForTemporalReuse;

    bool isValidPackedHitInfo(PackedHitInfo packed)
    {
        return packed.x != 0;
    }

    bool isValidScreenRegion(int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

    ShadingData getPixelShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinhole(pixel, params.frameDim);
        PrimaryHitPacked = vbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    ShadingData getPixelTemporalShadingData(int2 pixel, out PackedHitInfo PrimaryHitPacked)
    {
        ShadingData sd = {};
        Ray ray = gScene.camera.computeRayPinholePrevFrame(pixel, params.frameDim);
        PrimaryHitPacked = temporalVbuffer[pixel];
        if (isValidPackedHitInfo(PrimaryHitPacked))
        {
            HitInfo PrimaryHit; PrimaryHit.unpack(PrimaryHitPacked);
            sd = loadShadingData(PrimaryHit, -ray.dir, true);
        }
        return sd;
    }

    /*
    * Build a reservoir of paths whose suffix can be merged to a pixel's prefix.
    * This method is optimized for small windows such that we can run the entire
    * algorithm in one pass.
    */
    void ReSTIR(const uint2 pixel)
    {
        // TODO: How to seed efficiently?
        var sg = TinyUniformSampleGenerator(pixel, (kCandidateSamples + 1 + gNumSpatialRounds) * params.seed + kCandidateSamples);

        // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
        const uint centralOffset = params.getReservoirOffset(pixel);
        PathReservoir centralReservoir = outputReservoirs[centralOffset];

        PackedHitInfo centralPrimaryHitPacked;
        ShadingData centralPrimarySd = getPixelShadingData(pixel, centralPrimaryHitPacked);
        if (!isValidPackedHitInfo(centralPrimaryHitPacked)) return;

        // color averaging test

        // fetch temporal reservoir
        float2 motionVector = motionVectors[pixel];
        int2 prevPixel = pixel;

        bool foundTemporalSurface = true;

        if (gEnableTemporalReprojection)
        {
            prevPixel = pixel + motionVector * params.frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        }

        if (!isValidScreenRegion(prevPixel) || !foundTemporalSurface) return;

        PackedHitInfo temporalPrimaryHitPacked;
        ShadingData temporalPrimarySd = getPixelTemporalShadingData(prevPixel, temporalPrimaryHitPacked);
        if (!isValidPackedHitInfo(temporalPrimaryHitPacked)) return;

        PathReservoir temporalReservoir = temporalReservoirs[params.getReservoirOffset(prevPixel)];

        // talbot MIS
        // compute mis weight for current pixel

        HitInfo dstRcPrevVertexHit;
        HitInfo dstRcPrevVertexHit2;
        float3 dstRcPrevVertexWo;
        float3 dstRcPrevVertexWo2;
        float3 Tp;
        float3 Tp2;

        // write to memory
        if (centralReservoir.pathFlags.rcVertexLength() > 1)
        {
            Tp = traceHybridShiftRays(params, true, temporalPrimaryHitPacked, temporalPrimarySd, centralReservoir, dstRcPrevVertexHit, dstRcPrevVertexWo);
            reconnectionDataBuffer[centralOffset].data[0] = ReconnectionData(dstRcPrevVertexHit, dstRcPrevVertexWo, Tp);
        }
        if (temporalReservoir.pathFlags.rcVertexLength() > 1)
        {
            Tp2 = traceHybridShiftRays(params, false, centralPrimaryHitPacked, centralPrimarySd, temporalReservoir, dstRcPrevVertexHit2, dstRcPrevVertexWo2);
            reconnectionDataBuffer[centralOffset].data[1] = ReconnectionData(dstRcPrevVertexHit2, dstRcPrevVertexWo2, Tp2);
        }
    }

    /** Entry point for merge pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 _pixel)
    {
        const uint2 pixel = _pixel;

        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);
        logSetPixel(pixel);

        ReSTIR(pixel);
    }
};

cbuffer CB
{
    PathRetracePass gPathRetracePass;
}


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPathRetracePass.execute(dispatchThreadId.xy);
}
