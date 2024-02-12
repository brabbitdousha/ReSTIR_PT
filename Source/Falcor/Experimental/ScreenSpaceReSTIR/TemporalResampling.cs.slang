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
import Utils.Attributes;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Debug.PixelDebug;
import SurfaceData;
import EvalContext;
import Lights;
import Resampling;
import Params;

/** Performs temporal resampling by combining the reservoir from the initial
    resampling pass with a reservoir from the last frame obtained through
    reprojection.
*/
struct TemporalResampling
{
    // Static configuration.
    static const uint kMaxHistoryLength = MAX_HISTORY_LENGTH;

    static const bool kUsePairwiseMIS = USE_PAIRWISE_MIS;
    static const bool kUnbiased = UNBIASED;

    static const DebugOutput kDebugOutput = DebugOutput(DEBUG_OUTPUT);

    [root] StructuredBuffer<PackedSurfaceData> surfaceData;
    [root] StructuredBuffer<PackedSurfaceData> prevSurfaceData;
    Texture2D<float2> motionVectors;
    RWStructuredBuffer<PackedReservoir> reservoirs;
    StructuredBuffer<PackedReservoir> prevReservoirs;
    RWTexture2D<float4> debugOutput;

    Lights lights;

    uint2 frameDim;
    uint frameIndex;

    float normalThreshold;
    float depthThreshold;

    uint getPixelIndex(uint2 pixel) { return pixel.y * frameDim.x + pixel.x; }

    bool isValidPixel(int2 pixel) { return all(pixel >= 0 && pixel < frameDim); }

    void process(uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        printSetPixel(pixel);

        float2 mv = motionVectors[pixel];

        if (kDebugOutput == DebugOutput::TemporalReuse) debugOutput[pixel] = float4(1.0f, 0.f, 0.f, 0.f);

        const uint pixelIndex = getPixelIndex(pixel);

        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, frameIndex);

        // Back-project this pixel to previous frame.
        const int2 prevPixel = pixel + motionVectors[pixel] * frameDim + (sampleNext2D(sg) * 1.f - 0.f);
        if (!isValidPixel(prevPixel)) return;
        const uint prevPixelIndex = getPixelIndex(prevPixel);

        // Setup evaluation context for back-projected pixel.
        const EvalContext prevContext = EvalContext::create(prevPixel, frameDim, SurfaceData::createFromPacked(prevSurfaceData[prevPixelIndex]));
        if (!prevContext.isValid()) return;

        // Write debug output.
        if (kDebugOutput == DebugOutput::TemporalReuse) debugOutput[pixel] = float4(prevContext.N * 0.5f + 0.5f, 0.f);

        // Setup evaluation context for current pixel.
        const EvalContext currentContext = EvalContext::create(pixel, frameDim, SurfaceData::createFromPacked(surfaceData[pixelIndex]));
        if (!currentContext.isValid()) return;

        // Load reservoirs.
        Reservoir currentReservoir = Reservoir::createFromPacked(reservoirs[pixelIndex]);
        Reservoir prevReservoir = Reservoir::createFromPacked(prevReservoirs[prevPixelIndex]);
        prevReservoir.M = min(prevReservoir.M, currentReservoir.M * kMaxHistoryLength);

        // Setup resampling.
        RisState risState = RisState::createEmpty();

        if (kUsePairwiseMIS)
        {
            const LoadedLightSample currentLightSample = lights.loadLightSample(currentReservoir.lightSample);
            float currentTargetPdf = currentContext.evalTargetFunction(currentLightSample);
            streamingResampleStepMis(risState, currentReservoir, currentLightSample, currentTargetPdf, currentContext, prevReservoir, prevContext, 1, lights, sg);
            streamingResampleFinalizeMis(risState, currentReservoir, currentTargetPdf, sg);
            risState.weight = risState.weight > 0.f ? risState.weightSum / risState.weight : 0.f;
        }
        else
        {
            // Check if previous sample is valid for reuse.
            if (!kUnbiased && !currentContext.isValidNeighbor(prevContext, normalThreshold, depthThreshold)) return;

            // Resample current and previous reservoirs.
            streamingResampleStep(risState, currentReservoir, currentContext, lights, sg);
            bool usedPrev = streamingResampleStep(risState, prevReservoir, currentContext, lights, sg);

            // Compute the unbiased normalization term (instead of using 1/M).
            const LoadedLightSample lightSample = lights.loadLightSample(risState.lightSample);
            float currentPdf = currentContext.evalTargetFunction(lightSample, kUnbiased);
            float prevPdf = prevContext.evalTargetFunction(lightSample, kUnbiased);
            float normalization = (usedPrev ? prevPdf : currentPdf) / (currentReservoir.M * currentPdf + prevReservoir.M * prevPdf);
            risState.weight = risState.weight > 0.f ? (risState.weightSum * normalization) / risState.weight : 0.f;
        }

        // Save reservoir.
        reservoirs[pixelIndex] = risState.toReservoir().pack();
    }
};

cbuffer CB
{
    TemporalResampling gTemporalResampling;
}

[numthreads(16, 4, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gTemporalResampling.process(dispatchThreadId.xy);
}
