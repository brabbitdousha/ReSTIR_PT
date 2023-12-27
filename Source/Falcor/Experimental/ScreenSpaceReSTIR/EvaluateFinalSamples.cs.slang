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
import Utils.Attributes;
import Utils.Debug.PixelDebug;
import EvalContext;
import Lights;
import Reservoir;
import FinalSample;
import Params;

/** Evaluates reservoirs for final shading and stores the result in the finalSamples buffer.
*/
struct EvaluateFinalSamples
{
    // Static configuration.
    static const bool kUseVisibility = USE_VISIBILITY;
    static const bool kReuseVisibility = REUSE_VISIBILITY;

    static const DebugOutput kDebugOutput = DebugOutput(DEBUG_OUTPUT);

    // Resources.
    [root] StructuredBuffer<PackedSurfaceData> surfaceData;
    RWStructuredBuffer<PackedReservoir> reservoirs;
    RWStructuredBuffer<PackedFinalSample> finalSamples;
    RWTexture2D<float4> debugOutput;

    Lights lights;

    uint2 frameDim;     ///< Frame dimension (pixels).

    uint getPixelIndex(uint2 pixel) { return pixel.y * frameDim.x + pixel.x; }

    void process(uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        printSetPixel(pixel);

        uint pixelIndex = getPixelIndex(pixel);

        FinalSample finalSample = {};

        // Load reservoir.
        Reservoir reservoir = Reservoir::createFromPacked(reservoirs[pixelIndex]);

        if (reservoir.isValid())
        {
            // Create evaluation context.
            EvalContext evalContext = EvalContext::create(pixel, frameDim, SurfaceData::createFromPacked(surfaceData[pixelIndex]));

            // Evaluate light sample.
            EvaluatedLightSample evaluatedLightSample = lights.evalLightSample(reservoir.lightSample, evalContext.pos);

            // Evaluate visibility.
            float visibility = 1.f;
            if (kUseVisibility) visibility = evalContext.evalVisibility(evaluatedLightSample);

            if (visibility > 0.f)
            {
                finalSample.dir = evaluatedLightSample.dir;
                finalSample.distance = evaluatedLightSample.distance;
                finalSample.Li = reservoir.weight * evaluatedLightSample.geomFactor * lights.evalEmission(reservoir.lightSample);
            }
            else
            {
                if (kReuseVisibility)
                {
                    reservoirs[pixelIndex] = Reservoir::createEmpty().pack();
                }
            }
        }

        // Write final sample.
        finalSamples[pixelIndex] = finalSample.pack();

        // Write debug output.
        switch (kDebugOutput)
        {
        case DebugOutput::FinalSampleDir:
            debugOutput[pixel] = float4(finalSample.dir * 0.5f + 0.5f, 0.f);
            break;
        case DebugOutput::FinalSampleDistance:
            debugOutput[pixel] = float4(float3(finalSample.distance), 0.f);
            break;
        case DebugOutput::FinalSampleLi:
            debugOutput[pixel] = float4(finalSample.Li, 0.f);
            break;
        }
    }
};

cbuffer CB
{
    EvaluateFinalSamples gEvaluateFinalSamples;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gEvaluateFinalSamples.process(dispatchThreadId.xy);
}
