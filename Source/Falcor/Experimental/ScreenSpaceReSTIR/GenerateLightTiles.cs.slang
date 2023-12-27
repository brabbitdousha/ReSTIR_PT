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
import Lights;
import Utils.Sampling.TinyUniformSampleGenerator;

/** Generates random light samples and stores them into light tiles.
    These light samples are later resampled in the initial resampling pass.
    Using light samples from small light tiles highly improves performance
    due to reduced cache trashing.
*/
struct GenerateLightTiles
{
    // Static configuration.
    static const uint kLightTileCount = LIGHT_TILE_COUNT;
    static const uint kLightTileSize = LIGHT_TILE_SIZE;

    static const uint kEnvLightSampleCount = ENV_LIGHT_SAMPLE_COUNT;
    static const uint kEmissiveLightSampleCount = EMISSIVE_LIGHT_SAMPLE_COUNT;
    static const uint kAnalyticLightSampleCount = ANALYTIC_LIGHT_SAMPLE_COUNT;

    // Resources.
    RWStructuredBuffer<PackedLoadedLightSample> lightTileData;

    Lights lights;

    uint frameIndex;

    float getStratifiedRandom(uint sampleIndex, uint sampleCount, float offset)
    {
        return frac(offset + float(sampleIndex) / float(sampleCount));
    }

    void process(uint tileIndex, uint sampleIndex)
    {
        if (tileIndex >= kLightTileCount || sampleIndex >= kLightTileSize) return;

        // Compute index into light tile buffer.
        uint bufferIndex = tileIndex * kLightTileSize + sampleIndex;

        // Setup per-tile random generator and pick random offset.
        TinyUniformSampleGenerator tileSg = TinyUniformSampleGenerator(tileIndex, frameIndex);
        float offset = sampleNext1D(tileSg);

        // Setup per-thread random generator.
        TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(bufferIndex, frameIndex + 1);

        LightSample lightSample = LightSample::createInvalid();

        if (sampleIndex < kEnvLightSampleCount)
        {
            // Environment map.
            float random = getStratifiedRandom(sampleIndex, kEnvLightSampleCount, offset);
            lightSample = lights.sampleEnv(random, sg);
        }
        else if (sampleIndex < kEnvLightSampleCount + kEmissiveLightSampleCount)
        {
            // Emissive light.
            float random = getStratifiedRandom(sampleIndex, kEmissiveLightSampleCount, offset);
            lightSample = lights.sampleEmissive(random, sg);
        }
        else /* if (sampleIndex < kEnvLightSampleCount + kEmissiveLightSampleCount + kAnalyticLightSampleCount) */
        {
            // Analytic light.
            float random = getStratifiedRandom(sampleIndex, kAnalyticLightSampleCount, offset);
            lightSample = lights.sampleAnalytic(random, sg);
        }

        // Write light sample.
        lightTileData[bufferIndex] = lights.loadLightSample(lightSample).pack();
    }
};

cbuffer CB
{
    GenerateLightTiles gGenerateLightTiles;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gGenerateLightTiles.process(dispatchThreadId.y, dispatchThreadId.x);
}
