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
import LoadShadingData;
import Rendering.Materials.StandardMaterial;
import Rendering.Materials.Microfacet;
import Experimental.ScreenSpaceReSTIR.ScreenSpaceReSTIR;

RWTexture2D<float4> gColor;
RWTexture2D<float4> gEmission;
RWTexture2D<float4> gDiffuseIllumination;
RWTexture2D<float4> gDiffuseReflectance;
RWTexture2D<float4> gSpecularIllumination;
RWTexture2D<float4> gSpecularReflectance;
RWTexture2D<float4> gDebug;

#define is_valid(name) (is_valid_##name != 0)

/** Perform shading with final samples from ReSTIR.
*/
struct FinalShading
{
    // Static configuration.
    static const bool kUseEnvBackground = USE_ENV_BACKGROUND;

    Texture2D<PackedHitInfo> vbuffer;

    ScreenSpaceReSTIR screenSpaceReSTIR;
    StandardMaterial standardMaterial;

    uint2 frameDim;

    uint numReSTIRInstances;
    uint ReSTIRInstanceID;

    void execute(uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        float3 color = {};
        float3 emission = {};
        float3 diffuseReflectance = {};
        float3 diffuseIllumination = {};
        float3 specularReflectance = {};
        float3 specularIllumination = {};
        float hitT = 10000.f; // TODO: Remove magic number also used in SpatioTemporalResampling pass.
        float3 test_dir = float3(0.f);
        float3 test_Li = float3(0.f);
        float test_distance = 0.f;
        ShadingData sd;
        if (loadShadingData(pixel, frameDim, gScene.camera, vbuffer, sd))
        {
            // Get final sample from ReSTIR.

            float3 dir;
            float distance;
            float3 Li;
            bool valid = screenSpaceReSTIR.getFinalSample(pixel, dir, distance, Li);

            test_dir = (dir + 1)*0.5;
            test_distance = distance;
            test_Li = Li;

            // Evaluate diffuse and specular lobes.
            float3 diffuse = {};
            float3 specular = {};
            if (valid)
            {
                sd.setActiveLobes((uint)LobeType::DiffuseReflection);
                diffuse = standardMaterial.eval(sd, dir) * Li;

                sd.setActiveLobes((uint)LobeType::SpecularReflection);
                specular = standardMaterial.eval(sd, dir) * Li;

                hitT = distance;
            }

            emission = sd.emissive;

            color += diffuse + specular + emission;

            // Demodulate diffuse reflectance (albedo) from diffuse color.
            diffuseReflectance += sd.diffuse;
            float3 diffuseFactor = diffuseReflectance <= 0.f ? 0.f : 1.f / diffuseReflectance;
            diffuseIllumination += diffuse * diffuseFactor;

            // Demodulate preintegrated specular reflectance from specular color.
            specularReflectance += approxSpecularIntegralGGX(sd.specular, sd.linearRoughness * sd.linearRoughness, sd.NdotV);
            float3 specularFactor = specularReflectance <= 0.f ? 0.f : 1.f / specularReflectance;
            specularIllumination += specular * specularFactor;
        }
        else
        {
            if (kUseEnvBackground)
            {
                float3 dir = gScene.camera.computeRayPinhole(pixel, frameDim).dir;
                emission = gScene.envMap.eval(dir);
                color = emission;
            }
        }

        // Write active outputs.

        if (ReSTIRInstanceID == 0)
        {
            if (is_valid(gColor)) gColor[pixel] = float4(color, 1.f) / numReSTIRInstances;
            if (is_valid(gEmission)) gEmission[pixel] = float4(float3(test_Li),1.f);//float4(emission, 1.f) / numReSTIRInstances;
            if (is_valid(gDiffuseReflectance)) gDiffuseReflectance[pixel] = float4(diffuseReflectance, 1.f) / numReSTIRInstances;
            if (is_valid(gDiffuseIllumination)) gDiffuseIllumination[pixel] = float4(diffuseIllumination, hitT) / numReSTIRInstances;
            if (is_valid(gSpecularReflectance)) gSpecularReflectance[pixel] = float4(specularReflectance, 1.f) / numReSTIRInstances;
            if (is_valid(gSpecularIllumination)) gSpecularIllumination[pixel] = float4(specularIllumination, hitT) / numReSTIRInstances;
        }
        else
        {
            if (is_valid(gColor)) gColor[pixel] += float4(color, 1.f) / numReSTIRInstances;
            if (is_valid(gEmission)) gEmission[pixel] += float4(emission, 1.f) / numReSTIRInstances;
            if (is_valid(gDiffuseReflectance)) gDiffuseReflectance[pixel] += float4(diffuseReflectance, 1.f) / numReSTIRInstances;
            if (is_valid(gDiffuseIllumination)) gDiffuseIllumination[pixel] += float4(diffuseIllumination, hitT) / numReSTIRInstances;
            if (is_valid(gSpecularReflectance)) gSpecularReflectance[pixel] += float4(specularReflectance, 1.f) / numReSTIRInstances;
            if (is_valid(gSpecularIllumination)) gSpecularIllumination[pixel] += float4(specularIllumination, hitT) / numReSTIRInstances;
        }
    }
};

cbuffer CB
{
    FinalShading gFinalShading;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gFinalShading.execute(dispatchThreadId.xy);
}
