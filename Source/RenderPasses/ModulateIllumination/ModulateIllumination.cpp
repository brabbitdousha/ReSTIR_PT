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
#include "ModulateIllumination.h"
#include "RenderGraph/RenderPassHelpers.h"

namespace
{
    const char kDesc[] = "Modulate illumination pass";

    const std::string kShaderFile("RenderPasses/ModulateIllumination/ModulateIllumination.cs.slang");

    const Falcor::ChannelList kInputChannels =
    {
        { "emission",               "gEmission",                "Emission",                 true /* optional */ },
        { "diffuseReflectance",     "gDiffuseReflectance",      "Diffuse Reflectance",      true /* optional */, ResourceFormat::RGBA16Float },
        { "diffuseRadiance",        "gDiffuseRadiance",         "Diffuse Radiance",         true /* optional */, ResourceFormat::RGBA16Float },
        { "specularReflectance",    "gSpecularReflectance",     "Specular Reflectance",     true /* optional */, ResourceFormat::RGBA16Float },
        { "specularRadiance",       "gSpecularRadiance",        "Specular Radiance",        true /* optional */, ResourceFormat::RGBA16Float },
        { "residualRadiance",       "gResidualRadiance",        "Residual Radiance",        true /* optional */, ResourceFormat::RGBA16Float },
    };

    const std::string kOutput = "output";

    // Serialized parameters.
    const char kUseEmission[] = "useEmission";
    const char kUseDiffuseReflectance[] = "useDiffuseReflectance";
    const char kUseDiffuseRadiance[] = "useDiffuseRadiance";
    const char kUseSpecularReflectance[] = "useSpecularReflectance";
    const char kUseSpecularRadiance[] = "useSpecularRadiance";
    const char kUseResidualRadiance[] = "useResidualRadiance";
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" __declspec(dllexport) const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary & lib)
{
    lib.registerClass("ModulateIllumination", kDesc, ModulateIllumination::create);
}

ModulateIllumination::SharedPtr ModulateIllumination::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new ModulateIllumination(dict));
}

std::string ModulateIllumination::getDesc() { return kDesc; }

ModulateIllumination::ModulateIllumination(const Dictionary& dict)
{
    mpModulateIlluminationPass = ComputePass::create(kShaderFile, "main", Program::DefineList(), false);

    // Deserialize pass from dictionary.
    for (const auto& [key, value] : dict)
    {
        if (key == kUseEmission) mUseEmission = value;
        else if (key == kUseDiffuseReflectance) mUseDiffuseReflectance = value;
        else if (key == kUseDiffuseRadiance) mUseDiffuseRadiance = value;
        else if (key == kUseSpecularReflectance) mUseSpecularReflectance = value;
        else if (key == kUseSpecularRadiance) mUseSpecularRadiance = value;
        else if (key == kUseResidualRadiance) mUseResidualRadiance = value;
        else
        {
            logWarning("Unknown field '" + key + "' in ModulateIllumination dictionary");
        }
    }
}

Falcor::Dictionary ModulateIllumination::getScriptingDictionary()
{
    Dictionary dict;
    dict[kUseEmission] = mUseEmission;
    dict[kUseDiffuseReflectance] = mUseDiffuseReflectance;
    dict[kUseDiffuseRadiance] = mUseDiffuseRadiance;
    dict[kUseSpecularReflectance] = mUseSpecularReflectance;
    dict[kUseSpecularRadiance] = mUseSpecularRadiance;
    dict[kUseResidualRadiance] = mUseResidualRadiance;
    return dict;
}

RenderPassReflection ModulateIllumination::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    // TODO: Allow user to specify output format
    reflector.addOutput(kOutput, "output").bindFlags(ResourceBindFlags::UnorderedAccess).format(ResourceFormat::RGBA32Float);
    return reflector;
}

void ModulateIllumination::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void ModulateIllumination::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    Program::DefineList defineList = getValidResourceDefines(kInputChannels, renderData);

    // Override defines.
    if (!mUseEmission) defineList["is_valid_gEmission"] = "0";
    if (!mUseDiffuseReflectance) defineList["is_valid_gDiffuseReflectance"] = "0";
    if (!mUseDiffuseRadiance) defineList["is_valid_gDiffuseRadiance"] = "0";
    if (!mUseSpecularReflectance) defineList["is_valid_gSpecularReflectance"] = "0";
    if (!mUseSpecularRadiance) defineList["is_valid_gSpecularRadiance"] = "0";
    if (!mUseResidualRadiance) defineList["is_valid_gResidualRadiance"] = "0";

    if (mpModulateIlluminationPass->getProgram()->addDefines(defineList))
    {
        mpModulateIlluminationPass->setVars(nullptr);
    }

    mpModulateIlluminationPass["CB"]["frameDim"] = mFrameDim;

    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            mpModulateIlluminationPass[desc.texname] = renderData[desc.name]->asTexture();
        }
    };
    for (const auto& channel : kInputChannels) bind(channel);

    mpModulateIlluminationPass["gOutput"] = renderData[kOutput]->asTexture();

    mpModulateIlluminationPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
}

void ModulateIllumination::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Emission", mUseEmission);
    widget.checkbox("Diffuse Reflectance", mUseDiffuseReflectance);
    widget.checkbox("Diffuse Radiance", mUseDiffuseRadiance);
    widget.checkbox("Specular Reflectance", mUseSpecularReflectance);
    widget.checkbox("Specular Radiance", mUseSpecularRadiance);
    widget.checkbox("Residual Radiance", mUseResidualRadiance);
}
