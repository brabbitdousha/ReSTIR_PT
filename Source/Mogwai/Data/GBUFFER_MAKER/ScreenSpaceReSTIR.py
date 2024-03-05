from falcor import *

def render_graph_ScreenSpaceReSTIRGraph():
    g = RenderGraph("ScreenSpaceReSTIR")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ScreenSpaceReSTIRPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    GBufferRT = createPass("GBufferRT")
    g.addPass(GBufferRT, "GBufferRT")
    ScreenSpaceReSTIRPass = createPass("ScreenSpaceReSTIRPass")
    g.addPass(ScreenSpaceReSTIRPass, "ScreenSpaceReSTIRPass")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0, 'operator': ToneMapOp.Linear})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("GBufferRT.vbuffer", "ScreenSpaceReSTIRPass.vbuffer")
    g.addEdge("GBufferRT.mvec", "ScreenSpaceReSTIRPass.motionVectors")
    g.addEdge("ScreenSpaceReSTIRPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    #g.markOutput("ToneMapper.dst")
    g.markOutput("ScreenSpaceReSTIRPass.debug")
    return g

ScreenSpaceReSTIRGraph = render_graph_ScreenSpaceReSTIRGraph()
try: m.addGraph(ScreenSpaceReSTIRGraph)
except NameError: None
#m.resizeSwapChain(800, 800)
m.resizeSwapChain(800, 800)
m.loadScene('rabbit/rabbit.pyscene')

#caputre
#'''
m.clock.pause()
m.frameCapture.outputDir = "D:/test_space/falcor_test/output"

frames = [0, 20, 100]
for i in range(101):
    m.renderFrame()
    if i in frames:
        m.frameCapture.baseFilename = f"new_pos-{i:04d}"
        m.frameCapture.capture()
exit()
#'''
