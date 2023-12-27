def render_graph_ExampleBlitPass():
    g = RenderGraph("ExampleBlitPass Example")
    ExampleBlitPass = createPass("ExampleBlitPass")
    g.addPass(ExampleBlitPass, "MyBlitPass")
    ImageLoader = createPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg',
                             'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoader, "ImageLoader")
    g.addEdge("ImageLoader.dst", "MyBlitPass.input")
    g.markOutput("MyBlitPass.output")
    return g

try: m.addGraph(render_graph_ExampleBlitPass())
except NameError: None