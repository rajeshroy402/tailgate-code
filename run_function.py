# 4.2
#Import necessary libraries
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call
import pyds

def run(input_file_path):
    global inference_output
    inference_output=[]
    Gst.init(None)

    # Create element that will form a pipeline
    print("Creating Pipeline")
    pipeline=Gst.Pipeline()
    
    source=Gst.ElementFactory.make("filesrc", "file-source")
    source.set_property('location', input_file_path)
    h264parser=Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder=Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    
    streammux=Gst.ElementFactory.make("nvstreammux", "Stream-muxer")    
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    
    pgie=Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', os.environ['SPEC_FILE'])
    
    nvvidconv1=Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd=Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv2=Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    capsfilter=Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps=Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)
    
    encoder=Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    encoder.set_property("bitrate", 2000000)
    
    sink=Gst.ElementFactory.make("filesink", 'filesink')
    sink.set_property('location', 'output.mpeg4')
    sink.set_property("sync", 1)
    
    # Add the elements to the pipeline
    print("Adding elements to Pipeline")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(sink)

    # Link the elements together
    print("Linking elements in the Pipeline")
    source.link(h264parser)
    h264parser.link(decoder)
    decoder.get_static_pad('src').link(streammux.get_request_pad("sink_0"))
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(sink)
    
    # Attach probe to OSD sink pad
    osdsinkpad=nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Create an event loop and feed gstreamer bus mesages to it
    loop=GLib.MainLoop()
    bus=pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start play back and listen to events
    print("Starting pipeline")
    
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    
    pipeline.set_state(Gst.State.NULL)
    return inference_output