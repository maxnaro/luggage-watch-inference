import ctypes
import os
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import numpy as np
import pyds

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

CONFIDENCE_THRESHOLD = 0.25
LABELS = ["person", "luggage"]
MAX_DETECTIONS = 300


def pgie_src_pad_buffer_probe(pad, info, u_data):
    """Parse YOLO output tensor [1, 300, 6] and create NvDsObjectMeta entries."""
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                try:
                    l_user = l_user.next
                except StopIteration:
                    break
                continue

            tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

            # Access first (and only) output layer: [300, 6]
            layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            detections = np.ctypeslib.as_array(ptr, shape=(MAX_DETECTIONS, 6))

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                class_id = int(class_id)

                obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                obj_meta.unique_component_id = tensor_meta.unique_id
                obj_meta.confidence = float(conf)
                obj_meta.class_id = class_id

                rect = obj_meta.rect_params
                rect.left = float(x1)
                rect.top = float(y1)
                rect.width = float(x2 - x1)
                rect.height = float(y2 - y1)
                rect.has_bg_color = 0
                rect.border_width = 2
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)

                label = LABELS[class_id] if class_id < len(LABELS) else str(class_id)
                obj_meta.obj_label = label

                txt = obj_meta.text_params
                txt.display_text = f"{label} {conf:.2f}"
                txt.x_offset = int(x1)
                txt.y_offset = max(0, int(y1) - 10)
                txt.font_params.font_name = "Serif"
                txt.font_params.font_size = 12
                txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                txt.set_bg_clr = 1
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)

                pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

            try:
                l_user = l_user.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main():
    Gst.init(None)

    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make(
        "uridecodebin", "source"
    )  # Use GstV4l2Src for live camera input
    muxer = Gst.ElementFactory.make("nvstreammux", "muxer")
    primary_infer = Gst.ElementFactory.make("nvinfer", "primary-infer")
    object_tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    # TODO: Metadata probe (abandonment logic)
    osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    # Set stream and muxer properties
    source.set_property("uri", "file:///home/max/source/AVSS_E2.avi")  # Test video
    muxer.set_property("width", 1280)
    muxer.set_property("height", 1280)
    muxer.set_property("batch-size", 1)

    # Set inference and tracker configurations
    primary_infer.set_property("config-file-path", os.path.join(CONFIG_DIR, "config_infer_primary.txt"))
    object_tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
    )
    object_tracker.set_property("ll-config-file", os.path.join(CONFIG_DIR, "config_tracker_NvDCF_perf.yml"))

    for element in [source, muxer, primary_infer, object_tracker, osd, sink]:
        pipeline.add(element)

    muxer.link(primary_infer)
    primary_infer.link(object_tracker)
    object_tracker.link(osd)
    osd.link(sink)

    # Attach probe to parse YOLO output tensors into object metadata
    pgie_src_pad = primary_infer.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    def on_pad_added(src, new_pad):
        sink_pad = muxer.request_pad_simple("sink_0")
        new_pad.link(sink_pad)

    source.connect("pad-added", on_pad_added)

    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus_message(bus, msg):
        msg_type = msg.type
        match msg_type:
            case Gst.MessageType.EOS:
                print("End of stream")
                loop.quit()
            case Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"ERROR from {msg.src.get_name()}: {err.message}")
                if debug:
                    print(f"  Debug info: {debug}")
                loop.quit()
            case Gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                print(f"WARNING from {msg.src.get_name()}: {err.message}")
                if debug:
                    print(f"  Debug info: {debug}")
            case Gst.MessageType.STATE_CHANGED:
                if msg.src == pipeline:
                    old, new, pending = msg.parse_state_changed()
                    print(f"Pipeline state: {old.value_nick} -> {new.value_nick}")

    bus.connect("message", on_bus_message)

    loop.run()

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
