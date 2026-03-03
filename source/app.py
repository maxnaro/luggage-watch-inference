import ctypes
import os

import gi

GSTREAMER_PACKAGE = "Gst"
GSTREAMER_VERSION = "1.0"

gi.require_version(GSTREAMER_PACKAGE, GSTREAMER_VERSION)
from gi.repository import Gst, GLib  # type: ignore

import numpy as np
import pyds

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

CONFIDENCE_THRESHOLD = 0.25
LABELS = ["person", "luggage"]
MAX_DETECTIONS = 300
DETECTION_VALUES_COUNT = 6
OUTPUT_LAYER_INDEX = 0

SOURCE_URI = "file:///home/max/source/AVSS_E2.avi"

MUXER_WIDTH = 1280
MUXER_HEIGHT = 1280
MUXER_BATCH_SIZE = 1
MUXER_SINK_PAD_NAME = "sink_0"

PRIMARY_INFER_CONFIG_FILE = "config_infer_primary.txt"
TRACKER_LIB_FILE = "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CONFIG_FILE = "config_tracker_NvDCF_perf.yml"

SOURCE_ELEMENT_NAME = "source"
MUXER_ELEMENT_NAME = "muxer"
PRIMARY_INFER_ELEMENT_NAME = "primary-infer"
TRACKER_ELEMENT_NAME = "tracker"
OSD_ELEMENT_NAME = "onscreendisplay"
SINK_ELEMENT_NAME = "sink"

URI_DECODE_BIN = "uridecodebin"
NVSTREAMMUX = "nvstreammux"
NVINFER = "nvinfer"
NVTRACKER = "nvtracker"
NVDSOSD = "nvdsosd"
NVEGLGLESSINK = "nveglglessink"

PROPERTY_URI = "uri"
PROPERTY_WIDTH = "width"
PROPERTY_HEIGHT = "height"
PROPERTY_BATCH_SIZE = "batch-size"
PROPERTY_CONFIG_FILE_PATH = "config-file-path"
PROPERTY_LL_LIB_FILE = "ll-lib-file"
PROPERTY_LL_CONFIG_FILE = "ll-config-file"

PGIE_SRC_PAD_NAME = "src"
PAD_ADDED_SIGNAL = "pad-added"
BUS_MESSAGE_SIGNAL = "message"

RECT_HAS_BG_COLOR = 0
RECT_BORDER_WIDTH = 2
RECT_BORDER_COLOR = (0.0, 1.0, 0.0, 1.0)

TEXT_Y_OFFSET = 10
TEXT_FONT_NAME = "Serif"
TEXT_FONT_SIZE = 12
TEXT_FONT_COLOR = (1.0, 1.0, 1.0, 1.0)
TEXT_SET_BG_COLOR = 1
TEXT_BG_COLOR = (0.0, 0.0, 0.0, 0.5)

EOS_MESSAGE = "End of stream"
ERROR_PREFIX = "ERROR"
WARNING_PREFIX = "WARNING"
DEBUG_INFO_PREFIX = "  Debug info"
PIPELINE_STATE_PREFIX = "Pipeline state"


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
            layer = pyds.get_nvds_LayerInfo(tensor_meta, OUTPUT_LAYER_INDEX)
            ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
            detections = np.ctypeslib.as_array(ptr, shape=(MAX_DETECTIONS, DETECTION_VALUES_COUNT))

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
                rect.has_bg_color = RECT_HAS_BG_COLOR
                rect.border_width = RECT_BORDER_WIDTH
                rect.border_color.set(*RECT_BORDER_COLOR)

                label = LABELS[class_id] if class_id < len(LABELS) else str(class_id)
                obj_meta.obj_label = label

                txt = obj_meta.text_params
                txt.display_text = f"{label} {conf:.2f}"
                txt.x_offset = int(x1)
                txt.y_offset = max(0, int(y1) - TEXT_Y_OFFSET)
                txt.font_params.font_name = TEXT_FONT_NAME
                txt.font_params.font_size = TEXT_FONT_SIZE
                txt.font_params.font_color.set(*TEXT_FONT_COLOR)
                txt.set_bg_clr = TEXT_SET_BG_COLOR
                txt.text_bg_clr.set(*TEXT_BG_COLOR)

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
        URI_DECODE_BIN, SOURCE_ELEMENT_NAME
    )  # Use GstV4l2Src for live camera input
    muxer = Gst.ElementFactory.make(NVSTREAMMUX, MUXER_ELEMENT_NAME)
    primary_infer = Gst.ElementFactory.make(NVINFER, PRIMARY_INFER_ELEMENT_NAME)
    object_tracker = Gst.ElementFactory.make(NVTRACKER, TRACKER_ELEMENT_NAME)
    # TODO: Metadata probe (abandonment logic)
    osd = Gst.ElementFactory.make(NVDSOSD, OSD_ELEMENT_NAME)
    sink = Gst.ElementFactory.make(NVEGLGLESSINK, SINK_ELEMENT_NAME)

    # Set stream and muxer properties
    source.set_property(PROPERTY_URI, SOURCE_URI)  # Test video
    muxer.set_property(PROPERTY_WIDTH, MUXER_WIDTH)
    muxer.set_property(PROPERTY_HEIGHT, MUXER_HEIGHT)
    muxer.set_property(PROPERTY_BATCH_SIZE, MUXER_BATCH_SIZE)

    # Set inference and tracker configurations
    primary_infer.set_property(PROPERTY_CONFIG_FILE_PATH, os.path.join(CONFIG_DIR, PRIMARY_INFER_CONFIG_FILE))
    object_tracker.set_property(
        PROPERTY_LL_LIB_FILE,
        TRACKER_LIB_FILE,
    )
    object_tracker.set_property(PROPERTY_LL_CONFIG_FILE, os.path.join(CONFIG_DIR, TRACKER_CONFIG_FILE))

    for element in [source, muxer, primary_infer, object_tracker, osd, sink]:
        pipeline.add(element)

    muxer.link(primary_infer)
    primary_infer.link(object_tracker)
    object_tracker.link(osd)
    osd.link(sink)

    # Attach probe to parse YOLO output tensors into object metadata
    pgie_src_pad = primary_infer.get_static_pad(PGIE_SRC_PAD_NAME)
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, 0)

    def on_pad_added(src, new_pad):
        sink_pad = muxer.request_pad_simple(MUXER_SINK_PAD_NAME)
        new_pad.link(sink_pad)

    source.connect(PAD_ADDED_SIGNAL, on_pad_added)

    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus_message(bus, msg):
        msg_type = msg.type
        match msg_type:
            case Gst.MessageType.EOS:
                print(EOS_MESSAGE)
                loop.quit()
            case Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                print(f"{ERROR_PREFIX} from {msg.src.get_name()}: {err.message}")
                if debug:
                    print(f"{DEBUG_INFO_PREFIX}: {debug}")
                loop.quit()
            case Gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                print(f"{WARNING_PREFIX} from {msg.src.get_name()}: {err.message}")
                if debug:
                    print(f"{DEBUG_INFO_PREFIX}: {debug}")
            case Gst.MessageType.STATE_CHANGED:
                if msg.src == pipeline:
                    old, new, pending = msg.parse_state_changed()
                    print(f"{PIPELINE_STATE_PREFIX}: {old.value_nick} -> {new.value_nick}")

    bus.connect(BUS_MESSAGE_SIGNAL, on_bus_message)

    loop.run()

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
