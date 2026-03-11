import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

GSTREAMER_PACKAGE = "Gst"
GSTREAMER_VERSION = "1.0"

CONFIDENCE_THRESHOLD = 0.25
LABELS = ["person", "luggage"]
MAX_DETECTIONS = 300
DETECTION_VALUES_COUNT = 6
OUTPUT_LAYER_INDEX = 0

SOURCE_URI = "file:///home/max/source/AVSS_E2.avi"

MUXER_WIDTH = 1280
MUXER_HEIGHT = 1280
TRACKER_WIDTH = 640
TRACKER_HEIGHT = 640
MUXER_BATCH_SIZE = 1
MUXER_SINK_PAD_NAME = "sink_0"
OSD_PROCESS_MODE = 1

PRIMARY_INFER_CONFIG_FILE = "config_infer_primary.txt"
TRACKER_LIB_FILE = (
    "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so"
)
TRACKER_CONFIG_FILE = "config_tracker_NvDeepSORT.yml"

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
QUEUE = "queue"

PROPERTY_URI = "uri"
PROPERTY_WIDTH = "width"
PROPERTY_HEIGHT = "height"
PROPERTY_BATCH_SIZE = "batch-size"
PROPERTY_CONFIG_FILE_PATH = "config-file-path"
PROPERTY_LL_LIB_FILE = "ll-lib-file"
PROPERTY_LL_CONFIG_FILE = "ll-config-file"
PROPERTY_TRACKER_WIDTH = "tracker-width"
PROPERTY_TRACKER_HEIGHT = "tracker-height"
PROPERTY_PROCESS_MODE = "process-mode"

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

# Untracked objects in DeepStream pyds are assigned the max 64-bit uint
UNTRACKED_OBJECT_ID = 18446744073709551615
PERSON_CLASS_ID = 0
LUGGAGE_CLASS_ID = 1

# Tracking parameters # TODO: replace these with some sort of config or args
ABANDONMENT_TIMEOUT_SECONDS = 30
OWNER_RADIUS_PX = 200
MOVEMENT_THRESHOLD_PX = 10
SPATIAL_TOLERANCE_PX = 50
CONTEXT_TTL_SECONDS = 5

COLORS = {
    "person": (0.0, 0.5, 1.0, 1.0),  # Blue
    "Attended": (0.0, 1.0, 0.0, 1.0),  # Green
    "Unattended": (1.0, 0.6, 0.0, 1.0),  # Orange
    "Abandoned": (1.0, 0.0, 0.0, 1.0),  # Red
    "default": (1.0, 1.0, 1.0, 1.0),  # White
}

DISPLAY_META_LIMIT = 16