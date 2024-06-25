from monai.utils.enums import StrEnum
from monai.utils import look_up_option


class FlowDirection(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


class NNDatasetMode(StrEnum):
    COMPLETE = "complete"
    TRAINING = "train"
    VALIDATION = "val"
    TEST = "test"

# flow_dir = look_up_option("forward", FlowDirection)
# print(flow_dir)
# print(flow_dir == FlowDirection.FORWARD)
# print(flow_dir == "forward")
# print("forward" == FlowDirection.FORWARD)
