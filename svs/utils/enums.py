from monai.utils.enums import StrEnum
from monai.utils import look_up_option


class FlowDirection(StrEnum):
    FORWARD = "forward"
    BACKWARD = "backward"


# flow_dir = look_up_option("forward", FlowDirection)
# print(flow_dir)
# print(flow_dir == FlowDirection.FORWARD)
# print(flow_dir == "forward")
# print("forward" == FlowDirection.FORWARD)
