PATIENT_NAME_KEY = "patient"
IMAGE_KEY = "image"
TI_KEY = "ti"
TF_KEY = "tf"
TED_KEY = "ed_time"
TES_KEY = "es_time"
MI_KEY = "mi"
MF_KEY = "mf"
MES_KEY = "mes"
MED_KEY = "med"
OFFSET_KEY = "offset"
FWD_TS_KEY = "fwd_times"
BWD_TS_KEY = "bwd_times"

METADATA_SUBFIX = '_meta'
FLOW_SUBFIX = '_flow'
FWD_FLOW_KEY = "fwd" + FLOW_SUBFIX
BWD_FLOW_KEY = "bwd" + FLOW_SUBFIX

MASKS_KEYS = {MI_KEY, MF_KEY, MED_KEY, MES_KEY}
FLOWS_KEYS = {FWD_FLOW_KEY, BWD_FLOW_KEY}


# Loss constants
TOTAL_LOSS_KEY = "t"
SUPERVISED_LOSS_KEY = "s"
UNSUPERVISED_LOSS_KEY = "u"
PENALIZATION_LOSS_KEY = "p"
