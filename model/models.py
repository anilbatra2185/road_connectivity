from model.linknet import LinkNet34, LinkNet34MTL
from model.stack_module import StackHourglassNetMTL


MODELS = {"LinkNet34MTL": LinkNet34MTL, "StackHourglassNetMTL": StackHourglassNetMTL}

MODELS_REFINE = {"LinkNet34": LinkNet34}
