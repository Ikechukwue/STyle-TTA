from enum import Enum

class Scheduler_Type(Enum):
    DDIM = "ddim"
    EULER = "euler"
    LCM = "lcm"

class Model_Type(Enum):
    SDXL = "sdxl"
    SDXL_Turbo = "sdxl_turbo"
    LCM_SDXL = "lcm_sdxl"
    SD15 = "sd15"
    SD14 = "sd14"
    SD21 = "sd21"
    SD21_Turbo = "sd21_turbo"
    SD3 = "sd3"
