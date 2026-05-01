from dataclasses import dataclass, field
from gui_core.analysis_model import AnalysisParams
from gui_core.multivid_reg_model import MultiVid_Reg_Params
@dataclass
class GenSettings:
    main_folder: str = ""
    data_extension: str = ""
    frame_rate: int = 20
    ops_path: str = ""
    groups: list = field(default_factory=list)
    exp_condition: dict = field(default_factory=dict)
    bin_width: int = 5
    experiment_duration: int = 180
    analysis_params: AnalysisParams = field(default_factory=AnalysisParams)
    multivid_params: MultiVid_Reg_Params = field(default_factory = MultiVid_Reg_Params)

    def to_dict(self):
        return {
            "general_settings": {
                "main_folder": self.main_folder,
                "data_extension": self.data_extension,
                "frame_rate": self.frame_rate,
                "ops_path": self.ops_path,
                "groups": self.groups,
                "exp_condition": self.exp_condition,
                "BIN_WIDTH": self.bin_width,
                "EXPERIMENT_DURATION": self.experiment_duration,
            },
            "analysis_params": self.analysis_params.to_dict(),
            "multivid_params": self.multivid_params.to_dict()
        }

    @staticmethod
    def from_dict(data):
        gs = data.get("general_settings", {})
        ap = data.get("analysis_params", {})
        vp = data.get("multivid_params", {})
        return GenSettings(
            main_folder=gs.get("main_folder", ""),
            data_extension=gs.get("data_extension", ""),
            frame_rate=gs.get("frame_rate", 20),
            ops_path=gs.get("ops_path", ""),
            groups=gs.get("groups", []),
            exp_condition=gs.get("exp_condition", {}),
            bin_width=gs.get("BIN_WIDTH", 5),
            experiment_duration=gs.get("EXPERIMENT_DURATION", 180),
            analysis_params=AnalysisParams.from_dict(ap),
            multivid_params = MultiVid_Reg_Params.from_dict(vp)
        )