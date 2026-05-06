from dataclasses import dataclass

@dataclass
class AnalysisParams:
    overwrite_suite2p: bool = False
    multivid_processing: bool = False
    use_suite2p_ROI_classifier: bool = False
    update_suite2p_iscell: bool = True
    Img_Overlay: str = "max_proj"
    return_decay_times: bool = True
    skew_threshold: float = 1.0
    compactness_threshold: float = 1.4
    baseline_correction: str = "airPLS"
    lambda_window: int = 100
    MAD_baseline_filter_threshold: float = 2.0
    peak_detection_threshold: float = 4.5
    peak_count_threshold: int = 1

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data: dict):
        return AnalysisParams(**data)