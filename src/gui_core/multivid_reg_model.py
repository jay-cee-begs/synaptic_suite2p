from dataclasses import dataclass, field

@dataclass
class MultiVid_Reg_Params:
    
    Treatment_No: int = 2
    equal_baseline_and_treatments: bool = True
    unequal_treatment_lengths: list = field(default_factory=list)
    treatment_length_units: str = "frames"



    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data: dict):
        return MultiVid_Reg_Params(**data)