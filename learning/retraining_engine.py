from dataclasses import dataclass
@dataclass
class RetrainingEngine:
    min_new_outcomes:int=250
    def should_retrain(self,new_outcomes:int,current_expectancy_r:float):
        return new_outcomes>=self.min_new_outcomes or current_expectancy_r<0
