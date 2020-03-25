import os
from models import OutcomeModel, YardDistributionModel, TimeRunoffModel, FieldGoalModel, SmallYardDistributionModel

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# om = OutcomeModel()
# om.train_model(1)
# yd = YardDistributionModel()
# yd.train_model(1)

# om.generate_buckets()
# om.evaluate()
# yd.generate_buckets()
# yd.evaluate()

fg = FieldGoalModel()
fg.train_model()
fg.evaluate()

# sm = SmallYardDistributionModel()
# sm.train_model()
# sm.evaluate()