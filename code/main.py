import os
from models import OutcomeModel, YardDistributionModel

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
om = OutcomeModel()
om.train_model(1)
om.generate_buckets()
om.evaluate()

yd = YardDistributionModel()
yd.train_model(1)
yd.generate_buckets()
yd.evaluate()
