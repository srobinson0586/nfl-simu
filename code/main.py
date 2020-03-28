from models import OutcomeModel, YardDistributionModel, TimeRunoffModel, FieldGoalModel, SmallYardDistributionModel, PuntModel, TurnoverFieldPosModel

om = OutcomeModel()
om.train_model(1)

yd = YardDistributionModel()
yd.train_model(1)

# om.generate_buckets()
# om.evaluate()
# yd.generate_buckets()
# yd.evaluate()

fg = FieldGoalModel()
fg.train_model()
# fg.evaluate()

sm = SmallYardDistributionModel()
sm.train_model()
# sm.evaluate()

pm = PuntModel()
pm.train_model(50)
#pm.evaluate()

tr = TimeRunoffModel()
tr.train_model(1)
# tr.generate_buckets()
# tr.evaluate()

to = TurnoverFieldPosModel()
to.train_model(1)
# to.generate_buckets()
# to.evaluate()

om.evaluate()
yd.evaluate()
fg.evaluate()
sm.evaluate()
pm.evaluate()
tr.evaluate()
to.evaluate()