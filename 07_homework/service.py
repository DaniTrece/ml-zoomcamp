import numpy as np

from pydantic import BaseModel

import bentoml
from bentoml.io import JSON

# check that the model is correct, 
# initialize service: bentoml serve .\service.py:svc
# show models: bentoml models list
# get model: bentoml models get credit_risk_model:rtuvzlsr6wf5gha3
# crear un bentofile
# dockerizar: bentoml containerize credit_risk_model:rtuvzlsr6wf5gha3
# bentoml containerize mlzoomcamp_homework:qtzdz3slg6mwwdu5
class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str 
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int

model_ref = bentoml.xgboost.get("credit_risk_model:rtuvzlsr6wf5gha3")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])


# @svc.api(input=NumpyNdarray(shape=(-1,29), dtype=np.float, enforce_dtype=True, enforce_shape=True), output=JSON())
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
async def classify(credit_apllication):
    application_data = credit_apllication.dict()
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    result = prediction[0]

    if result > 0.5:
        return {
            "status": "DECLINED"
        }
    elif result > 0.25:
        return {
            "status": "MAYBE"
        }
    else:
        return {
            "status": "APPROVED"
        }