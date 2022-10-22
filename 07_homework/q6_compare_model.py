import numpy as np


import bentoml
from bentoml.io import JSON

# Model 1
model_ref_1 = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
# Model 2
model_ref_2 = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")


model_1_runner = model_ref_1.to_runner()
model_2_runner = model_ref_2.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[model_1_runner])


@svc.api(input=bentoml.io.NumpyNdarray(), output=JSON())
async def classify(application_data):
    prediction = await model_1_runner.predict.async_run(application_data)
    print(prediction)
    result = prediction[0]
    
    return {"result": result}
