import numpy as np


import bentoml
from bentoml.io import JSON


model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])


@svc.api(input=bentoml.io.NumpyNdarray(), output=JSON())
async def classify(application_data):
    prediction = await model_runner.predict.async_run(application_data)
    print(prediction)
    result = prediction[0]
    
    return {"result": result}
