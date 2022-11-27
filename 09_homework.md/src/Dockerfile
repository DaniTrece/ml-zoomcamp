FROM svizor42/zoomcamp-dino-dragon-lambda:v2

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp38-cp38-linux_x86_64.whl

ENV MODEL_NAME=dino-vs-dragon-v2.tflite
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]