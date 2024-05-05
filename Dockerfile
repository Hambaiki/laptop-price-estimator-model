FROM public.ecr.aws/lambda/python:3.11

COPY requirements-lf.txt ${LAMBDA_TASK_ROOT}

COPY models/gradient_boosting.pkl ${LAMBDA_TASK_ROOT}/models/
COPY models/linear_regression.pkl ${LAMBDA_TASK_ROOT}/models/
COPY models/random_forest.pkl ${LAMBDA_TASK_ROOT}/models/

RUN pip install -r requirements-lf.txt

COPY app/ ${LAMBDA_TASK_ROOT}/app/

CMD [ "app.main.handler" ]
