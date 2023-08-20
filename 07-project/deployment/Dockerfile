FROM python:3.10-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "./" ]

ENV RUN_ID=307dee98dcf84aeba7b600a806dec2db

EXPOSE 9696

#CMD ["python", "predict.py"]
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]