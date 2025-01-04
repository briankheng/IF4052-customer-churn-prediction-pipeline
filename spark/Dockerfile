FROM bitnami/spark:3.5.1
USER root
ADD requirements.txt .
RUN pip install -r requirements.txt
ENV PYARROW_IGNORE_TIMEZONE=1
RUN export PYARROW_IGNORE_TIMEZONE