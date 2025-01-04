FROM apache/airflow:2.10.4
USER root
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y git \
    && apt-get install -y --no-install-recommends openjdk-17-jdk-headless \
    && apt-get autoremove -yqq --purge \
    && apt-get clean
USER airflow
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN export JAVA_HOME
ADD requirements.txt .
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt