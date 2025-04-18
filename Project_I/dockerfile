FROM ubuntu:18.04

# Update system and install dependencies
RUN apt-get update && apt-get -y upgrade && \
    apt install -y openjdk-8-jre-headless scala wget screen curl python3 python3-pip

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz && \
    tar xvf spark-2.4.3-bin-hadoop2.7.tgz && \
    mv spark-2.4.3-bin-hadoop2.7 /usr/local/spark

# Set environment variables
ENV SPARK_HOME="/usr/local/spark"
ENV PATH="${PATH}:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"
ENV SPARK_NO_DAEMONIZE="true"

# Start Spark services when the container runs
CMD $SPARK_HOME/sbin/start-master.sh && \
    $SPARK_HOME/sbin/start-slave.sh spark://sparkmaster:7077 && \
    bash
