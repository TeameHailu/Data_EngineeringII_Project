FROM ubuntu:18.04
RUN apt-get update
RUN apt-get -y upgrade
RUN apt install -y openjdk-8-jre-headless
RUN apt install -y scala
RUN apt install -y wget
RUN apt install -y screen
RUN wget https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz
RUN tar xvf spark-2.4.3-bin-hadoop2.7.tgz
RUN mv spark-2.4.3-bin-hadoop2.7/ /usr/local/spark
ENV PATH="${PATH}:$SPARK_HOME/bin"
ENV SPARK_HOME="/usr/local/spark"
ENV SPARK_NO_DAEMONIZE="true"
RUN sleep 5
CMD screen -d -m $SPARK_HOME/sbin/start-master.sh ;
$SPARK_HOME/sbin/start-slave.sh spark://sparkmaster:7077
