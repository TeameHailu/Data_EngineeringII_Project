FROM ubuntu:22.04
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install sl
ENV PATH="${PATH}:/usr/games/"
CMD ["echo", "Data Engineering-II."]
