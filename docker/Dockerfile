FROM ipython/scipyserver

RUN apt-get update && apt-get install -y wget git vim unzip

ADD get_ipopt.sh /tmp/
ADD get_cyipopt.sh /tmp/
RUN bash /tmp/get_ipopt.sh
ENV IPOPTPATH /opt/ipopt
ENV PKG_CONFIG_PATH $PKG_CONFIG_PATH:$IPOPTPATH/lib/pkgconfig
ENV PATH $PATH:$IPOPTPATH/bin
RUN bash --login /tmp/get_cyipopt.sh
RUN rm /tmp/get_ipopt.sh /tmp/get_cyipopt.sh
