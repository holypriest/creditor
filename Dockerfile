FROM ubuntu:latest

ENV CREDITOR_MAIN_MODEL askr.pkl
ENV CREDITOR_MAIN_DATAFILE  training_set.parquet
ENV PYTHONPATH $PYTHONPATH:/deploy/creditor

RUN apt-get update && apt-get install -y python python-dev python-pip gunicorn supervisor nginx

RUN mkdir -p /deploy/creditor
COPY . /deploy/creditor
WORKDIR /deploy/creditor
RUN pip install -r /deploy/creditor/requirements.txt

RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/sites-available/
RUN ln -s /etc/nginx/sites-available/nginx.conf /etc/nginx/sites-enabled/nginx.conf
RUN echo "daemon off;" >> /etc/nginx/nginx.conf

RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
