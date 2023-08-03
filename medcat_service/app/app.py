#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys

import flask_injector
import injector
from flask import Flask

from medcat_service.api import api
from medcat_service.nlp_processor import MedCatProcessor
from medcat_service.nlp_service import MedCatService, NlpService


def setup_logging():
    """
    Configure and setup a default logging handler to print messages to stdout
    """
    log_format = '[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'

    log_handler = logging.StreamHandler(sys.stdout)
    log_handler.setFormatter(logging.Formatter(fmt=log_format))
    log_handler.setLevel(level=os.getenv("APP_LOG_LEVEL", logging.INFO))

    root_logger = logging.getLogger()

    # only add the handler if a previous one does not exists
    handler_exists = False
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler) and h.level is log_handler.level:
            handler_exists = True
            break

    if not handler_exists:
        root_logger.addHandler(log_handler)

def setup_cuda():
    log = logging.getLogger("CUDA resource allocation")
    # this variabloe should be set by a post fork hook
    # variables need to be set before torch is imported
    worker_age = int(os.getenv("GUNICORN_WORKER_AGE", -1))
    cuda_device_count = int(os.getenv("APP_CUDA_DEVICE_COUNT", -1))

    if worker_age >= 0 and cuda_device_count > 0:
        # set variables for cuda resource allocation
        # Needs to be done before loading models
        # The number of devices to use should be set via
        # APP_CUDA_DEVICE_COUNT in env_app and the docker compose
        # file should allocate cards to the container
        cudaid = worker_age % cuda_device_count
        log.info("Setting cuda device " + str(cudaid))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaid)
    else:
        log.info("worker age or cuda device variables not set")


def create_app():
    """
    Creates the Flask application using the factory method
    :return: Flask application
    """
    setup_logging()

    setup_cuda()
    # create flask app and register API
    app = Flask(__name__)
    app.register_blueprint(api)

    # provide the dependent modules via dependency injection
    def configure(binder):
        binder.bind(MedCatProcessor, to=MedCatProcessor, scope=injector.singleton)
        binder.bind(NlpService, to=MedCatService, scope=injector.singleton)

    flask_injector.FlaskInjector(
        app=app,
        modules=[configure])

    # remember to return the app
    return app
