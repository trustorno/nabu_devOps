#!/usr/bin/env bash

docker build -t nabu .
docker tag nabu 546041135204.dkr.ecr.eu-west-1.amazonaws.com/nabu
aws ecr get-login --no-include-email --region eu-west-1 |bash
docker push 546041135204.dkr.ecr.eu-west-1.amazonaws.com/nabu
