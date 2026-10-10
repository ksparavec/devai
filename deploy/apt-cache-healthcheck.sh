#!/bin/bash
# devai-apt-cache's health check (compose: test ["CMD", "/usr/local/bin/apt-cache-healthcheck"]):
# one GET of the report page over bash's /dev/tcp, healthy on HTTP 200.
# A script, because podman-compose splits a CMD array's arguments on spaces,
# and not the package's `acngtool curl`, which takes about 3.5 s per call.
exec 3<>/dev/tcp/127.0.0.1/3142 || exit 1
printf 'GET /acng-report.html HTTP/1.0\r\n\r\n' >&3
IFS= read -r -t 2 status <&3 || exit 1
[[ $status == *" 200 "* ]]
