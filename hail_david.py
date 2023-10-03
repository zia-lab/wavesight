#!/usr/bin/env python3

import http.client, urllib
import sys
import argparse
try:
  from dave import secrets
  push_token = secrets['pushover_token']
except:
  print("Ask David about secrets.")

def send_message(message, app='Ringer'):
    token_dict = {'Ringer': push_token}
    app_token = token_dict[app]
    conn = http.client.HTTPSConnection("api.pushover.net",443)
    endpoint = "/1/messages.json"
    conn.request("POST", endpoint,
      urllib.parse.urlencode({
        "token": app_token,
        "user": "uqhx6qfvn87dtfz5dhk71hf2xh1iwu",
        "message": message,
      }), { "Content-type": "application/x-www-form-urlencoded" })
    return conn.getresponse().read().decode()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send me a message.')
    parser.add_argument('message', type=str, help='Message to be sent.')
    args = parser.parse_args()
    send_message(args.message)