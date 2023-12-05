#!/usr/bin/env python3

import time
import requests
import json
import io

try:
    from dave import secrets
    slack_token = secrets['slack_token']
except:
    print("Ask David about secrets.")
    pass

WORKSPACE_url = 'https://zialab.slack.com'
default_slack_channel = '#datahose'
slack_icon_emoji = ':see_no_evil:'
slack_user_name = 'labbot'

def post_message_to_slack(text, blocks = None, thread_ts = None, slack_channel = default_slack_channel, max_retries = 20):
    dt = 1
    ok = False
    num_tries = 0
    max_wait_time = 60
    try:
        if thread_ts == None:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post('https://slack.com/api/chat.postMessage', {
                    'token': slack_token,
                    'channel': slack_channel,
                    'text': text,
                    'icon_emoji': slack_icon_emoji,
                    'username': slack_user_name,
                    'blocks': json.dumps(blocks) if blocks else None
                }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        time.sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
        else:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post('https://slack.com/api/chat.postMessage', {
                    'token': slack_token,
                    'channel': slack_channel,
                    'text': text,
                    'icon_emoji': slack_icon_emoji,
                    'username': slack_user_name,
                    'thread_ts': thread_ts,
                    'blocks': json.dumps(blocks) if blocks else None
                }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        time.sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
    except:
        pass

def post_file_to_slack(text, file_name, file_bytes, file_type=None, title=None, thread_ts = None, slack_channel=default_slack_channel, max_retries=20):
    dt = 1
    ok = False
    num_tries = 0
    max_wait_time = 60
    try:
        if thread_ts == None:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post(
                'https://slack.com/api/files.upload',
                {
                    'token': slack_token,
                    'filename': file_name,
                    'channels': slack_channel,
                    'filetype': file_type,
                    'initial_comment': text,
                    'title': title
                },
                files = { 'file': file_bytes }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        time.sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
        else:
            while (not ok) and (num_tries <= max_retries):
                req = requests.post(
                'https://slack.com/api/files.upload',
                {
                    'token': slack_token,
                    'filename': file_name,
                    'channels': slack_channel,
                    'filetype': file_type,
                    'initial_comment': text,
                    'thread_ts': thread_ts,
                    'title': title
                },
                files = { 'file': file_bytes }).json()
                ok = 'error' not in req
                if not ok:
                    if req['error'] == 'ratelimited':
                        print("rate limited, waiting")
                        time.sleep(dt)
                        dt = dt*2
                        if dt >= max_wait_time:
                            dt = max_wait_time
                        num_tries += 1
                    else:
                        print("Error in request")
                        return req
                if num_tries >= max_retries:
                    print("Max retries reached.")
                    return req
            return req
    except:
        pass

def send_fig_to_slack(fig, slack_channel, info_msg, shortfname, thread_ts = None, format='png'):
    '''
    Use to send a matplotlib figure to Slack.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        A figure object from matplotlib.
    slack_channel : str
        Name of Slack channel to send to.
    info_msg : str
        A string message to be send together with the figure.
    shortfname : str
        The string that is used in Slack to refer to the image, with no extension.
    thread_ts : str, optional
        The timestamp of the thread to which to post to inside of given channel. The default is None,
        which means that the message will be posted to the channel directly.
    
    Returns
    -------
    None.
    '''
    try:
        buf = io.BytesIO()
        if format in ['jpg','jpeg','png']:
            fig.savefig(buf, format=format, dpi=200)
        elif format in ['pdf']:
            fig.savefig(buf, format=format)
        buf.seek(0)
        post_file_to_slack(info_msg, shortfname, buf.read(), slack_channel=slack_channel, thread_ts = thread_ts)
    except:
        pass

def delete_thread(link, username='labbot', verbose=False):
    '''
    Parameters
    ----------
    link: str
        Looks like https://zialab.slack.com/archives/CFEE5US5V/p1696259584464219 and can
        be obtained from the root message of the thread to be deleted.
    username: str
        The username of the bot used to post the thread messages.
    verbose: bool
        Whether to print every reply from the API request to delete each of the messages.
    
    Returns
    -------
    responses: list
        A list of dictionaries corresponding to the jsonified responses from Slack.
                
    '''
    channel_id, thread_ts = link.split('/')[-2:]
    thread_ts = thread_ts[1:]
    thread_ts = thread_ts[:-6] + '.' + thread_ts[-6:]
    delete_url = 'https://slack.com/api/chat.delete'
    responses = []
    try:
        # Fetch the replies of the thread
        response = requests.post('https://slack.com/api/conversations.replies', {
                        'token'    : slack_token,
                        'channel'  : channel_id,
                        'ts'       : thread_ts,
                        'username' : username,
                        'blocks'   : None
                    })
        response.raise_for_status()
        messages = response.json().get('messages', [])
        responses.append(response.json())
        for message in messages:
            if verbose:
                print(message)
            msg_ts = message['ts']
            response = requests.post(delete_url,
                                    {'channel'  : channel_id, 
                                     'token'    : slack_token,
                                     'username' : username, 
                                     'ts'       : msg_ts})
            response.raise_for_status()
            responses.append(response.json())
            # Sleep for a short duration to avoid hitting rate limits
            time.sleep(60/50.)
            
    except requests.exceptions.RequestException as e:
        print(f"Error deleting thread: {e}")
    return responses

def search_threads(channel_id, username, search_for, max_age_in_hours = 10**5):
    '''
    Parameters
    ----------
    channel_id : str
        The alpha-numeric string that identifies a channel.
    username : str
        Name of the user of app with privileges.
    search_for : str
        The string that the text of the returned meessages must contain.
    max_age_in_hours : float, optional
        The maximum age of the messages in hours. The default is 10**5.
    Return
    ------
    filtered_links : list
        A list of urls of the resultant messages.
    '''
    response = requests.post('https://slack.com/api/conversations.history', {
                    'token'    : slack_token,
                    'channel'  : channel_id,
                    'username' : username,
                    'blocks'   : None
                })
    response.raise_for_status()
    response = response.json()
    filtered_links = []
    for message in response['messages']:
        if search_for in message['text']:
            try:
                ts = message['thread_ts']
            except:
                ts = message['ts']
            now = time.time()
            if now - float(ts) > max_age_in_hours*3600:
                print("Skipping message because it is too old.")
                continue
            links = '%s/archives/%s/p%s' % (WORKSPACE_url, channel_id, ts.replace('.',''))
            filtered_links.append(links)
    return filtered_links
