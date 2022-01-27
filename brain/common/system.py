import json
import subprocess
import traceback

from tornado.httpclient import HTTPRequest, AsyncHTTPClient, HTTPError
from brain.common import pylog

@pylog.logit
def sysCommand(command: str, cwd: str = "./"):
    '''Run system command with subprocess.run and return result
    '''
    result = subprocess.run(
        command,
        shell=True,
        close_fds=True,
        cwd=cwd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    suc = (result.returncode == 0)
    out = result.stdout.decode('UTF-8', 'strict').strip()
    error = result.stderr.decode('UTF-8', 'strict').strip()

    if not suc:
        return suc, error
    else:
        return suc, out


async def HTTPPost(api: str, ip: str, port: str, data: dict):
    url = 'http://{ip}:{port}/{api}'.format(
        ip=ip,
        port=port,
        api=api,
    )
    try:
        http_client = AsyncHTTPClient()

        response = await http_client.fetch(HTTPRequest(
            url=url,
            method="POST",
            body=json.dumps(data),
        ))

    except RuntimeError as e:
        return False, "{},{}".format(e, traceback.format_exc())

    except HTTPError as e:
        return False, "{},{}".format(e, traceback.format_exc())

    except Exception as e:
        return False, "{},{}".format(e, traceback.format_exc())

    else:
        if response.code == 200:
            return True, ""
        else:
            return False, response.reason

    finally:
        http_client.close()
