import requests
import subprocess
import json

from brain.common.pylog import logger


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


def httpResponse(response_data, response_ip, response_port, response_api):
    logger.info("send response to {ip}:{port}:{data}".format(
        ip = response_ip,
        port = response_port,
        data = response_data
    ))

    try:
        requests.post(
            url = "http://{ip}:{port}/{api}".format(ip = response_ip, port = response_port, api = response_api),
            data = json.dumps(response_data),
            timeout = 3)
        
    except requests.exceptions.ConnectTimeout:
        logger.warning("send response timeout!")