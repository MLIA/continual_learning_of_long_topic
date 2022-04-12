import sys
import tqdm
import requests 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', dest='url', type=str,
                    help='URL from download the file')
parser.add_argument('-d', dest='destination', type=str,
                    help='file to download the file')


def _download_large_file(url, filepath, file_size=None):
    r = requests.get(url, stream=True)

    initial_pos = 0

    with open(filepath, 'wb') as f:
        with tqdm.tqdm(total=file_size, unit='B',
                unit_scale=True, unit_divisor=1024,
                desc=filepath, initial=initial_pos,
                miniters=1) as pbar:
            for chunk in r.iter_content(32 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

def _get_file_size(url):
    requests_instance = requests.head(url)
    file_size = int(requests_instance.headers.get('content-length', 0))
    return file_size

def download_to(url, destination):
    file_size = _get_file_size(url)
    _download_large_file(url, destination, file_size=file_size)

if __name__ == "__main__":
    args = parser.parse_args()
    url, destination = args.url, args.destination
    file_size = _get_file_size(url)
    _download_large_file(url, destination, file_size=file_size)
