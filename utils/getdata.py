import gzip
import urllib
import hashlib
#fileMd5, file_content = md5Gzip(urllib.urlretrieve(data['fileUrl'])[0])

def get_file(file_url,file_path,md5_hash):
    md5, file_content = md5Gzip(urllib.urlretrieve(file_url,file_path)[0])
    if file_md5 ! = md5:
        raise Exception("downloaded file {} failed the checksum".format(save_file_path))
def md5Gzip(fname):
    hash_md5 = hashlib.md5()
    file_content = None
    with gzip.open(fname, 'rb') as f:
        # Make an iterable of the file and divide into 4096 byte chunks
        # The iteration ends when we hit an empty byte string (b"")
        for chunk in iter(lambda: f.read(4096), b""):
            # Update the MD5 hash with the chunk
            hash_md5.update(chunk)
        # get file content
        f.seek(0)
        file_content = f.read()

    return hash_md5.hexdigest(), file_content